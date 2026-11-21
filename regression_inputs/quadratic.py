"""Degree-two models. Generate wide polynomial designs in column blocks."""
import numpy as np
from scipy.linalg import eigh, solve
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def layout(p):
    # Same linear, square and pair terms as sklearn PolynomialFeatures.
    i, j = np.triu_indices(p)
    return np.r_[np.arange(p), i], np.r_[np.full(p,-1), j]


def terms(X, left, right, ids):
    i, j = left[ids], right[ids]
    out = X[:, i].copy()
    pair = j >= 0
    out[:, pair] *= X[:, j[pair]]
    return out


def quadratic_predict(name, A, y, B):
    p = A.shape[1]
    size = p + p*(p+1)//2
    ridge_alpha = np.logspace(-1,6,20)[11]
    if size * (len(A)+len(B)) < 12_000_000:
        poly = PolynomialFeatures(degree=2, include_bias=name == 'Poly')
        train = poly.fit_transform(A)
        valid = poly.transform(B)
        if name == 'Poly':
            return LinearRegression().fit(train, y).predict(valid)
        scale = StandardScaler()
        train, valid = scale.fit_transform(train), scale.transform(valid)
        model = (Ridge(alpha=ridge_alpha) if name == 'Poly Ridge'
                 else Lasso(alpha=.1, max_iter=10000, tol=1e-3))
        return model.fit(train, y).predict(valid)

    centered_y = y - y.mean()
    if name == 'Poly':
        # Dot product of the unscaled explicit [x_i, x_i*x_j (i<=j)] design.
        def kernel(L, R):
            dot = L @ R.T
            return dot + .5*(dot**2 + (L**2) @ (R**2).T)
        gram = kernel(A,A)
        cross = kernel(B,A)
        means, grand = gram.mean(axis=0), gram.mean()
        gram -= means[:,None] + means[None,:] - grand
        cross -= means[None,:] + cross.mean(axis=1)[:,None] - grand
        values, vectors = eigh(gram)
        keep = values > values[-1]*len(A)*np.finfo(float).eps
        dual = vectors[:,keep] @ ((vectors[:,keep].T @ centered_y)/values[keep])
        return y.mean() + cross @ dual

    left, right = layout(p)
    if name == 'Poly Ridge':
        # Each polynomial column is scaled using its training-fold moments.
        gram = np.zeros((len(A),len(A)))
        cross = np.zeros((len(B),len(A)))
        for start in range(0,size,256):
            ids = slice(start,min(size,start+256))
            block = terms(A,left,right,ids)
            scale = StandardScaler()
            block = scale.fit_transform(block)
            future = scale.transform(terms(B,left,right,ids))
            gram += block @ block.T
            cross += future @ block.T
        gram.flat[::len(A)+1] += ridge_alpha
        return y.mean() + cross @ solve(gram,centered_y,assume_a='pos')

    # Working-set Lasso: add excluded columns violating the L1 optimality
    # condition until none remain. Avoid storing all interactions at once.
    active = np.array([], dtype=int)
    residual = centered_y.copy()
    while True:
        candidates = []
        for start in range(0,size,256):
            ids = np.arange(start,min(size,start+256))
            ids = ids[~np.isin(ids,active)]
            if not len(ids):
                continue
            block = StandardScaler().fit_transform(terms(A,left,right,ids))
            corr = np.abs(block.T @ residual)/len(A)
            bad = corr > .1 + 1e-5
            candidates.extend(zip(corr[bad].tolist(),ids[bad].tolist()))
        if not candidates:
            if not len(active):
                return np.full(len(B),y.mean())
            return model.predict(scale.transform(terms(B,left,right,active)))
        candidates.sort(reverse=True)
        active = np.sort(np.r_[active,[j for _,j in candidates[:128]]])
        scale = StandardScaler()
        train = scale.fit_transform(terms(A,left,right,active))
        model = Lasso(alpha=.1, max_iter=10000,tol=1e-6).fit(train,y)
        residual = y - model.predict(train)
