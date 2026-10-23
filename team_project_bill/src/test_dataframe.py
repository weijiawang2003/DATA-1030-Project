from IPython.display import display


def dataframe_test(df):
    """Print shape, dtypes, missing values, duplicated rows and summary statistics of df."""
    print("shape:", df.shape)
    print("\ndtypes:\n", df.dtypes)
    print("\nmissing values:\n", df.isna().sum())
    print("\nduplicated rows:", df.duplicated().sum())
    display(df.describe())
