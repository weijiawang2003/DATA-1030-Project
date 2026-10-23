import os
import textwrap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations
import pandas as pd


def singular_attribute_EDA(column, setting, save_folder):
    """
    Plot one column and save the figure to save_folder/<column name>.jpg

    column:  a single column, e.g. data["cloud_cover_mean"]
    setting: one entry of singular_attribute_EDA_input, a dict with
             kind        = "continuous" / "categorical" / "ordinal"
             bin_set     = continuous: [bins] or [] (default bins)
                           categorical / ordinal: category list or []
             title       = figure headline
             xlabel      = x axis label (with unit)
             ylabel      = y axis label
             log_y       = True: use log scale on the y axis
             description = caption under the figure
    save_folder: folder to save the figure in (already created)
    """
    kind = setting["kind"]
    bin_set = setting["bin_set"]

    # report style: font and font sizes
    plt.rcParams["font.family"] = "serif"
    title_size = 15
    label_size = 12
    tick_size = 10
    caption_size = 10

    fig, ax = plt.subplots(figsize = (8, 5))

    # ---------- continuous: histogram, blue ----------
    if kind == "continuous":
        color = "steelblue"
        if len(bin_set) != 0:
            # bin_set = [number of bins]
            column.plot.hist(bins = bin_set[0], ax = ax, color = color, edgecolor = "white")
        else:
            # not set: matplotlib default bins
            column.plot.hist(ax = ax, color = color, edgecolor = "white")

    # ---------- categorical: one bar per category, dark green ----------
    elif kind == "categorical":
        color = "darkgreen"
        counts = column.value_counts()
        if len(bin_set) != 0:
            # bin_set = category list, bars follow this order
            counts = counts.reindex(bin_set, fill_value = 0)
        else:
            # not set: most common category first
            counts = counts.sort_values(ascending = False)
        counts.plot.bar(ax = ax, color = color, edgecolor = "white", rot = 0)

    # ---------- ordinal: one bar per level, wine red ----------
    elif kind == "ordinal":
        color = "#722F37"
        counts = column.value_counts()
        if len(bin_set) != 0:
            # bin_set = level list from low to high, bars follow this order
            counts = counts.reindex(bin_set, fill_value = 0)
        else:
            # not set: levels sorted from low to high
            counts = counts.sort_index()
        counts.plot.bar(ax = ax, color = color, edgecolor = "white", rot = 0)

    if setting["log_y"]:
        ax.set_yscale("log")

    ax.set_title(setting["title"], fontsize = title_size, fontweight = "bold")
    ax.set_xlabel(setting["xlabel"], fontsize = label_size)
    ax.set_ylabel(setting["ylabel"], fontsize = label_size)
    ax.tick_params(axis = "both", labelsize = tick_size)

    # light horizontal grid, no top/right frame
    ax.grid(axis = "y", alpha = 0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # caption under the figure, wrapped to the figure width
    caption = textwrap.fill(setting["description"], width = 100)
    fig.text(0.5, -0.02, caption, ha = "center", va = "top", fontsize = caption_size)

    fig.savefig(os.path.join(save_folder, column.name + ".jpg"), dpi = 300, bbox_inches = "tight")
    plt.show()



def correlation_EDA(data, X_list, y_list, title, figure_name, save_folder, names):
    """
    Heatmap of |correlation| between the continuous columns X_list + y_list.
    Saved to save_folder/<figure_name>.jpg

    data:        the whole DataFrame
    X_list:      continuous feature columns, e.g. X1_continuous
    y_list:      continuous target columns, e.g. y1_continuous (one or more)
    title:       figure headline
    figure_name: file name without .jpg, e.g. "X1_y1"
    save_folder: folder to save the figure in (already created)
    names:       readable name (with unit) of each column
    """
    # check if X and y are empty
    X_empty = len(X_list) == 0
    y_empty = len(y_list) == 0
    if X_empty:
        return

    # y always comes after X
    columns = X_list + y_list
    corr = data[columns].corr().abs()

    # report style: font and font sizes
    plt.rcParams["font.family"] = "serif"
    title_size = 15
    tick_size = 9
    number_size = 7

    # color bar: blue (0) -> red (1), smooth, no white in the middle
    blue_red = LinearSegmentedColormap.from_list("blue_red", ["#2c7bb6", "#d7191c"])

    fig, ax = plt.subplots(figsize = (11, 9))
    image = ax.imshow(corr.values, cmap = blue_red, vmin = 0, vmax = 1)

    # color bar always 0-1
    color_bar = fig.colorbar(image, ax = ax, fraction = 0.046, pad = 0.04)
    color_bar.set_label("|correlation|", fontsize = 12)
    color_bar.ax.tick_params(labelsize = tick_size)

    # readable names on both axes
    labels = []
    for column in columns:
        labels.append(names[column])
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(labels, rotation = 90, fontsize = tick_size)
    ax.set_yticklabels(labels, fontsize = tick_size)

    # number in every cell
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha = "center", va = "center",
                    color = "white", fontsize = number_size)

    # thick black line between X and y (only when y is not empty)
    if not y_empty:
        split = len(X_list) - 0.5
        ax.axhline(split, color = "black", linewidth = 3)
        ax.axvline(split, color = "black", linewidth = 3)

    ax.set_title(title, fontsize = title_size, fontweight = "bold")

    fig.savefig(os.path.join(save_folder, figure_name + ".jpg"), dpi = 300, bbox_inches = "tight")
    plt.show()



class EDA00:

    def __init__(self, data, names, dates):
        self.data = data
        self.names = names
        self.dates = dates

    def XX_EDA(self, X_cont = [], X_cat = [], X_ord = [], loglist = [], folder_name = ""):
        """
        Plot the relation between every pair of X columns, save to folder_name.
        Titles and axis labels use self.names (readable name with unit).

        cont - cont: scatter        cat - cat: hist (stacked count bars)
        ord  - ord : bar            cont - cat: box
        cont - ord : box            ord  - cat: bar

        X_cont, X_cat, X_ord: continuous / categorical / ordinal X column names
        loglist:     column names that use a log axis (x or y)
                     symlog is used so that the 0 values (e.g. dry days) stay on the plot
        folder_name: folder to save the figures in, created if it does not exist
        """
        Xcont = self.data[X_cont]
        Xcat = self.data[X_cat]
        Xord = self.data[X_ord]

        # create folder_name if it does not exist yet
        os.makedirs(folder_name, exist_ok = True)

        # report style: font and font sizes
        plt.rcParams["font.family"] = "serif"
        title_size = 15
        label_size = 12
        tick_size = 10

        # ---------- cont - cont: scatter, blue ----------
        if len(X_cont) != 0:
            pairs = list(combinations(X_cont, 2))
            for pair in pairs:
                x = pair[0]
                y = pair[1]
                fig, ax = plt.subplots(figsize = (8, 6))
                ax.scatter(Xcont[x], Xcont[y], s = 4, alpha = 0.3, color = "steelblue")
                if x in loglist:
                    ax.set_xscale("symlog", linthresh = 1)
                    ax.set_xlim(left = 0)
                if y in loglist:
                    ax.set_yscale("symlog", linthresh = 1)
                    ax.set_ylim(bottom = 0)
                ax.set_title(self.names[y] + " vs " + self.names[x], fontsize = title_size, fontweight = "bold")
                ax.set_xlabel(self.names[x], fontsize = label_size)
                ax.set_ylabel(self.names[y], fontsize = label_size)
                ax.tick_params(axis = "both", labelsize = tick_size)
                ax.grid(alpha = 0.3)
                fig.savefig(os.path.join(folder_name, x + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                plt.show()
                plt.close(fig)

        # ---------- cat - cat: hist (stacked count bars) ----------
        if len(X_cat) != 0:
            pairs = list(combinations(X_cat, 2))
            for pair in pairs:
                a = pair[0]
                b = pair[1]
                counts = pd.crosstab(Xcat[a], Xcat[b])
                fig, ax = plt.subplots(figsize = (10, 6))
                counts.plot.bar(ax = ax, stacked = True, colormap = "tab20", edgecolor = "white", rot = 0)
                ax.set_title("Days by " + self.names[a] + " and " + self.names[b], fontsize = title_size, fontweight = "bold")
                ax.set_xlabel(self.names[a], fontsize = label_size)
                ax.set_ylabel("count (days)", fontsize = label_size)
                ax.tick_params(axis = "both", labelsize = tick_size)
                ax.legend(title = self.names[b], bbox_to_anchor = (1.01, 1), loc = "upper left")
                ax.grid(axis = "y", alpha = 0.3)
                ax.set_axisbelow(True)
                fig.savefig(os.path.join(folder_name, a + "__" + b + ".jpg"), dpi = 300, bbox_inches = "tight")
                plt.show()
                plt.close(fig)

        # ---------- ord - ord: bar ----------
        if len(X_ord) != 0:
            pairs = list(combinations(X_ord, 2))
            for pair in pairs:
                a = pair[0]
                b = pair[1]
                counts = pd.crosstab(Xord[a], Xord[b]).sort_index()
                fig, ax = plt.subplots(figsize = (10, 6))
                counts.plot.bar(ax = ax, colormap = "Reds", edgecolor = "white", rot = 0)
                ax.set_title("Days by " + self.names[a] + " and " + self.names[b], fontsize = title_size, fontweight = "bold")
                ax.set_xlabel(self.names[a], fontsize = label_size)
                ax.set_ylabel("count (days)", fontsize = label_size)
                ax.tick_params(axis = "both", labelsize = tick_size)
                ax.legend(title = self.names[b], bbox_to_anchor = (1.01, 1), loc = "upper left")
                ax.grid(axis = "y", alpha = 0.3)
                ax.set_axisbelow(True)
                fig.savefig(os.path.join(folder_name, a + "__" + b + ".jpg"), dpi = 300, bbox_inches = "tight")
                plt.show()
                plt.close(fig)

        # ---------- cont - cat: box, dark green ----------
        if len(X_cont) != 0 and len(X_cat) != 0:
            for cont in X_cont:
                for cat in X_cat:
                    levels = sorted(Xcat[cat].unique())
                    boxes = []
                    for level in levels:
                        boxes.append(Xcont[cont][Xcat[cat] == level])
                    fig, ax = plt.subplots(figsize = (10, 6))
                    ax.boxplot(boxes, patch_artist = True,
                               boxprops = dict(facecolor = "darkgreen", alpha = 0.6),
                               medianprops = dict(color = "black"),
                               flierprops = dict(marker = ".", markersize = 2, alpha = 0.3))
                    ax.set_xticks(range(1, len(levels) + 1))
                    ax.set_xticklabels(levels)
                    if cont in loglist:
                        ax.set_yscale("symlog", linthresh = 1)
                        ax.set_ylim(bottom = 0)
                    ax.set_title(self.names[cont] + " by " + self.names[cat], fontsize = title_size, fontweight = "bold")
                    ax.set_xlabel(self.names[cat], fontsize = label_size)
                    ax.set_ylabel(self.names[cont], fontsize = label_size)
                    ax.tick_params(axis = "both", labelsize = tick_size)
                    ax.grid(axis = "y", alpha = 0.3)
                    fig.savefig(os.path.join(folder_name, cont + "__" + cat + ".jpg"), dpi = 300, bbox_inches = "tight")
                    plt.show()
                    plt.close(fig)

        # ---------- cont - ord: box, wine red ----------
        if len(X_cont) != 0 and len(X_ord) != 0:
            for cont in X_cont:
                for ord_col in X_ord:
                    levels = sorted(Xord[ord_col].unique())
                    boxes = []
                    for level in levels:
                        boxes.append(Xcont[cont][Xord[ord_col] == level])
                    fig, ax = plt.subplots(figsize = (10, 6))
                    ax.boxplot(boxes, patch_artist = True,
                               boxprops = dict(facecolor = "#722F37", alpha = 0.6),
                               medianprops = dict(color = "black"),
                               flierprops = dict(marker = ".", markersize = 2, alpha = 0.3))
                    ax.set_xticks(range(1, len(levels) + 1))
                    ax.set_xticklabels(levels)
                    if cont in loglist:
                        ax.set_yscale("symlog", linthresh = 1)
                        ax.set_ylim(bottom = 0)
                    ax.set_title(self.names[cont] + " by " + self.names[ord_col], fontsize = title_size, fontweight = "bold")
                    ax.set_xlabel(self.names[ord_col], fontsize = label_size)
                    ax.set_ylabel(self.names[cont], fontsize = label_size)
                    ax.tick_params(axis = "both", labelsize = tick_size)
                    ax.grid(axis = "y", alpha = 0.3)
                    fig.savefig(os.path.join(folder_name, cont + "__" + ord_col + ".jpg"), dpi = 300, bbox_inches = "tight")
                    plt.show()
                    plt.close(fig)

        # ---------- ord - cat: bar ----------
        if len(X_ord) != 0 and len(X_cat) != 0:
            for ord_col in X_ord:
                for cat in X_cat:
                    counts = pd.crosstab(Xord[ord_col], Xcat[cat]).sort_index()
                    fig, ax = plt.subplots(figsize = (10, 6))
                    counts.plot.bar(ax = ax, colormap = "tab20", edgecolor = "white", rot = 0)
                    ax.set_title("Days by " + self.names[ord_col] + " and " + self.names[cat], fontsize = title_size, fontweight = "bold")
                    ax.set_xlabel(self.names[ord_col], fontsize = label_size)
                    ax.set_ylabel("count (days)", fontsize = label_size)
                    ax.tick_params(axis = "both", labelsize = tick_size)
                    ax.legend(title = self.names[cat], bbox_to_anchor = (1.01, 1), loc = "upper left")
                    ax.grid(axis = "y", alpha = 0.3)
                    ax.set_axisbelow(True)
                    fig.savefig(os.path.join(folder_name, ord_col + "__" + cat + ".jpg"), dpi = 300, bbox_inches = "tight")
                    plt.show()
                    plt.close(fig)

    def Xy_EDA(self, X_cont = [], X_cat = [], X_ord = [], y_cont = [], y_cat = [], y_ord = [], loglist = [], folder_name = ""):
        """
        Plot the relation between every X column and every y column, save to folder_name.
        Titles and axis labels use self.names (readable name with unit).

        cont - cont: scatter        cat - cat: hist (stacked count bars)
        ord  - ord : bar            cont - cat: box
        cont - ord : box            ord  - cat: bar

        X_cont, X_cat, X_ord: continuous / categorical / ordinal X column names
        y_cont, y_cat, y_ord: continuous / categorical / ordinal y column names
        loglist:     column names that use a log axis (X and y names are all different)
                     symlog is used so that the 0 values (e.g. dry days) stay on the plot
        folder_name: folder to save the figures in, created if it does not exist
        """
        Xcont = self.data[X_cont]
        Xcat = self.data[X_cat]
        Xord = self.data[X_ord]
        ycont = self.data[y_cont]
        ycat = self.data[y_cat]
        yord = self.data[y_ord]

        # create folder_name if it does not exist yet
        os.makedirs(folder_name, exist_ok = True)

        # report style: font and font sizes
        plt.rcParams["font.family"] = "serif"
        title_size = 15
        label_size = 12
        tick_size = 10

        # ---------- X cont - y cont: scatter, blue ----------
        if X_cont != []:
            if y_cont != []:
                for X in X_cont:
                    for y in y_cont:
                        fig, ax = plt.subplots(figsize = (8, 6))
                        ax.scatter(Xcont[X], ycont[y], s = 4, alpha = 0.3, color = "steelblue")
                        if X in loglist:
                            ax.set_xscale("symlog", linthresh = 1)
                            ax.set_xlim(left = 0)
                        if y in loglist:
                            ax.set_yscale("symlog", linthresh = 1)
                            ax.set_ylim(bottom = 0)
                        ax.set_title(self.names[y] + " vs " + self.names[X], fontsize = title_size, fontweight = "bold")
                        ax.set_xlabel(self.names[X], fontsize = label_size)
                        ax.set_ylabel(self.names[y], fontsize = label_size)
                        ax.tick_params(axis = "both", labelsize = tick_size)
                        ax.grid(alpha = 0.3)
                        fig.savefig(os.path.join(folder_name, X + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                        plt.show()
                        plt.close(fig)

        # ---------- X cont - y cat: box of X for each y category, dark green ----------
        if X_cont != []:
            if y_cat != []:
                for X in X_cont:
                    for y in y_cat:
                        levels = sorted(ycat[y].unique())
                        boxes = []
                        for level in levels:
                            boxes.append(Xcont[X][ycat[y] == level])
                        fig, ax = plt.subplots(figsize = (10, 6))
                        ax.boxplot(boxes, patch_artist = True,
                                   boxprops = dict(facecolor = "darkgreen", alpha = 0.6),
                                   medianprops = dict(color = "black"),
                                   flierprops = dict(marker = ".", markersize = 2, alpha = 0.3))
                        ax.set_xticks(range(1, len(levels) + 1))
                        ax.set_xticklabels(levels)
                        if X in loglist:
                            ax.set_yscale("symlog", linthresh = 1)
                            ax.set_ylim(bottom = 0)
                        ax.set_title(self.names[X] + " by " + self.names[y], fontsize = title_size, fontweight = "bold")
                        ax.set_xlabel(self.names[y], fontsize = label_size)
                        ax.set_ylabel(self.names[X], fontsize = label_size)
                        ax.tick_params(axis = "both", labelsize = tick_size)
                        ax.grid(axis = "y", alpha = 0.3)
                        fig.savefig(os.path.join(folder_name, X + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                        plt.show()
                        plt.close(fig)

        # ---------- X cont - y ord: box of X for each y level, wine red ----------
        if X_cont != []:
            if y_ord != []:
                for X in X_cont:
                    for y in y_ord:
                        levels = sorted(yord[y].unique())
                        boxes = []
                        for level in levels:
                            boxes.append(Xcont[X][yord[y] == level])
                        fig, ax = plt.subplots(figsize = (10, 6))
                        ax.boxplot(boxes, patch_artist = True,
                                   boxprops = dict(facecolor = "#722F37", alpha = 0.6),
                                   medianprops = dict(color = "black"),
                                   flierprops = dict(marker = ".", markersize = 2, alpha = 0.3))
                        ax.set_xticks(range(1, len(levels) + 1))
                        ax.set_xticklabels(levels)
                        if X in loglist:
                            ax.set_yscale("symlog", linthresh = 1)
                            ax.set_ylim(bottom = 0)
                        ax.set_title(self.names[X] + " by " + self.names[y], fontsize = title_size, fontweight = "bold")
                        ax.set_xlabel(self.names[y], fontsize = label_size)
                        ax.set_ylabel(self.names[X], fontsize = label_size)
                        ax.tick_params(axis = "both", labelsize = tick_size)
                        ax.grid(axis = "y", alpha = 0.3)
                        fig.savefig(os.path.join(folder_name, X + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                        plt.show()
                        plt.close(fig)

        # ---------- X cat - y cont: box of y for each X category, dark green ----------
        if X_cat != []:
            if y_cont != []:
                for X in X_cat:
                    for y in y_cont:
                        levels = sorted(Xcat[X].unique())
                        boxes = []
                        for level in levels:
                            boxes.append(ycont[y][Xcat[X] == level])
                        fig, ax = plt.subplots(figsize = (10, 6))
                        ax.boxplot(boxes, patch_artist = True,
                                   boxprops = dict(facecolor = "darkgreen", alpha = 0.6),
                                   medianprops = dict(color = "black"),
                                   flierprops = dict(marker = ".", markersize = 2, alpha = 0.3))
                        ax.set_xticks(range(1, len(levels) + 1))
                        ax.set_xticklabels(levels)
                        if y in loglist:
                            ax.set_yscale("symlog", linthresh = 1)
                            ax.set_ylim(bottom = 0)
                        ax.set_title(self.names[y] + " by " + self.names[X], fontsize = title_size, fontweight = "bold")
                        ax.set_xlabel(self.names[X], fontsize = label_size)
                        ax.set_ylabel(self.names[y], fontsize = label_size)
                        ax.tick_params(axis = "both", labelsize = tick_size)
                        ax.grid(axis = "y", alpha = 0.3)
                        fig.savefig(os.path.join(folder_name, X + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                        plt.show()
                        plt.close(fig)

        # ---------- X cat - y cat: hist (stacked count bars) ----------
        if X_cat != []:
            if y_cat != []:
                for X in X_cat:
                    for y in y_cat:
                        counts = pd.crosstab(Xcat[X], ycat[y])
                        fig, ax = plt.subplots(figsize = (10, 6))
                        counts.plot.bar(ax = ax, stacked = True, colormap = "tab20", edgecolor = "white", rot = 0)
                        ax.set_title("Days by " + self.names[X] + " and " + self.names[y], fontsize = title_size, fontweight = "bold")
                        ax.set_xlabel(self.names[X], fontsize = label_size)
                        ax.set_ylabel("count (days)", fontsize = label_size)
                        ax.tick_params(axis = "both", labelsize = tick_size)
                        ax.legend(title = self.names[y], bbox_to_anchor = (1.01, 1), loc = "upper left")
                        ax.grid(axis = "y", alpha = 0.3)
                        ax.set_axisbelow(True)
                        fig.savefig(os.path.join(folder_name, X + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                        plt.show()
                        plt.close(fig)

        # ---------- X cat - y ord: bar, y levels on the x axis ----------
        if X_cat != []:
            if y_ord != []:
                for X in X_cat:
                    for y in y_ord:
                        counts = pd.crosstab(yord[y], Xcat[X]).sort_index()
                        fig, ax = plt.subplots(figsize = (10, 6))
                        counts.plot.bar(ax = ax, colormap = "tab20", edgecolor = "white", rot = 0)
                        ax.set_title("Days by " + self.names[y] + " and " + self.names[X], fontsize = title_size, fontweight = "bold")
                        ax.set_xlabel(self.names[y], fontsize = label_size)
                        ax.set_ylabel("count (days)", fontsize = label_size)
                        ax.tick_params(axis = "both", labelsize = tick_size)
                        ax.legend(title = self.names[X], bbox_to_anchor = (1.01, 1), loc = "upper left")
                        ax.grid(axis = "y", alpha = 0.3)
                        ax.set_axisbelow(True)
                        fig.savefig(os.path.join(folder_name, X + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                        plt.show()
                        plt.close(fig)

        # ---------- X ord - y cont: box of y for each X level, wine red ----------
        if X_ord != []:
            if y_cont != []:
                for X in X_ord:
                    for y in y_cont:
                        levels = sorted(Xord[X].unique())
                        boxes = []
                        for level in levels:
                            boxes.append(ycont[y][Xord[X] == level])
                        fig, ax = plt.subplots(figsize = (10, 6))
                        ax.boxplot(boxes, patch_artist = True,
                                   boxprops = dict(facecolor = "#722F37", alpha = 0.6),
                                   medianprops = dict(color = "black"),
                                   flierprops = dict(marker = ".", markersize = 2, alpha = 0.3))
                        ax.set_xticks(range(1, len(levels) + 1))
                        ax.set_xticklabels(levels)
                        if y in loglist:
                            ax.set_yscale("symlog", linthresh = 1)
                            ax.set_ylim(bottom = 0)
                        ax.set_title(self.names[y] + " by " + self.names[X], fontsize = title_size, fontweight = "bold")
                        ax.set_xlabel(self.names[X], fontsize = label_size)
                        ax.set_ylabel(self.names[y], fontsize = label_size)
                        ax.tick_params(axis = "both", labelsize = tick_size)
                        ax.grid(axis = "y", alpha = 0.3)
                        fig.savefig(os.path.join(folder_name, X + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                        plt.show()
                        plt.close(fig)

        # ---------- X ord - y cat: bar, X levels on the x axis ----------
        if X_ord != []:
            if y_cat != []:
                for X in X_ord:
                    for y in y_cat:
                        counts = pd.crosstab(Xord[X], ycat[y]).sort_index()
                        fig, ax = plt.subplots(figsize = (10, 6))
                        counts.plot.bar(ax = ax, colormap = "tab20", edgecolor = "white", rot = 0)
                        ax.set_title("Days by " + self.names[X] + " and " + self.names[y], fontsize = title_size, fontweight = "bold")
                        ax.set_xlabel(self.names[X], fontsize = label_size)
                        ax.set_ylabel("count (days)", fontsize = label_size)
                        ax.tick_params(axis = "both", labelsize = tick_size)
                        ax.legend(title = self.names[y], bbox_to_anchor = (1.01, 1), loc = "upper left")
                        ax.grid(axis = "y", alpha = 0.3)
                        ax.set_axisbelow(True)
                        fig.savefig(os.path.join(folder_name, X + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                        plt.show()
                        plt.close(fig)

        # ---------- X ord - y ord: bar ----------
        if X_ord != []:
            if y_ord != []:
                for X in X_ord:
                    for y in y_ord:
                        counts = pd.crosstab(Xord[X], yord[y]).sort_index()
                        fig, ax = plt.subplots(figsize = (10, 6))
                        counts.plot.bar(ax = ax, colormap = "Reds", edgecolor = "white", rot = 0)
                        ax.set_title("Days by " + self.names[X] + " and " + self.names[y], fontsize = title_size, fontweight = "bold")
                        ax.set_xlabel(self.names[X], fontsize = label_size)
                        ax.set_ylabel("count (days)", fontsize = label_size)
                        ax.tick_params(axis = "both", labelsize = tick_size)
                        ax.legend(title = self.names[y], bbox_to_anchor = (1.01, 1), loc = "upper left")
                        ax.grid(axis = "y", alpha = 0.3)
                        ax.set_axisbelow(True)
                        fig.savefig(os.path.join(folder_name, X + "__" + y + ".jpg"), dpi = 300, bbox_inches = "tight")
                        plt.show()
                        plt.close(fig)

    def timeseries_EDA(self, X_cont = [], X_cat = [], X_ord = [], y_cont = [], y_cat = [], y_ord = [], folder_name = ""):
        """
        Plot every X and y column over time, save to folder_name.

        cont:      line plot, blue
        cat / ord: scatter plot, dark green / wine red,
                   y axis fixed beforehand: categories sorted from small to large

        X_cont, X_cat, X_ord: continuous / categorical / ordinal X column names
        y_cont, y_cat, y_ord: continuous / categorical / ordinal y column names
        folder_name: folder to save the figures in, created if it does not exist
        Titles and axis labels use self.names (readable name with unit).
        """
        # create folder_name if it does not exist yet
        os.makedirs(folder_name, exist_ok = True)

        # dates as real dates for the x axis
        dates = pd.to_datetime(self.dates)

        # report style: font and font sizes
        plt.rcParams["font.family"] = "serif"
        title_size = 15
        label_size = 12
        tick_size = 10

        # X and y are plotted the same way
        cont_list = X_cont + y_cont
        cat_list = X_cat + y_cat
        ord_list = X_ord + y_ord

        # ---------- cont: line plot, blue ----------
        for x in cont_list:
            fig, ax = plt.subplots(figsize = (42, 5))
            ax.plot(dates, self.data[x], linewidth = 0.6, color = "steelblue")
            ax.set_title(self.names[x] + " over time", fontsize = title_size, fontweight = "bold")
            ax.set_xlabel("Date", fontsize = label_size)
            ax.set_ylabel(self.names[x], fontsize = label_size)
            ax.tick_params(axis = "both", labelsize = tick_size)
            ax.grid(alpha = 0.3)
            fig.savefig(os.path.join(folder_name, x + ".jpg"), dpi = 300, bbox_inches = "tight")
            plt.show()
            plt.close(fig)

        # ---------- cat: scatter plot, dark green ----------
        for x in cat_list:
            # y axis fixed beforehand: one row per category, sorted from small to large
            levels = sorted(self.data[x].unique())
            positions = []
            for value in self.data[x]:
                positions.append(levels.index(value))
            fig, ax = plt.subplots(figsize = (42, 5))
            ax.scatter(dates, positions, s = 4, alpha = 0.5, color = "darkgreen")
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels(levels)
            ax.set_title(self.names[x] + " over time", fontsize = title_size, fontweight = "bold")
            ax.set_xlabel("Date", fontsize = label_size)
            ax.set_ylabel(self.names[x], fontsize = label_size)
            ax.tick_params(axis = "both", labelsize = tick_size)
            ax.grid(alpha = 0.3)
            fig.savefig(os.path.join(folder_name, x + ".jpg"), dpi = 300, bbox_inches = "tight")
            plt.show()
            plt.close(fig)

        # ---------- ord: scatter plot, wine red ----------
        for x in ord_list:
            # y axis fixed beforehand: one row per level, sorted from small to large
            levels = sorted(self.data[x].unique())
            positions = []
            for value in self.data[x]:
                positions.append(levels.index(value))
            fig, ax = plt.subplots(figsize = (42, 5))
            ax.scatter(dates, positions, s = 4, alpha = 0.5, color = "#722F37")
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels(levels)
            ax.set_title(self.names[x] + " over time", fontsize = title_size, fontweight = "bold")
            ax.set_xlabel("Date", fontsize = label_size)
            ax.set_ylabel(self.names[x], fontsize = label_size)
            ax.tick_params(axis = "both", labelsize = tick_size)
            ax.grid(alpha = 0.3)
            fig.savefig(os.path.join(folder_name, x + ".jpg"), dpi = 300, bbox_inches = "tight")
            plt.show()
            plt.close(fig)
