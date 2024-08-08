import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import seaborn as sns
import anndata
import pandas as pd
import numpy as np

# global parameters for plotting

plt.rcParams['pdf.fonttype'] = 42  # for saving PDF with changeable text
plt.rcParams['ps.fonttype'] = 42  # for saving PDF with changeable text

var_cmap = matplotlib.cm.get_cmap("YlOrRd")
var_cmap.set_bad(color='lightgrey')

cell_cmap = matplotlib.cm.get_cmap("YlGnBu")
cell_cmap.set_bad(color='lightgrey')

color_map = {"cancer": "#ca0020", "healthy": "#b2abd2", "undetermined": "lightgrey"}


def get_high_conf_cells(adata, Cdiff_sub_cells=.3):
    test_list = np.arange(0, adata.obsm["C"].shape[1])
    permutations = [(a, b) for idx, a in enumerate(test_list) for b in test_list[idx + 1:]]
    sub_cells = np.zeros(adata.obsm["C"].shape[0]).astype(bool)
    for p in permutations:
        sub_cells |= np.abs(adata.obsm["C"][:, p[0]] - adata.obsm["C"][:, p[1]]) > Cdiff_sub_cells
    return adata.obs_names[sub_cells]


def cell_subset(adata, sub_cells=None, Cdiff_sub_cells=None, cell_sort=None):
    adata = adata[sub_cells] if sub_cells is not None else adata
    if Cdiff_sub_cells is not None:
        adata = adata[get_high_conf_cells(adata, Cdiff_sub_cells)]
    if cell_sort is None:
        cell_df = pd.DataFrame(adata.obsm["C"] > .5, index=adata.obs_names)
        row_ind = cell_df.sort_values(cell_df.columns.tolist(), ascending=False).index
        adata = adata[row_ind]
    else:
        adata[cell_sort]
    return adata


def vaf_subset(adata, sub_var=None, FC_sub_var=None, cov_sub_var=None):
    adata = adata[:, sub_var] if sub_var is not None else adata
    if FC_sub_var is not None:

        V = adata.varm["V"].T
        clones = np.argmax(adata.obsm["C"], axis=1)

        test_list = np.arange(0, adata.obsm["C"].shape[1])
        permutations = [(a, b) for idx, a in enumerate(test_list) for b in test_list[idx + 1:]]

        sub_vars = np.zeros(adata.shape[1]).astype(bool)

        for p in permutations:
            sub_vars_FC = np.abs(V[p[0]] - V[p[1]]) > FC_sub_var
            sub_vars_FC &= (V[p[0]] > .2) | (V[p[1]] > .2)
            if cov_sub_var is not None:
                sub_vars_FC &= np.sum(adata[clones == p[0]].X >= 2, axis=0) > (cov_sub_var * np.sum(clones == p[0]))
                sub_vars_FC &= np.sum(adata[clones == p[1]].X >= 2, axis=0) > (cov_sub_var * np.sum(clones == p[1]))
            sub_vars |= sub_vars_FC

        adata = adata[:, sub_vars]
    return adata


def set_border(g):
    for _, spine in g.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)


def make_labels(adata):
    idx = [adata.var.chr.tolist().index(i) for i in np.unique(adata.var.chr)]
    ylab = np.repeat(None, adata.shape[1])
    ylab[idx] = np.unique(adata.var.chr)
    return ylab


def VAF_plot(adata, sub_var=None, FC_sub_var=None, cov_sub_var=None, sub_cells=None, Cdiff_sub_cells=None,
             cell_sort=None, figsize=(7, 6), y_ticks="chr", VAF="full", save_path=None):
    factor_ticks = [str(i) + " (" + adata.uns['factor_labels'][i] + ")" for i in range(adata.varm["V"].shape[1])]
    print(factor_ticks)
    print(adata.varm["V"].shape[1])

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, .1], 'width_ratios': [.1, 1]},
                                                 figsize=figsize)

    # let's plot the cells
    adata = cell_subset(adata, sub_cells, Cdiff_sub_cells, cell_sort)
    g0 = sns.heatmap(adata.obsm["C"], cmap=cell_cmap, cbar_ax=ax2, ax=ax0, yticklabels=[],  # xticklabels=factor_ticks,
                     vmin=0, vmax=1)
    set_border(g0)
    ax0.set_title("cell factors")

    ###########################
    # let's plot the variants #
    ###########################
    # subset the variants
    adata = vaf_subset(adata, sub_var, FC_sub_var, cov_sub_var)

    # create horizontal labels
    if y_ticks == "chr":
        ylab = make_labels(adata)
    elif y_ticks == "full":
        ylab = [adata[:, i].var.chr[0] + "_" + str(adata[:, i].var.pos[0]) for i in adata.var_names]
    else:
        ylab = []
    # plot
    g1 = sns.heatmap(adata.varm["V"].T, cmap=var_cmap, cbar_ax=ax2, ax=ax3, xticklabels=ylab, vmin=0, vmax=1)
    ax3.yaxis.tick_right()
    ax3.set_yticks([i + 0.5 for i in range(adata.varm["V"].shape[1])])
    ax3.set_yticklabels(factor_ticks, rotation=0)
    for _, spine in g1.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

        # plot the VAF matrix
    if VAF == "full":
        to_plot = adata.layers["ALT"].copy() / adata.X
        to_plot[adata.X < 2] = np.nan
    else:
        to_plot = adata.layers["M"].copy()
        to_plot[adata.X < 2] = np.nan
    g1 = sns.heatmap(to_plot, cmap=var_cmap, cbar_ax=ax2,
                     ax=ax1, yticklabels=[], xticklabels=[], vmin=0, vmax=1)
    ax1.set_title("VAF of selected variants")
    for _, spine in g1.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    plt.close()


def VAF_compare(adata, sub_var=None, FC_sub_var=None, cov_sub_var=None, figsize=(6, 6), save_path=None):
    # todo add description
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # subset the variants
    adata = vaf_subset(adata, sub_var, FC_sub_var, cov_sub_var)

    plt.hist(adata.varm["V"], color=[color_map[i] for i in adata.uns["factor_labels"]],
             label=adata.uns["factor_labels"])
    plt.legend()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    plt.close()


def binarize_categorical(to_plot):
    if (len(to_plot.shape)) > 1:
        if to_plot.shape[1] == 1:
            to_plot = to_plot[:, 0]
    if (len(to_plot.shape)) == 1:
        if (to_plot.dtype != int) & (to_plot.dtype != float):
            to_plot = np.array(to_plot.tolist())
            labs = np.unique(to_plot)
            to_plot = pd.DataFrame([to_plot == i for i in labs], index=labs).T
        else:
            to_plot = pd.DataFrame(to_plot)
    return to_plot


def aligned_heatmap(adata, obsm=[], obs=[], sub_cells=None, Cdiff_sub_cells=None, figsize=(7, 8), save_path=None):
    n_maps = len(obsm) + len(obs) + 2

    width_ratios = np.ones(n_maps)
    width_ratios[-1] = .1  # for cbar
    fig, (axs) = plt.subplots(1, n_maps, gridspec_kw={'width_ratios': width_ratios}, figsize=figsize)

    # let's subset the cells
    adata = cell_subset(adata, sub_cells, Cdiff_sub_cells, cell_sort=None)

    # first heatmap is C if we have no cell labels, else "cell_labels"
    if "cell_labels" in adata.obs.columns:
        to_plot = binarize_categorical(np.array(adata.obs["cell_labels"]))
        to_plot.index = adata.obs_names
        row_ind = (to_plot > .5).sort_values(to_plot.columns.tolist(), ascending=False).index
        adata = adata[row_ind]
        to_plot = to_plot.loc[row_ind]
    else:
        to_plot = pd.DataFrame(adata.obsm["C"])

    g0 = sns.heatmap(to_plot, cmap=cell_cmap, cbar_ax=axs[-1], ax=axs[0], yticklabels=[], vmin=0, vmax=1)
    set_border(g0)

    ax_idx = 1

    for observation in obs:
        to_plot = binarize_categorical(np.array(adata.obs[observation]))
        g0 = sns.heatmap(to_plot, cmap=cell_cmap, cbar_ax=axs[-1], ax=axs[ax_idx], yticklabels=[], vmin=0, vmax=1)
        set_border(g0)
        ax_idx += 1

    for observation in obsm:
        to_plot = binarize_categorical(np.array(adata.obsm[observation]))
        g0 = sns.heatmap(to_plot, cmap=cell_cmap, cbar_ax=axs[-1], ax=axs[ax_idx], yticklabels=[], vmin=0, vmax=1)
        set_border(g0)
        ax_idx += 1

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    plt.close()


def umap_clones(adata, factors=False, sub_cells=None, Cdiff_sub_cells=None, figsize=None, save_path=None,
                s=5):
    # let's subset the cells
    adata = cell_subset(adata, sub_cells, Cdiff_sub_cells, cell_sort=None)
    umap = adata.obsm["umap"]

    if factors:
        to_plot = adata.obsm["C"]
        n_maps = adata.obsm["C"].shape[1]
        fig, (axs) = plt.subplots(1, n_maps, figsize=(4 * n_maps, 4) if figsize is None else figsize)

        for i in range(n_maps):
            axs[i].scatter(umap["UMAP_1"], umap["UMAP_2"], c=to_plot[:, i], s=s,
                           cmap=cell_cmap, vmax=1, vmin=0)
            # axs[i].set_yticklabels([]), axs[i].set_xticklabels([])
            axs[i].set_yticks([]), axs[i].set_xticks([])
            axs[i].set_title("factor " + str(i))
    else:
        to_plot = binarize_categorical(adata.obs["cell_labels"])
        fig, ax = plt.subplots(1, 1, figsize=(4, 4) if figsize is None else figsize)
        for i in to_plot.columns.tolist()[::-1]:
            sub = (to_plot[i] == 1).tolist()
            ax.scatter(umap["UMAP_1"].loc[sub], umap["UMAP_2"].loc[sub], c=color_map[i], s=s, label=i)
        ax.set_yticks([]), ax.set_xticks([])
        plt.legend()

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    plt.close()
