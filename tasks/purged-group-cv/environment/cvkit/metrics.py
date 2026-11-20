"""Group-wise AUC and cluster bootstrap. TODO: implement (see instruction)."""


def group_auc(y_true, y_score, groups):
    raise NotImplementedError


def macro_auc(y_true, y_score, groups):
    raise NotImplementedError


def cluster_bootstrap_ci(y_true, y_score, groups, n_boot=200, alpha=0.05, seed=0):
    raise NotImplementedError
