import pandas as pd
import numpy as np
from passenger.preprocess.filter import get_pars_from_line, filter_vars


def get_raw_data(raw_prefix, filter_germline=None, filter_ref_alt_cells=None):
    ALT = pd.read_csv(raw_prefix + "ALT.csv", index_col=0)
    REF = pd.read_csv(raw_prefix + "REF.csv", index_col=0)
    meta = pd.read_csv(raw_prefix + "meta.csv", index_col=0)

    if filter_germline is not None and filter_ref_alt_cells is not None:
        REF, ALT, meta = filter_vars(REF, ALT, meta, filter_germline=filter_germline,
                                     filter_ref_alt_cells=filter_ref_alt_cells)

    return REF, ALT, meta


def get_run_data(file_prefix):
    C = np.loadtxt(file_prefix + "_C.csv", delimiter=",")
    C_std = np.loadtxt(file_prefix + "_C_std.csv", delimiter=",")
    V = np.loadtxt(file_prefix + "_V.csv", delimiter=",")
    V_std = np.loadtxt(file_prefix + "_V_std.csv", delimiter=",")
    meta = pd.read_csv(file_prefix + "_meta.csv", delimiter=",", index_col=0)
    return C, C_std, V, V_std, meta


def get_best_run(run_prefix, raw_prefix, parfile, k=2):
    best_score, best_line = 0, -1
    C, C_std, V, V_std, meta = None, None, None, None, None

    for i in np.arange(1, 7):
        filter_germline, filter_ref_alt_cells = get_pars_from_line(parfile, i)
        file_prefix = run_prefix + "_" + str(filter_germline) + "_" + str(filter_ref_alt_cells)
        C_, C_std_, V_, V_std_, meta_ = get_run_data(file_prefix)

        if V_.shape[0] > 100:
            c2_ = (C_ / np.sum(C_, axis=0))
            new_score = np.nanmean(np.abs((1 / k) - c2_))

            if best_score < new_score:
                C, C_std, V, V_std, meta = C_, C_std_, V_, V_std_, meta_
                best_score = new_score
                best_line = i

    readme = (best_line, best_score)

    filter_germline, filter_ref_alt_cells = get_pars_from_line(parfile, best_line)
    REF, ALT, _ = get_raw_data(raw_prefix, filter_germline, filter_ref_alt_cells)
    cell_assignments = pd.DataFrame(C.T, index=ALT.columns)

    return cell_assignments, C_std, V, V_std, REF, ALT, meta, readme