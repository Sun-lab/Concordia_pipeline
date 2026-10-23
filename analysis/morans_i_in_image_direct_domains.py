#!/usr/bin/env python
# coding: utf-8


import os
import argparse

import numpy as np
import pandas as pd

import scanpy as sc


parser = argparse.ArgumentParser(
    description="Moran's I per image for the direct-domains features")

parser.add_argument('--method', required=True, type=str,
                    choices=['extended', 'basic', 'local', 'spatial_lda', 'spatopic'],
                    help='which feature set to measure')
parser.add_argument('--n_neighbors', default=15, type=int,
                    help='neighbours of the spatial graph Moran\'s I is computed against')

MODEL_RUN_ID = "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20"
N_REPEATS = 10
EPOCH_ID = 1000

RAW_DIR = "../../data/Cords_data/raw_data"
REGION_LIST = "../../data/Cords_data/region_list.csv"

ENSEMBLE_ROOT = ("../code/results_rs_ensemble_scaled_direct_domains/"
                 + MODEL_RUN_ID + "__ensemble_" + str(N_REPEATS) + "/")
RESULTS_RS = "../code/results_rs"

SPATIAL_LDA_WEIGHT_FILE = "../../SpatialLDA_copy/results/cords/cords_topic_weights_radius_27_d_0.25.tsv"
SPATOPIC_WEIGHT_FILE = "../../SpaTopic_copy/results/cords/radius_62_sigma_8/cell_topic_weights.csv"


def load_scale_factors(graph_type):
    """the per-run standard deviations the ensemble k-means recorded"""
    runs_csv = os.path.join(ENSEMBLE_ROOT, graph_type, "ensemble_runs.csv")
    if not os.path.isfile(runs_csv):
        raise FileNotFoundError("missing " + runs_csv)

    df_runs = pd.read_csv(runs_csv)
    assert len(df_runs) == N_REPEATS

    return dict(zip(df_runs["repeat_index"], df_runs["overall_sd"]))


def load_weight_table(weight_file, sep, parse_cell_id_from_label):
    """a whole-dataset topic weight table, indexed by CELL_ID"""
    if not os.path.isfile(weight_file):
        raise FileNotFoundError("missing " + weight_file)

    df = pd.read_csv(weight_file, sep=sep)
    df = df.rename(columns={df.columns[0]: "cell_label"})

    if parse_cell_id_from_label:
        cell_id = df["cell_label"].str.replace(r"^[^.]*\.(.*)_[0-9]+$", r"\1",
                                               regex=True).astype(np.int64)
    else:
        cell_id = df["cell_label"].astype(np.int64)

    df = df.drop(columns=["cell_label"])
    df.index = cell_id

    assert not df.index.has_duplicates, "repeated cell id in " + weight_file

    return df


def features_for_image(method, region_id, cell_ids, scale_of_run, weight_table):
    """the feature matrix of one image, rows in the order of its raw file"""

    if method in ("extended", "basic", "local"):

        blocks = []

        for repeat_index in range(1, N_REPEATS + 1):

            cur_file = os.path.join(
                RESULTS_RS, MODEL_RUN_ID + "__repeat_" + str(repeat_index),
                method, "epoch_" + str(EPOCH_ID), "linear1",
                "linear1_" + region_id + ".csv")

            cur = pd.read_csv(cur_file, header=0).to_numpy(dtype=np.float64)

            if cur.shape[0] != len(cell_ids):
                raise ValueError("row count mismatch for " + region_id
                                 + " in repeat " + str(repeat_index))

            blocks.append(cur / scale_of_run[repeat_index])

        return np.concatenate(blocks, axis=1)

    return weight_table.loc[cell_ids].to_numpy(dtype=np.float64)


def main():

    args = parser.parse_args()
    print(args)

    method = args.method

    out_dir = os.path.join("figures", "morans_i_in_image_direct_domains", method)
    value_dir = os.path.join(out_dir, "moran_i_values")
    os.makedirs(value_dir, exist_ok=True)

    region_ids = pd.read_csv(REGION_LIST)["region_ID"].tolist()
    print("images:", len(region_ids))

    scale_of_run = None
    weight_table = None

    if method in ("extended", "basic", "local"):
        scale_of_run = load_scale_factors(method)
        print("scale factors read for", len(scale_of_run), "runs")
    elif method == "spatial_lda":
        weight_table = load_weight_table(SPATIAL_LDA_WEIGHT_FILE, "\t", False)
        print("topic weights:", weight_table.shape)
    else:
        weight_table = load_weight_table(SPATOPIC_WEIGHT_FILE, ",", True)
        print("topic weights:", weight_table.shape)

    moran_rows = []

    for image_ind, cur_region in enumerate(region_ids):

        df_cur = pd.read_csv(os.path.join(RAW_DIR, cur_region + ".csv"), header=0)

        cur_coords = df_cur[["X", "Y"]].to_numpy(dtype=np.float64)
        cell_ids = df_cur["CELL_ID"].to_numpy()

        np_embed = features_for_image(method, cur_region, cell_ids,
                                      scale_of_run, weight_table)

        assert np_embed.shape[0] == df_cur.shape[0]

        np_embed = np_embed - np.mean(np_embed, axis=0)

        adata = sc.AnnData(cur_coords)
        adata.obsm["X_embed"] = np_embed

        sc.pp.neighbors(adata, use_rep="X", n_neighbors=args.n_neighbors)

        morans_i_values = sc.metrics.morans_i(adata, obsm="X_embed")

        morans_i_df = pd.DataFrame(morans_i_values, columns=["morans_i_value"])
        morans_i_df.to_csv(os.path.join(value_dir, "morans_i_" + cur_region + ".csv"),
                           na_rep="NA", index=False)

        moran_rows.append(morans_i_values)

        if image_ind % 100 == 0:
            print("done with index i = " + str(image_ind), flush=True)

    moran_mat = np.vstack(moran_rows)
    print("moran matrix:", moran_mat.shape, "(images x dimensions)")

    median_df = pd.DataFrame({
        "setting": method,
        "feature_dim": ["dim_" + str(i + 1) for i in range(moran_mat.shape[1])],
        "morans_i": np.nanmedian(moran_mat, axis=0)})

    median_file = os.path.join(out_dir, "morans_i_median_by_dim.csv")
    median_df.to_csv(median_file, na_rep="NA", index=False)

    print("median Moran's I over dimensions: %.3f (range %.3f to %.3f)"
          % (median_df["morans_i"].median(), median_df["morans_i"].min(),
             median_df["morans_i"].max()))
    print("saved", value_dir)
    print("saved", median_file)


if __name__ == "__main__":
    main()
