#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from datetime import datetime

import os
import sys
import argparse

from collections import defaultdict
from collections import Counter

from data_utilities import data_features, make_model_run_id


parser = argparse.ArgumentParser(description="k-means on concatenated embeddings from repeated runs")

parser.add_argument('--data_name', default="cords_2024", type=str, help='the name of the dataset to use')
parser.add_argument('--graph_type', default="extended", type=str,
                                    help='name for the feature and graph combination.',
                                    choices=['extended', 'basic', 'local'])                               
parser.add_argument('--n_kmeans_clusters', default=40, type=int, help='the number of clusters to have from kmeans')
parser.add_argument('--epoch_limit', default=1000, type=int, help='the number of epochs to train the model for, ' \
'which determines the epoch id of the embeddings to run k-means on')
parser.add_argument('--data_dir', default="./data/Cords_data", type=str, help='the overall folder of the dataset')
parser.add_argument('--dist_cutoff', type=float, default=16, help='distance cutoff for constructing the base graph')
parser.add_argument('--expanded_edge_cutoff', type=float, default=48, help='search radius for candidate cells in the first extension step')
parser.add_argument('--top_k', type=int, default=4, help='number of nearest candidate cells to add in the first extension step')
parser.add_argument('--ctg_comp_dist_cutoff', type=float, default=0.176, help='cell type group composition distance cutoff for adding edges')
parser.add_argument('--degree_limit', type=int, default=20, help='desired average degree of nodes in each fully extended graph')
parser.add_argument('--group_scheme', type=str, default="default", help='fine-to-coarse cell type grouping scheme')
parser.add_argument('--model_run_id', type=str, default=None, help='dataset-aware identifier for this graph/model parameter set')
parser.add_argument("--n_repeats", default=10, type=int, help="number of repeated runs to concatenate")

def run_k_means_ensemble(data_name="cords_2024",
                graph_type="extended",
                n_kmeans_clusters=40,
                epoch_limit=1000,
                data_dir="./data/Cords_data",
                dist_cutoff=16,
                expanded_edge_cutoff=48,
                top_k=4,
                ctg_comp_dist_cutoff=0.176,
                degree_limit=20,
                group_scheme="default",
                model_run_id=None,
                n_repeats=10):

    input_args = locals()
    print("input args are", input_args)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start running K-means on concatenated embeddings: ", current_time)

    n_kmeans_clusters = int(n_kmeans_clusters)

    epoch_id = epoch_limit

    if model_run_id is None:
        model_run_id = make_model_run_id(data_name, group_scheme, dist_cutoff,
                                         expanded_edge_cutoff, top_k,
                                         ctg_comp_dist_cutoff, degree_limit)

    data_dicts = data_features(data_name, graph_type,
                               group_scheme=group_scheme,
                               dist_cutoff=dist_cutoff,
                               expanded_edge_cutoff=expanded_edge_cutoff,
                               top_k=top_k,
                               ctg_comp_dist_cutoff=ctg_comp_dist_cutoff,
                               degree_limit=degree_limit,
                               model_run_id=model_run_id,
                               data_dir=data_dir)

    train_images = data_dicts.train_images

    raw_dir = data_dicts.raw_dir

    repeat_run_ids = [model_run_id + "__repeat_" + str(i)
                      for i in range(1, n_repeats + 1)]
    embedding_dirs = ["./results_rs/" + run_id + "/" + graph_type +
                      "/epoch_" + str(epoch_id)
                      for run_id in repeat_run_ids]

    for embedding_dir in embedding_dirs:
        if not os.path.isdir(embedding_dir + "/linear1"):
            raise FileNotFoundError("missing embedding folder: " + embedding_dir + "/linear1")

    ensemble_run_id = model_run_id + "__ensemble_" + str(n_repeats)
    output_dir = ("./results_rs_ensemble_scaled/" + ensemble_run_id + "/" +
                  graph_type + "/clusters_k" + str(n_kmeans_clusters))
    os.makedirs(output_dir, exist_ok=True)

    print("using per-model-scaled embeddings from:")
    for run_id in repeat_run_ids:
        print("  " + run_id)
    print("writing results to " + output_dir)

    layer_chunk = []
    num_id_list = []

    for i in range(len(train_images)):

        cur_region = train_images[i]

        df_cur = pd.read_csv(raw_dir+"/"+cur_region+".csv",
                            header=0)

        num_id_list += df_cur["CELL_ID"].tolist()

        region_embeddings = []
        n_embedding_columns = None
        for repeat_index, embedding_dir in enumerate(embedding_dirs):
            df_embedding = pd.read_csv(
                embedding_dir + "/linear1/linear1_" + cur_region + ".csv",
                header=0)

            if df_cur.shape[0] != df_embedding.shape[0]:
                raise ValueError(
                    "row count mismatch for " + cur_region + " in " +
                    repeat_run_ids[repeat_index])
            if n_embedding_columns is None:
                n_embedding_columns = df_embedding.shape[1]
            elif df_embedding.shape[1] != n_embedding_columns:
                raise ValueError(
                    "embedding dimension mismatch for " + cur_region + " in " +
                    repeat_run_ids[repeat_index])

            region_embeddings.append(df_embedding.to_numpy(dtype=np.float32))

        layer_chunk.append(np.concatenate(region_embeddings, axis=1))

        if i%100 == 0:
            print("done with index i = "+str(i))


    X = np.concatenate(layer_chunk, axis=0)
    assert len(num_id_list) == X.shape[0]

    if not np.isfinite(X).all():
        raise ValueError("the concatenated embeddings contain NaN or infinite values")

    print("concatenated embedding matrix shape:", X.shape)

    run_scales = []

    for r in range(n_repeats):

        cur_block = X[:, (r * n_embedding_columns):((r + 1) * n_embedding_columns)]
        cur_scale = float(cur_block.std())

        if not np.isfinite(cur_scale) or cur_scale == 0:
            raise ValueError("degenerate standard deviation " + str(cur_scale) +
                             " for " + repeat_run_ids[r])

        cur_block /= cur_scale
        run_scales.append(cur_scale)

    print("overall standard deviation of each model, used to scale its block:")
    for run_id, cur_scale in zip(repeat_run_ids, run_scales):
        print("  " + run_id + ": " + str(cur_scale))

    current_time = datetime.now()
    print(current_time)
    k_means = KMeans(n_clusters=n_kmeans_clusters, random_state=0, n_init="auto").fit(X)
    current_time = datetime.now()
    print(current_time)

    print(k_means.inertia_)

    assert len(set(num_id_list)) == len(num_id_list), "duplicated cell IDs (even if not in the same image) are not allowed"

    k_means_pred = k_means.labels_
    k_means_pred.shape

    k_means_pred = k_means_pred.tolist()
    k_means_pred[:10]

    print(len(set(k_means_pred)))
    list(set(k_means_pred))[:10]

    max(Counter(k_means_pred).values())
    min(Counter(k_means_pred).values())

    len(num_id_list)

    num_id_list[:10]

    df_kmeans = pd.DataFrame(list(zip(num_id_list, k_means_pred)), 
                            columns=["CELL_ID", "kmeans_cluster"])
    df_kmeans.shape

    df_kmeans.to_csv(output_dir + "/kmeans_cluster.csv", index=False)

    df_runs = pd.DataFrame({"repeat_index": list(range(1, n_repeats + 1)),
                            "model_run_id": repeat_run_ids,
                            "overall_sd": run_scales})
    df_runs.to_csv(output_dir + "/ensemble_runs.csv", index=False)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Finish running K-means on concatenated embeddings: ", current_time)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    run_k_means_ensemble(**vars(args))
