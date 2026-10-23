#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix

from sklearn.cluster import KMeans
from datetime import datetime

import os
import sys
import argparse

from collections import defaultdict
from collections import Counter

from data_utilities import data_features, make_model_run_id


parser = argparse.ArgumentParser(description='compute embed and phyisical space distance on concatenated embeddings')

parser.add_argument('--data_name', default="cords_2024", type=str, help='the name of the dataset to use')
parser.add_argument('--graph_type', default="extended", type=str,
                                    help='name for the feature and graph combination.',
                                    choices=['extended', 'basic', 'local'])
parser.add_argument('--region_index', type=int, default=0,
                                    help='the index of the region to compute the between-cluster distances for')
parser.add_argument('--n_kmeans_clusters', default=10, type=int,
                                    help='the number of clusters to have from kmeans')
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

def get_dist_in_image_ensemble(data_name="cords_2024", graph_type="extended", region_index=0,
                               n_kmeans_clusters=10, epoch_limit=1000,
                               data_dir="./data/Cords_data", dist_cutoff=16,
                               expanded_edge_cutoff=48, top_k=4,
                               ctg_comp_dist_cutoff=0.176, degree_limit=20,
                               group_scheme="default", model_run_id=None,
                               n_repeats=10):

    input_args = locals()
    print("input args are", input_args)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start computing embedding distance and physical distance based on given image: ", current_time)

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
    result_subfolder = data_dicts.result_subfolder
    n_cells_threshold = data_dicts.n_cells_threshold

    raw_dir = data_dicts.raw_dir

    repeat_run_ids = [result_subfolder + "__repeat_" + str(i)
                      for i in range(1, n_repeats + 1)]
    embedding_dirs = ["./results_rs/" + run_id + "/" + graph_type +
                      "/epoch_" + str(epoch_id)
                      for run_id in repeat_run_ids]

    for embedding_dir in embedding_dirs:
        if not os.path.isdir(embedding_dir + "/linear1"):
            raise FileNotFoundError("missing embedding folder: " + embedding_dir + "/linear1")

    ensemble_run_id = result_subfolder + "__ensemble_" + str(n_repeats)

    output_dir = ("./results_rs_ensemble_scaled_direct_domains/" +
                  ensemble_run_id + "/" + graph_type)

    if not os.path.isdir(output_dir):
        raise FileNotFoundError("missing " + output_dir +
                                "; run k_means_on_embeddings_rs_ensemble_scaled_direct_domains.py first")

    train_images.sort()

    runs_csv = output_dir + "/ensemble_runs.csv"
    if not os.path.isfile(runs_csv):
        raise FileNotFoundError("missing " + runs_csv +
                                "; run k_means_on_embeddings_rs_ensemble_scaled_direct_domains.py first")

    df_runs = pd.read_csv(runs_csv)
    if "overall_sd" not in df_runs.columns:
        raise ValueError(runs_csv + " has no overall_sd column, so it was not written by "
                         "the scaled k-means")

    sd_by_run = dict(zip(df_runs["model_run_id"], df_runs["overall_sd"]))

    missing_runs = [run_id for run_id in repeat_run_ids if run_id not in sd_by_run]
    if len(missing_runs) > 0:
        raise ValueError("no recorded overall_sd for: " + ", ".join(missing_runs))

    run_scales = [float(sd_by_run[run_id]) for run_id in repeat_run_ids]

    print("per-model scale factors read from " + runs_csv + ":")
    for run_id, cur_scale in zip(repeat_run_ids, run_scales):
        print("  " + run_id + ": " + str(cur_scale))


    cur_region = train_images[region_index]

    print("using per-model-scaled embeddings of " + str(n_repeats) + " runs for region " + cur_region)

    df_cur = pd.read_csv(raw_dir+"/"+cur_region+".csv",
                        header=0)

    cur_coords = np.array([[x,y] for x,y in zip(df_cur["X"].tolist(),
                                                df_cur["Y"].tolist())])
    cur_coords_mat = distance_matrix(cur_coords, cur_coords, p=2)

    region_embeddings = []
    n_embedding_columns = None

    for repeat_index, embedding_dir in enumerate(embedding_dirs):

        df_embedding = pd.read_csv(embedding_dir+"/linear1/linear1_"+cur_region+".csv",
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

        cur_block = df_embedding.to_numpy(dtype=np.float32)
        cur_block = cur_block / np.float32(run_scales[repeat_index])
        region_embeddings.append(cur_block)

    np_embed = np.concatenate(region_embeddings, axis=1)

    assert df_cur.shape[0]==np_embed.shape[0]

    print("concatenated embedding matrix shape:", np_embed.shape)

    domain_dir = os.path.join(output_dir, "domains_k" + str(n_kmeans_clusters))

    domain_csv = os.path.join(domain_dir, "domain_annotation_for_cells.csv")

    if not os.path.isfile(domain_csv):
        raise FileNotFoundError("missing " + domain_csv +
                                "; run k_means_on_embeddings_rs_ensemble_scaled_direct_domains.py first")

    df_cluster = pd.read_csv(domain_csv, header=0)

    if df_cluster["domain"].min() != 1 or df_cluster["domain"].max() != n_kmeans_clusters:
        raise ValueError("expected domains 1.." + str(n_kmeans_clusters) + " in " +
                         domain_csv + ", found " + str(df_cluster["domain"].min()) +
                         ".." + str(df_cluster["domain"].max()))

    cluster_dict = dict(zip(df_cluster["CELL_ID"].tolist(),
                            (df_cluster["domain"] - 1).tolist()))

    cur_kmeans_clusters = [cluster_dict[x] for x in df_cur["CELL_ID"].tolist()]

    embed_mat = np.full((n_kmeans_clusters, n_kmeans_clusters), np.nan)
    coord_mat = np.full((n_kmeans_clusters, n_kmeans_clusters), np.nan)

    counter_cluster = Counter(cur_kmeans_clusters)

    for i in range(n_kmeans_clusters-1):
        if counter_cluster[i] >= n_cells_threshold:
            rows_i = [x for x,y in enumerate(cur_kmeans_clusters) if y==i]
            embed_slice_i = np_embed[rows_i]
            embed_i = np.mean(embed_slice_i, axis=0)
            for j in range(i+1, n_kmeans_clusters):
                if counter_cluster[j] >= n_cells_threshold:
                    rows_j = [x for x,y in enumerate(cur_kmeans_clusters) if y==j]
                    embed_slice_j = np_embed[rows_j]
                    embed_j = np.mean(embed_slice_j, axis=0)
                    embed_l2 = np.linalg.norm(embed_i - embed_j)
                    coords_mat_sub = cur_coords_mat[rows_i][:, rows_j]
                    min_i = np.min(coords_mat_sub, axis=1)
                    min_j = np.min(coords_mat_sub, axis=0)
                    q_i = np.quantile(min_i, 0.9)
                    q_j = np.quantile(min_j, 0.9)
                    min_q = min([q_i, q_j])
                    embed_mat[i,j] = embed_l2
                    coord_mat[i,j] = min_q
                    embed_mat[j,i] = embed_l2
                    coord_mat[j,i] = min_q

    embed_output_dir = os.path.join(domain_dir, "embedding_dist_in_image")
    coord_output_dir = os.path.join(domain_dir, "coord_dist_in_image")
    os.makedirs(embed_output_dir, exist_ok=True)
    os.makedirs(coord_output_dir, exist_ok=True)

    np.savetxt(os.path.join(embed_output_dir, f"embedding_dist_{cur_region}.csv"),
            embed_mat, delimiter=",")

    np.savetxt(os.path.join(coord_output_dir, f"coord_dist_{cur_region}.csv"),
            coord_mat, delimiter=",")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Finish computing embedding distance and physical distance based on given image: ", current_time)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    get_dist_in_image_ensemble(**vars(args))
