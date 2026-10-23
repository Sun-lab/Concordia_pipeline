#!/usr/bin/env python
# coding: utf-8


import os
import glob
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from data_utilities import data_features


parser = argparse.ArgumentParser()

parser.add_argument('--data_name', default="mpfc_2022", type=str, help='the name of the dataset to use')
parser.add_argument('--data_dir', default="../../data/mpfc_data", type=str, help='the overall folder of the dataset')
parser.add_argument('--results_dir', type=str,
                    default="../../SpatialLDA_copy/results/mpfc_16topics",
                    help='the folder holding the SpatialLDA result and components tables')
parser.add_argument('--radius', type=str, default="690",
                    help='the radius whose SpatialLDA run to read; a string, so it names the file exactly')
parser.add_argument('--n_topics', type=int, default=16,
                    help='the number of topics, which sizes both matrices and must match the tables read')
parser.add_argument('--region_index', type=int, default=0, help='which image of the sorted image list to do')


def get_dist_in_image_spatial_lda(data_name="mpfc_2022",
                                  data_dir="../../data/mpfc_data",
                                  results_dir="../../SpatialLDA_copy/results/mpfc_16topics",
                                  radius="690",
                                  n_topics=16,
                                  region_index=0):

    now = datetime.now()
    print("Start computing composition distance and physical distance: " +
          now.strftime("%H:%M:%S"))

    data_dicts = data_features(data_name, "extended",
                               group_scheme="default",
                               dist_cutoff=1,
                               expanded_edge_cutoff=1,
                               top_k=1,
                               ctg_comp_dist_cutoff=0.1,
                               degree_limit=1,
                               model_run_id=None,
                               data_dir=data_dir)

    train_images = data_dicts.train_images
    n_cells_threshold = data_dicts.n_cells_threshold
    raw_dir = data_dicts.raw_dir

    raw_basenames = sorted(os.path.basename(x)[:-4]
                           for x in glob.glob(os.path.join(raw_dir, "*.csv")))
    if sorted(train_images) != raw_basenames:
        raise ValueError("the image list from data_features is not the contents of "
                         + raw_dir + "; the placeholder graph arguments are no "
                         "longer inert")

    train_images.sort()
    cur_region = train_images[region_index]

    print("region " + str(region_index) + " of " + str(len(train_images)) +
          ": " + cur_region + ", threshold " + str(n_cells_threshold) + " cells")

    results_file = os.path.join(results_dir,
                                "mpfc_results_radius_" + str(radius) + ".tsv")
    if not os.path.isfile(results_file):
        raise FileNotFoundError("missing " + results_file)

    df_topic = pd.read_csv(results_file, sep="\t", header=0)

    for cur_column in ["sample", "cell_id", "predicted_cluster"]:
        if cur_column not in df_topic.columns:
            raise ValueError(results_file + " has no column " + cur_column)

    if not df_topic["predicted_cluster"].isin(range(n_topics)).all():
        raise ValueError("topics outside 0 to " + str(n_topics - 1) + " in " +
                         results_file + "; n_topics does not match this run")

    components_file = os.path.join(
        results_dir, "mpfc_components_normalized_radius_" + str(radius) + ".tsv")
    if not os.path.isfile(components_file):
        raise FileNotFoundError("missing " + components_file)

    df_components = pd.read_csv(components_file, sep="\t", header=0, index_col=0)

    if df_components.shape[0] != n_topics:
        raise ValueError(components_file + " has " + str(df_components.shape[0]) +
                         " topics, not " + str(n_topics))

    topic_index = [int(str(x).replace("Topic-", "")) for x in df_components.index]

    if sorted(topic_index) != list(range(n_topics)):
        raise ValueError("the topic names in " + components_file +
                         " are not Topic-0 to Topic-" + str(n_topics - 1))

    composition = np.full((n_topics, df_components.shape[1]), np.nan)
    composition[topic_index, :] = df_components.to_numpy(dtype=np.float64)

    if np.isnan(composition).any():
        raise ValueError("a topic has no composition row in " + components_file)

    row_sums = composition.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("the rows of " + components_file +
                         " do not sum to 1; they are not normalized compositions")

    print("read " + str(n_topics) + " topics over " +
          str(df_components.shape[1]) + " cell types from " + components_file)

    df_cur = pd.read_csv(os.path.join(raw_dir, cur_region + ".csv"), header=0)

    cur_coords = np.array([[x, y] for x, y in zip(df_cur["X"].tolist(),
                                                  df_cur["Y"].tolist())])
    cur_coords_mat = distance_matrix(cur_coords, cur_coords, p=2)

    df_topic_cur = df_topic[df_topic["sample"] == cur_region]

    if df_topic_cur.shape[0] == 0:
        raise ValueError("no rows for sample " + cur_region + " in " + results_file)

    topic_dict = dict(zip(df_topic_cur["cell_id"].tolist(),
                          df_topic_cur["predicted_cluster"].tolist()))

    missing = [x for x in df_cur["CELL_ID"].tolist() if x not in topic_dict]
    if len(missing) > 0:
        raise ValueError(str(len(missing)) + " cells of " + cur_region +
                         " have no topic in " + results_file +
                         ", the first being " + str(missing[0]))

    cur_topics = [topic_dict[x] for x in df_cur["CELL_ID"].tolist()]

    print("image holds " + str(len(cur_topics)) + " cells over " +
          str(len(set(cur_topics))) + " of the " + str(n_topics) + " topics")

    embed_mat = np.full((n_topics, n_topics), np.nan)
    coord_mat = np.full((n_topics, n_topics), np.nan)

    counter_cluster = Counter(cur_topics)

    for i in range(n_topics - 1):
        if counter_cluster[i] >= n_cells_threshold:
            rows_i = [x for x, y in enumerate(cur_topics) if y == i]
            for j in range(i + 1, n_topics):
                if counter_cluster[j] >= n_cells_threshold:
                    rows_j = [x for x, y in enumerate(cur_topics) if y == j]

                    embed_l2 = np.linalg.norm(composition[i] - composition[j])

                    coords_mat_sub = cur_coords_mat[rows_i][:, rows_j]
                    min_i = np.min(coords_mat_sub, axis=1)
                    min_j = np.min(coords_mat_sub, axis=0)
                    q_i = np.quantile(min_i, 0.9)
                    q_j = np.quantile(min_j, 0.9)
                    min_q = min([q_i, q_j])

                    embed_mat[i, j] = embed_l2
                    coord_mat[i, j] = min_q
                    embed_mat[j, i] = embed_l2
                    coord_mat[j, i] = min_q

    n_pairs = int(np.sum(~np.isnan(embed_mat)) / 2)
    print("wrote " + str(n_pairs) + " of the " +
          str(n_topics * (n_topics - 1) // 2) + " topic pairs; the rest are NaN " +
          "because a topic is under the " + str(n_cells_threshold) + " cell floor")

    output_dir = os.path.join(results_dir, "radius_" + str(radius))

    embed_output_dir = os.path.join(output_dir, "embedding_dist_in_image")
    coord_output_dir = os.path.join(output_dir, "coord_dist_in_image")
    os.makedirs(embed_output_dir, exist_ok=True)
    os.makedirs(coord_output_dir, exist_ok=True)

    np.savetxt(os.path.join(embed_output_dir, f"embedding_dist_{cur_region}.csv"),
               embed_mat, delimiter=",")

    np.savetxt(os.path.join(coord_output_dir, f"coord_dist_{cur_region}.csv"),
               coord_mat, delimiter=",")

    now = datetime.now()
    print("Finish computing composition distance and physical distance based on given image: " +
          now.strftime("%H:%M:%S"))


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    get_dist_in_image_spatial_lda(**vars(args))
