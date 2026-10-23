#!/usr/bin/env python
# coding: utf-8


import os
import sys

import numpy as np
import pandas as pd
import networkx as nx

from datetime import datetime
from collections import Counter
from collections import defaultdict


module_path = os.path.abspath(os.path.join("..", "code"))

if module_path not in sys.path:
    sys.path.append(module_path)

from graph_data_class import load_cell_data
from graph_data_class import build_graph_from_cell_coords

from data_utilities import data_features


figure_dir = "figures"
os.makedirs(figure_dir, exist_ok=True)

data_name = "cords_2024"
data_dir = "../../data/Cords_data"
graph_type = "extended"
group_scheme = "default"
dist_cutoff = 16
expanded_edge_cutoff = 3 * dist_cutoff
top_k = 4
ctg_comp_dist_cutoff = 0.176
degree_limit = 20

model_run_id = (data_name + "__group_" + group_scheme + "__base" + str(dist_cutoff) +
                "__rad" + str(expanded_edge_cutoff) + "__topk" + str(top_k) +
                "__ctg" + str(ctg_comp_dist_cutoff).replace(".", "p") +
                "__deg" + str(degree_limit))

raw_kmeans_csv = "../../results/cords_2024/extended/raw_kmeans/kmeans_cluster.csv"

output_csv = os.path.join(figure_dir, "n_connected_components_raw_kmeans.csv")
by_domain_csv = os.path.join(figure_dir, "n_connected_components_raw_kmeans_by_domain.csv")


data_dicts = data_features(data_name, graph_type,
                           group_scheme=group_scheme,
                           dist_cutoff=dist_cutoff,
                           expanded_edge_cutoff=expanded_edge_cutoff,
                           top_k=top_k,
                           ctg_comp_dist_cutoff=ctg_comp_dist_cutoff,
                           degree_limit=degree_limit,
                           model_run_id=model_run_id,
                           data_dir=data_dir)

raw_dir = data_dicts.raw_dir
n_cells_threshold = data_dicts.n_cells_threshold

neighbor_edge_cutoff = data_dicts.dist_cutoff

train_images = data_dicts.train_images

print("images: " + str(len(train_images)))
print("raw_dir: " + raw_dir)
print("basic graph edge cutoff: " + str(neighbor_edge_cutoff))
print("a domain needs at least " + str(n_cells_threshold) + " cells in an image to count")


if not os.path.isfile(raw_kmeans_csv):
    raise FileNotFoundError("missing " + raw_kmeans_csv)

df_kmeans = pd.read_csv(raw_kmeans_csv, header=0)

print("")
print("read " + str(len(df_kmeans)) + " cells from " + raw_kmeans_csv +
      ", labels " + str(sorted(df_kmeans["kmeans_cluster"].unique())))

cell_to_domain_dict = dict(zip(df_kmeans["CELL_ID"].tolist(),
                               df_kmeans["kmeans_cluster"].tolist()))


region_n_connected_components_dict = defaultdict(list)

avg_n_connected_components_list = []
n_domains_kept_list = []
n_cells_list = []

by_domain_rows = []

for idx_region, region_id in enumerate(train_images):

    cell_data_file = raw_dir + f"/{region_id}.csv"
    cell_data = load_cell_data(cell_data_file)

    G, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, neighbor_edge_cutoff)

    cur_domains = [cell_to_domain_dict[x] for x in cell_data["CELL_ID"].tolist()]

    domain_counter = Counter(cur_domains)
    existing_domains = list(set(cur_domains))
    kept_domains = sorted(x for x in existing_domains
                          if domain_counter[x] >= n_cells_threshold)

    domain_to_cells_dict = defaultdict(list)

    for row_i, domain in enumerate(cur_domains):
        domain_to_cells_dict[domain] += [row_i]

    for cur_domain in kept_domains:

        cell_indices = domain_to_cells_dict[cur_domain]
        subgraph = G.subgraph(cell_indices)
        n_connected_components = nx.number_connected_components(subgraph)

        region_n_connected_components_dict[region_id] += [n_connected_components]

        by_domain_rows.append({"region_id": region_id,
                               "domain": int(cur_domain),
                               "n_cells": len(cell_indices),
                               "n_connected_components": n_connected_components,
                               "cells_per_component": len(cell_indices) / n_connected_components})

    if len(region_n_connected_components_dict[region_id]) > 0:
        avg_n_connected_components_list += [np.mean(region_n_connected_components_dict[region_id])]
    else:
        avg_n_connected_components_list += [np.nan]

    n_domains_kept_list += [len(kept_domains)]
    n_cells_list += [len(cell_data)]

    if idx_region % 100 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"finished region {idx_region} of {len(train_images)} at {current_time}")


df_avg = pd.DataFrame({"region_id": train_images,
                       "n_cells": n_cells_list,
                       "n_domains_kept": n_domains_kept_list,
                       "avg_n_connected_components": avg_n_connected_components_list})

df_avg.to_csv(output_csv, index=False)

df_by_domain = pd.DataFrame(by_domain_rows)
df_by_domain.to_csv(by_domain_csv, index=False)


avg_values = df_avg["avg_n_connected_components"].to_numpy()
avg_values = avg_values[np.isfinite(avg_values)]

print("")
print("==== raw k-means ====")
print("images with at least one domain over the threshold: " + str(len(avg_values)) +
      " of " + str(len(train_images)))
print("domains kept per image: mean " + str(df_avg["n_domains_kept"].mean()) +
      ", range " + str(df_avg["n_domains_kept"].min()) + " to " +
      str(df_avg["n_domains_kept"].max()))
print("")
print("average number of connected components per image: mean " + str(avg_values.mean()) +
      ", median " + str(np.median(avg_values)) +
      ", range " + str(avg_values.min()) + " to " + str(avg_values.max()))
print("")
print("over the " + str(len(df_by_domain)) + " domain-image groups: median " +
      str(df_by_domain["n_connected_components"].median()) +
      ", mean " + str(df_by_domain["n_connected_components"].mean()) +
      ", max " + str(df_by_domain["n_connected_components"].max()))
print("median cells per connected component: " +
      str(df_by_domain["cells_per_component"].median()))

print("")
print("saved " + output_csv)
print("saved " + by_domain_csv)
