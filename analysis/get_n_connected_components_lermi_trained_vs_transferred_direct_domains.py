#!/usr/bin/env python


import os
import sys

import numpy as np
import pandas as pd
import networkx as nx

from collections import Counter
from collections import defaultdict


module_path = os.path.abspath(os.path.join("..", "code"))

if module_path not in sys.path:
    sys.path.append(module_path)

from graph_data_class import load_cell_data
from graph_data_class import build_graph_from_cell_coords

from data_utilities import data_features, make_model_run_id


figure_dir = "figures"
os.makedirs(figure_dir, exist_ok=True)

data_name = "lermi_2025"
data_dir = "../../data/Lermi_data"
graph_type = "extended"
group_scheme = "default"
dist_cutoff = 16
expanded_edge_cutoff = 3 * dist_cutoff
top_k = 4
ctg_comp_dist_cutoff = 0.176
degree_limit = 20

model_run_id = make_model_run_id(data_name, group_scheme, dist_cutoff, expanded_edge_cutoff,
                                 top_k, ctg_comp_dist_cutoff, degree_limit)

source_list = [
    {"tag": "lermi_trained",
     "label": "trained on Lermi",
     "csv": ("../code/results_rs_ensemble_scaled_direct_domains/"
             "lermi_2025__group_default__base16__rad48__topk4__ctg0p176__deg20__ensemble_10/"
             "extended/domains_k10/domain_annotation_for_cells.csv")},
    {"tag": "cords_transferred",
     "label": "transferred from Cords",
     "csv": ("../code/results_rs_transfer/"
             "lermi_2025__group_default__base16__rad48__topk4__ctg0p176__deg20/"
             "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20__ensemble_10/"
             "extended/domains_k10/domain_annotation_for_cells.csv")},
]


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
print("basic graph edge cutoff: " + str(neighbor_edge_cutoff))
print("a domain needs at least " + str(n_cells_threshold) + " cells in an image to count")


cell_to_domain_of = {}

for source in source_list:
    df_domain = pd.read_csv(source["csv"], header=0)
    assert not df_domain["CELL_ID"].duplicated().any()
    cell_to_domain_of[source["tag"]] = dict(zip(df_domain["CELL_ID"].tolist(),
                                                df_domain["domain"].tolist()))
    print("read " + str(len(df_domain)) + " cells, " + source["label"] +
          ", labels " + str(sorted(df_domain["domain"].unique())))


avg_rows = {source["tag"]: [] for source in source_list}
by_domain_rows = {source["tag"]: [] for source in source_list}

for region_id in train_images:

    cell_data = load_cell_data(raw_dir + f"/{region_id}.csv")

    G, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, neighbor_edge_cutoff)

    for source in source_list:

        tag = source["tag"]
        cur_domains = [cell_to_domain_of[tag][x] for x in cell_data["CELL_ID"].tolist()]

        domain_counter = Counter(cur_domains)
        kept_domains = sorted(x for x in domain_counter if domain_counter[x] >= n_cells_threshold)

        domain_to_cells_dict = defaultdict(list)
        for row_i, domain in enumerate(cur_domains):
            domain_to_cells_dict[domain] += [row_i]

        n_components_list = []

        for cur_domain in kept_domains:

            cell_indices = domain_to_cells_dict[cur_domain]
            n_connected_components = nx.number_connected_components(G.subgraph(cell_indices))

            n_components_list += [n_connected_components]

            by_domain_rows[tag].append({"region_id": region_id,
                                        "domain": int(cur_domain),
                                        "n_cells": len(cell_indices),
                                        "n_connected_components": n_connected_components,
                                        "cells_per_component": len(cell_indices) / n_connected_components})

        avg_rows[tag].append({"region_id": region_id,
                              "n_cells": len(cell_data),
                              "n_domains_kept": len(kept_domains),
                              "avg_n_connected_components":
                                  np.mean(n_components_list) if n_components_list else np.nan})

    print("finished " + region_id)


df_avg_of = {}

for source in source_list:

    tag = source["tag"]

    output_csv = os.path.join(figure_dir, "n_connected_components_lermi_" + tag + "_direct_domains.csv")
    by_domain_csv = os.path.join(figure_dir,
                                 "n_connected_components_lermi_" + tag + "_direct_domains_by_domain.csv")

    df_avg = pd.DataFrame(avg_rows[tag])
    df_by_domain = pd.DataFrame(by_domain_rows[tag])

    df_avg.to_csv(output_csv, index=False)
    df_by_domain.to_csv(by_domain_csv, index=False)

    df_avg_of[tag] = df_avg

    avg_values = df_avg["avg_n_connected_components"].to_numpy()
    avg_values = avg_values[np.isfinite(avg_values)]

    print("")
    print("==== " + source["label"] + " ====")
    print("domains kept per image: mean " + str(df_avg["n_domains_kept"].mean()) +
          ", range " + str(df_avg["n_domains_kept"].min()) + " to " +
          str(df_avg["n_domains_kept"].max()))
    print("average number of connected components per image: mean " + str(avg_values.mean()) +
          ", median " + str(np.median(avg_values)) +
          ", range " + str(avg_values.min()) + " to " + str(avg_values.max()))
    print("over the " + str(len(df_by_domain)) + " domain-image groups: median " +
          str(df_by_domain["n_connected_components"].median()) +
          ", mean " + str(df_by_domain["n_connected_components"].mean()) +
          ", max " + str(df_by_domain["n_connected_components"].max()))
    print("median cells per connected component: " +
          str(df_by_domain["cells_per_component"].median()))
    print("saved " + output_csv)
    print("saved " + by_domain_csv)


df_side = pd.DataFrame({"region_id": train_images})
for source in source_list:
    df_cur = df_avg_of[source["tag"]]
    df_side[source["tag"] + "_n_domains_kept"] = df_cur["n_domains_kept"].to_numpy()
    df_side[source["tag"] + "_avg_n_cc"] = df_cur["avg_n_connected_components"].round(2).to_numpy()

df_side["region_id"] = df_side["region_id"].str.replace(r".*_(core\d+)$", r"\1", regex=True)

print("")
print("per image, side by side:")
print(df_side.to_string(index=False))
