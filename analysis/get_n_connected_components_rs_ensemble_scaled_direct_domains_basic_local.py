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

ensemble_run_id = model_run_id + "__ensemble_10"

direct_root = ("../code/results_rs_ensemble_scaled_direct_domains/" +
               ensemble_run_id + "/")

annotations = [
    {"tag": "basic", "label": "basic graph",
     "path": os.path.join(direct_root, "basic", "domains_k10",
                          "domain_annotation_for_cells.csv")},
    {"tag": "local", "label": "local graph",
     "path": os.path.join(direct_root, "local", "domains_k10",
                          "domain_annotation_for_cells.csv")},
]

reference_csv = os.path.join(
    figure_dir, "n_connected_components_rs_ensemble_scaled_direct_domains.csv")
reference_label = "extended graph"


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
print("basic graph edge cutoff: " + str(neighbor_edge_cutoff) +
      " (the measurement graph, the same for every annotation)")
print("a domain needs at least " + str(n_cells_threshold) + " cells in an image to count")


for annotation in annotations:

    if not os.path.isfile(annotation["path"]):
        raise FileNotFoundError(
            "missing " + annotation["path"] +
            "; run k_means_on_embeddings_rs_ensemble_scaled_direct_domains_basic_local.sh first")

    df_domain = pd.read_csv(annotation["path"], header=0)

    print("")
    print("read " + str(len(df_domain)) + " cells from " + annotation["path"] +
          ", labels " + str(sorted(df_domain["domain"].unique())))

    annotation["cell_to_domain_dict"] = dict(zip(df_domain["CELL_ID"].tolist(),
                                                 df_domain["domain"].tolist()))

    annotation["avg_n_connected_components_list"] = []
    annotation["n_domains_kept_list"] = []
    annotation["by_domain_rows"] = []

if (set(annotations[0]["cell_to_domain_dict"].keys()) !=
        set(annotations[1]["cell_to_domain_dict"].keys())):
    raise ValueError("the two annotations do not cover the same cells")


n_cells_list = []

for idx_region, region_id in enumerate(train_images):

    cell_data_file = raw_dir + f"/{region_id}.csv"
    cell_data = load_cell_data(cell_data_file)

    G, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, neighbor_edge_cutoff)

    cell_id_list = cell_data["CELL_ID"].tolist()

    n_cells_list += [len(cell_data)]

    for annotation in annotations:

        cur_domains = [annotation["cell_to_domain_dict"][x] for x in cell_id_list]

        domain_counter = Counter(cur_domains)
        existing_domains = list(set(cur_domains))
        kept_domains = sorted(x for x in existing_domains
                              if domain_counter[x] >= n_cells_threshold)

        domain_to_cells_dict = defaultdict(list)

        for row_i, domain in enumerate(cur_domains):
            domain_to_cells_dict[domain] += [row_i]

        cur_counts = []

        for cur_domain in kept_domains:

            cell_indices = domain_to_cells_dict[cur_domain]
            subgraph = G.subgraph(cell_indices)
            n_connected_components = nx.number_connected_components(subgraph)

            cur_counts += [n_connected_components]

            annotation["by_domain_rows"].append({
                "region_id": region_id,
                "domain": int(cur_domain),
                "n_cells": len(cell_indices),
                "n_connected_components": n_connected_components,
                "cells_per_component": len(cell_indices) / n_connected_components})

        if len(cur_counts) > 0:
            annotation["avg_n_connected_components_list"] += [np.mean(cur_counts)]
        else:
            annotation["avg_n_connected_components_list"] += [np.nan]

        annotation["n_domains_kept_list"] += [len(kept_domains)]

    if idx_region % 100 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"finished region {idx_region} of {len(train_images)} at {current_time}")


summary_frames = []

for annotation in annotations:

    df_avg = pd.DataFrame({
        "region_id": train_images,
        "n_cells": n_cells_list,
        "n_domains_kept": annotation["n_domains_kept_list"],
        "avg_n_connected_components": annotation["avg_n_connected_components_list"]})

    output_csv = os.path.join(
        figure_dir, "n_connected_components_rs_ensemble_scaled_direct_domains_" +
        annotation["tag"] + ".csv")
    by_domain_csv = os.path.join(
        figure_dir, "n_connected_components_rs_ensemble_scaled_direct_domains_" +
        annotation["tag"] + "_by_domain.csv")

    df_avg.to_csv(output_csv, index=False)

    df_by_domain = pd.DataFrame(annotation["by_domain_rows"])
    df_by_domain.to_csv(by_domain_csv, index=False)

    avg_values = df_avg["avg_n_connected_components"].to_numpy()
    avg_values = avg_values[np.isfinite(avg_values)]

    print("")
    print("==== direct k = 10, " + annotation["label"] + " ====")
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

    summary_frames.append({
        "graph_type": annotation["label"],
        "n_images_reported": int(len(avg_values)),
        "mean_domains_kept": float(df_avg["n_domains_kept"].mean()),
        "mean_avg_components": float(avg_values.mean()),
        "median_avg_components": float(np.median(avg_values)),
        "n_groups": int(len(df_by_domain)),
        "median_components": float(df_by_domain["n_connected_components"].median()),
        "median_cells_per_component": float(df_by_domain["cells_per_component"].median())})


if os.path.isfile(reference_csv):

    df_reference = pd.read_csv(reference_csv)

    if list(df_reference["region_id"]) != list(train_images):
        raise ValueError(reference_csv + " covers a different set of images, so its "
                         "numbers cannot be put beside these")

    reference_values = df_reference["avg_n_connected_components"].to_numpy()
    reference_values = reference_values[np.isfinite(reference_values)]

    summary_frames = [{
        "graph_type": reference_label,
        "n_images_reported": int(len(reference_values)),
        "mean_domains_kept": float(df_reference["n_domains_kept"].mean()),
        "mean_avg_components": float(reference_values.mean()),
        "median_avg_components": float(np.median(reference_values)),
        "n_groups": np.nan,
        "median_components": np.nan,
        "median_cells_per_component": np.nan}] + summary_frames

    print("")
    print("read the extended annotation back from " + reference_csv +
          " for the comparison below; the two by-group columns are left empty for it "
          "because that file holds one row per image, not per domain")

else:
    print("")
    print("no extended result at " + reference_csv +
          ", the comparison shows basic and local alone")

print("")
print("==== the graph types side by side ====")
print(pd.DataFrame(summary_frames).to_string(index=False))
