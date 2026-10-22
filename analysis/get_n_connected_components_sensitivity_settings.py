import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import networkx as nx

module_path = os.path.abspath(os.path.join("..", "code"))
if module_path not in sys.path:
    sys.path.append(module_path)

from graph_data_class import load_cell_data
from graph_data_class import build_graph_from_cell_coords
from data_utilities import data_features


parser = argparse.ArgumentParser(description="connected components for every setting")
parser.add_argument("--time_only", action="store_true",
                    help="time a sample of images and extrapolate, writing nothing")
parser.add_argument("--n_test_images", type=int, default=30,
                    help="how many images --time_only times")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()


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

direct_root = "../code/results_rs_ensemble_scaled_direct_domains/"
n_repeats = 10

output_csv = os.path.join(figure_dir, "n_connected_components_sensitivity_settings.csv")
by_domain_csv = os.path.join(figure_dir,
                             "n_connected_components_sensitivity_settings_by_domain.csv")

# the default's own counts, from the script this varies, to check against
reference_by_domain_csv = os.path.join(
    figure_dir, "n_connected_components_rs_ensemble_scaled_direct_domains_by_domain.csv")


def run_id(data_name="cords_2024", group_scheme="default", dist_cutoff="16",
           radius="48", top_k="4", ctg="0p176", degree_limit="20"):
    return (data_name + "__group_" + group_scheme + "__base" + dist_cutoff +
            "__rad" + radius + "__topk" + top_k + "__ctg" + ctg +
            "__deg" + degree_limit)


def setting(tag, label, group, model_run_id, k=10):
    return {"tag": tag, "label": label, "group": group, "k": k,
            "path": os.path.join(direct_root, model_run_id + "__ensemble_" + str(n_repeats),
                                 "extended", "domains_k" + str(k),
                                 "domain_annotation_for_cells.csv")}


settings = [
    setting("default", "default", "default", run_id()),

    setting("dist_14p5", "distance cutoff 14.5, radius 43.5", "sensitivity",
            run_id(dist_cutoff="14p5", radius="43p5")),
    setting("dist_17", "distance cutoff 17, radius 51", "sensitivity",
            run_id(dist_cutoff="17", radius="51")),
    setting("rad_32", "extension radius 32", "sensitivity", run_id(radius="32")),
    setting("rad_64", "extension radius 64", "sensitivity", run_id(radius="64")),
    setting("topk_3", "top k 3", "sensitivity", run_id(top_k="3")),
    setting("topk_5", "top k 5", "sensitivity", run_id(top_k="5")),
    setting("ctg_0p157", "composition cutoff 0.157", "sensitivity", run_id(ctg="0p157")),
    setting("ctg_0p202", "composition cutoff 0.202", "sensitivity", run_id(ctg="0p202")),
    setting("deg_16", "degree limit 16", "sensitivity", run_id(degree_limit="16")),
    setting("deg_24", "degree limit 24", "sensitivity", run_id(degree_limit="24")),
    setting("group8", "cell type groups group8", "sensitivity", run_id(group_scheme="group8")),
    setting("permutate5pct", "permuted data, permutate5pct", "sensitivity",
            run_id(data_name="cords_permutate5pct")),
    setting("permutate", "permuted data, permutate", "sensitivity",
            run_id(data_name="cords_permutate")),

    setting("k8", "k = 8", "n_domains", run_id(), k=8),
    setting("k12", "k = 12", "n_domains", run_id(), k=12),
    setting("k16", "k = 16", "n_domains", run_id(), k=16),
    setting("k20", "k = 20", "n_domains", run_id(), k=20),
]

if len(set(s["tag"] for s in settings)) != len(settings):
    raise ValueError("two settings share a tag")

for cur_setting in settings:
    if not os.path.isfile(cur_setting["path"]):
        raise FileNotFoundError("missing " + cur_setting["path"] + " for " + cur_setting["tag"])


# ---- the default dataset, graph cutoff and image list, for every setting ----

model_run_id = run_id()

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
train_images = list(data_dicts.train_images)

if neighbor_edge_cutoff != dist_cutoff:
    raise ValueError("the basic graph cutoff is " + str(neighbor_edge_cutoff) +
                     ", not the default " + str(dist_cutoff))

print("images: " + str(len(train_images)) + ", raw_dir: " + raw_dir)
print("basic graph edge cutoff: " + str(neighbor_edge_cutoff) + ", for every setting")
print("a domain needs at least " + str(n_cells_threshold) + " cells in an image to count")


# ---- every annotation, as one column of a table keyed by CELL_ID ----

t_load = time.time()

label_table = None

for cur_setting in settings:

    df_cur = pd.read_csv(cur_setting["path"], usecols=["CELL_ID", "domain"])

    if df_cur["CELL_ID"].duplicated().any():
        raise ValueError("duplicated CELL_ID in " + cur_setting["path"])

    if not df_cur["domain"].isin(range(1, cur_setting["k"] + 1)).all():
        raise ValueError("labels outside 1 to " + str(cur_setting["k"]) + " in " +
                         cur_setting["path"])

    df_cur = df_cur.set_index("CELL_ID")["domain"].astype(np.int16)

    if label_table is None:
        label_table = df_cur.to_frame(cur_setting["tag"])
    else:
        if not df_cur.index.sort_values().equals(label_table.index.sort_values()):
            raise ValueError(cur_setting["tag"] + " does not annotate the same cells as the default")
        label_table[cur_setting["tag"]] = df_cur.reindex(label_table.index).to_numpy()

setting_tags = [s["tag"] for s in settings]
label_matrix = label_table[setting_tags].to_numpy()
cell_index = label_table.index

print("read " + str(len(settings)) + " annotations of " + str(len(label_table)) +
      " cells in " + format(time.time() - t_load, ".0f") + " s")


# ---- the components of every domain of every annotation, on one graph ----

def count_image(region_id):

    cell_data = load_cell_data(raw_dir + "/" + region_id + ".csv")
    n_cells = len(cell_data)

    t_graph = time.time()
    G, _ = build_graph_from_cell_coords(cell_data, neighbor_edge_cutoff)
    t_graph = time.time() - t_graph

    t_count = time.time()

    positions = cell_index.get_indexer(cell_data["CELL_ID"].to_numpy())
    if (positions < 0).any():
        raise ValueError("cells of " + region_id + " missing from the annotations")

    image_labels = label_matrix[positions, :]

    image_rows = []
    domain_rows = []

    for setting_index, cur_setting in enumerate(settings):

        labels = image_labels[:, setting_index].astype(np.int64)

        domain_values, domain_sizes = np.unique(labels, return_counts=True)

        kept = [(int(d), int(n)) for d, n in zip(domain_values, domain_sizes)
                if n >= n_cells_threshold]

        # the graph is indexed by row position, so the cells of a domain are its rows
        for cur_domain, cur_size in kept:

            subgraph = G.subgraph(np.flatnonzero(labels == cur_domain).tolist())
            n_cc = nx.number_connected_components(subgraph)

            domain_rows.append({"setting": cur_setting["tag"], "region_id": region_id,
                                "domain": cur_domain, "n_cells": cur_size,
                                "n_connected_components": n_cc,
                                "cells_per_component": cur_size / n_cc})

        cur_counts = [r["n_connected_components"] for r in domain_rows
                      if r["setting"] == cur_setting["tag"]]

        image_rows.append({"group": cur_setting["group"], "setting": cur_setting["tag"],
                           "setting_label": cur_setting["label"], "n_domains": cur_setting["k"],
                           "region_id": region_id, "n_cells": n_cells,
                           "n_domains_kept": len(kept),
                           "avg_n_connected_components": (float(np.mean(cur_counts))
                                                          if len(cur_counts) > 0 else np.nan)})

    return image_rows, domain_rows, n_cells, t_graph, time.time() - t_count


# ---- the timing run ----

if args.time_only:

    rng = np.random.default_rng(args.seed)
    test_images = sorted(rng.choice(train_images, size=min(args.n_test_images, len(train_images)),
                                    replace=False).tolist())

    all_n_cells = []
    for region_id in train_images:
        with open(raw_dir + "/" + region_id + ".csv") as cur_file:
            all_n_cells.append(sum(1 for _ in cur_file) - 1)
    all_n_cells = np.array(all_n_cells, dtype=float)

    test_n_cells, test_graph, test_count, test_domain_rows = [], [], [], []

    for region_id in test_images:
        _, domain_rows, n_cells, t_graph, t_count = count_image(region_id)
        test_n_cells.append(n_cells)
        test_graph.append(t_graph)
        test_count.append(t_count)
        test_domain_rows += domain_rows
        print("  " + region_id + ": " + str(n_cells) + " cells, graph " +
              format(t_graph, ".2f") + " s, 18 annotations counted in " + format(t_count, ".2f") + " s")

    test_n_cells = np.array(test_n_cells, dtype=float)
    test_total = np.array(test_graph) + np.array(test_count)

    # the graph build is dominated by a dense n by n distance matrix, so the time
    # of an image is fitted as a + b n + c n^2 and summed over every image
    design = np.stack([np.ones_like(test_n_cells), test_n_cells, test_n_cells ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(design, test_total, rcond=None)
    predicted = np.stack([np.ones_like(all_n_cells), all_n_cells, all_n_cells ** 2], axis=1) @ coef

    print("")
    print("timed " + str(len(test_images)) + " images, " + str(int(test_n_cells.min())) + " to " +
          str(int(test_n_cells.max())) + " cells (all images: " + str(int(all_n_cells.min())) +
          " to " + str(int(all_n_cells.max())) + ", median " + str(int(np.median(all_n_cells))) + ")")
    print("per image: graph " + format(np.mean(test_graph), ".2f") + " s and counting " +
          format(np.mean(test_count), ".2f") + " s on average")
    print("estimate for all " + str(len(train_images)) + " images: " +
          format(np.mean(test_total) * len(train_images) / 60, ".0f") + " min from the mean, " +
          format(max(predicted.sum(), 0) / 60, ".0f") + " min from the size fit, plus " +
          format(time.time() - t_load, ".0f") + " s to load the annotations")

    if os.path.isfile(reference_by_domain_csv):
        df_reference = pd.read_csv(reference_by_domain_csv)
        df_test = pd.DataFrame(test_domain_rows)
        df_test = df_test[df_test["setting"] == "default"]
        df_join = df_test.merge(df_reference, on=["region_id", "domain"], suffixes=("", "_reference"))
        if len(df_join) != len(df_test):
            raise ValueError("the default's domains differ from " + reference_by_domain_csv)
        if not (df_join["n_connected_components"] == df_join["n_connected_components_reference"]).all():
            raise ValueError("the default's counts differ from " + reference_by_domain_csv)
        print("the default's counts on these images match " + reference_by_domain_csv +
              " for all " + str(len(df_join)) + " domains")

    sys.exit(0)


# ---- the full run ----

all_image_rows = []
all_domain_rows = []

t_start = time.time()

for idx_region, region_id in enumerate(train_images):

    image_rows, domain_rows, _, _, _ = count_image(region_id)

    all_image_rows += image_rows
    all_domain_rows += domain_rows

    if idx_region % 100 == 0:
        print("finished image " + str(idx_region) + " of " + str(len(train_images)) +
              " after " + format((time.time() - t_start) / 60, ".1f") + " min")

df_avg = pd.DataFrame(all_image_rows)
df_by_domain = pd.DataFrame(all_domain_rows)

df_avg.to_csv(output_csv, index=False)
df_by_domain.to_csv(by_domain_csv, index=False)

if os.path.isfile(reference_by_domain_csv):
    df_reference = pd.read_csv(reference_by_domain_csv)
    df_default = df_by_domain[df_by_domain["setting"] == "default"]
    df_join = df_default.merge(df_reference, on=["region_id", "domain"], suffixes=("", "_reference"))
    if len(df_join) != len(df_default) or len(df_join) != len(df_reference) or \
            not (df_join["n_connected_components"] == df_join["n_connected_components_reference"]).all():
        raise ValueError("the default's counts differ from " + reference_by_domain_csv)
    print("the default's counts match " + reference_by_domain_csv +
          " for all " + str(len(df_join)) + " domain-image groups")

print("")
print(df_avg.groupby(["group", "setting_label"], sort=False)["avg_n_connected_components"]
      .median().round(2).to_string())

print("")
print("finished in " + format((time.time() - t_start) / 60, ".1f") + " min")
print("saved " + output_csv)
print("saved " + by_domain_csv)
