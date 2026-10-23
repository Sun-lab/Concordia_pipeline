#!/usr/bin/env python
# coding: utf-8


import os
import sys
import glob
import time

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree


module_path = os.path.abspath(os.path.join("..", "code"))

if module_path not in sys.path:
    sys.path.append(module_path)

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

tumor_group = "tumor"

min_tumor_cells = 30

min_qualified_tumor = 30

n_images = 100
random_seed = 1
min_image_cells = 3300

example_images = ["86_C6_10", "86_C2_4"]

reference_csvs = [
    os.path.join(figure_dir, "local_pca_dimension_cell_min_all_methods_random_sample.csv"),
]

output_csv = os.path.join(
    figure_dir, "tumor_neighbor_tumor_fraction_sampled_images_random_sample.csv")


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

neighbor_edge_cutoff = data_dicts.dist_cutoff

group_ct_mapping = data_dicts.group_ct_mapping

if tumor_group not in group_ct_mapping:
    raise ValueError("no cell type group named " + tumor_group + "; the groups are " +
                     str(sorted(group_ct_mapping.keys())))

tumor_cell_types = set(group_ct_mapping[tumor_group])

all_grouped = [ct for grp in group_ct_mapping.values() for ct in grp]

if len(all_grouped) != len(set(all_grouped)):
    raise ValueError("a cell type falls in more than one group")

all_grouped = set(all_grouped)

print("neighbour cutoff: " + str(neighbor_edge_cutoff) + ", inclusive, as in the basic graph")
print("tumour group holds " + str(len(tumor_cell_types)) + " cell types: " +
      str(sorted(tumor_cell_types)))


all_image_paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
all_images = [os.path.basename(p)[:-4] for p in all_image_paths]

print("")
print("found " + str(len(all_images)) + " images in " + raw_dir)

image_cell_counts = {}

for cur_name, cur_path in zip(all_images, all_image_paths):
    with open(cur_path) as cur_file:
        image_cell_counts[cur_name] = sum(1 for _ in cur_file) - 1

eligible_images = sorted(name for name in all_images
                         if image_cell_counts[name] >= min_image_cells)

print("of these, " + str(len(eligible_images)) + " hold at least " +
      str(min_image_cells) + " cells")

if len(eligible_images) < n_images:
    raise ValueError("only " + str(len(eligible_images)) + " images are eligible, " +
                     "fewer than the " + str(n_images) + " to be drawn")

rng = np.random.default_rng(random_seed)
sampled_images = sorted(rng.choice(eligible_images, size=n_images,
                                   replace=False).tolist())

if len(set(sampled_images)) != n_images:
    raise ValueError("drew " + str(len(set(sampled_images))) + " distinct images, " +
                     "expected " + str(n_images))

print("drew all " + str(n_images) + " images at random from the eligible set " +
      "with seed " + str(random_seed) + ", nothing forced")

for cur_name in example_images:
    print("  the example image " + cur_name + " is " +
          ("in" if cur_name in sampled_images else "NOT in") + " this sample")

n_checked = 0

for cur_csv in reference_csvs:

    if not os.path.isfile(cur_csv):
        continue

    reference_images = set(pd.read_csv(cur_csv, usecols=["region_id"])["region_id"].unique())

    if reference_images != set(sampled_images):
        raise ValueError(
            "the sample built here does not match the images in " + cur_csv + ": " +
            str(len(set(sampled_images) - reference_images)) + " here are missing from it and " +
            str(len(reference_images - set(sampled_images))) + " of its are missing here")

    n_checked += 1

if n_checked == 0:
    print("no random-sample cell_min csv on disk to check the sample against")
else:
    print("the sample matches the images of " + str(n_checked) + " cell_min csv file(s)")


result_rows = []

t_start = time.time()

for image_position, region_id in enumerate(sampled_images):

    df_cells = pd.read_csv(os.path.join(raw_dir, region_id + ".csv"))[
        ["CELL_ID", "X", "Y", "CELL_TYPE"]]

    n_cells_image = len(df_cells)

    unknown = set(df_cells["CELL_TYPE"].unique()) - all_grouped
    if len(unknown) > 0:
        raise ValueError("cell types of " + region_id + " outside the grouping: " +
                         str(sorted(unknown)))

    is_tumor = df_cells["CELL_TYPE"].isin(tumor_cell_types).to_numpy()

    n_tumor = int(is_tumor.sum())

    coords = df_cells[["X", "Y"]].to_numpy(float)

    tumor_positions = np.flatnonzero(is_tumor)

    if n_tumor < min_tumor_cells:

        result_rows.append({
            "region_id": region_id,
            "n_cells_image": n_cells_image,
            "n_tumor": n_tumor,
            "n_tumor_with_neighbors": np.nan,
            "mean_prop_t": np.nan,
            "median_prop_t": np.nan,
            "not_measured_because": "fewer than " + str(min_tumor_cells) + " tumour cells"})

        print("[" + str(image_position + 1) + "/" + str(len(sampled_images)) + "] " +
              region_id + ": only " + str(n_tumor) + " tumour cells, not measured")
        continue

    tree = cKDTree(coords)

    neighbor_lists = tree.query_ball_point(coords[tumor_positions],
                                           r=neighbor_edge_cutoff)

    n_neighbors = np.zeros(len(tumor_positions), dtype=int)
    prop_t = np.full(len(tumor_positions), np.nan)

    for k, (centre_index, neighbor_index) in enumerate(zip(tumor_positions, neighbor_lists)):

        neighbor_index = [x for x in neighbor_index if x != centre_index]

        n_neighbors[k] = len(neighbor_index)

        if len(neighbor_index) == 0:
            continue

        n_tumor_neighbors = int(np.sum(is_tumor[neighbor_index]))
        prop_t[k] = n_tumor_neighbors / len(neighbor_index)

    qualified = n_neighbors >= 1

    n_qualified = int(qualified.sum())

    if n_qualified >= min_qualified_tumor:
        cur_mean = float(np.mean(prop_t[qualified]))
        not_measured_because = ""
    else:
        cur_mean = np.nan
        not_measured_because = ("fewer than " + str(min_qualified_tumor) +
                                " tumour cells with a neighbour")

    result_rows.append({
        "region_id": region_id,
        "n_cells_image": n_cells_image,
        "n_tumor": n_tumor,
        "n_tumor_with_neighbors": n_qualified,
        "mean_prop_t": cur_mean,
        "median_prop_t": (float(np.median(prop_t[qualified]))
                          if n_qualified >= min_qualified_tumor else np.nan),
        "not_measured_because": not_measured_because})

    print("[" + str(image_position + 1) + "/" + str(len(sampled_images)) + "] " +
          region_id + ": " + str(n_tumor) + " tumour cells, " + str(n_qualified) +
          " with a neighbour, " +
          ("mean prop_t = " + format(cur_mean, ".3f")
           if n_qualified >= min_qualified_tumor else "not reported") +
          ", " + format(time.time() - t_start, ".0f") + " s elapsed")

df_result = pd.DataFrame(result_rows)

df_result.to_csv(output_csv, index=False)


df_valid = df_result.dropna(subset=["mean_prop_t"])

print("")
print("==== " + str(len(sampled_images)) + " sampled images ====")
print("measured: " + str(len(df_valid)) + ", not measured: " +
      str(len(df_result) - len(df_valid)))

if len(df_result) > len(df_valid):
    print("reasons a value is missing:")
    reasons = df_result.loc[df_result["mean_prop_t"].isna(),
                            "not_measured_because"].value_counts()
    for cur_reason, cur_count in reasons.items():
        print("  " + str(cur_count) + ": " + cur_reason)

if len(df_valid) > 0:
    cur_values = df_valid["mean_prop_t"]
    print("")
    print("tumour cells per image: median " + str(int(df_valid["n_tumor"].median())) +
          ", range " + str(int(df_valid["n_tumor"].min())) + " to " +
          str(int(df_valid["n_tumor"].max())))
    print("of them with at least one neighbour: median " +
          format(100.0 * (df_valid["n_tumor_with_neighbors"] / df_valid["n_tumor"]).median(), ".1f") + "%")
    print("")
    print("mean prop_t over the qualifying tumour cells of an image:")
    print("  mean " + format(cur_values.mean(), ".3f") +
          ", median " + format(cur_values.median(), ".3f") +
          ", range " + format(cur_values.min(), ".3f") +
          " to " + format(cur_values.max(), ".3f"))
    print("  quartiles " + format(cur_values.quantile(0.25), ".3f") +
          " and " + format(cur_values.quantile(0.75), ".3f"))

print("")
print("saved " + output_csv)
