#!/usr/bin/env python
# coding: utf-8


import os
import time

import numpy as np
import pandas as pd

from local_pca_dimension_cell_min import local_pca_dimension_cell_min


figure_dir = "figures"
os.makedirs(figure_dir, exist_ok=True)

raw_dir = "../../data/Cords_data/raw_data"

region_ids = ["86_C6_10", "175_B7_15"]

if len(set(region_ids)) != len(region_ids):
    raise ValueError("an image is named twice in region_ids")

min_cells = 30

radii = np.array([32.0, 48.0])

min_neighbors = 10
threshold = 1.5

min_counted = 30

n_domains = 10

direct_root = ("../code/results_rs_ensemble_scaled_direct_domains/"
               "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20__ensemble_10/")


def direct_method(tag, label, graph_type):
    return {"tag": tag, "label": label,
            "kind": "global",
            "path": os.path.join(direct_root, graph_type, "domains_k" + str(n_domains),
                                 "domain_annotation_for_cells.csv"),
            "sep": ",", "id_col": "CELL_ID", "label_col": "domain",
            "label_offset": 0}


methods = [
    direct_method("direct_extended", "direct k = 10, extended graph", "extended"),
    direct_method("direct_basic", "direct k = 10, basic graph", "basic"),
    direct_method("direct_local", "direct k = 10, local graph", "local"),
    {"tag": "raw_kmeans", "label": "raw k-means, k = 10",
     "kind": "global",
     "path": "../../results/cords_2024/extended/raw_kmeans/kmeans_cluster.csv",
     "sep": ",", "id_col": "CELL_ID", "label_col": "kmeans_cluster",
     "label_offset": 1},
    {"tag": "spatial_lda", "label": "SpatialLDA, radius 27, d 0.25",
     "kind": "global",
     "path": "../../SpatialLDA_copy/results/cords/cords_results_radius_27_d_0.25.tsv",
     "sep": "\t", "id_col": "cell_id", "label_col": "predicted_cluster",
     "label_offset": 1},
    {"tag": "spatopic_radius_27_sigma_5", "label": "SpaTopic, radius 27, sigma 5",
     "kind": "per_image",
     "path": "../../SpaTopic_copy/results/cords/radius_27_sigma_5",
     "suffix": "_spatopic_results.csv", "label_col": "topic",
     "label_offset": 0},
    {"tag": "spatopic_radius_62_sigma_8", "label": "SpaTopic, radius 62, sigma 8",
     "kind": "per_image",
     "path": "../../SpaTopic_copy/results/cords/radius_62_sigma_8",
     "suffix": "_spatopic_results.csv", "label_col": "topic",
     "label_offset": 0},
]

if len(set(m["tag"] for m in methods)) != len(methods):
    raise ValueError("two methods share a tag, which would merge their rows")

for method in methods:
    cur_exists = (os.path.isdir(method["path"]) if method["kind"] == "per_image"
                  else os.path.isfile(method["path"]))
    if not cur_exists:
        raise FileNotFoundError("missing " + method["path"] + " for " + method["tag"])

print("all " + str(len(methods)) + " annotations are on disk")


for cur_region_id in region_ids:

    cur_raw_csv = os.path.join(raw_dir, cur_region_id + ".csv")

    if not os.path.isfile(cur_raw_csv):
        raise FileNotFoundError("missing " + cur_raw_csv +
                                "; there is no image named " + cur_region_id)

    with open(cur_raw_csv) as cur_file:
        n_cells_named = sum(1 for _ in cur_file) - 1

    print("measuring " + cur_region_id + ", " + str(n_cells_named) + " cells")

sampled_images = list(region_ids)


result_rows = []

for method in methods:

    print("")
    print("==== " + method["tag"] + ": " + method["label"] + " ====")

    t_start = time.time()

    df_method_domain = None

    if method["kind"] == "global":

        print("reading " + method["path"])

        df_method_domain = pd.read_csv(method["path"], sep=method["sep"])
        df_method_domain = df_method_domain[[method["id_col"], method["label_col"]]]
        df_method_domain.columns = ["CELL_ID", "domain"]
        df_method_domain["domain"] = (df_method_domain["domain"].astype(int) +
                                      method["label_offset"])

        if not df_method_domain["domain"].isin(range(1, n_domains + 1)).all():
            raise ValueError("labels outside 1 to " + str(n_domains) + " after the " +
                             "offset of " + str(method["label_offset"]) + ", in " +
                             method["path"])

        print("read a domain for " + str(len(df_method_domain)) + " cells, labels " +
              str(sorted(df_method_domain["domain"].unique())))

    else:
        print("reading one file per image from " + method["path"])

    n_skipped_small = 0

    for region_id in sampled_images:

        df_cells = pd.read_csv(os.path.join(raw_dir, region_id + ".csv"))[
            ["CELL_ID", "X", "Y"]]
        n_cells_image = len(df_cells)

        if method["kind"] == "global":

            df_cells = df_cells.merge(df_method_domain, on="CELL_ID", how="inner")

            if len(df_cells) != n_cells_image:
                raise ValueError("some cells of " + region_id + " have no domain under " +
                                 method["tag"])

        else:

            cur_path = os.path.join(method["path"], region_id + method["suffix"])

            if not os.path.isfile(cur_path):
                raise FileNotFoundError("missing " + cur_path)

            df_method = pd.read_csv(cur_path)

            if len(df_method) != n_cells_image:
                raise ValueError(cur_path + " has " + str(len(df_method)) +
                                 " rows, the image has " + str(n_cells_image) + " cells")

            if not np.allclose(df_method["X"].to_numpy(), df_cells["X"].to_numpy()):
                raise ValueError("X coordinates of " + cur_path + " do not match the raw file")
            if not np.allclose(df_method["Y"].to_numpy(), df_cells["Y"].to_numpy()):
                raise ValueError("Y coordinates of " + cur_path + " do not match the raw file")

            df_cells["domain"] = (df_method[method["label_col"]].to_numpy().astype(int) +
                                  method["label_offset"])

            if not df_cells["domain"].isin(range(1, n_domains + 1)).all():
                raise ValueError("labels outside 1 to " + str(n_domains) + " in " + cur_path)

        domain_sizes = df_cells["domain"].value_counts().sort_index()

        for cur_domain in sorted(domain_sizes.index):

            if domain_sizes[cur_domain] < min_cells:
                n_skipped_small += 1
                continue

            cur_coords = df_cells.loc[df_cells["domain"] == cur_domain,
                                      ["X", "Y"]].to_numpy(float)

            df_per_cell = local_pca_dimension_cell_min(cur_coords, radii,
                                                       min_neighbors=min_neighbors)

            n_cells_domain = len(df_per_cell)
            n_counted = int(df_per_cell["usable_any"].sum())

            cur_values = df_per_cell.loc[df_per_cell["usable_any"], "d_min_cell"]

            n_below = int((cur_values < threshold).sum())

            result_rows.append({
                "method": method["tag"],
                "method_label": method["label"],
                "region_id": region_id,
                "n_cells_image": n_cells_image,
                "domain": int(cur_domain),
                "n_cells": n_cells_domain,
                "n_counted": n_counted,
                "fraction_counted": n_counted / n_cells_domain,
                "n_below_threshold": n_below,
                "proportion_below_threshold": (n_below / n_counted
                                               if n_counted >= min_counted else np.nan),
                "median_d_min_cell": (float(cur_values.median())
                                      if n_counted > 0 else np.nan),
                "n_usable_at_both_radii": int((df_per_cell["n_usable_radii"] == 2).sum())})

    n_method_rows = sum(1 for r in result_rows if r["method"] == method["tag"])

    print("measured " + str(n_method_rows) + " domain-image groups, skipped " +
          str(n_skipped_small) + " below " + str(min_cells) + " cells, in " +
          format(time.time() - t_start, ".0f") + " s")

df_all = pd.DataFrame(result_rows)

for cur_method in df_all["method"].unique():
    cur_images = set(df_all.loc[df_all["method"] == cur_method, "region_id"])
    if not cur_images.issubset(set(sampled_images)):
        raise ValueError(cur_method + " has rows for images outside the sample")

for cur_region_id in region_ids:

    cur_tag = "cell_min_all_methods_" + cur_region_id

    df_cur_image = df_all.loc[df_all["region_id"] == cur_region_id]

    if len(df_cur_image) == 0:
        raise ValueError("no domain of any annotation cleared min_cells in " +
                         cur_region_id + ", so there is nothing to write for it")

    result_file = os.path.join(figure_dir, "local_pca_dimension_" + cur_tag + ".csv")
    df_cur_image.to_csv(result_file, index=False)

    method_rows = []

    for cur_method in df_cur_image["method"].unique():

        df_cur = df_cur_image.loc[df_cur_image["method"] == cur_method]
        df_valid = df_cur.dropna(subset=["proportion_below_threshold"])

        method_rows.append({
            "method": cur_method,
            "method_label": df_cur["method_label"].iloc[0],
            "region_id": cur_region_id,
            "n_groups": int(len(df_cur)),
            "n_domains_seen": int(df_cur["domain"].nunique()),
            "median_n_cells": float(df_cur["n_cells"].median()),
            "median_fraction_counted": float(df_cur["fraction_counted"].median()),
            "median_proportion_below": float(df_valid["proportion_below_threshold"].median()),
            "q25_proportion_below": float(df_valid["proportion_below_threshold"].quantile(0.25)),
            "q75_proportion_below": float(df_valid["proportion_below_threshold"].quantile(0.75)),
            "range_of_domain_medians": float(
                df_valid["proportion_below_threshold"].max() -
                df_valid["proportion_below_threshold"].min())})

    df_method_summary = pd.DataFrame(method_rows)

    method_summary_file = os.path.join(
        figure_dir, "local_pca_dimension_" + cur_tag + "_by_method.csv")
    df_method_summary.to_csv(method_summary_file, index=False)

    print("")
    print("==== " + cur_region_id + ": each method over its domains ====")
    print(df_method_summary.to_string(index=False))
    print("saved " + result_file)
    print("saved " + method_summary_file)
