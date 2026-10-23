#!/usr/bin/env python
# coding: utf-8


import os
import glob
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from local_pca_dimension_cell_min import local_pca_dimension_cell_min


figure_dir = "figures"
os.makedirs(figure_dir, exist_ok=True)

raw_dir = "../../data/Cords_data/raw_data"

output_tag = "cell_min_all_methods_random_sample"

n_images = 100
random_seed = 1
min_image_cells = 3300

example_images = ["86_C6_10", "86_C2_4"]

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


all_image_paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
all_images = [os.path.basename(p)[:-4] for p in all_image_paths]

print("found " + str(len(all_images)) + " images in " + raw_dir)

image_cell_counts = {}

for cur_name, cur_path in zip(all_images, all_image_paths):
    with open(cur_path) as cur_file:
        image_cell_counts[cur_name] = sum(1 for _ in cur_file) - 1

eligible_images = sorted(name for name in all_images
                         if image_cell_counts[name] >= min_image_cells)

count_values = np.array([image_cell_counts[name] for name in eligible_images])

print("of these, " + str(len(eligible_images)) + " hold at least " +
      str(min_image_cells) + " cells, from " + str(count_values.min()) + " to " +
      str(count_values.max()) + ", median " + str(int(np.median(count_values))))

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

result_file = os.path.join(figure_dir, "local_pca_dimension_" + output_tag + ".csv")
df_all.to_csv(result_file, index=False)


method_rows = []

for cur_method in df_all["method"].unique():

    df_cur = df_all.loc[df_all["method"] == cur_method]
    df_valid = df_cur.dropna(subset=["proportion_below_threshold"])

    method_rows.append({
        "method": cur_method,
        "method_label": df_cur["method_label"].iloc[0],
        "n_groups": int(len(df_cur)),
        "n_domains_seen": int(df_cur["domain"].nunique()),
        "groups_per_image": float(len(df_cur) / df_cur["region_id"].nunique()),
        "median_n_cells": float(df_cur["n_cells"].median()),
        "median_fraction_counted": float(df_cur["fraction_counted"].median()),
        "median_proportion_below": float(df_valid["proportion_below_threshold"].median()),
        "q25_proportion_below": float(df_valid["proportion_below_threshold"].quantile(0.25)),
        "q75_proportion_below": float(df_valid["proportion_below_threshold"].quantile(0.75)),
        "range_of_domain_medians": float(
            df_valid.groupby("domain")["proportion_below_threshold"].median().max() -
            df_valid.groupby("domain")["proportion_below_threshold"].median().min()),
        "spearman_proportion_n_cells": (float(df_valid[
            ["proportion_below_threshold", "n_cells"]].corr(
            method="spearman").iloc[0, 1]) if len(df_valid) > 2 else np.nan)})

df_method_summary = pd.DataFrame(method_rows)

method_summary_file = os.path.join(
    figure_dir, "local_pca_dimension_" + output_tag + "_by_method.csv")
df_method_summary.to_csv(method_summary_file, index=False)

domain_summary_rows = []

for cur_method in df_all["method"].unique():
    for cur_domain in range(1, n_domains + 1):

        df_cur = df_all.loc[(df_all["method"] == cur_method) &
                            (df_all["domain"] == cur_domain)]

        if len(df_cur) == 0:
            continue

        cur_values = df_cur["proportion_below_threshold"].dropna()

        domain_summary_rows.append({
            "method": cur_method,
            "domain": cur_domain,
            "n_images": int(len(df_cur)),
            "median_n_cells": float(df_cur["n_cells"].median()),
            "median_fraction_counted": float(df_cur["fraction_counted"].median()),
            "median_proportion_below": (float(cur_values.median())
                                        if len(cur_values) > 0 else np.nan),
            "min_proportion_below": (float(cur_values.min())
                                     if len(cur_values) > 0 else np.nan),
            "max_proportion_below": (float(cur_values.max())
                                     if len(cur_values) > 0 else np.nan)})

df_domain_summary = pd.DataFrame(domain_summary_rows)

domain_summary_file = os.path.join(
    figure_dir, "local_pca_dimension_" + output_tag + "_by_domain.csv")
df_domain_summary.to_csv(domain_summary_file, index=False)

print("")
print("==== each method over all its domain-image groups ====")
print(df_method_summary.to_string(index=False))


jitter_rng = np.random.default_rng(random_seed + 1)

method_order = [m["tag"] for m in methods]
method_labels = [df_all.loc[df_all["method"] == m, "method_label"].iloc[0]
                 for m in method_order]

fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))

axes[0].boxplot([df_all.loc[df_all["method"] == m,
                            "proportion_below_threshold"].dropna().to_numpy()
                 for m in method_order],
                labels=[str(i + 1) for i in range(len(method_order))],
                showfliers=False)

for cur_position, cur_method in enumerate(method_order):
    cur_values = df_all.loc[df_all["method"] == cur_method,
                            "proportion_below_threshold"].dropna().to_numpy()
    axes[0].plot(np.full(len(cur_values), cur_position + 1) +
                 jitter_rng.uniform(-0.16, 0.16, len(cur_values)),
                 cur_values, ".", markersize=2, alpha=0.3, color="#1f77b4")

axes[0].set_ylabel("proportion of counted cells with D_min < " + format(threshold, ".1f"))
axes[0].set_xlabel("method")
axes[0].set_title("how much of a domain ever looks line-like\n" +
                  "one point per domain-image group")

for cur_position, cur_method in enumerate(method_order):
    cur_medians = df_all.loc[df_all["method"] == cur_method].dropna(
        subset=["proportion_below_threshold"]).groupby(
        "domain")["proportion_below_threshold"].median().to_numpy()
    axes[1].plot(np.full(len(cur_medians), cur_position + 1), cur_medians,
                 "o", markersize=5, alpha=0.8, color="#d62728")
    if len(cur_medians) > 0:
        axes[1].plot([cur_position + 1, cur_position + 1],
                     [cur_medians.min(), cur_medians.max()],
                     "-", linewidth=1, color="#d62728", alpha=0.5)

axes[1].set_xticks(range(1, len(method_order) + 1))
axes[1].set_xticklabels([str(i + 1) for i in range(len(method_order))])
axes[1].set_ylabel("median proportion, one point per domain")
axes[1].set_xlabel("method")
axes[1].set_title("where each method's ten domains sit\n" +
                  "a short span means the domains look alike on this scale")

axes[2].bar(range(1, len(method_order) + 1),
            [df_method_summary.loc[df_method_summary["method"] == m,
                                   "spearman_proportion_n_cells"].iloc[0]
             for m in method_order],
            color="#7f7f7f")
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_xticks(range(1, len(method_order) + 1))
axes[2].set_xticklabels([str(i + 1) for i in range(len(method_order))])
axes[2].set_ylabel("Spearman(proportion, domain cell count)")
axes[2].set_xlabel("method")
axes[2].set_title("how much the proportion tracks domain size\n" +
                  "a large value means the metric is partly reporting size")

fig.suptitle("D_min per cell, minimum over radii " +
             str(list(radii)) + ", threshold " + format(threshold, ".1f") +
             "; " + str(n_images) + " images drawn at random, nothing forced" +
             "\n" + ";  ".join(str(i + 1) + " = " + cur_label
                               for i, cur_label in enumerate(method_labels)),
             fontsize=11)

fig.tight_layout(rect=(0, 0, 1, 0.90))

figure_file = os.path.join(figure_dir, "local_pca_dimension_" + output_tag + ".pdf")
fig.savefig(figure_file, bbox_inches="tight")
plt.close(fig)

print("")
print("saved " + result_file)
print("saved " + method_summary_file)
print("saved " + domain_summary_file)
print("saved " + figure_file)
