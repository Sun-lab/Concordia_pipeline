import os
import glob
import time

import numpy as np
import pandas as pd

from local_pca_dimension_cell_min import local_pca_dimension_cell_min


figure_dir = "figures"
os.makedirs(figure_dir, exist_ok=True)

raw_dir = "../../data/Cords_data/raw_data"

output_tag = "cell_min_default_domain10_matches_random_sample"

n_images = 100
random_seed = 1
min_image_cells = 3300

min_cells = 30
radii = np.array([32.0, 48.0])
min_neighbors = 10
threshold = 1.5
min_counted = 30

default_k = 10

direct_root = "../code/results_rs_ensemble_scaled_direct_domains/"
n_repeats = 10

default_run_id = "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20"

match_csv = ("../../revision_figures/supp_default_domain10_matches_across_settings/"
             "default_domain10_matches.csv")

reference_csv = os.path.join(figure_dir,
                             "local_pca_dimension_cell_min_all_methods_random_sample.csv")


def sensitivity_run_id(data_name="cords_2024", group_scheme="default",
                       dist_cutoff="16", radius="48", top_k="4",
                       ctg="0p176", degree_limit="20"):
    return (data_name + "__group_" + group_scheme + "__base" + dist_cutoff +
            "__rad" + radius + "__topk" + top_k + "__ctg" + ctg +
            "__deg" + degree_limit)


run_id_of_tag = {
    "dist_14p5": sensitivity_run_id(dist_cutoff="14p5", radius="43p5"),
    "dist_17": sensitivity_run_id(dist_cutoff="17", radius="51"),
    "rad_32": sensitivity_run_id(radius="32"),
    "rad_64": sensitivity_run_id(radius="64"),
    "topk_3": sensitivity_run_id(top_k="3"),
    "topk_5": sensitivity_run_id(top_k="5"),
    "ctg_0p157": sensitivity_run_id(ctg="0p157"),
    "ctg_0p202": sensitivity_run_id(ctg="0p202"),
    "deg_16": sensitivity_run_id(degree_limit="16"),
    "deg_24": sensitivity_run_id(degree_limit="24"),
    "group8": sensitivity_run_id(group_scheme="group8"),
    "permutate5pct": sensitivity_run_id(data_name="cords_permutate5pct"),
    "permutate": sensitivity_run_id(data_name="cords_permutate"),
    "k8": default_run_id,
    "k12": default_run_id,
    "k16": default_run_id,
    "k20": default_run_id,
}


def annotation_path(tag, k):
    return os.path.join(direct_root, run_id_of_tag[tag] + "__ensemble_" + str(n_repeats),
                        "extended", "domains_k" + str(k),
                        "domain_annotation_for_cells.csv")


df_match = pd.read_csv(match_csv, dtype={"domain": str})
df_match = df_match[df_match["criterion"] != "no match"].copy()
df_match["domain"] = df_match["domain"].astype(int)

if df_match.duplicated(subset=["tag", "domain"]).any():
    raise ValueError("a (setting, domain) pair is listed twice in " + match_csv)

unknown_tags = sorted(set(df_match["tag"]) - set(run_id_of_tag))
if len(unknown_tags) > 0:
    raise ValueError("no annotation is known for the settings " + str(unknown_tags))

settings = []

for cur_tag, df_cur in df_match.groupby("tag", sort=False):

    cur_k = int(df_cur["n_domains"].iloc[0])
    cur_path = annotation_path(cur_tag, cur_k)

    if not os.path.isfile(cur_path):
        raise FileNotFoundError("missing " + cur_path + " for " + cur_tag)

    settings.append({"tag": cur_tag,
                     "label": df_cur["setting"].iloc[0],
                     "group": df_cur["group"].iloc[0],
                     "k": cur_k,
                     "path": cur_path,
                     "domains": sorted(df_cur["domain"].tolist()),
                     "criterion_of": dict(zip(df_cur["domain"], df_cur["criterion"]))})

print("read " + str(len(df_match)) + " (setting, domain) pairs over " +
      str(len(settings)) + " settings from " + match_csv)


all_image_paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
all_images = [os.path.basename(p)[:-4] for p in all_image_paths]

image_cell_counts = {}

for cur_name, cur_path in zip(all_images, all_image_paths):
    with open(cur_path) as cur_file:
        image_cell_counts[cur_name] = sum(1 for _ in cur_file) - 1

eligible_images = sorted(name for name in all_images
                         if image_cell_counts[name] >= min_image_cells)

if len(eligible_images) < n_images:
    raise ValueError("only " + str(len(eligible_images)) + " images are eligible, " +
                     "fewer than the " + str(n_images) + " to be drawn")

rng = np.random.default_rng(random_seed)
sampled_images = sorted(rng.choice(eligible_images, size=n_images,
                                   replace=False).tolist())

if len(set(sampled_images)) != n_images:
    raise ValueError("drew " + str(len(set(sampled_images))) + " distinct images, " +
                     "expected " + str(n_images))

print("drew " + str(n_images) + " images at random from the " + str(len(eligible_images)) +
      " holding at least " + str(min_image_cells) + " cells, with seed " + str(random_seed))

if os.path.isfile(reference_csv):
    reference_images = set(pd.read_csv(reference_csv, usecols=["region_id"])["region_id"])
    if reference_images != set(sampled_images):
        raise ValueError("the sample does not match the images of " + reference_csv)
    print("the sample matches the images of " + reference_csv)
else:
    print("no " + reference_csv + " on disk to check the sample against")


raw_cells_of = {}

for region_id in sampled_images:
    raw_cells_of[region_id] = pd.read_csv(os.path.join(raw_dir, region_id + ".csv"))[
        ["CELL_ID", "X", "Y"]]


result_rows = []

for setting in settings:

    print("")
    print("==== " + setting["tag"] + ": " + setting["label"] + ", domains " +
          str(setting["domains"]) + " ====")

    t_start = time.time()

    df_domain = pd.read_csv(setting["path"], usecols=["CELL_ID", "domain"])
    df_domain["domain"] = df_domain["domain"].astype(int)

    if df_domain["CELL_ID"].duplicated().any():
        raise ValueError("duplicated CELL_ID in " + setting["path"])

    if not df_domain["domain"].isin(range(1, setting["k"] + 1)).all():
        raise ValueError("labels outside 1 to " + str(setting["k"]) + " in " + setting["path"])

    n_skipped_small = 0

    for region_id in sampled_images:

        df_cells = raw_cells_of[region_id]
        n_cells_image = len(df_cells)

        df_cells = df_cells.merge(df_domain, on="CELL_ID", how="inner")

        if len(df_cells) != n_cells_image:
            raise ValueError("some cells of " + region_id + " have no domain under " +
                             setting["tag"])

        for cur_domain in setting["domains"]:

            cur_coords = df_cells.loc[df_cells["domain"] == cur_domain,
                                      ["X", "Y"]].to_numpy(float)

            if len(cur_coords) < min_cells:
                n_skipped_small += 1
                continue

            df_per_cell = local_pca_dimension_cell_min(cur_coords, radii,
                                                       min_neighbors=min_neighbors)

            n_cells_domain = len(df_per_cell)
            n_counted = int(df_per_cell["usable_any"].sum())

            cur_values = df_per_cell.loc[df_per_cell["usable_any"], "d_min_cell"]
            n_below = int((cur_values < threshold).sum())

            result_rows.append({
                "group": setting["group"],
                "setting": setting["tag"],
                "setting_label": setting["label"],
                "n_domains": setting["k"],
                "criterion": setting["criterion_of"][cur_domain],
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

    n_setting_rows = sum(1 for r in result_rows if r["setting"] == setting["tag"])

    print("measured " + str(n_setting_rows) + " domain-image groups, skipped " +
          str(n_skipped_small) + " below " + str(min_cells) + " cells, in " +
          format(time.time() - t_start, ".0f") + " s")


df_all = pd.DataFrame(result_rows)

result_file = os.path.join(figure_dir, "local_pca_dimension_" + output_tag + ".csv")
df_all.to_csv(result_file, index=False)


summary_rows = []

for _, cur_pair in df_match.iterrows():

    df_cur = df_all[(df_all["setting"] == cur_pair["tag"]) &
                    (df_all["domain"] == cur_pair["domain"])]
    cur_values = df_cur["proportion_below_threshold"].dropna()

    summary_rows.append({
        "group": cur_pair["group"],
        "setting": cur_pair["tag"],
        "setting_label": cur_pair["setting"],
        "n_domains": cur_pair["n_domains"],
        "domain": cur_pair["domain"],
        "criterion": cur_pair["criterion"],
        "n_images_measured": len(df_cur),
        "n_images_with_value": len(cur_values),
        "median_proportion_below": (float(cur_values.median())
                                    if len(cur_values) > 0 else np.nan),
        "q25_proportion_below": (float(cur_values.quantile(0.25))
                                 if len(cur_values) > 0 else np.nan),
        "q75_proportion_below": (float(cur_values.quantile(0.75))
                                 if len(cur_values) > 0 else np.nan)})

df_summary = pd.DataFrame(summary_rows)

summary_file = os.path.join(figure_dir, "local_pca_dimension_" + output_tag + "_by_pair.csv")
df_summary.to_csv(summary_file, index=False)

if os.path.isfile(reference_csv):
    df_reference = pd.read_csv(reference_csv)
    reference_values = df_reference.loc[(df_reference["method"] == "direct_extended") &
                                        (df_reference["domain"] == default_k),
                                        "proportion_below_threshold"].dropna()
    print("")
    print("for reference, default domain 10 in " + reference_csv + ": " +
          str(len(reference_values)) + " images with a value, median " +
          format(reference_values.median(), ".3f"))

print("")
pd.set_option("display.width", 200)
print(df_summary[["setting_label", "domain", "criterion", "n_images_measured",
                  "n_images_with_value", "median_proportion_below"]].to_string(index=False))

print("")
print("saved " + result_file)
print("saved " + summary_file)
