args = commandArgs(trailingOnly=TRUE)
args

if (length(args) < 5) {
  message("at least five arguments are expected: data_name, results_dir, radius, ",
          "n_kmeans_clusters, n_domains; data_dir is optional\n")
  quit(save="no")
}else{
  for (arg in args){
    eval(parse(text=arg))
  }
}

for (required in c("data_name", "results_dir", "radius",
                   "n_kmeans_clusters", "n_domains")){
  if (!exists(required)){
    stop("missing argument: ", required)
  }
}

if (!exists("model_run_id")){
  model_run_id = NULL
}

if (!exists("data_dir")){
  data_dir = NULL
}

radius = as.character(radius)

n_kmeans_clusters = as.integer(n_kmeans_clusters)
n_domains = as.integer(n_domains)

data_name
results_dir
radius
n_kmeans_clusters
n_domains

library(tidyr)
library(dplyr)
library(data.table)
library(ggplot2)
library(ggpubr)
library(ggpointdensity)
theme_set(theme_classic())

library(ggExtra)
library(stringr)
library(gtools)
library(RColorBrewer)

options(bitmapType = "cairo")

source("data_utilities.R")

source("_get_n_cells_in_clusters.R")

source("_save_domain_annotations_drop_absent.R")


feature_list = data_features(data_name, model_run_id, data_dir)

df = feature_list$df
print(head(df))

region_ids = df$region_ID

print(length(unique(region_ids)))

raw_dir = feature_list$raw_dir

n_cells_threshold = feature_list$n_cells_threshold
n_images_threshold = feature_list$n_images_threshold

dir = file.path(results_dir, paste0("radius_", radius))
domain_dir = file.path(dir, paste0("domains_k", n_domains))

if (!dir.exists(dir)){
  stop("missing ", dir,
       "; run cluster_dist_in_image_spatial_lda_celltype_composition.py first")
}

for (needed in c("embedding_dist_in_image", "coord_dist_in_image")){
  if (!dir.exists(file.path(dir, needed))){
    stop("missing ", file.path(dir, needed),
         "; run cluster_dist_in_image_spatial_lda_celltype_composition.py first")
  }
}


results_file = file.path(results_dir,
                         paste0("mpfc_results_radius_", radius, ".tsv"))

if (!file.exists(results_file)){
  stop("missing ", results_file)
}

df_topic = fread(results_file, sep="\t", header=TRUE, data.table=FALSE)

for (needed in c("sample", "cell_id", "predicted_cluster")){
  if (!(needed %in% colnames(df_topic))){
    stop(results_file, " has no column ", needed)
  }
}

if (!all(df_topic$predicted_cluster %in% 0:(n_kmeans_clusters-1))){
  stop("topics outside 0 to ", n_kmeans_clusters-1, " in ", results_file,
       "; n_kmeans_clusters does not match this run")
}

df_cluster = data.frame(CELL_ID = df_topic$cell_id,
                        kmeans_cluster = df_topic$predicted_cluster)

if (any(duplicated(df_cluster$CELL_ID))){
  stop("duplicated cell ids in ", results_file,
       "; the conversion assumes one row per cell over all images")
}

all_raw_ids = NULL
for (cur_region in region_ids){
  df_raw_cur = read.csv(file.path(raw_dir, paste0(cur_region, ".csv")), header=TRUE)
  all_raw_ids = c(all_raw_ids, df_raw_cur$CELL_ID)
}

missing_ids = setdiff(all_raw_ids, df_cluster$CELL_ID)
if (length(missing_ids) > 0){
  stop(length(missing_ids), " cells of the raw files have no topic in ",
       results_file, ", the first being ", missing_ids[1])
}

cluster_file = file.path(dir, "kmeans_cluster.csv")
write.csv(df_cluster, cluster_file, row.names=FALSE)

cat("converted", nrow(df_cluster), "cells over", n_kmeans_clusters,
    "topics into", cluster_file, "\n")
print(table(df_cluster$kmeans_cluster))


get_n_cells_in_clusters(raw_dir, dir, region_ids, n_kmeans_clusters)

save_domain_annotations_drop_absent(dir, domain_dir, region_ids, n_kmeans_clusters,
                                    n_cells_threshold, n_images_threshold,
                                    n_domains)




gc()
sessionInfo()

q(save="no")
