args = commandArgs(trailingOnly=TRUE)
args

if (length(args) < 6) {
  message("at least six arguments are expected: data_name, results_dir, radius, ",
          "sigma, n_kmeans_clusters, n_domains; data_dir is optional\n")
  quit(save="no")
}else{
  for (arg in args){
    eval(parse(text=arg))
  }
}

for (required in c("data_name", "results_dir", "radius", "sigma",
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
sigma = as.character(sigma)

n_kmeans_clusters = as.integer(n_kmeans_clusters)
n_domains = as.integer(n_domains)

data_name
results_dir
radius
sigma
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

dir = file.path(results_dir, paste0("radius_", radius, "_sigma_", sigma))
domain_dir = file.path(dir, paste0("domains_k", n_domains))

if (!dir.exists(dir)){
  stop("missing ", dir,
       "; check the radius and sigma pair against step2_spatopic_mpfc_16topics.R")
}

for (needed in c("embedding_dist_in_image", "coord_dist_in_image")){
  if (!dir.exists(file.path(dir, needed))){
    stop("missing ", file.path(dir, needed),
         "; run cluster_dist_in_image_spatopic_celltype_composition.py first")
  }
}


df_cluster = NULL

for (cur_region in region_ids){

  results_file = file.path(dir, paste0(cur_region, "_spatopic_results.csv"))

  if (!file.exists(results_file)){
    stop("missing ", results_file)
  }

  df_raw_cur = read.csv(file.path(raw_dir, paste0(cur_region, ".csv")), header=TRUE)
  df_topic_cur = read.csv(results_file, header=TRUE)

  for (needed in c("X", "Y", "topic")){
    if (!(needed %in% colnames(df_topic_cur))){
      stop(results_file, " has no column ", needed)
    }
  }

  if (nrow(df_topic_cur) != nrow(df_raw_cur)){
    stop("row count mismatch for ", cur_region, ": ", nrow(df_raw_cur),
         " raw cells against ", nrow(df_topic_cur), " in ", results_file)
  }

  if (!isTRUE(all.equal(df_topic_cur$X, df_raw_cur$X))){
    stop("X does not line up for ", cur_region, " in ", results_file,
         "; the rows are not the raw file's rows")
  }

  if (!isTRUE(all.equal(df_topic_cur$Y, df_raw_cur$Y))){
    stop("Y does not line up for ", cur_region, " in ", results_file,
         "; the rows are not the raw file's rows")
  }

  if (!all(df_topic_cur$topic %in% 1:n_kmeans_clusters)){
    stop("topics outside 1 to ", n_kmeans_clusters, " in ", results_file,
         "; n_kmeans_clusters does not match this run")
  }

  df_cluster = rbind(df_cluster,
                     data.frame(CELL_ID = df_raw_cur$CELL_ID,
                                kmeans_cluster = df_topic_cur$topic - 1L))
}

if (any(duplicated(df_cluster$CELL_ID))){
  stop("duplicated cell ids after joining the per-image tables; the raw files ",
       "do not number their cells uniquely across images")
}

stopifnot(all(df_cluster$kmeans_cluster %in% 0:(n_kmeans_clusters-1)))

cluster_file = file.path(dir, "kmeans_cluster.csv")
write.csv(df_cluster, cluster_file, row.names=FALSE)

cat("converted", nrow(df_cluster), "cells over", length(region_ids),
    "images into", cluster_file, "\n")
cat("topics 1 to", n_kmeans_clusters, "in the files became 0 to",
    n_kmeans_clusters-1, "here\n")
print(table(df_cluster$kmeans_cluster))


get_n_cells_in_clusters(raw_dir, dir, region_ids, n_kmeans_clusters)

save_domain_annotations_drop_absent(dir, domain_dir, region_ids, n_kmeans_clusters,
                                    n_cells_threshold, n_images_threshold,
                                    n_domains)




gc()
sessionInfo()

q(save="no")
