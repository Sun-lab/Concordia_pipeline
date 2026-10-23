args = commandArgs(trailingOnly=TRUE)
args

if (!(length(args) %in% c(4, 5, 6, 7))) {
  message("four to seven arguments are expected.\n")
  quit(save="no")
}else{
  for (arg in args){
    eval(parse(text=arg))
  }
}

if (!exists("model_run_id")){
  model_run_id = NULL
}

if (!exists("data_dir")){
  data_dir = NULL
}

if (!exists("n_repeats")){
  n_repeats = 10
}

n_kmeans_clusters = as.integer(n_kmeans_clusters)
n_domains = as.integer(n_domains)
n_repeats = as.integer(n_repeats)

model_run_id

data_name
graph_type
n_kmeans_clusters
n_domains
n_repeats

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
source("_save_domain_annotations.R")


feature_list = data_features(data_name, model_run_id, data_dir)

df = feature_list$df
print(head(df))

region_ids = df$region_ID

print(length(unique(region_ids)))

result_subfolder = feature_list$result_subfolder
raw_dir = feature_list$raw_dir

n_cells_threshold = feature_list$n_cells_threshold
n_images_threshold = feature_list$n_images_threshold

ensemble_run_id = paste0(result_subfolder, "__ensemble_", n_repeats)

dir = file.path("./results_rs_ensemble_scaled", ensemble_run_id, graph_type,
                paste0("clusters_k", n_kmeans_clusters))
domain_dir = file.path(dir, paste0("domains_k", n_domains))

print(paste("reading the ensemble k-means and per-image distances from", dir))

get_n_cells_in_clusters(raw_dir, dir, region_ids, n_kmeans_clusters)

save_domain_annotations(dir, domain_dir, region_ids, n_kmeans_clusters,
                        n_cells_threshold, n_images_threshold,
                        n_domains)




gc()
sessionInfo()

q(save="no")
