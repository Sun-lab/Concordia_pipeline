library(data.table)
library(mclust)

table_dir = "../revision_tables"
if (!dir.exists(table_dir)){
  dir.create(table_dir, recursive = TRUE)
}

table_file = file.path(table_dir, "tableS1_ari_with_merged.csv")

raw_dir = "../data/mpfc_data/raw_data"
region_id_list = c("20180417_BZ5_control", "20180419_BZ9_control",
                   "20180424_BZ14_control")

n_domains = 4
n_kmeans_clusters = 16

merge_root = paste0("../revision/code/results_rs_ensemble_scaled/mpfc_2022_cpu/",
                    "mpfc_2022__group_default__base380__rad1140__topk4__ctg0p176__deg20__ensemble_10/")
direct_root = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                     "mpfc_2022__group_default__base380__rad1140__topk4__ctg0p176__deg20__ensemble_10/")

lda_radii = c("490", "690", "1600")
topic_settings = c("490" = "radius_490_sigma_22", "690" = "radius_690_sigma_26",
                   "1600" = "radius_1600_sigma_40")

per_cell <- function(label, path){
  list(label = label, kind = "per_cell", path = path)
}

direct_method <- function(label, graph_type){
  per_cell(label, paste0(direct_root, graph_type, "/domains_k", n_domains,
                         "/domain_annotation_for_cells.csv"))
}

merge_method <- function(label, graph_type){
  per_cell(paste0(label, ", merge"),
           paste0(merge_root, graph_type, "/clusters_k", n_kmeans_clusters,
                  "/domains_k", n_domains, "/domain_annotation_for_cells.csv"))
}

direct_list = c(
  list(list(label = "K-means", kind = "kmeans_membership",
            path = "../results/mpfc_2022/extended/raw_kmeans")),
  list(direct_method("Concordia-Basic", "basic"),
       direct_method("Concordia-Local", "local"),
       direct_method("Concordia", "extended")),
  list(list(label = "CytoCommunity", kind = "per_image_rows",
            path = "../CytoCommunity_copy/Tutorial/Unsupervised_mpfc/Step3_Output")),
  lapply(lda_radii, function(r){
    list(label = paste0("Spatial LDA (radius_", r, ")"), kind = "spatial_lda",
         path = paste0("../SpatialLDA_copy/results/mpfc/mpfc_results_radius_", r, ".tsv"))
  }),
  lapply(lda_radii, function(r){
    list(label = paste0("Spatial Topic (radius_", r, ")"), kind = "per_image_coords",
         path = paste0("../SpaTopic_copy/results/mpfc/", topic_settings[[r]]),
         suffix = "_spatopic_results.csv", label_col = "topic")
  })
)

merged_list = c(
  list(list(label = "K-means, merge",
            kind = "kmeans_membership",
            path = "../results/mpfc_2022/extended/group_composition_2nd_basic")),
  list(merge_method("Concordia-Basic", "basic"),
       merge_method("Concordia-Local", "local"),
       merge_method("Concordia", "extended")),
  lapply(lda_radii, function(r){
    per_cell(paste0("Spatial LDA (radius_", r, "), merge"),
             paste0("../SpatialLDA_copy/results/mpfc_16topics/radius_", r,
                    "/domains_k", n_domains, "/domain_annotation_for_cells.csv"))
  }),
  lapply(lda_radii, function(r){
    per_cell(paste0("Spatial Topic (radius_", r, "), merge"),
             paste0("../SpaTopic_copy/results/mpfc_16topics/", topic_settings[[r]],
                    "/domains_k", n_domains, "/domain_annotation_for_cells.csv"))
  })
)

methods = c(direct_list, merged_list)
method_levels = sapply(methods, function(x) x$label)

stopifnot(!any(duplicated(method_levels)))

cell_label_of = list()

for (method in methods){

  if (method$kind == "per_cell"){

    if (!file.exists(method$path)){
      stop("missing ", method$path, " for ", method$label)
    }
    cur = fread(method$path, select = c("CELL_ID", "domain"))
    setnames(cur, c("CELL_ID", "label"))
    cell_label_of[[method$label]] = cur

  }else if (method$kind == "kmeans_membership"){

    cluster_csv = file.path(method$path, "kmeans_cluster.csv")
    member_csv = file.path(method$path,
                           "domain_weighted_dist_clustering_membership_training_kmeans.csv")
    for (cur_path in c(cluster_csv, member_csv)){
      if (!file.exists(cur_path)){
        stop("missing ", cur_path, " for ", method$label)
      }
    }
    df_cluster = fread(cluster_csv, select = c("CELL_ID", "kmeans_cluster"))
    df_member = fread(member_csv)
    df_cluster[, cluster := paste0("cluster_", kmeans_cluster)]
    cur = merge(df_cluster, df_member[, .(cluster, domain)], by = "cluster")
    stopifnot(nrow(cur) == nrow(df_cluster))
    cell_label_of[[method$label]] = cur[, .(CELL_ID, label = domain)]

  }else if (method$kind == "spatial_lda"){

    if (!file.exists(method$path)){
      stop("missing ", method$path, " for ", method$label)
    }
    cell_label_of[[method$label]] = fread(method$path, sep = "\t")
  }
}

ari_rows = NULL

for (region_id in region_id_list){

  df_cell = fread(file.path(raw_dir, paste0(region_id, ".csv")),
                  select = c("CELL_ID", "X", "Y", "DOMAIN"))
  n_cells = nrow(df_cell)

  for (method in methods){

    if (method$kind %in% c("per_cell", "kmeans_membership")){

      cur = merge(df_cell, cell_label_of[[method$label]], by = "CELL_ID")
      stopifnot(nrow(cur) == n_cells)

    }else if (method$kind == "spatial_lda"){

      df_lda = cell_label_of[[method$label]][sample == region_id]
      cur = merge(df_cell, df_lda[, .(CELL_ID = cell_id, label = predicted_cluster)],
                  by = "CELL_ID")
      stopifnot(nrow(cur) == n_cells)

    }else if (method$kind == "per_image_rows"){

      label_csv = file.path(method$path, paste0("Step3_Output_", region_id),
                            "TCNLabel_MajorityVoting.csv")
      if (!file.exists(label_csv)){
        stop("missing ", label_csv)
      }
      df_label = fread(label_csv, header = FALSE)
      stopifnot(nrow(df_label) == n_cells)
      cur = copy(df_cell)
      cur[, label := df_label$V1]

    }else{

      label_csv = file.path(method$path, paste0(region_id, method$suffix))
      if (!file.exists(label_csv)){
        stop("missing ", label_csv)
      }
      df_label = fread(label_csv)
      stopifnot(nrow(df_label) == n_cells)
      stopifnot(isTRUE(all.equal(df_label$X, df_cell$X)))
      stopifnot(isTRUE(all.equal(df_label$Y, df_cell$Y)))
      cur = copy(df_cell)
      cur[, label := df_label[[method$label_col]]]
    }

    ari_rows = rbind(ari_rows, data.frame(
      image_id = region_id, method = method$label,
      ARI = mclust::adjustedRandIndex(as.character(cur$label), as.character(cur$DOMAIN)),
      stringsAsFactors = FALSE))
  }
}

stopifnot(nrow(ari_rows) == length(region_id_list) * length(methods))

df_table = dcast(as.data.table(ari_rows), method ~ image_id, value.var = "ARI")
df_table = df_table[match(method_levels, df_table$method)]

df_out = as.data.frame(df_table[, ..region_id_list])
df_out[] = lapply(df_out, function(x) sprintf("%.3f", x))
rownames(df_out) = df_table$method

write.csv(df_out, table_file)

print(df_out)

cat("\nsaved", table_file, "\n")
