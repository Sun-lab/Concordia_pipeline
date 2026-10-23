library(data.table)
library(tidyr)
library(dplyr)
library(ggplot2)
library(mclust)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = "../revision_figures"
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

file_prefix = "figure2_ari_all_methods_k16_merged_topics"

raw_dir = "../data/mpfc_data/raw_data"

region_id_list = c("20180417_BZ5_control", "20180419_BZ9_control",
                   "20180424_BZ14_control")

n_domains = 4

n_kmeans_clusters = 16

merge_root = paste0("../revision/code/results_rs_ensemble_scaled/mpfc_2022_cpu/",
                    "mpfc_2022__group_default__base380__rad1140__topk4__ctg0p176__deg20__ensemble_10/")

merge_method <- function(label, graph_type){
  list(label = paste0(label, ", k = ", n_kmeans_clusters), kind = "per_cell",
       path = paste0(merge_root, graph_type, "/clusters_k", n_kmeans_clusters,
                     "/domains_k", n_domains, "/domain_annotation_for_cells.csv"),
       sep = ",", id_col = "CELL_ID", label_col = "domain")
}

merged_topic_method <- function(label, path){
  list(label = paste0(label, ", k = ", n_kmeans_clusters), kind = "per_cell",
       path = paste0(path, "/domains_k", n_domains,
                     "/domain_annotation_for_cells.csv"),
       sep = ",", id_col = "CELL_ID", label_col = "domain")
}

direct_root = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                     "mpfc_2022__group_default__base380__rad1140__topk4__ctg0p176__deg20__ensemble_10/")

direct_method <- function(label, graph_type){
  list(label = label, kind = "per_cell",
       path = paste0(direct_root, graph_type, "/domains_k", n_domains,
                     "/domain_annotation_for_cells.csv"),
       sep = ",", id_col = "CELL_ID", label_col = "domain")
}

methods = list(
  list(label = "K-means", kind = "kmeans_membership",
       path = "../results/mpfc_2022/extended/raw_kmeans"),
  list(label = paste0("K-means, k = ", n_kmeans_clusters),
       kind = "kmeans_membership",
       path = "../results/mpfc_2022/extended/group_composition_2nd_basic"),

  list(label = "CytoCommunity", kind = "per_image_rows",
       path = "../CytoCommunity_copy/Tutorial/Unsupervised_mpfc/Step3_Output"),

  list(label = "Spatial-LDA", kind = "spatial_lda",
       path = "../SpatialLDA_copy/results/mpfc/mpfc_results_radius_690.tsv"),
  merged_topic_method("Spatial-LDA",
                      "../SpatialLDA_copy/results/mpfc_16topics/radius_690"),

  list(label = "SpatialTopic", kind = "per_image_coords",
       path = "../SpaTopic_copy/results/mpfc/radius_690_sigma_26",
       suffix = "_spatopic_results.csv", label_col = "topic"),
  merged_topic_method("SpatialTopic",
                      "../SpaTopic_copy/results/mpfc_16topics/radius_690_sigma_26"),

  direct_method("Concordia-Basic", "basic"),
  merge_method("Concordia-Basic", "basic"),
  direct_method("Concordia-Local", "local"),
  merge_method("Concordia-Local", "local"),
  direct_method("Concordia", "extended"),
  merge_method("Concordia", "extended")
)

method_levels = sapply(methods, function(x) x$label)



cell_label_of = list()

for (method in methods){

  if (method$kind == "per_cell"){

    if (!file.exists(method$path)){
      stop("missing ", method$path, " for ", method$label)
    }

    cur = fread(method$path, sep = method$sep,
                select = c(method$id_col, method$label_col))
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

cat("read the whole-dataset annotations\n\n")



ari_rows = NULL

for (region_id in region_id_list){

  cat("====", region_id, "====\n")

  df_cell = fread(file.path(raw_dir, paste0(region_id, ".csv")),
                  select = c("CELL_ID", "X", "Y", "DOMAIN"))

  n_cells = nrow(df_cell)

  for (method in methods){

    if (method$kind %in% c("per_cell", "kmeans_membership")){

      cur = merge(df_cell[, .(CELL_ID, X, Y, DOMAIN)],
                  cell_label_of[[method$label]], by = "CELL_ID")
      stopifnot(nrow(cur) == n_cells)

    }else if (method$kind == "spatial_lda"){

      df_lda = cell_label_of[[method$label]]
      df_lda = df_lda[sample == region_id]

      cur = merge(df_cell[, .(CELL_ID, X, Y, DOMAIN)],
                  df_lda[, .(CELL_ID = cell_id, label = predicted_cluster)],
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

      cur = copy(df_cell[, .(CELL_ID, X, Y, DOMAIN)])
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

      cur = copy(df_cell[, .(CELL_ID, X, Y, DOMAIN)])
      cur[, label := df_label[[method$label_col]]]
    }

    cur_ari = mclust::adjustedRandIndex(as.character(cur$label),
                                        as.character(cur$DOMAIN))

    ari_rows = rbind(ari_rows, data.frame(image_id = region_id,
                                          method = method$label,
                                          ARI = cur_ari,
                                          stringsAsFactors = FALSE))

    cat(sprintf("  %-22s ARI %.3f\n", method$label, cur_ari))
  }
}

stopifnot(nrow(ari_rows) == length(region_id_list) * length(methods))

ari_rows$method = factor(ari_rows$method, levels = method_levels)



df_wide = as.data.frame(ari_rows %>%
                          pivot_wider(names_from = method, values_from = ARI))

table_file = file.path(figure_dir, paste0(file_prefix, "_per_image.csv"))
write.csv(df_wide, table_file, row.names = FALSE)

cat("\n==== ARI against the manual annotation, by method ====\n")
print(ari_rows %>%
        group_by(method) %>%
        summarise(mean_ARI = mean(ARI), min_ARI = min(ARI), max_ARI = max(ARI),
                  .groups = "drop") %>%
        as.data.frame(), row.names = FALSE)



y_low = min(ari_rows$ARI)
y_high = max(ari_rows$ARI)
y_pad = 0.05 * (y_high - y_low)

ari_rows$method_group = sub(", k = [0-9]+$", "", as.character(ari_rows$method))
ari_rows$merged = ifelse(grepl(", k = ", as.character(ari_rows$method)),
                         "Yes", "No")

group_levels = unique(sub(", k = [0-9]+$", "", method_levels))
ari_rows$method_group = factor(ari_rows$method_group, levels = group_levels)
ari_rows$merged = factor(ari_rows$merged, levels = c("Yes", "No"))

stopifnot(!any(is.na(ari_rows$method_group)))
stopifnot(!any(is.na(ari_rows$merged)))

group_colors = setNames(c("#E41A1C", "#FF7F00", "#4DAF4A", "#984EA3",
                          "#377EB8", "#A65628", "#F781BF"), group_levels)
stopifnot(length(group_levels) == length(group_colors))

merge_shapes = setNames(c(16, 15), c("Yes", "No"))

p1 <- ggplot(ari_rows, aes(x = method, y = ARI, color = method_group,
                           shape = merged)) +
  geom_point(position = position_jitter(width = 0.06, height = 0), size = 1, alpha = 0.9) +
  scale_color_manual(values = group_colors) +
  scale_shape_manual(values = merge_shapes) +
  guides(color = guide_legend(title = NULL,
                              override.aes = list(size = 2, shape = 16),
                              ncol = 2, byrow = FALSE),
         shape = guide_legend(title = "merge",
                              override.aes = list(size = 2, color = "black"))) +
  labs(title = "  ") + ylim(y_low - y_pad, y_high + y_pad) +
  theme(legend.text = element_text(size = 12),
        legend.title = element_text(size = 12),
        legend.box = "horizontal",
        legend.box.just = "top",
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

figure_file = file.path(figure_dir, paste0(file_prefix, "_legend.pdf"))
pdf(file = figure_file, width = 8.1, height = 2.0)
print(p1)
dev.off()

figure_file_no_legend = file.path(figure_dir, paste0(file_prefix, "_no_legend.pdf"))
pdf(file = figure_file_no_legend, width = 2.7, height = 1.5)
print(p1 + theme(legend.position = "none"))
dev.off()

cat("\nsaved", table_file, "\n")
cat("saved", figure_file, "\n")
cat("saved", figure_file_no_legend, "\n")
