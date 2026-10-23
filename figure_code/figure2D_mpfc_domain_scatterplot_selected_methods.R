library(data.table)
library(ggplot2)
library(ggpubr)
library(clue)
library(grid)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = "../revision_figures"
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

file_prefix = "figure2_mpfc_domain_scatterplot_selected_methods"

raw_dir = "../data/mpfc_data/raw_data"

region_id_list = c("20180417_BZ5_control", "20180419_BZ9_control",
                   "20180424_BZ14_control")

n_domains = 4

n_kmeans_clusters = 16

all_region_types = c("1", "2", "3", "4")
all_layer_names = c("L1", "L2/3", "L5", "L6")

region_colors = setNames(c("#E41A1CFF", "#984EA3FF", "#4DAF4AFF", "#377EB8FF"),
                         all_layer_names)

merge_root = paste0("../revision/code/results_rs_ensemble_scaled/mpfc_2022_cpu/",
                    "mpfc_2022__group_default__base380__rad1140__topk4__ctg0p176__deg20__ensemble_10/")

merge_method <- function(label, graph_type){
  list(label = paste0(label, ", merge"), kind = "per_cell",
       path = paste0(merge_root, graph_type, "/clusters_k", n_kmeans_clusters,
                     "/domains_k", n_domains, "/domain_annotation_for_cells.csv"),
       sep = ",", id_col = "CELL_ID", label_col = "domain")
}

methods = list(
  list(label = "K-means", kind = "kmeans_membership",
       path = "../results/mpfc_2022/extended/raw_kmeans"),

  list(label = "CytoCommunity", kind = "per_image_rows",
       path = "../CytoCommunity_copy/Tutorial/Unsupervised_mpfc/Step3_Output"),

  list(label = "Spatial-LDA", kind = "spatial_lda",
       path = "../SpatialLDA_copy/results/mpfc/mpfc_results_radius_690.tsv"),

  list(label = "SpatialTopic", kind = "per_image_coords",
       path = "../SpaTopic_copy/results/mpfc/radius_690_sigma_26",
       suffix = "_spatopic_results.csv", label_col = "topic"),

  merge_method("Concordia-Basic", "basic"),
  merge_method("Concordia-Local", "local"),
  merge_method("Concordia", "extended")
)



match_labels_to_truth <- function(method_label, truth_layer, all_layer_names){

  stopifnot(length(method_label) == length(truth_layer))

  contingency = table(factor(as.character(method_label),
                             levels = sort(unique(as.character(method_label)))),
                      factor(as.character(truth_layer), levels = all_layer_names))

  if (nrow(contingency) < ncol(contingency)){
    padding = matrix(0, nrow = ncol(contingency) - nrow(contingency),
                     ncol = ncol(contingency))
    rownames(padding) = paste0("__empty_", seq_len(nrow(padding)))
    colnames(padding) = colnames(contingency)
    contingency = rbind(contingency, padding)
  }

  stopifnot(nrow(contingency) == ncol(contingency))

  assignment = clue::solve_LSAP(max(contingency) - contingency)

  mapped = setNames(colnames(contingency)[as.integer(assignment)],
                    rownames(contingency))

  mapped = mapped[!grepl("^__empty_", names(mapped))]

  return(mapped)
}



sort_domain_ids <- function(ids){

  ids = unique(as.character(ids))

  suppressWarnings(as_number <- as.numeric(ids))

  if (!any(is.na(as_number))){
    return(ids[order(as_number)])
  }

  return(sort(ids))
}



build_panel <- function(df_plot, panel_title, panel_colors){

  ggplot(df_plot, aes(x = X, y = Y, color = group)) +
    geom_point(size = 0.2) +
    scale_color_manual(values = panel_colors) +
    labs(title = panel_title) +
    guides(color = guide_legend(title = NULL, override.aes = list(size = 2.5))) +
    theme(plot.title = element_text(size = 11, hjust = 0.5),
          legend.position = "right",
          legend.text = element_text(size = 10),
          legend.key.height = unit(0.9, "lines"),
          legend.box.spacing = unit(2, "pt"),
          legend.margin = margin(0, 0, 0, 0),
          legend.key.width = unit(0.7, "lines"),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
}



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



for (region_id in region_id_list){

  cat("====", region_id, "====\n")

  df_cell = fread(file.path(raw_dir, paste0(region_id, ".csv")),
                  select = c("CELL_ID", "X", "Y", "DOMAIN"))

  n_cells = nrow(df_cell)

  stopifnot(all(as.character(df_cell$DOMAIN) %in% all_region_types))

  df_cell[, layer := factor(all_layer_names[match(as.character(DOMAIN), all_region_types)],
                            levels = all_layer_names)]

  cat("cells:", n_cells, "\n")

  df_cell[, group := factor(as.character(layer),
                            levels = all_layer_names[all_layer_names %in%
                                                       as.character(layer)])]

  ground_truth_panel = build_panel(as.data.frame(df_cell), "Manual annotation",
                                   region_colors[levels(df_cell$group)])

  method_panels = list()

  for (method in methods){

    if (method$kind %in% c("per_cell", "kmeans_membership")){

      cur = merge(df_cell[, .(CELL_ID, X, Y, layer)],
                  cell_label_of[[method$label]], by = "CELL_ID")
      stopifnot(nrow(cur) == n_cells)

    }else if (method$kind == "spatial_lda"){

      df_lda = cell_label_of[[method$label]]
      df_lda = df_lda[sample == region_id]

      cur = merge(df_cell[, .(CELL_ID, X, Y, layer)],
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

      cur = copy(df_cell[, .(CELL_ID, X, Y, layer)])
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

      cur = copy(df_cell[, .(CELL_ID, X, Y, layer)])
      cur[, label := df_label[[method$label_col]]]
    }

    layer_of_id = match_labels_to_truth(cur$label, cur$layer, all_layer_names)

    domain_ids = sort_domain_ids(cur$label)

    stopifnot(all(domain_ids %in% names(layer_of_id)))

    cur[, group := factor(as.character(label), levels = domain_ids)]

    panel_colors = setNames(unname(region_colors[layer_of_id[domain_ids]]),
                            domain_ids)

    stopifnot(!any(is.na(panel_colors)))

    method_panels[[method$label]] = build_panel(as.data.frame(cur), method$label,
                                                panel_colors)

    cat("  drew", method$label, "\n")
  }

  stopifnot(length(method_panels) == length(methods))

  n_col = 3
  n_row = ceiling((1 + length(methods)) / n_col)

  plot_list = c(list(ground_truth_panel), method_panels)

  stopifnot(length(plot_list) == 1 + length(methods))
  stopifnot(length(plot_list) <= n_col * n_row)

  figure_file = file.path(figure_dir, paste0(file_prefix, "_", region_id, ".pdf"))

  inch_per_col = 5.6 * (1 / 2)
  inch_per_row = 2.6 * (1 / 2) + 0.3

  pdf(figure_file, width = inch_per_col * n_col, height = inch_per_row * n_row)
  print(ggarrange(plotlist = plot_list, ncol = n_col, nrow = n_row))
  dev.off()

  cat("saved", figure_file, "\n\n")
}

cat("wrote", length(region_id_list), "figures into", figure_dir, "\n")
