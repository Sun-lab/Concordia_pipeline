library(data.table)
library(ggplot2)
library(gridExtra)
library(grid)
library(RColorBrewer)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure5_example_image_domain_scatterplot_six_methods")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

n_domains = 10

n_columns = 3
n_rows = 2

panel_width = 3
panel_height = 3

title_font_size = 16
legend_font_size = 13

stopifnot(title_font_size > legend_font_size)

region_id_list = c("86_C6_10", "86_C2_4", "175_B7_15", "178_B2_8", "87_C2_4")

raw_dir = "../data/Cords_data/raw_data"

direct_root = "../revision/code/results_rs_ensemble_scaled_direct_domains/"
n_repeats = 10



domain_colors = setNames(c(brewer.pal(9, "Set1"), "#000000"),
                         as.character(1:n_domains))

stopifnot(length(domain_colors) == n_domains)
stopifnot(!any(duplicated(domain_colors)))

cat("domain colours:\n")
cat(paste0("  domain ", format(names(domain_colors), width = 2), "  ",
           domain_colors, collapse = "\n"), "\n\n", sep = "")



build_domain_scatterplot <- function(df_cell, setting_label, n_domains){

  df_cell$domain = factor(as.character(df_cell$domain),
                          levels = as.character(1:n_domains))

  stopifnot(!any(is.na(df_cell$domain)))

  ggplot(df_cell, aes(x = X, y = Y, color = domain)) +
    geom_point(size = 0.2) +
    scale_color_manual(values = domain_colors) +
    guides(color = guide_legend(title = NULL, override.aes = list(size = 3))) +
    labs(title = setting_label) +
    theme(legend.text = element_text(size = legend_font_size,
                                     margin = margin(l = 1, unit = "pt")),
          legend.key.height = unit(0.7, "lines"),
          legend.box.spacing = unit(2, "pt"),
          legend.margin = margin(0, 0, 0, 0),
          plot.title = element_text(size = title_font_size, hjust = 0.5),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
}



default_run_id = "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20"

direct_setting <- function(label, model_run_id, graph_type = "extended"){
  list(label = label,
       kind = "per_cell",
       path = paste0(direct_root, model_run_id, "__ensemble_", n_repeats, "/",
                     graph_type, "/domains_k", n_domains,
                     "/domain_annotation_for_cells.csv"),
       sep = ",", id_col = "CELL_ID", label_col = "domain", label_offset = 0)
}

settings_list = list(

  direct_setting("Concordia-Basic\nDomains", default_run_id, "basic"),
  direct_setting("Concordia-Local\nDomains", default_run_id, "local"),
  direct_setting("Concordia\nDomains", default_run_id, "extended"),

  list(label = "K-means\nDomains",
       kind = "per_cell",
       path = "../results/cords_2024/extended/raw_kmeans/kmeans_cluster.csv",
       sep = ",", id_col = "CELL_ID", label_col = "kmeans_cluster", label_offset = 1),

  list(label = "Spatial-LDA\nDomains",
       kind = "per_cell",
       path = "../SpatialLDA_copy/results/cords/cords_results_radius_27_d_0.25.tsv",
       sep = "\t", id_col = "cell_id", label_col = "predicted_cluster", label_offset = 1),
  list(label = "SpatialTopic\nDomains",
       kind = "per_image",
       path = "../SpaTopic_copy/results/cords/radius_62_sigma_8",
       suffix = "_spatopic_results.csv", label_col = "topic", label_offset = 0)
)

stopifnot(length(settings_list) <= n_columns * n_rows)
stopifnot(length(settings_list) > n_columns * (n_rows - 1))

n_blank = n_columns * n_rows - length(settings_list)



raw_cell_of = list()

for (region_id in region_id_list){

  raw_csv = file.path(raw_dir, paste0(region_id, ".csv"))

  if (!file.exists(raw_csv)){
    stop("missing ", raw_csv)
  }

  raw_cell_of[[region_id]] = fread(raw_csv, select = c("CELL_ID", "X", "Y"))
}

cat("read", length(raw_cell_of), "images:",
    paste(sapply(raw_cell_of, nrow), collapse = ", "), "cells\n\n")



panel_of = list()

for (setting in settings_list){

  if (setting$kind == "per_image"){
    if (!dir.exists(setting$path)){
      stop("missing ", setting$path, " for ", gsub("\n", ", ", setting$label))
    }
    cell_domain = NULL
  }else{
    if (!file.exists(setting$path)){
      stop("missing ", setting$path, " for ", gsub("\n", ", ", setting$label))
    }

    cell_domain = fread(setting$path, sep = setting$sep,
                        select = c(setting$id_col, setting$label_col))
    setnames(cell_domain, c("CELL_ID", "domain"))
    cell_domain[, domain := as.integer(domain) + setting$label_offset]

    stopifnot(all(cell_domain$domain %in% 1:n_domains))
  }

  n_domains_present = c()

  for (region_id in region_id_list){

    raw_cell = raw_cell_of[[region_id]]

    if (setting$kind == "per_image"){

      method_csv = file.path(setting$path, paste0(region_id, setting$suffix))
      if (!file.exists(method_csv)){
        stop("missing ", method_csv)
      }

      df_method = fread(method_csv)

      stopifnot(nrow(df_method) == nrow(raw_cell))
      stopifnot(isTRUE(all.equal(df_method$X, raw_cell$X)))
      stopifnot(isTRUE(all.equal(df_method$Y, raw_cell$Y)))

      cur_cell = copy(raw_cell)
      cur_cell[, domain := as.integer(df_method[[setting$label_col]]) +
                 setting$label_offset]

    }else{

      cur_cell = merge(raw_cell, cell_domain, by = "CELL_ID")
      stopifnot(nrow(cur_cell) == nrow(raw_cell))
    }

    stopifnot(all(cur_cell$domain %in% 1:n_domains))

    n_domains_present = c(n_domains_present, length(unique(cur_cell$domain)))

    panel_of[[paste0(region_id, "|", setting$label)]] =
      build_domain_scatterplot(as.data.frame(cur_cell), setting$label, n_domains)
  }

  cat("done: ", gsub("\n", ", ", setting$label),
      " | domains used, per image: ", paste(n_domains_present, collapse = ", "),
      "\n", sep = "")
}



setting_labels = sapply(settings_list, function(x) x$label)

for (region_id in region_id_list){

  plot_list = list()

  for (cur_label in setting_labels){
    plot_list[[cur_label]] = panel_of[[paste0(region_id, "|", cur_label)]]
  }

  for (blank_index in seq_len(n_blank)){
    plot_list[[paste0("blank_", blank_index)]] = nullGrob()
  }

  stopifnot(length(plot_list) == n_rows * n_columns)

  combined = arrangeGrob(grobs = plot_list, ncol = n_columns)

  figure_file = file.path(figure_dir, paste0(region_id, ".pdf"))

  ggsave(figure_file, plot = combined,
         width = panel_width * n_columns, height = panel_height * n_rows)

  cat("saved", figure_file, "\n")
}

cat("\nwrote", length(region_id_list), "figures into", figure_dir, "\n")
