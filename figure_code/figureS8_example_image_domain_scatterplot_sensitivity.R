library(data.table)
library(ggplot2)
library(gridExtra)
library(grid)
library(RColorBrewer)
library(patchwork)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure5_example_image_domain_scatterplot_sensitivity")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

n_domains = 10

n_columns = 4

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



group_ct_mapping = list()

group_ct_mapping[["immune"]] = c("Bcell", "CD4", "CD4_Treg", "CD8",
                                 "IDO_CD4", "IDO_CD8", "ki67_CD4", "ki67_CD8",
                                 "Myeloid", "Neutrophil", "PD1_CD4",
                                 "TCF1/7_CD4", "TCF1/7_CD8")
group_ct_mapping[["tumor"]] = c("hypoxic", "normal")
group_ct_mapping[["CAF"]] = c("Collagen_CAF", "dCAF", "hypoxic_CAF",
                              "hypoxic_tpCAF", "iCAF", "IDO_CAF",
                              "mCAF", "PDPN_CAF", "SMA_CAF", "tpCAF",
                              "vCAF")
group_ct_mapping[["vessel"]] = c("Blood", "HEV", "Lymphatic")
group_ct_mapping[["Other"]] = c("Other")

ct_group_colors = setNames(c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"),
                           names(group_ct_mapping))

ct_group_matching = rbindlist(lapply(names(group_ct_mapping), function(x){
  data.table(ct_group = x, CELL_TYPE = group_ct_mapping[[x]])
}))

stopifnot(!any(duplicated(ct_group_matching$CELL_TYPE)))

ct_panel_label = "Cell type"



panel_theme = theme(
  legend.text = element_text(size = legend_font_size,
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



build_ct_group_panel <- function(df_cell){

  ggplot(df_cell, aes(x = X, y = Y, color = ct_group)) +
    geom_point(size = 0.2) +
    scale_color_manual(values = ct_group_colors) +
    guides(color = guide_legend(title = NULL, override.aes = list(size = 3))) +
    labs(title = ct_panel_label) +
    panel_theme
}



build_domain_scatterplot <- function(df_cell, setting_label, n_domains){

  df_cell$domain = factor(as.character(df_cell$domain),
                          levels = as.character(1:n_domains))

  stopifnot(!any(is.na(df_cell$domain)))

  ggplot(df_cell, aes(x = X, y = Y, color = domain)) +
    geom_point(size = 0.2) +
    scale_color_manual(values = domain_colors) +
    guides(color = guide_legend(title = NULL, override.aes = list(size = 3))) +
    labs(title = setting_label) +
    panel_theme
}



sensitivity_run_id <- function(data_name = "cords_2024", group_scheme = "default",
                               dist_cutoff = "16", radius = "48", top_k = "4",
                               ctg = "0p176", degree_limit = "20"){
  paste0(data_name, "__group_", group_scheme, "__base", dist_cutoff, "__rad", radius,
         "__topk", top_k, "__ctg", ctg, "__deg", degree_limit)
}

direct_setting <- function(label, model_run_id, graph_type = "extended"){
  list(label = label,
       kind = "per_cell",
       path = paste0(direct_root, model_run_id, "__ensemble_", n_repeats, "/",
                     graph_type, "/domains_k", n_domains,
                     "/domain_annotation_for_cells.csv"),
       sep = ",", id_col = "CELL_ID", label_col = "domain", label_offset = 0)
}

settings_list = list(

  direct_setting("Concordia\ndefault", sensitivity_run_id()),

  direct_setting("Distance cutoff\n14.5, radius 43.5",
                 sensitivity_run_id(dist_cutoff = "14p5", radius = "43p5")),
  direct_setting("Distance cutoff\n17, radius 51",
                 sensitivity_run_id(dist_cutoff = "17", radius = "51")),

  direct_setting("Extension radius\n32", sensitivity_run_id(radius = "32")),
  direct_setting("Extension radius\n64", sensitivity_run_id(radius = "64")),

  direct_setting("Top k\n3", sensitivity_run_id(top_k = "3")),
  direct_setting("Top k\n5", sensitivity_run_id(top_k = "5")),

  direct_setting("Composition cutoff\n0.157", sensitivity_run_id(ctg = "0p157")),
  direct_setting("Composition cutoff\n0.202", sensitivity_run_id(ctg = "0p202")),

  direct_setting("Degree limit\n16", sensitivity_run_id(degree_limit = "16")),
  direct_setting("Degree limit\n24", sensitivity_run_id(degree_limit = "24")),

  direct_setting("Cell type groups\ngroup8", sensitivity_run_id(group_scheme = "group8")),

  direct_setting("Permuted data\npermutate5pct",
                 sensitivity_run_id(data_name = "cords_permutate5pct")),
  direct_setting("Permuted data\npermutate10pct",
                 sensitivity_run_id(data_name = "cords_permutate"))
)

stopifnot(!any(duplicated(sapply(settings_list, function(x) x$label))))


panel_layout = list(
  c(ct_panel_label, "Concordia\ndefault", NA, NA),
  c("Distance cutoff\n14.5, radius 43.5", "Distance cutoff\n17, radius 51",
    "Extension radius\n32", "Extension radius\n64"),
  c("Top k\n3", "Top k\n5", "Composition cutoff\n0.157", "Composition cutoff\n0.202"),
  c("Degree limit\n16", "Degree limit\n24", "Cell type groups\ngroup8", NA),
  c("Permuted data\npermutate5pct", "Permuted data\npermutate10pct", NA, NA)
)

stopifnot(all(sapply(panel_layout, length) == n_columns))

n_rows = length(panel_layout)

placed_labels = unlist(panel_layout)
placed_labels = placed_labels[!is.na(placed_labels)]

stopifnot(!any(duplicated(placed_labels)))
stopifnot(setequal(placed_labels,
                   c(ct_panel_label, sapply(settings_list, function(x) x$label))))



raw_cell_of = list()

for (region_id in region_id_list){

  raw_csv = file.path(raw_dir, paste0(region_id, ".csv"))

  if (!file.exists(raw_csv)){
    stop("missing ", raw_csv)
  }

  cur_cell = fread(raw_csv, select = c("CELL_ID", "X", "Y", "CELL_TYPE"))

  cur_cell[, ct_group := factor(
    ct_group_matching$ct_group[match(CELL_TYPE, ct_group_matching$CELL_TYPE)],
    levels = names(group_ct_mapping))]

  stopifnot(!any(is.na(cur_cell$ct_group)))

  raw_cell_of[[region_id]] = cur_cell
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



for (region_id in region_id_list){

  plot_list = list()

  for (cur_row in panel_layout){
    for (cur_label in cur_row){
      if (is.na(cur_label)){
        cur_panel = plot_spacer()
      }else if (cur_label == ct_panel_label){
        cur_panel = build_ct_group_panel(as.data.frame(raw_cell_of[[region_id]]))
      }else{
        cur_panel = panel_of[[paste0(region_id, "|", cur_label)]]
        stopifnot(!is.null(cur_panel))
      }
      plot_list[[length(plot_list) + 1]] = cur_panel
    }
  }

  stopifnot(length(plot_list) == n_rows * n_columns)

  combined = wrap_plots(plot_list, ncol = n_columns)

  figure_file = file.path(figure_dir, paste0(region_id, ".pdf"))

  ggsave(figure_file, plot = combined,
         width = panel_width * n_columns, height = panel_height * n_rows)

  cat("saved", figure_file, "\n")
}

cat("\nwrote", length(region_id_list), "figures into", figure_dir, "\n")
