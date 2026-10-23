library(data.table)
library(ggplot2)
library(grid)
library(RColorBrewer)
library(patchwork)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure5_example_image_domain_scatterplot_n_domains")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

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

default_run_id = "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20"

other_half_color = "#DDDDDD"



base_colors = c(brewer.pal(9, "Set1"), "#000000")

n_colors = length(base_colors)

stopifnot(n_colors == 10)
stopifnot(!any(duplicated(base_colors)))



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



build_domain_panel <- function(df_cell, panel_title, shown_domains){

  stopifnot(length(shown_domains) <= n_colors)

  shown_colors = setNames(base_colors[seq_along(shown_domains)],
                          as.character(shown_domains))

  in_shown = df_cell$domain %in% shown_domains

  df_shown = df_cell[in_shown, ]
  df_shown$domain = factor(as.character(df_shown$domain),
                           levels = as.character(shown_domains))

  p1 = ggplot(df_cell, aes(x = X, y = Y))

  if (any(!in_shown)){
    p1 = p1 + geom_point(data = df_cell[!in_shown, ], size = 0.2,
                         color = other_half_color)
  }

  p1 +
    geom_point(data = df_shown, aes(color = domain), size = 0.2) +
    scale_color_manual(values = shown_colors) +
    guides(color = guide_legend(title = NULL, override.aes = list(size = 3))) +
    labs(title = panel_title) +
    panel_theme
}



annotation_path <- function(k){
  paste0(direct_root, default_run_id, "__ensemble_", n_repeats,
         "/extended/domains_k", k, "/domain_annotation_for_cells.csv")
}

k_setting <- function(k, all_label = paste0("k = ", k, "\nall domains")){

  if (k <= n_colors){
    panels = list(list(label = all_label, shown = 1:k))
  }else{
    stopifnot(k %% 2 == 0, k / 2 <= n_colors)
    half = k / 2
    panels = list(
      list(label = paste0("k = ", k, "\ndomains 1 to ", half), shown = 1:half),
      list(label = paste0("k = ", k, "\ndomains ", half + 1, " to ", k),
           shown = (half + 1):k))
  }

  list(k = k, path = annotation_path(k), panels = panels)
}

settings_list = list(
  k_setting(10, all_label = "Concordia\ndefault"),
  k_setting(8),
  k_setting(12),
  k_setting(16),
  k_setting(20)
)

panel_labels = unlist(lapply(settings_list, function(x) sapply(x$panels, function(y) y$label)))

stopifnot(!any(duplicated(panel_labels)))



panel_layout = list(
  c(ct_panel_label, "Concordia\ndefault", NA, NA),
  c("k = 8\nall domains", "k = 12\ndomains 1 to 6", "k = 12\ndomains 7 to 12", NA),
  c("k = 16\ndomains 1 to 8", "k = 16\ndomains 9 to 16",
    "k = 20\ndomains 1 to 10", "k = 20\ndomains 11 to 20")
)

stopifnot(all(sapply(panel_layout, length) == n_columns))

n_rows = length(panel_layout)

placed_labels = unlist(panel_layout)
placed_labels = placed_labels[!is.na(placed_labels)]

stopifnot(!any(duplicated(placed_labels)))
stopifnot(setequal(placed_labels, c(ct_panel_label, panel_labels)))



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

  if (!file.exists(setting$path)){
    stop("missing ", setting$path, " for k = ", setting$k)
  }

  cell_domain = fread(setting$path, select = c("CELL_ID", "domain"))
  cell_domain[, domain := as.integer(domain)]

  stopifnot(!any(duplicated(cell_domain$CELL_ID)))
  stopifnot(all(cell_domain$domain %in% 1:setting$k))

  n_domains_present = c()

  for (region_id in region_id_list){

    raw_cell = raw_cell_of[[region_id]]

    cur_cell = merge(raw_cell, cell_domain, by = "CELL_ID")
    stopifnot(nrow(cur_cell) == nrow(raw_cell))

    n_domains_present = c(n_domains_present, length(unique(cur_cell$domain)))

    for (cur_panel in setting$panels){
      panel_of[[paste0(region_id, "|", cur_panel$label)]] =
        build_domain_panel(as.data.frame(cur_cell), cur_panel$label, cur_panel$shown)
    }
  }

  cat("done: k = ", setting$k, " | domains used, per image: ",
      paste(n_domains_present, collapse = ", "), "\n", sep = "")
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
