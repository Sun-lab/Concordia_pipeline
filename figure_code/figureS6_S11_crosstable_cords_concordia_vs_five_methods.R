library(data.table)
library(ggplot2)
library(patchwork)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "supp_crosstable_cords_concordia_vs_five_methods")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

n_domains = 10
domain_levels = as.character(1:n_domains)

raw_dir = "../data/Cords_data/raw_data"
region_list_csv = "../data/Cords_data/region_list.csv"

direct_root = "../revision/code/results_rs_ensemble_scaled_direct_domains/"
default_run_id = "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20"
n_repeats = 10

figure_width = 7.5
row_height = 3.45
extra_height = 1.1

row_gap = 26
title_gap = 12
base_font_size = 10
title_font_size = 10
value_font_size = 2.2
legend_font_size = 9

fill_low = "#075AFF"
fill_mid = "#FFFFCC"
fill_high = "#FF0000"
fill_midpoint = 0.5



direct_setting <- function(label, tag, graph_type, figure = 1){
  list(label = label, tag = tag, figure = figure, kind = "per_cell",
       path = paste0(direct_root, default_run_id, "__ensemble_", n_repeats, "/",
                     graph_type, "/domains_k", n_domains,
                     "/domain_annotation_for_cells.csv"),
       sep = ",", id_col = "CELL_ID", label_col = "domain", label_offset = 0)
}

reference_setting = direct_setting("Concordia", "concordia", "extended")

comparison_list = list(
  list(label = "K-means", tag = "kmeans", figure = 1, kind = "per_cell",
       path = "../results/cords_2024/extended/raw_kmeans/kmeans_cluster.csv",
       sep = ",", id_col = "CELL_ID", label_col = "kmeans_cluster", label_offset = 1),

  direct_setting("Concordia-Basic", "concordia_basic", "basic"),
  direct_setting("Concordia-Local", "concordia_local", "local"),

  list(label = "Spatial-LDA", tag = "spatial_lda", figure = 2, kind = "per_cell",
       path = "../SpatialLDA_copy/results/cords/cords_results_radius_27_d_0.25.tsv",
       sep = "\t", id_col = "cell_id", label_col = "predicted_cluster", label_offset = 1),

  list(label = "SpatialTopic", tag = "spatialtopic", figure = 2, kind = "per_image",
       path = "../SpaTopic_copy/results/cords/radius_62_sigma_8",
       suffix = "_spatopic_results.csv", label_col = "topic", label_offset = 0)
)

stopifnot(!any(duplicated(sapply(comparison_list, function(x) x$tag))))

figure_of = sapply(comparison_list, function(x) x$figure)



region_ids = read.csv(region_list_csv, header = TRUE)$region_ID

stopifnot(!any(duplicated(region_ids)))

read_annotation <- function(setting){

  if (setting$kind == "per_cell"){

    if (!file.exists(setting$path)){
      stop("missing ", setting$path, " for ", setting$label)
    }

    cell_domain = fread(setting$path, sep = setting$sep,
                        select = c(setting$id_col, setting$label_col))
    setnames(cell_domain, c("CELL_ID", "domain"))

  }else{

    if (!dir.exists(setting$path)){
      stop("missing ", setting$path, " for ", setting$label)
    }

    cell_domain = rbindlist(lapply(region_ids, function(cur_region_id){

      raw_cell = fread(file.path(raw_dir, paste0(cur_region_id, ".csv")),
                       select = c("CELL_ID", "X", "Y"))

      method_csv = file.path(setting$path, paste0(cur_region_id, setting$suffix))
      if (!file.exists(method_csv)){
        stop("missing ", method_csv)
      }

      df_method = fread(method_csv)

      stopifnot(nrow(df_method) == nrow(raw_cell))
      stopifnot(isTRUE(all.equal(df_method$X, raw_cell$X)))
      stopifnot(isTRUE(all.equal(df_method$Y, raw_cell$Y)))

      data.table(CELL_ID = raw_cell$CELL_ID, domain = df_method[[setting$label_col]])
    }))
  }

  cell_domain[, domain := as.integer(domain) + setting$label_offset]

  stopifnot(!any(duplicated(cell_domain$CELL_ID)))
  stopifnot(all(cell_domain$domain %in% 1:n_domains))

  cat(setting$label, ": ", nrow(cell_domain), " cells, domains ",
      min(cell_domain$domain), " to ", max(cell_domain$domain),
      " after an offset of ", setting$label_offset, "\n", sep = "")

  return(cell_domain)
}



build_heatmap <- function(cur_table, x_label, panel_title){

  df_plot = as.data.frame(as.table(cur_table))
  colnames(df_plot) = c("reference", "other", "value")

  ggplot(df_plot, aes(x = other, y = reference, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", value)), size = value_font_size) +
    scale_fill_gradient2(low = fill_low, mid = fill_mid, high = fill_high,
                         midpoint = fill_midpoint, limits = c(0, 1),
                         name = NULL) +
    guides(fill = guide_colorbar(barwidth = unit(6, "cm"),
                                 barheight = unit(0.4, "cm"))) +
    scale_x_discrete(expand = expansion(add = 0)) +
    scale_y_discrete(limits = rev(domain_levels), expand = expansion(add = 0)) +
    coord_fixed() +
    labs(title = panel_title, x = x_label, y = "Concordia domain") +
    theme_minimal(base_size = base_font_size) +
    theme(panel.grid = element_blank(),
          plot.title = element_text(size = title_font_size, hjust = 0.5,
                                    margin = margin(b = title_gap)),
          plot.margin = margin(row_gap / 2, 5.5, row_gap / 2, 5.5))
}



df_reference = read_annotation(reference_setting)
setnames(df_reference, "domain", "reference")

plot_list_of = lapply(sort(unique(figure_of)), function(x) list())
names(plot_list_of) = as.character(sort(unique(figure_of)))

for (comparison_index in seq_along(comparison_list)){

  setting = comparison_list[[comparison_index]]

  df_other = read_annotation(setting)
  setnames(df_other, "domain", "other")

  df_both = merge(df_reference, df_other, by = "CELL_ID")

  stopifnot(nrow(df_both) == nrow(df_reference))
  stopifnot(nrow(df_both) == nrow(df_other))

  counts = table(reference = factor(as.character(df_both$reference), levels = domain_levels),
                 other = factor(as.character(df_both$other), levels = domain_levels))

  stopifnot(sum(counts) == nrow(df_both))

  by_row = sweep(counts, 1, rowSums(counts), "/")
  by_col = sweep(counts, 2, colSums(counts), "/")

  stopifnot(!any(is.na(by_row)), !any(is.na(by_col)))
  stopifnot(all(abs(rowSums(by_row) - 1) < 1e-9))
  stopifnot(all(abs(colSums(by_col) - 1) < 1e-9))

  cur_figure = as.character(setting$figure)
  is_first = length(plot_list_of[[cur_figure]]) == 0

  plot_list_of[[cur_figure]][[paste0(setting$tag, "|row")]] = build_heatmap(
    by_row, paste(setting$label, "domain"),
    if (is_first) "Share of each Concordia domain\n(rows sum to 1)" else NULL)

  plot_list_of[[cur_figure]][[paste0(setting$tag, "|col")]] = build_heatmap(
    by_col, paste(setting$label, "domain"),
    if (is_first) "Share of each comparison domain\n(columns sum to 1)" else NULL)

  rm(df_other, df_both)
  gc()
}



for (cur_figure in names(plot_list_of)){

  cur_plot_list = plot_list_of[[cur_figure]]
  n_rows = length(cur_plot_list) / 2

  stopifnot(n_rows == sum(figure_of == as.integer(cur_figure)))

  combined = wrap_plots(cur_plot_list, ncol = 2, byrow = TRUE) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom",
          legend.text = element_text(size = legend_font_size))

  figure_file = file.path(figure_dir, paste0("crosstables_part_", cur_figure, ".pdf"))

  ggsave(figure_file, plot = combined, width = figure_width,
         height = row_height * n_rows + extra_height)

  cat("saved", figure_file, "\n")
}
