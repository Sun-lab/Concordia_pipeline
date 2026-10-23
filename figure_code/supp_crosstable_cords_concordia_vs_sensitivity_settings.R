library(data.table)
library(ggplot2)
library(patchwork)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "supp_crosstable_cords_concordia_vs_sensitivity_settings")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

n_domains = 10
domain_levels = as.character(1:n_domains)

direct_root = "../revision/code/results_rs_ensemble_scaled_direct_domains/"
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



sensitivity_run_id <- function(data_name = "cords_2024", group_scheme = "default",
                               dist_cutoff = "16", radius = "48", top_k = "4",
                               ctg = "0p176", degree_limit = "20"){
  paste0(data_name, "__group_", group_scheme, "__base", dist_cutoff, "__rad", radius,
         "__topk", top_k, "__ctg", ctg, "__deg", degree_limit)
}

direct_setting <- function(label, tag, model_run_id, figure = NA){
  list(label = label, tag = tag, figure = figure,
       path = paste0(direct_root, model_run_id, "__ensemble_", n_repeats,
                     "/extended/domains_k", n_domains,
                     "/domain_annotation_for_cells.csv"))
}

reference_setting = direct_setting("default", "default", sensitivity_run_id())

comparison_list = list(
  direct_setting("distance cutoff = 14.5, radius = 43.5", "dist_14p5",
                 sensitivity_run_id(dist_cutoff = "14p5", radius = "43p5"), 1),
  direct_setting("distance cutoff = 17, radius = 51", "dist_17",
                 sensitivity_run_id(dist_cutoff = "17", radius = "51"), 1),
  direct_setting("extension radius = 32", "rad_32", sensitivity_run_id(radius = "32"), 1),
  direct_setting("extension radius = 64", "rad_64", sensitivity_run_id(radius = "64"), 1),

  direct_setting("top k = 3", "topk_3", sensitivity_run_id(top_k = "3"), 2),
  direct_setting("top k = 5", "topk_5", sensitivity_run_id(top_k = "5"), 2),
  direct_setting("composition cutoff = 0.157", "ctg_0p157",
                 sensitivity_run_id(ctg = "0p157"), 2),
  direct_setting("composition cutoff = 0.202", "ctg_0p202",
                 sensitivity_run_id(ctg = "0p202"), 2),

  direct_setting("degree limit = 16", "deg_16", sensitivity_run_id(degree_limit = "16"), 3),
  direct_setting("degree limit = 24", "deg_24", sensitivity_run_id(degree_limit = "24"), 3),
  direct_setting("cell type groups = group8", "group8",
                 sensitivity_run_id(group_scheme = "group8"), 3),

  direct_setting("permuted data, permutate5pct", "permutate5pct",
                 sensitivity_run_id(data_name = "cords_permutate5pct"), 4),
  direct_setting("permuted data, permutate", "permutate",
                 sensitivity_run_id(data_name = "cords_permutate"), 4)
)

stopifnot(!any(duplicated(sapply(comparison_list, function(x) x$tag))))

figure_of = sapply(comparison_list, function(x) x$figure)

stopifnot(!any(is.na(figure_of)))



read_annotation <- function(setting){

  if (!file.exists(setting$path)){
    stop("missing ", setting$path, " for ", setting$label)
  }

  cell_domain = fread(setting$path, select = c("CELL_ID", "domain"))
  cell_domain[, domain := as.integer(domain)]

  stopifnot(!any(duplicated(cell_domain$CELL_ID)))
  stopifnot(all(cell_domain$domain %in% 1:n_domains))

  cat(setting$label, ": ", nrow(cell_domain), " cells, domains ",
      min(cell_domain$domain), " to ", max(cell_domain$domain), "\n", sep = "")

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
    labs(title = panel_title, x = x_label, y = "Domain, default") +
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

  counts = table(default = factor(as.character(df_both$reference), levels = domain_levels),
                 setting = factor(as.character(df_both$other), levels = domain_levels))

  stopifnot(sum(counts) == nrow(df_both))

  by_row = sweep(counts, 1, rowSums(counts), "/")
  by_col = sweep(counts, 2, colSums(counts), "/")

  stopifnot(!any(is.na(by_row)), !any(is.na(by_col)))
  stopifnot(all(abs(rowSums(by_row) - 1) < 1e-9))
  stopifnot(all(abs(colSums(by_col) - 1) < 1e-9))

  write.csv(as.data.frame.matrix(counts),
            file.path(figure_dir, paste0("counts_", setting$tag, ".csv")))
  write.csv(as.data.frame.matrix(by_row),
            file.path(figure_dir, paste0("by_row_sum_", setting$tag, ".csv")))
  write.csv(as.data.frame.matrix(by_col),
            file.path(figure_dir, paste0("by_column_sum_", setting$tag, ".csv")))

  cur_figure = as.character(setting$figure)
  is_first = length(plot_list_of[[cur_figure]]) == 0

  x_label = paste0("Domain, ", setting$label)

  plot_list_of[[cur_figure]][[paste0(setting$tag, "|row")]] = build_heatmap(
    by_row, x_label,
    if (is_first) "Share of each default domain\n(rows sum to 1)" else NULL)

  plot_list_of[[cur_figure]][[paste0(setting$tag, "|col")]] = build_heatmap(
    by_col, x_label,
    if (is_first) "Share of each sensitivity-setting domain\n(columns sum to 1)" else NULL)

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
