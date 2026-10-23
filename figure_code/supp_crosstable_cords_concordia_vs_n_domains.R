library(data.table)
library(ggplot2)
library(patchwork)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "supp_crosstable_cords_concordia_vs_n_domains")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

direct_root = "../revision/code/results_rs_ensemble_scaled_direct_domains/"
default_run_id = "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20"
n_repeats = 10

default_k = 10
compared_k = c(8, 12, 16, 20)

tile_size = 0.32
side_room = 1.0
panel_vertical_room = 1.0
legend_room = 0.6

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


annotation_path <- function(k){
  paste0(direct_root, default_run_id, "__ensemble_", n_repeats,
         "/extended/domains_k", k, "/domain_annotation_for_cells.csv")
}

read_annotation <- function(k){

  cur_path = annotation_path(k)

  if (!file.exists(cur_path)){
    stop("missing ", cur_path, " for k = ", k)
  }

  cell_domain = fread(cur_path, select = c("CELL_ID", "domain"))
  cell_domain[, domain := as.integer(domain)]

  stopifnot(!any(duplicated(cell_domain$CELL_ID)))
  stopifnot(all(cell_domain$domain %in% 1:k))

  cat("k = ", k, ": ", nrow(cell_domain), " cells, domains ",
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
    scale_y_discrete(limits = rev(as.character(1:default_k)),
                     expand = expansion(add = 0)) +
    coord_fixed() +
    labs(title = panel_title, x = x_label,
         y = paste0("Domain, default (k = ", default_k, ")")) +
    theme_minimal(base_size = base_font_size) +
    theme(panel.grid = element_blank(),
          plot.title = element_text(size = title_font_size, hjust = 0.5,
                                    margin = margin(b = title_gap)),
          plot.margin = margin(row_gap / 2, 5.5, row_gap / 2, 5.5))
}


df_reference = read_annotation(default_k)
setnames(df_reference, "domain", "reference")

reference_levels = as.character(1:default_k)

for (cur_k in compared_k){

  df_other = read_annotation(cur_k)
  setnames(df_other, "domain", "other")

  df_both = merge(df_reference, df_other, by = "CELL_ID")

  stopifnot(nrow(df_both) == nrow(df_reference))
  stopifnot(nrow(df_both) == nrow(df_other))

  other_levels = as.character(1:cur_k)

  counts = table(default = factor(as.character(df_both$reference), levels = reference_levels),
                 setting = factor(as.character(df_both$other), levels = other_levels))

  stopifnot(sum(counts) == nrow(df_both))

  by_row = sweep(counts, 1, rowSums(counts), "/")
  by_col = sweep(counts, 2, colSums(counts), "/")

  stopifnot(!any(is.na(by_row)), !any(is.na(by_col)))
  stopifnot(all(abs(rowSums(by_row) - 1) < 1e-9))
  stopifnot(all(abs(colSums(by_col) - 1) < 1e-9))

  k_tag = sprintf("k%02d", cur_k)

  write.csv(as.data.frame.matrix(counts),
            file.path(figure_dir, paste0("counts_k", cur_k, ".csv")))
  write.csv(as.data.frame.matrix(by_row),
            file.path(figure_dir, paste0("by_row_sum_k", cur_k, ".csv")))
  write.csv(as.data.frame.matrix(by_col),
            file.path(figure_dir, paste0("by_column_sum_k", cur_k, ".csv")))

  x_label = paste0("Domain, k = ", cur_k)

  p_row = build_heatmap(by_row, x_label,
                        "Share of each default domain\n(rows sum to 1)")
  p_col = build_heatmap(by_col, x_label,
                        paste0("Share of each k = ", cur_k, " domain\n(columns sum to 1)"))

  combined = wrap_plots(list(p_row, p_col), ncol = 1) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom",
          legend.text = element_text(size = legend_font_size))

  figure_file = file.path(figure_dir, paste0("crosstables_", k_tag, ".pdf"))

  ggsave(figure_file, plot = combined,
         width = cur_k * tile_size + side_room,
         height = 2 * (default_k * tile_size + panel_vertical_room) + legend_room)

  cat("saved", figure_file, "\n")

  rm(df_other, df_both)
  gc()
}
