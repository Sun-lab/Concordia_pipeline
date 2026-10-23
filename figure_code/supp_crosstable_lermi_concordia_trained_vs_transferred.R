library(data.table)
library(ggplot2)
library(patchwork)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "supp_crosstable_lermi_concordia_trained_vs_transferred")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

n_domains = 10
domain_levels = as.character(1:n_domains)

trained_path = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                      "lermi_2025__group_default__base16__rad48__topk4__ctg0p176__deg20__ensemble_10/",
                      "extended/domains_k", n_domains, "/domain_annotation_for_cells.csv")

transferred_path = paste0("../revision/code/results_rs_transfer/",
                          "lermi_2025__group_default__base16__rad48__topk4__ctg0p176__deg20/",
                          "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20__ensemble_10/",
                          "extended/domains_k", n_domains, "/domain_annotation_for_cells.csv")

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


read_annotation <- function(cur_path, label){

  if (!file.exists(cur_path)){
    stop("missing ", cur_path, " for ", label)
  }

  cell_domain = fread(cur_path, select = c("CELL_ID", "domain"))
  cell_domain[, domain := as.integer(domain)]

  stopifnot(!any(duplicated(cell_domain$CELL_ID)))
  stopifnot(all(cell_domain$domain %in% 1:n_domains))

  cat(label, ": ", nrow(cell_domain), " cells, domains ",
      min(cell_domain$domain), " to ", max(cell_domain$domain), "\n", sep = "")

  return(cell_domain)
}


build_heatmap <- function(cur_table, panel_title){

  df_plot = as.data.frame(as.table(cur_table))
  colnames(df_plot) = c("trained", "transferred", "value")

  ggplot(df_plot, aes(x = transferred, y = trained, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", value)), size = value_font_size) +
    scale_fill_gradient2(low = fill_low, mid = fill_mid, high = fill_high,
                         midpoint = fill_midpoint, limits = c(0, 1),
                         name = NULL) +
    guides(fill = guide_colorbar(barwidth = unit(6, "cm"),
                                 barheight = unit(0.4, "cm"))) +
    scale_x_discrete(expand = expansion(add = 0)) +
    scale_y_discrete(limits = rev(domain_levels),
                     expand = expansion(add = 0)) +
    coord_fixed() +
    labs(title = panel_title,
         x = "Domain, transferred from Cords",
         y = "Domain, trained on Lermi") +
    theme_minimal(base_size = base_font_size) +
    theme(panel.grid = element_blank(),
          plot.title = element_text(size = title_font_size, hjust = 0.5,
                                    margin = margin(b = title_gap)),
          plot.margin = margin(row_gap / 2, 5.5, row_gap / 2, 5.5))
}


df_trained = read_annotation(trained_path, "trained on Lermi")
setnames(df_trained, "domain", "trained")

df_transferred = read_annotation(transferred_path, "transferred from Cords")
setnames(df_transferred, "domain", "transferred")

df_both = merge(df_trained, df_transferred, by = "CELL_ID")

stopifnot(nrow(df_both) == nrow(df_trained))
stopifnot(nrow(df_both) == nrow(df_transferred))

counts = table(trained = factor(as.character(df_both$trained), levels = domain_levels),
               transferred = factor(as.character(df_both$transferred), levels = domain_levels))

stopifnot(sum(counts) == nrow(df_both))

by_row = sweep(counts, 1, rowSums(counts), "/")
by_col = sweep(counts, 2, colSums(counts), "/")

stopifnot(!any(is.na(by_row)), !any(is.na(by_col)))
stopifnot(all(abs(rowSums(by_row) - 1) < 1e-9))
stopifnot(all(abs(colSums(by_col) - 1) < 1e-9))

write.csv(as.data.frame.matrix(counts), file.path(figure_dir, "counts.csv"))
write.csv(as.data.frame.matrix(by_row), file.path(figure_dir, "by_row_sum.csv"))
write.csv(as.data.frame.matrix(by_col), file.path(figure_dir, "by_column_sum.csv"))

p_row = build_heatmap(by_row, "Share of each Lermi-trained domain\n(rows sum to 1)")
p_col = build_heatmap(by_col, "Share of each transferred domain\n(columns sum to 1)")

combined = wrap_plots(list(p_row, p_col), ncol = 1) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom",
        legend.text = element_text(size = legend_font_size))

figure_file = file.path(figure_dir, "crosstables.pdf")

ggsave(figure_file, plot = combined,
       width = n_domains * tile_size + side_room,
       height = 2 * (n_domains * tile_size + panel_vertical_room) + legend_room)

cat("saved", figure_file, "\n")

sessionInfo()
