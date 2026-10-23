library(data.table)
library(ggplot2)
library(RColorBrewer)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure5_example_image_domain_scatterplot_spatialtopic_radius27")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

n_domains = 10

panel_width = 3
panel_height = 3

title_font_size = 16
legend_font_size = 13

stopifnot(title_font_size > legend_font_size)

region_id = "86_C6_10"

raw_dir = "../data/Cords_data/raw_data"

setting_label = "SpatialTopic (radius=27)\nDomains"
spatopic_dir = "../SpaTopic_copy/results/cords/radius_27_sigma_5"
spatopic_suffix = "_spatopic_results.csv"
label_col = "topic"
label_offset = 0



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



raw_csv = file.path(raw_dir, paste0(region_id, ".csv"))
if (!file.exists(raw_csv)){
  stop("missing ", raw_csv)
}

raw_cell = fread(raw_csv, select = c("CELL_ID", "X", "Y"))

cat("read", region_id, ":", nrow(raw_cell), "cells\n")



method_csv = file.path(spatopic_dir, paste0(region_id, spatopic_suffix))
if (!file.exists(method_csv)){
  stop("missing ", method_csv)
}

df_method = fread(method_csv)

stopifnot(nrow(df_method) == nrow(raw_cell))
stopifnot(isTRUE(all.equal(df_method$X, raw_cell$X)))
stopifnot(isTRUE(all.equal(df_method$Y, raw_cell$Y)))

cur_cell = copy(raw_cell)
cur_cell[, domain := as.integer(df_method[[label_col]]) + label_offset]

stopifnot(all(cur_cell$domain %in% 1:n_domains))

cat("cells per domain:\n")
print(table(domain = cur_cell$domain))



p = build_domain_scatterplot(as.data.frame(cur_cell), setting_label, n_domains)

figure_file = file.path(figure_dir, paste0(region_id, ".pdf"))

ggsave(figure_file, plot = p, width = panel_width, height = panel_height)

cat("\nsaved", figure_file, "\n")
