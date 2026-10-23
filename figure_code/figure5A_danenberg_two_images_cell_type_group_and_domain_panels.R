library(data.table)
library(ggplot2)
library(RColorBrewer)
library(patchwork)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "danenberg_two_images_cell_type_group_and_domain_panels")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

data_name = "danenberg_2022"
data_dir = "../data/Danenberg_data"

region_id_list = c("MB-0580-455", "MB-3688-634")

stopifnot(!any(duplicated(region_id_list)))

n_domains = 10
highlight_domain = 9

domain_csv = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                    "danenberg_2022__group_default__base20__rad60__topk4__ctg0p176__deg20",
                    "__ensemble_10/extended/domains_k10/domain_annotation_for_cells.csv")

point_size_of_row = c(0.6, 0.6)
highlight_point_size_of_row = c(0.8, 0.8)

stopifnot(length(point_size_of_row) == length(region_id_list),
          length(highlight_point_size_of_row) == length(region_id_list))
other_color = "#DDDDDD"
legend_font_size = 13
title_font_size = 16

figure_width = 10.6
figure_height = 6.4


source("../revision/code/data_utilities.R")

feature_list = data_features(data_name = data_name, data_dir = data_dir)

raw_dir = feature_list$raw_dir
group_ct_mapping = feature_list$group_ct_mapping

all_ct_groups = names(group_ct_mapping)

not_training = setdiff(region_id_list, feature_list$df$region_ID)
if (length(not_training) > 0){
  stop("not in the training set: ", paste(not_training, collapse = ", "))
}

base_palette = c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00")

stopifnot(length(all_ct_groups) <= length(base_palette))

ct_group_colors = setNames(base_palette[seq_along(all_ct_groups)], all_ct_groups)

ct_group_of = unlist(lapply(all_ct_groups, function(ct_grp){
  setNames(rep(ct_grp, length(group_ct_mapping[[ct_grp]])), group_ct_mapping[[ct_grp]])
}))

stopifnot(!any(duplicated(names(ct_group_of))))

ct_group_labels = setNames(all_ct_groups, all_ct_groups)
ct_group_labels[ct_group_labels == "stroma"] = "fibroblast"

domain_palette = brewer.pal(9, "Set1")
domain_palette[which(domain_palette == "#4DAF4A")] = "#1B9E77"
domain_palette[which(domain_palette == "#FFFF33")] = "#66A61E"
domain_palette = c(domain_palette, "#000000")

domain_colors = setNames(domain_palette, as.character(1:n_domains))

stopifnot(length(domain_colors) == n_domains)
stopifnot(!any(duplicated(domain_colors)))
stopifnot(highlight_domain %in% 1:n_domains)


if (!file.exists(domain_csv)){
  stop("missing ", domain_csv)
}

cell_domain = fread(domain_csv, select = c("CELL_ID", "domain"))
cell_domain[, domain := as.integer(domain)]

stopifnot(!any(duplicated(cell_domain$CELL_ID)))
stopifnot(all(cell_domain$domain %in% 1:n_domains))

region_cell_of = list()

for (cur_region_id in region_id_list){

  raw_csv = file.path(raw_dir, paste0(cur_region_id, ".csv"))

  if (!file.exists(raw_csv)){
    stop("missing ", raw_csv)
  }

  cur_cells = fread(raw_csv, select = c("CELL_ID", "X", "Y", "CELL_TYPE"))

  cur_cells[, ct_group := ct_group_of[CELL_TYPE]]
  stopifnot(!any(is.na(cur_cells$ct_group)))
  cur_cells[, ct_group := factor(ct_group, levels = all_ct_groups)]

  cur_domain = cell_domain$domain[match(cur_cells$CELL_ID, cell_domain$CELL_ID)]
  stopifnot(!any(is.na(cur_domain)))
  cur_cells[, domain := factor(as.character(cur_domain), levels = as.character(1:n_domains))]

  region_cell_of[[cur_region_id]] = cur_cells

  cat(cur_region_id, ": ", nrow(cur_cells), " cells, ",
      length(unique(cur_domain)), " of ", n_domains, " domains present, ",
      sum(cur_domain == highlight_domain), " cells in domain ", highlight_domain, "\n", sep = "")
}


panel_theme = theme(plot.title = element_text(size = title_font_size, hjust = 0.5),
                    legend.text = element_text(size = legend_font_size,
                                               margin = margin(l = 1, unit = "pt")),
                    legend.key.height = unit(0.7, "lines"),
                    legend.box.spacing = unit(2, "pt"),
                    legend.margin = margin(0, 0, 0, 0),
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    axis.text.x = element_blank(),
                    axis.ticks.x = element_blank(),
                    axis.text.y = element_blank(),
                    axis.ticks.y = element_blank())

build_panel <- function(region_cell, color_column, color_values, panel_title,
                        point_size, color_labels = waiver()){
  ggplot(region_cell, aes(x = X, y = Y, color = .data[[color_column]])) +
    geom_point(size = point_size, stroke = 0) +
    scale_color_manual(values = color_values, labels = color_labels) +
    guides(color = guide_legend(title = NULL, override.aes = list(size = 3))) +
    coord_fixed() +
    labs(title = panel_title) +
    panel_theme
}

build_highlight_panel <- function(region_cell, panel_title, point_size, highlight_point_size){
  in_domain = region_cell$domain == as.character(highlight_domain)
  ggplot(region_cell, aes(x = X, y = Y)) +
    geom_point(aes(color = ct_group), size = 0, stroke = 0, alpha = 0) +
    geom_point(data = region_cell[!in_domain], size = point_size, stroke = 0,
               color = other_color) +
    geom_point(data = region_cell[in_domain], aes(color = ct_group),
               size = highlight_point_size, stroke = 0) +
    scale_color_manual(values = ct_group_colors, labels = ct_group_labels, drop = FALSE) +
    guides(color = guide_legend(title = NULL, override.aes = list(size = 3, alpha = 1))) +
    coord_fixed() +
    labs(title = panel_title) +
    panel_theme
}

plot_list = list()

for (row_index in seq_along(region_id_list)){

  region_cell = region_cell_of[[region_id_list[row_index]]]
  is_top = row_index == 1

  cur_point_size = point_size_of_row[row_index]
  cur_highlight_point_size = highlight_point_size_of_row[row_index]

  plot_list[[length(plot_list) + 1]] = build_panel(
    region_cell, "ct_group", ct_group_colors,
    if (is_top) "Cell type" else NULL, cur_point_size, ct_group_labels)

  plot_list[[length(plot_list) + 1]] = build_panel(
    region_cell, "domain", domain_colors,
    if (is_top) "Concordia\ndomains" else NULL, cur_point_size)

  plot_list[[length(plot_list) + 1]] = build_highlight_panel(
    region_cell, if (is_top) paste0("Concordia\ndomain ", highlight_domain) else NULL,
    cur_point_size, cur_highlight_point_size)
}

stopifnot(length(plot_list) == 3 * length(region_id_list))

combined = wrap_plots(plot_list, ncol = 3, byrow = TRUE)

figure_file = file.path(figure_dir, "six_panels.pdf")

ggsave(figure_file, plot = combined, width = figure_width, height = figure_height)

cat("saved", figure_file, "\n")
