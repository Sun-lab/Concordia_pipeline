library(data.table)
library(ggplot2)
library(RColorBrewer)
library(patchwork)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "lermi_core19_cell_type_group_and_domain_panels")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

data_dir = "../data/Lermi_data"
raw_dir = file.path(data_dir, "raw_data")

example_region_id = "GSM9052318_Lung_Adenocarcinoma_TMA2_Xenium_multi_modal_core19"
short_name = "core19"

n_domains = 10
n_cells_threshold = 30

targets = list(
  list(tag = "lermi_trained",
       title = "Concordia",
       domain = 9,
       annotation = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                           "lermi_2025__group_default__base16__rad48__topk4__ctg0p176__deg20",
                           "__ensemble_10/extended/domains_k10/domain_annotation_for_cells.csv")),
  list(tag = "cords_transferred",
       title = "Concordia transferred",
       domain = 10,
       annotation = paste0("../revision/code/results_rs_transfer/",
                           "lermi_2025__group_default__base16__rad48__topk4__ctg0p176__deg20/",
                           "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20",
                           "__ensemble_10/extended/domains_k10/domain_annotation_for_cells.csv"))
)

point_size = 0.45
highlight_point_size = 0.6
other_color = "#DDDDDD"

tissue_title_font_size = 16
tissue_legend_font_size = 14

title_font_size = 16
legend_font_size = 14

panel_margin_x = 1
panel_margin_y = 5

tissue_width = 4.2
tissue_height = 3.55

domain_column_width = 4.2
domain_figure_height = 7.0


group_ct_mapping = list()
group_ct_mapping[["immune"]] = c("CD4+ T cells", "CD8+ T cells", "B cells",
                                 "Plasma cells", "Alveolar macrophages",
                                 "Interstitial macrophages", "DC",
                                 "Classical monocytes", "Mast cells")
group_ct_mapping[["tumor"]] = c("Tumor cells")
group_ct_mapping[["Fibroblast"]] = c("Fibroblasts", "Smooth muscle cells")
group_ct_mapping[["vessel"]] = c("Venous systemic EC", "Lymphatic EC", "Pericytes")
group_ct_mapping[["Other"]] = c("AT2", "Other")

all_ct_groups = names(group_ct_mapping)

ct_group_colors = setNames(c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"),
                           all_ct_groups)
ct_group_labels = c(immune = "immune", tumor = "tumor", Fibroblast = "fibroblast",
                    vessel = "vessel", Other = "other")

ct_group_matching = rbindlist(lapply(all_ct_groups, function(x){
  data.table(ct_group = x, CELL_TYPE = group_ct_mapping[[x]])
}))

stopifnot(!any(duplicated(ct_group_matching$CELL_TYPE)))
stopifnot(nrow(ct_group_matching) == 17)

domain_palette = brewer.pal(9, "Set1")
domain_palette[which(domain_palette == "#4DAF4A")] = "#1B9E77"
domain_palette[which(domain_palette == "#FFFF33")] = "#66A61E"
domain_palette = c(domain_palette, "#000000")

domain_colors = setNames(domain_palette, as.character(1:n_domains))

stopifnot(length(domain_colors) == n_domains)
stopifnot(!any(duplicated(domain_colors)))


raw_csv = file.path(raw_dir, paste0(example_region_id, ".csv"))
if (!file.exists(raw_csv)){
  stop("missing ", raw_csv)
}

region_cell = fread(raw_csv, select = c("CELL_ID", "X", "Y", "CELL_TYPE"))
n_cells_region = nrow(region_cell)

stopifnot(all(region_cell$CELL_TYPE %in% ct_group_matching$CELL_TYPE))
region_cell = merge(region_cell, ct_group_matching, by = "CELL_TYPE")
stopifnot(nrow(region_cell) == n_cells_region)
region_cell[, ct_group := factor(ct_group, levels = all_ct_groups)]

cat(short_name, ": ", n_cells_region, " cells\n", sep = "")

for (target in targets){

  if (!file.exists(target$annotation)){
    stop("missing ", target$annotation, " for ", target$tag)
  }

  df_cur = fread(target$annotation, select = c("CELL_ID", "domain"))
  df_cur[, domain := as.integer(domain)]

  stopifnot(!any(duplicated(df_cur$CELL_ID)))
  stopifnot(all(df_cur$domain %in% 1:n_domains))

  cur_domain = df_cur$domain[match(region_cell$CELL_ID, df_cur$CELL_ID)]
  stopifnot(!any(is.na(cur_domain)))

  region_cell[, (paste0("domain_", target$tag)) :=
                factor(as.character(cur_domain), levels = as.character(1:n_domains))]

  cat("  ", target$tag, ", domain ", target$domain, ": ",
      sum(cur_domain == target$domain), " cells\n", sep = "")
}


panel_theme = theme(plot.title = element_text(size = title_font_size, hjust = 0.5),
                    legend.position = "right",
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
                    axis.ticks.y = element_blank(),
                    plot.margin = margin(panel_margin_y, panel_margin_x,
                                         panel_margin_y, panel_margin_x))

p_tissue = ggplot(region_cell, aes(x = X, y = Y, color = ct_group)) +
  geom_point(size = point_size, stroke = 0) +
  scale_color_manual(values = ct_group_colors, labels = ct_group_labels, drop = FALSE) +
  guides(color = guide_legend(title = NULL, override.aes = list(size = 3))) +
  coord_fixed() +
  labs(title = "Cell type") +
  panel_theme +
  theme(plot.title = element_text(size = tissue_title_font_size, hjust = 0.5),
        legend.text = element_text(size = tissue_legend_font_size))

build_domain_panel <- function(domain_column, panel_title){
  ggplot(region_cell, aes(x = X, y = Y, color = .data[[domain_column]])) +
    geom_point(size = point_size, stroke = 0) +
    scale_color_manual(values = domain_colors, drop = FALSE) +
    guides(color = guide_legend(title = NULL, override.aes = list(size = 3))) +
    coord_fixed() +
    labs(title = panel_title) +
    panel_theme
}

build_highlight_panel <- function(domain_column, cur_domain, panel_title){
  in_domain = region_cell[[domain_column]] == as.character(cur_domain)
  stopifnot(sum(in_domain) >= n_cells_threshold)
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

for (target in targets){

  domain_column = paste0("domain_", target$tag)

  plot_list[[length(plot_list) + 1]] = build_domain_panel(
    domain_column, paste0(target$title, "\ndomains"))

  plot_list[[length(plot_list) + 1]] = build_highlight_panel(
    domain_column, target$domain, paste0(target$title, "\ndomain ", target$domain))
}

combined = wrap_plots(plot_list, ncol = 2, byrow = TRUE)


tissue_file = file.path(figure_dir, paste0(short_name, "_cell_type_group.pdf"))
domain_file = file.path(figure_dir, paste0(short_name, "_domain_panels.pdf"))

while (!is.null(dev.list())) dev.off()

pdf(tissue_file, width = tissue_width, height = tissue_height)
print(p_tissue)
invisible(dev.off())

pdf(domain_file, width = 2 * domain_column_width, height = domain_figure_height)
print(combined)
invisible(dev.off())

cat("saved ", tissue_file, "\n", sep = "")
cat("saved ", domain_file, "\n", sep = "")
