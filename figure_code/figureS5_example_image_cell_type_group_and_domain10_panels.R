library(data.table)
library(ggplot2)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure5_example_image_cell_type_group_and_domain10_panels")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

example_region_ids = c("86_C2_4", "86_C7_4")

stopifnot(!any(duplicated(example_region_ids)))

n_domains = 10
target_domain = 10

panel_margin_x = 1
panel_margin_y = 5

panel_width = 3.6
panel_height = 3.6

raw_dir = "../data/Cords_data/raw_data"

annotation_csv = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                        "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20__ensemble_10/",
                        "extended/domains_k10/domain_annotation_for_cells.csv")

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


panel_theme = theme(legend.position = "none",
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    axis.text.x = element_blank(),
                    axis.ticks.x = element_blank(),
                    axis.text.y = element_blank(),
                    axis.ticks.y = element_blank(),
                    plot.margin = margin(panel_margin_y, panel_margin_x,
                                         panel_margin_y, panel_margin_x))


build_ct_group_panel <- function(region_cell){

  ggplot(region_cell, aes(x = X, y = Y, color = ct_group)) +
    geom_point(size = 0.2) +
    scale_color_manual(values = ct_group_colors, drop = FALSE) +
    coord_fixed() +
    panel_theme
}


build_domain_panel <- function(region_cell, cur_domain){

  ggplot(region_cell, aes(x = X, y = Y)) +
    geom_point(size = 0.2, color = "#DDDDDD") +
    geom_point(data = region_cell[domain == cur_domain],
               aes(x = X, y = Y, color = ct_group),
               size = 0.35) +
    scale_color_manual(values = ct_group_colors, drop = FALSE) +
    coord_fixed() +
    panel_theme
}


if (!file.exists(annotation_csv)){
  stop("missing ", annotation_csv)
}

cat("reading ", annotation_csv, "\n", sep = "")

cell_domain = fread(annotation_csv, select = c("CELL_ID", "domain"))
cell_domain[, domain := as.integer(domain)]

stopifnot(all(cell_domain$domain %in% 1:n_domains))
stopifnot(!any(duplicated(cell_domain$CELL_ID)))


build_figures_for_image <- function(cur_region_id){

  cat("\n==== ", cur_region_id, " ====\n", sep = "")

  raw_csv = file.path(raw_dir, paste0(cur_region_id, ".csv"))

  if (!file.exists(raw_csv)){
    stop("missing ", raw_csv)
  }

  region_cell = fread(raw_csv, select = c("CELL_ID", "X", "Y", "CELL_TYPE"))
  n_cells_region = nrow(region_cell)

  region_cell = merge(region_cell, cell_domain, by = "CELL_ID")
  stopifnot(nrow(region_cell) == n_cells_region)

  stopifnot(all(region_cell$CELL_TYPE %in% ct_group_matching$CELL_TYPE))

  region_cell = merge(region_cell, ct_group_matching, by = "CELL_TYPE")
  stopifnot(nrow(region_cell) == n_cells_region)

  region_cell[, ct_group := factor(ct_group, levels = names(group_ct_mapping))]

  n_cells_domain = sum(region_cell$domain == target_domain)

  cat("  ", n_cells_region, " cells, ", n_cells_domain, " in domain ",
      target_domain, "\n", sep = "")

  if (n_cells_domain == 0){
    stop("no cells of domain ", target_domain, " in ", cur_region_id)
  }

  tissue_file = file.path(figure_dir, paste0(cur_region_id, "_cell_type_group.pdf"))

  while (!is.null(dev.list())) dev.off()

  pdf(tissue_file, width = panel_width, height = panel_height)
  print(build_ct_group_panel(region_cell))
  invisible(dev.off())

  domain_file = file.path(figure_dir, paste0(cur_region_id, "_domain", target_domain, ".pdf"))

  pdf(domain_file, width = panel_width, height = panel_height)
  print(build_domain_panel(region_cell, target_domain))
  invisible(dev.off())

  cat("  saved ", tissue_file, "\n", sep = "")
  cat("  saved ", domain_file, "\n", sep = "")
}


for (cur_region_id in example_region_ids){
  build_figures_for_image(cur_region_id)
}
