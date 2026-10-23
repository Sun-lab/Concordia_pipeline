library(data.table)
library(ggplot2)
library(gridExtra)
library(grid)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure5_example_image_cell_type_group_and_domain_panels")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

example_region_ids = c("86_C3_6", "175_B7_15", "86_C6_10")

stopifnot(!any(duplicated(example_region_ids)))

n_domains = 10

n_cells_threshold = 30

title_font_size = 15
legend_font_size = 17

domain_panel_margin_x = 1
domain_panel_margin_y = 5

n_domain_columns = 2
n_domain_rows = 2

tissue_width = 5.4
tissue_height = 4.2

domain_panel_width = 3.6
domain_panel_height = 3.8

raw_dir = "../data/Cords_data/raw_data"

direct_root = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                     "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20__ensemble_10/")

metric_dir = "../revision/analysis/figures"

metric_csvs = c(
  file.path(metric_dir, "local_pca_dimension_cell_min_all_methods_random_sample.csv"),
  file.path(metric_dir, paste0("local_pca_dimension_cell_min_all_methods_",
                               example_region_ids, ".csv")))

targets = list(
  list(tag = "raw_kmeans",
       short = "K-means",
       domain = 6,
       metric_method = "raw_kmeans",
       annotation = "../results/cords_2024/extended/raw_kmeans/kmeans_cluster.csv",
       id_col = "CELL_ID", label_col = "kmeans_cluster", label_offset = 1),
  list(tag = "basic",
       short = "Concordia-Basic",
       domain = 7,
       metric_method = "direct_basic",
       annotation = paste0(direct_root, "basic/domains_k10/domain_annotation_for_cells.csv"),
       id_col = "CELL_ID", label_col = "domain", label_offset = 0),
  list(tag = "local",
       short = "Concordia-Local",
       domain = 6,
       metric_method = "direct_local",
       annotation = paste0(direct_root, "local/domains_k10/domain_annotation_for_cells.csv"),
       id_col = "CELL_ID", label_col = "domain", label_offset = 0),
  list(tag = "extended",
       short = "Concordia",
       domain = 10,
       metric_method = "direct_extended",
       annotation = paste0(direct_root, "extended/domains_k10/domain_annotation_for_cells.csv"),
       id_col = "CELL_ID", label_col = "domain", label_offset = 0)
)

stopifnot(!any(duplicated(sapply(targets, function(x) x$tag))))
stopifnot(length(targets) == n_domain_rows * n_domain_columns)



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



panel_theme = theme(plot.title = element_text(size = title_font_size, hjust = 0.5),
                    legend.position = "none",
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    axis.text.x = element_blank(),
                    axis.ticks.x = element_blank(),
                    axis.text.y = element_blank(),
                    axis.ticks.y = element_blank())



build_ct_group_panel <- function(region_cell){

  ggplot(region_cell, aes(x = X, y = Y, color = ct_group)) +
    geom_point(size = 0.2) +
    scale_color_manual(values = ct_group_colors, drop = FALSE) +
    coord_fixed() +
    guides(color = guide_legend(title = NULL,
                                override.aes = list(size = 3))) +
    panel_theme +
    theme(legend.position = "right",
          legend.text = element_text(size = legend_font_size))
}



build_domain_panel <- function(region_cell, domain_column, cur_domain,
                               panel_title, n_cells_threshold){

  cur_count = sum(region_cell[[domain_column]] == cur_domain)

  p1 = ggplot(region_cell, aes(x = X, y = Y)) +
    geom_point(size = 0.2, color = "#DDDDDD")

  if (cur_count >= n_cells_threshold){
    p1 = p1 + geom_point(data = region_cell[get(domain_column) == cur_domain],
                         aes(x = X, y = Y, color = ct_group),
                         size = 0.35) +
      scale_color_manual(values = ct_group_colors, drop = FALSE)
  }

  p1 +
    coord_fixed() +
    labs(title = panel_title) +
    panel_theme +
    theme(plot.margin = margin(domain_panel_margin_y, domain_panel_margin_x,
                               domain_panel_margin_y, domain_panel_margin_x))
}



present_metric_csvs = metric_csvs[file.exists(metric_csvs)]

if (length(present_metric_csvs) == 0){
  stop("none of these are on disk:\n  ", paste(metric_csvs, collapse = "\n  "),
       "\nrun local_pca_dimension_sampled_images_all_methods_cell_min_random_sample.py",
       " or local_pca_dimension_single_image_all_methods_cell_min.py first")
}

df_metric_all = rbindlist(lapply(present_metric_csvs, function(cur_path){
  fread(cur_path, select = c("method", "region_id", "domain", "n_cells",
                             "n_counted", "proportion_below_threshold"))
}))

df_metric_all = unique(df_metric_all, by = c("method", "region_id", "domain"))

cat("read f_lin from ", length(present_metric_csvs), " file(s), ",
    nrow(df_metric_all), " rows over ",
    length(unique(df_metric_all$region_id)), " images\n", sep = "")



f_lin_for_image <- function(cur_region_id){

  cur_f_lin = list()

  for (target in targets){

    df_cur = df_metric_all[method == target$metric_method &
                             region_id == cur_region_id &
                             domain == target$domain]

    if (nrow(df_cur) != 1){
      stop("expected one row for ", target$metric_method, " domain ", target$domain,
           " in ", cur_region_id, ", found ", nrow(df_cur))
    }

    if (is.na(df_cur$proportion_below_threshold)){
      stop("f_lin is missing for ", target$short, " domain ", target$domain,
           " in ", cur_region_id, "; the domain had too few counted cells to be scored")
    }

    cur_f_lin[[target$tag]] = df_cur$proportion_below_threshold

    cat("  ", target$short, ", domain ", target$domain, ": f_lin = ",
        format(round(df_cur$proportion_below_threshold, 3), nsmall = 3),
        " over ", df_cur$n_counted, " counted cells\n", sep = "")
  }

  return(cur_f_lin)
}



cell_domain = NULL

for (target in targets){

  if (!file.exists(target$annotation)){
    stop("missing ", target$annotation, " for ", target$short)
  }

  cat("reading ", target$annotation, " (this is the slow read)\n", sep = "")

  df_cur = fread(target$annotation, select = c(target$id_col, target$label_col))
  setnames(df_cur, c("CELL_ID", "domain"))

  df_cur[, domain := as.integer(domain) + target$label_offset]

  stopifnot(all(df_cur$domain %in% 1:n_domains))
  stopifnot(!any(duplicated(df_cur$CELL_ID)))

  setnames(df_cur, "domain", paste0("domain_", target$tag))

  if (is.null(cell_domain)){
    cell_domain = df_cur
  }else{
    cell_domain = merge(cell_domain, df_cur, by = "CELL_ID")
  }
}


build_figures_for_image <- function(cur_region_id){

  cat("\n==== ", cur_region_id, " ====\n", sep = "")

  f_lin_of = f_lin_for_image(cur_region_id)

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

  cat("  ", n_cells_region, " cells, all four annotations joined\n", sep = "")

  p_tissue = build_ct_group_panel(region_cell)

  tissue_file = file.path(figure_dir, paste0(cur_region_id, "_cell_type_group.pdf"))

  while (!is.null(dev.list())) dev.off()

  pdf(tissue_file, width = tissue_width, height = tissue_height)
  print(p_tissue)
  invisible(dev.off())

  domain_plots = lapply(targets, function(target){

    domain_column = paste0("domain_", target$tag)

    cur_f_lin = format(round(f_lin_of[[target$tag]], 3), nsmall = 3)

    first_line = paste0(target$short, ", domain ", target$domain)

    panel_title = bquote(atop(.(first_line), italic(f)[lin] == .(cur_f_lin)))

    build_domain_panel(region_cell, domain_column, target$domain,
                       panel_title, n_cells_threshold)
  })

  names(domain_plots) = sapply(targets, function(x) x$tag)

  domain_file = file.path(figure_dir, paste0(cur_region_id, "_domain_panels.pdf"))

  while (!is.null(dev.list())) dev.off()

  pdf(domain_file,
      width = domain_panel_width * n_domain_columns,
      height = domain_panel_height * n_domain_rows)
  grid.arrange(grobs = domain_plots, ncol = n_domain_columns)
  invisible(dev.off())

  cat("  saved ", tissue_file, "\n", sep = "")
  cat("  saved ", domain_file, "\n", sep = "")
}


for (cur_region_id in example_region_ids){
  build_figures_for_image(cur_region_id)
}
