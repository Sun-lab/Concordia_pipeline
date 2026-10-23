library(data.table)
library(ggplot2)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure3_domain_ct_group_prop_dotplot_lermi_trained_vs_transferred")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

figure_width = 5.2
figure_height = 3.6
base_font_size = 18

label_font_size = 12
legend_font_size = 11

n_domains = 10
n_cells_threshold = 30

raw_dir = "../data/Lermi_data/raw_data"
region_list_csv = "../data/Lermi_data/region_list.csv"

source_list = list(
  list(tag = "lermi_trained",
       csv = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                    "lermi_2025__group_default__base16__rad48__topk4__ctg0p176__deg20",
                    "__ensemble_10/extended/domains_k10/domain_annotation_for_cells.csv")),
  list(tag = "cords_transferred",
       csv = paste0("../revision/code/results_rs_transfer/",
                    "lermi_2025__group_default__base16__rad48__topk4__ctg0p176__deg20/",
                    "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20",
                    "__ensemble_10/extended/domains_k10/domain_annotation_for_cells.csv"))
)


fill_low = "#075AFF"
fill_mid = "#FFFFCC"
fill_high = "#FF0000"
fill_midpoint = 0.5

dot_max_size = 6
legend_breaks = c(0.25, 0.5, 0.75, 1)


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

ct_group_label = c(Fibroblast = "fibroblast", immune = "immune", tumor = "tumor",
                   vessel = "vessel", Other = "other")

ct_group_matching = rbindlist(lapply(all_ct_groups, function(ct_grp){
  data.table(CELL_TYPE = group_ct_mapping[[ct_grp]], ct_group = ct_grp)
}))

stopifnot(!any(duplicated(ct_group_matching$CELL_TYPE)))
stopifnot(nrow(ct_group_matching) == 17)


load_cell_ct_group_table <- function(raw_dir, region_ids){
  cell_list = lapply(region_ids, function(region_id){
    df_cur = fread(file.path(raw_dir, paste0(region_id, ".csv")),
                   select = c("CELL_ID", "CELL_TYPE"))
    df_cur[, region_id := region_id]
    df_cur
  })
  cell_table = rbindlist(cell_list)

  cell_table = merge(cell_table, ct_group_matching, by = "CELL_TYPE", all.x = TRUE)
  stopifnot(!any(is.na(cell_table$ct_group)))
  cell_table[, ct_group := factor(ct_group, levels = all_ct_groups)]

  return(cell_table)
}


compute_domain_ct_group_median <- function(cell_table, cell_domain){

  merged = merge(cell_table, cell_domain, by = "CELL_ID")

  stopifnot(nrow(merged) == nrow(cell_table))

  counts = merged[, .N, by = .(region_id, domain, ct_group)]
  counts_wide = dcast(counts, region_id + domain ~ ct_group, value.var = "N", fill = 0)

  for (col in setdiff(all_ct_groups, colnames(counts_wide))){
    counts_wide[, (col) := 0]
  }

  counts_wide[, total := rowSums(.SD), .SDcols = all_ct_groups]
  counts_wide = counts_wide[total >= n_cells_threshold]

  prop_cols = paste0(all_ct_groups, "__prop")
  counts_wide[, (prop_cols) := lapply(.SD, function(x) x / total), .SDcols = all_ct_groups]

  long = melt(counts_wide, id.vars = c("region_id", "domain"), measure.vars = prop_cols,
              variable.name = "ct_group", value.name = "prop")
  long[, ct_group := gsub("__prop$", "", ct_group)]

  domain_median = long[, .(median_prop = median(prop)), by = .(domain, ct_group)]
  domain_median = merge(CJ(domain = 1:n_domains, ct_group = all_ct_groups),
                        domain_median, by = c("domain", "ct_group"), all.x = TRUE)

  n_images = counts_wide[, .(n_images = .N), by = domain]
  n_images = merge(data.table(domain = 1:n_domains), n_images, by = "domain", all.x = TRUE)
  n_images[is.na(n_images), n_images := 0]

  return(list(median = as.data.frame(domain_median), n_images = n_images))
}


region_ids = read.csv(region_list_csv, header = TRUE)$region_ID
cell_table = load_cell_ct_group_table(raw_dir, region_ids)

cat("cell table: ", nrow(cell_table), " cells over ", length(region_ids),
    " images\n", sep = "")


for (cur_source in source_list){

  if (!file.exists(cur_source$csv)){
    stop("missing ", cur_source$csv)
  }

  cell_domain = fread(cur_source$csv, select = c("CELL_ID", "domain"))
  cell_domain[, domain := as.integer(domain)]

  stopifnot(!any(duplicated(cell_domain$CELL_ID)))
  stopifnot(all(cell_domain$domain %in% 1:n_domains))

  cat("\n==== ", cur_source$tag, ": ", nrow(cell_domain), " cells ====\n", sep = "")

  result = compute_domain_ct_group_median(cell_table, cell_domain)
  combined = result$median
  n_images = result$n_images

  combined$ct_group = factor(ct_group_label[combined$ct_group], levels = ct_group_label)

  domain_levels = paste0("domain ", 1:n_domains)

  combined$domain = factor(paste0("domain ", combined$domain), levels = rev(domain_levels))

  stopifnot(!any(is.na(combined$ct_group)))
  stopifnot(!any(is.na(combined$domain)))

  cat("images in which each domain holds at least ", n_cells_threshold, " cells:\n", sep = "")
  print(as.data.frame(n_images), row.names = FALSE)

  p1 = ggplot(combined, aes(x = ct_group, y = domain)) +
    geom_tile(fill = NA, color = "grey70") +
    geom_point(aes(size = median_prop, fill = median_prop),
               shape = 21, color = "grey40", stroke = 0.3, na.rm = TRUE) +
    scale_size_area(max_size = dot_max_size, limits = c(0, 1),
                    breaks = legend_breaks, name = NULL) +
    scale_fill_gradient2(low = fill_low, mid = fill_mid, high = fill_high,
                         midpoint = fill_midpoint, limits = c(0, 1),
                         breaks = legend_breaks, name = NULL, guide = "legend") +
    scale_x_discrete(expand = expansion(add = 0)) +
    scale_y_discrete(expand = expansion(add = 0), position = "right") +
    coord_cartesian(clip = "off") +
    labs(x = NULL, y = NULL) +
    theme_minimal(base_size = base_font_size) +
    theme(panel.grid = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1),
          axis.text.y.right = element_text(hjust = 0, size = label_font_size),
          legend.position = "bottom",
          legend.text = element_text(size = legend_font_size),
          legend.box.spacing = unit(2, "pt"),
          legend.justification = "left",
          plot.margin = margin(5.5, 5.5, 5.5, 26))

  figure_file = file.path(figure_dir, paste0(cur_source$tag, "_dotplot.pdf"))

  while (!is.null(dev.list())) dev.off()

  pdf(figure_file, width = figure_width, height = figure_height)
  print(p1)
  dev.off()

  table_file = file.path(figure_dir, paste0(cur_source$tag, "_domain_ct_group_prop_median.csv"))
  write.csv(merge(combined, data.frame(domain = paste0("domain ", n_images$domain),
                                       n_images = n_images$n_images), by = "domain"),
            table_file, row.names = FALSE)

  cat("saved", figure_file, "\n")
  cat("saved", table_file, "\n")
}
