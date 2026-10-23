library(data.table)
library(ggplot2)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure3_domain_ct_group_prop_dotplot_direct_domains")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

figure_width = 10.6
figure_height = 3.6
base_font_size = 18

description_font_size = 12
legend_font_size = 11

n_domains = 10
n_cells_threshold = 30

domain_description = c(
  "tumor core",
  "immune border outer layer mixed with non-immune cell types",
  "tumor border outer layer mixed with fibroblast",
  "immune border inner layer",
  "fibroblast mixed with immune, border for dense fibroblast stroma",
  "immune aggregation",
  "fibroblast stroma",
  "tumor border outer layer mixed with immune and fibroblast",
  "vessel mixed with immune and fibroblast",
  paste("tumor border inner layer for dense tumor, mixed with non-tumor",
        "cell types for diffused tumor"))

stopifnot(length(domain_description) == n_domains)

names(domain_description) = paste0("domain ", 1:n_domains)

raw_dir = "../data/Cords_data/raw_data"
region_list_csv = "../data/Cords_data/region_list.csv"

domain_csv = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                    "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20",
                    "__ensemble_10/extended/domains_k10/domain_annotation_for_cells.csv")


fill_low = "#075AFF"
fill_mid = "#FFFFCC"
fill_high = "#FF0000"
fill_midpoint = 0.5

dot_max_size = 6
legend_breaks = c(0.25, 0.5, 0.75, 1)


group_ct_mapping = list()
group_ct_mapping[["immune"]] = c('Bcell', 'CD4', 'CD4_Treg', 'CD8', 'IDO_CD4', 'IDO_CD8',
                                 'ki67_CD4', 'ki67_CD8', 'Myeloid', 'Neutrophil', 'PD1_CD4',
                                 'TCF1/7_CD4', 'TCF1/7_CD8')
group_ct_mapping[["tumor"]] = c('hypoxic', 'normal')
group_ct_mapping[["Fibroblast"]] = c('Collagen_CAF', 'dCAF', 'hypoxic_CAF', 'hypoxic_tpCAF',
                                     'iCAF', 'IDO_CAF', 'mCAF', 'PDPN_CAF', 'SMA_CAF', 'tpCAF', 'vCAF')
group_ct_mapping[["vessel"]] = c('Blood', 'HEV', 'Lymphatic')
group_ct_mapping[["Other"]] = c('Other')

all_ct_groups = names(group_ct_mapping)

ct_group_matching = rbindlist(lapply(all_ct_groups, function(ct_grp){
  data.table(CELL_TYPE = group_ct_mapping[[ct_grp]], ct_group = ct_grp)
}))

stopifnot(!any(duplicated(ct_group_matching$CELL_TYPE)))


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


compute_domain_ct_group_median <- function(cell_table, cell_domain, n_domains,
                                           n_cells_threshold, all_ct_groups){

  merged = merge(cell_table, cell_domain, by = "CELL_ID")

  stopifnot(nrow(merged) == nrow(cell_table))

  counts = merged[, .N, by = .(region_id, domain, ct_group)]
  counts_wide = dcast(counts, region_id + domain ~ ct_group, value.var = "N", fill = 0)

  missing_cols = setdiff(all_ct_groups, colnames(counts_wide))
  for (col in missing_cols){
    counts_wide[, (col) := 0]
  }

  counts_wide[, total := rowSums(.SD), .SDcols = all_ct_groups]

  prop_cols = paste0(all_ct_groups, "__prop")
  counts_wide[, (prop_cols) := lapply(.SD, function(x) x / total), .SDcols = all_ct_groups]

  counts_wide[total < n_cells_threshold, (prop_cols) := NA]

  long = melt(counts_wide, id.vars = c("region_id", "domain"), measure.vars = prop_cols,
              variable.name = "ct_group", value.name = "prop")
  long[, ct_group := gsub("__prop$", "", ct_group)]

  domain_median = long[, .(median_prop = median(prop, na.rm = TRUE)), by = .(domain, ct_group)]

  return(as.data.frame(domain_median))
}


if (!file.exists(domain_csv)){
  stop("missing ", domain_csv)
}

region_ids = read.csv(region_list_csv, header = TRUE)$region_ID
cell_table = load_cell_ct_group_table(raw_dir, region_ids)

cat("cell table: ", nrow(cell_table), " cells over ", length(region_ids),
    " images\n", sep = "")

cell_domain = fread(domain_csv, select = c("CELL_ID", "domain"))
cell_domain[, domain := as.integer(domain)]

stopifnot(all(cell_domain$domain %in% 1:n_domains))

cat("annotation: ", nrow(cell_domain), " cells, domains ",
    min(cell_domain$domain), " to ", max(cell_domain$domain), "\n", sep = "")

combined = compute_domain_ct_group_median(cell_table, cell_domain, n_domains,
                                          n_cells_threshold, all_ct_groups)


combined$ct_group[combined$ct_group == "Fibroblast"] = "CAF"
combined$ct_group[combined$ct_group == "Other"] = "other"

combined$ct_group = factor(combined$ct_group,
                           levels = c("CAF", "immune", "tumor", "vessel", "other"))

combined$domain = factor(paste0("domain ", combined$domain),
                         levels = rev(paste0("domain ", 1:n_domains)))

stopifnot(!any(is.na(combined$ct_group)))
stopifnot(!any(is.na(combined$domain)))

row_totals = tapply(combined$median_prop, combined$domain, sum, na.rm = TRUE)
cat("\nrow sums (a domain's five medians; medians need not sum to exactly 1):\n")
print(round(row_totals, 3))


domain_axis_label <- function(cur_levels){
  paste0(formatC(sub("^domain ", "", cur_levels), width = 2), "   ",
         domain_description[cur_levels])
}

stopifnot(all(levels(combined$domain) %in% names(domain_description)))

p1 = ggplot(combined, aes(x = ct_group, y = domain)) +
  geom_tile(fill = NA, color = "grey70") +
  geom_point(aes(size = median_prop, fill = median_prop),
             shape = 21, color = "grey40", stroke = 0.3) +
  scale_size_area(max_size = dot_max_size, limits = c(0, 1),
                  breaks = legend_breaks, name = NULL) +
  scale_fill_gradient2(low = fill_low, mid = fill_mid, high = fill_high,
                       midpoint = fill_midpoint, limits = c(0, 1),
                       breaks = legend_breaks, name = NULL, guide = "legend") +
  scale_x_discrete(expand = expansion(add = 0)) +
  scale_y_discrete(expand = expansion(add = 0), position = "right",
                   labels = domain_axis_label) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = base_font_size) +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y.right = element_text(hjust = 0,
                                         size = description_font_size),
        legend.position = "bottom",
        legend.text = element_text(size = legend_font_size),
        legend.box.spacing = unit(2, "pt"),
        legend.justification = "left",
        plot.margin = margin(5.5, 5.5, 5.5, 26))

figure_file = file.path(figure_dir, "dotplot.pdf")

while (!is.null(dev.list())) dev.off()

pdf(figure_file, width = figure_width, height = figure_height)
print(p1)
dev.off()

table_file = file.path(figure_dir, "domain_ct_group_prop_median.csv")
write.csv(combined, table_file, row.names = FALSE)

cat("\nsaved", figure_file, "\n")
cat("saved", table_file, "\n")
