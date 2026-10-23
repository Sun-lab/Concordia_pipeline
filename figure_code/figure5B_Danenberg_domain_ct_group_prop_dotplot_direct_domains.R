library(data.table)
library(ggplot2)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figureS3_Danenberg_domain_ct_group_prop_dotplot_direct_domains")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

figure_width = 10.0
figure_height = 3.6
base_font_size = 18

description_font_size = 12
legend_font_size = 11

n_domains = 10

domain_description = c(
  "tumor core",
  "balanced immune fibroblast mixture",
  "tumor-immune interface",
  "tumor border outer layer mixed with fibroblast",
  "immune aggregate slightly mixed with fibroblast",
  "fibroblast border mixed with non-fibroblast cell types",
  "fibroblast stroma, slightly mixed with immune",
  "endothelial mixed with non-endothelial cell types",
  paste("tumor border inner layer for dense tumor, mixed with non-tumor",
        "cell types for diffused tumor"),
  "tumor-fibroblast interface slightly mixed with immune")

stopifnot(length(domain_description) == n_domains)

names(domain_description) = paste0("domain ", 1:n_domains)

data_dir = "../data/Danenberg_data"

domain_csv = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                    "danenberg_2022__group_default__base20__rad60__topk4__ctg0p176__deg20",
                    "__ensemble_10/extended/domains_k10/domain_annotation_for_cells.csv")


fill_low = "#075AFF"
fill_mid = "#FFFFCC"
fill_high = "#FF0000"
fill_midpoint = 0.5

dot_max_size = 6
legend_breaks = c(0.25, 0.5, 0.75, 1)


source("../revision/code/data_utilities.R")

feature_list = data_features(data_name = "danenberg_2022", data_dir = data_dir)

group_ct_mapping = feature_list$group_ct_mapping
raw_dir = feature_list$raw_dir
region_ids = feature_list$df$region_ID
n_cells_threshold = feature_list$n_cells_threshold

all_ct_groups = names(group_ct_mapping)

ct_group_matching = rbindlist(lapply(all_ct_groups, function(ct_grp){
  data.table(CELL_TYPE = group_ct_mapping[[ct_grp]], ct_group = ct_grp)
}))

stopifnot(!any(duplicated(ct_group_matching$CELL_TYPE)))
stopifnot(!any(duplicated(region_ids)))

cat("groups: ", paste(all_ct_groups, collapse = ", "),
    "; cell threshold: ", n_cells_threshold, "\n", sep = "")


load_cell_ct_group_table <- function(raw_dir, region_ids){
  cell_list = lapply(region_ids, function(region_id){
    df_cur = fread(file.path(raw_dir, paste0(region_id, ".csv")),
                   select = c("CELL_ID", "CELL_TYPE"))
    df_cur[, region_id := region_id]
    df_cur
  })
  cell_table = rbindlist(cell_list)

  stopifnot(!any(duplicated(cell_table$CELL_ID)))

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

cell_table = load_cell_ct_group_table(raw_dir, region_ids)

cat("cell table: ", nrow(cell_table), " cells over ", length(region_ids),
    " images\n", sep = "")

cell_domain = fread(domain_csv, select = c("CELL_ID", "domain"))
cell_domain[, domain := as.integer(domain)]

stopifnot(all(cell_domain$domain %in% 1:n_domains))
stopifnot(!any(duplicated(cell_domain$CELL_ID)))

cat("annotation: ", nrow(cell_domain), " cells, domains ",
    min(cell_domain$domain), " to ", max(cell_domain$domain), "\n", sep = "")

combined = compute_domain_ct_group_median(cell_table, cell_domain, n_domains,
                                          n_cells_threshold, all_ct_groups)


combined$ct_group[combined$ct_group == "stroma"] = "fibroblast"

combined$ct_group = factor(combined$ct_group,
                           levels = c("fibroblast", "immune", "tumor", "endothelial"))

combined$domain = factor(paste0("domain ", combined$domain),
                         levels = rev(paste0("domain ", 1:n_domains)))

stopifnot(!any(is.na(combined$ct_group)))
stopifnot(!any(is.na(combined$domain)))
stopifnot(nrow(combined) == n_domains * nlevels(combined$ct_group))

row_totals = tapply(combined$median_prop, combined$domain, sum, na.rm = TRUE)
cat("\nrow sums (a domain's four medians; medians need not sum to exactly 1):\n")
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
