library(data.table)
library(ggplot2)
library(ggpubr)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figureS2_Cords_domain_ct_group_prop_boxplots_concordia_extended")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

raw_dir = "../data/Cords_data/raw_data"
region_list_csv = "../data/Cords_data/region_list.csv"

domain_csv = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                    "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20",
                    "__ensemble_10/extended/domains_k10/domain_annotation_for_cells.csv")

n_domains = 10

n_cells_threshold = 30

panel_width = 2.1
panel_height = 2.0
legend_height = 0.4
title_height = 0.7
n_panels_per_row = 5

base_font_size = 12
legend_font_size = 14



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

ct_group_levels = c("immune", "tumor", "CAF", "vessel", "other")
ct_colors = setNames(c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"),
                     ct_group_levels)



compute_domain_ct_group_prop <- function(cell_table, cell_domain, all_ct_groups,
                                         n_domains, n_cells_threshold){

  merged = merge(cell_table, cell_domain, by = "CELL_ID")

  stopifnot(nrow(merged) == nrow(cell_table))

  counts = merged[, .N, by = .(region_id, domain, ct_group)]
  counts_wide = dcast(counts, region_id + domain ~ ct_group, value.var = "N", fill = 0)

  missing_cols = setdiff(all_ct_groups, colnames(counts_wide))
  for (cur_col in missing_cols){
    counts_wide[, (cur_col) := 0]
  }

  full_grid = CJ(region_id = unique(cell_table$region_id), domain = 1:n_domains)
  counts_wide = merge(full_grid, counts_wide, by = c("region_id", "domain"), all.x = TRUE)

  for (cur_col in all_ct_groups){
    counts_wide[is.na(get(cur_col)), (cur_col) := 0]
  }

  counts_wide[, total := rowSums(.SD), .SDcols = all_ct_groups]

  prop_cols = paste0(all_ct_groups, "__prop")
  counts_wide[, (prop_cols) := lapply(.SD, function(x) x / total), .SDcols = all_ct_groups]

  counts_wide[total < n_cells_threshold, (prop_cols) := NA]

  cat("image-domain pairs boxed:", sum(counts_wide$total >= n_cells_threshold),
      " NA, fewer than", n_cells_threshold, "cells:",
      sum(counts_wide$total < n_cells_threshold),
      "(of which absent from the image:", sum(counts_wide$total == 0), ")\n")

  long = melt(counts_wide, id.vars = c("region_id", "domain"), measure.vars = prop_cols,
              variable.name = "ct_group", value.name = "prop")
  long[, ct_group := gsub("__prop$", "", ct_group)]

  return(as.data.frame(long))
}



if (!file.exists(domain_csv)){
  stop("missing ", domain_csv)
}

region_ids = read.csv(region_list_csv, header = TRUE)$region_ID

cell_table = rbindlist(lapply(region_ids, function(region_id){
  df_cur = fread(file.path(raw_dir, paste0(region_id, ".csv")),
                 select = c("CELL_ID", "CELL_TYPE"))
  df_cur[, region_id := region_id]
  df_cur
}))

cell_table = merge(cell_table, ct_group_matching, by = "CELL_TYPE", all.x = TRUE)
stopifnot(!any(is.na(cell_table$ct_group)))

cat("cell table:", nrow(cell_table), "cells over", length(region_ids), "images\n")

cell_domain = fread(domain_csv, select = c("CELL_ID", "domain"))
cell_domain[, domain := as.integer(domain)]

stopifnot(all(cell_domain$domain %in% 1:n_domains))
cat("domains found:", length(unique(cell_domain$domain)), "of", n_domains, "\n")

df_long = compute_domain_ct_group_prop(cell_table, cell_domain, all_ct_groups,
                                       n_domains, n_cells_threshold)

df_long$ct_group[df_long$ct_group == "Fibroblast"] = "CAF"
df_long$ct_group[df_long$ct_group == "Other"] = "other"
df_long$ct_group = factor(df_long$ct_group, levels = ct_group_levels)
stopifnot(!any(is.na(df_long$ct_group)))

domainID_list = paste0("domain ", 1:n_domains)
df_long$domain = factor(paste0("domain ", df_long$domain), levels = domainID_list)
stopifnot(!any(is.na(df_long$domain)))



p_list = lapply(domainID_list, function(domainID){
  ggplot(df_long[df_long$domain == domainID, ],
         aes(x = ct_group, y = prop, fill = ct_group)) +
    geom_boxplot(outlier.size = 0.6, linewidth = 0.4) +
    scale_fill_manual(values = ct_colors, drop = FALSE) +
    labs(title = domainID) + ylim(0, 1) +
    theme_classic(base_size = base_font_size) +
    theme(plot.title = element_text(hjust = 0.5),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.text = element_text(size = legend_font_size,
                                     margin = margin(l = 4, r = 14)),
          legend.key.size = grid::unit(0.6, "cm")) +
    guides(fill = guide_legend(title = NULL, nrow = 1))
})

n_row = ceiling(length(p_list) / n_panels_per_row)

figure_file = file.path(figure_dir, "boxplots.pdf")

while (!is.null(dev.list())) dev.off()

pdf(figure_file, width = panel_width * n_panels_per_row,
    height = panel_height * n_row + legend_height + title_height)
combined_plot = ggarrange(plotlist = p_list, ncol = n_panels_per_row, nrow = n_row,
                          common.legend = TRUE, legend = "top")
combined_plot = annotate_figure(combined_plot,
                                top = text_grob("Lung cancer dataset\nConcordia domains, cell type composition",
                                                face = "bold", size = 16))
print(combined_plot)
dev.off()

table_file = file.path(figure_dir, "domain_ct_group_prop_per_image.csv")
write.csv(df_long, table_file, row.names = FALSE)

cat("\nsaved", figure_file, "\n")
cat("saved", table_file, "\n")
