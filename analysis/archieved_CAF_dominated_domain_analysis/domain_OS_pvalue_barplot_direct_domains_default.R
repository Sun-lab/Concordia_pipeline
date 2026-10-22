library(data.table)
library(dplyr)
library(tidyr)
library(survival)
library(ggplot2)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = "figures"
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

n_domains = 10
n_repeats = 10

default_run_id = "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20"

setting_label = "direct k = 10\nextended (default)"

annotation_csv = paste0("../code/results_rs_ensemble_scaled_direct_domains/",
                        default_run_id, "__ensemble_", n_repeats,
                        "/extended/domains_k", n_domains,
                        "/domain_annotation_for_cells.csv")

if (!file.exists(annotation_csv)){
  stop("missing ", annotation_csv)
}


df_images = read.csv("../../data/Cords_data/patient_image_greq_1000_cells.csv", header = TRUE)
df_images$RoiID = gsub(",", "_", df_images$RoiID)
df_images = df_images[which(df_images$DX.name %in% c("Adenocarcinoma", "Squamous cell carcinoma")), ]

df_patients = read.csv("../../data/Cords_data/patient_features.csv", header = TRUE)
df_patients_subset = df_patients[which(df_patients$Patient_ID %in% df_images$Patient_ID), ]

df_patients_matched = df_patients_subset[match(df_images$Patient_ID, df_patients_subset$Patient_ID), ]
stopifnot(all(df_images$Patient_ID == df_patients_matched$Patient_ID))

df_roi = cbind(df_patients_matched, df_images[, c("RoiID", "n_cells")])


load_cell_region_table <- function(raw_dir, region_ids){
  cell_list = lapply(region_ids, function(region_id){
    df_cur = fread(file.path(raw_dir, paste0(region_id, ".csv")), select = c("CELL_ID"))
    df_cur[, region_id := region_id]
    df_cur
  })
  return(rbindlist(cell_list))
}


compute_domain_prop_per_image <- function(cell_region_table, cell_domain, n_domains){

  merged = merge(cell_region_table, cell_domain, by = "CELL_ID")

  stopifnot(nrow(merged) == nrow(cell_region_table))

  merged[, domain := factor(domain, levels = 1:n_domains)]

  counts = merged[, .N, by = .(region_id, domain)]
  counts_wide = dcast(counts, region_id ~ domain, value.var = "N", fill = 0, drop = FALSE)

  domain_cols = as.character(1:n_domains)
  counts_wide[, total := rowSums(.SD), .SDcols = domain_cols]

  prop_cols = paste0("domain_", domain_cols, "_prop")
  counts_wide[, (prop_cols) := lapply(.SD, function(x) x / total), .SDcols = domain_cols]

  return(as.data.frame(counts_wide[, c("region_id", prop_cols), with = FALSE]))
}


compute_domain_os_pvalue_long <- function(df_prop, setting_label){

  merged = merge(df_prop, df_roi, by.x = "region_id", by.y = "RoiID")

  prop_cols = grep("^domain_.*_prop$", colnames(merged), value = TRUE)
  prop_cols = prop_cols[order(as.integer(gsub("domain_|_prop", "", prop_cols)))]

  df_patients_prop <- merged %>%
    group_by(Patient_ID) %>%
    summarise(across(all_of(c(prop_cols, "n_cells")), ~ mean(.x, na.rm = TRUE), .names = "{.col}"),
              across(c("Ev.O", "OS", "DX.name", "Age", "Gender", "Grade", "Smok"), unique),
              .groups = "drop")
  df_patients_prop = as.data.frame(df_patients_prop)

  df_patients_prop$Gender = factor(df_patients_prop$Gender, levels = c(1, 2))
  df_patients_prop$Grade = factor(df_patients_prop$Grade, levels = c(1, 2, 3))
  df_patients_prop$Smok = factor(df_patients_prop$Smok, levels = c(0, 1, 2, 3))

  col_pvalue = c()
  col_pvalue_small_model = c()

  for (prop_str in prop_cols){

    df_prop_cur = data.frame(prop = df_patients_prop[, prop_str],
                             Ev.O = df_patients_prop$Ev.O,
                             OS = df_patients_prop$OS,
                             DX.name = df_patients_prop$DX.name,
                             Gender = df_patients_prop$Gender,
                             Age = df_patients_prop$Age,
                             Grade = df_patients_prop$Grade,
                             Smok = df_patients_prop$Smok)

    if (length(unique(df_prop_cur$prop)) < 2){
      col_pvalue = c(col_pvalue, NA_real_)
      col_pvalue_small_model = c(col_pvalue_small_model, NA_real_)
      next
    }

    cur_fit = coxph(Surv(OS, Ev.O) ~ prop + Age + Gender + DX.name + Grade + Smok,
                    data = df_prop_cur)
    col_pvalue = c(col_pvalue, summary(cur_fit)$coefficients[1, 5])

    cur_fit_small = coxph(Surv(OS, Ev.O) ~ prop, data = df_prop_cur)
    col_pvalue_small_model = c(col_pvalue_small_model, summary(cur_fit_small)$coefficients[1, 5])
  }

  df_pvalue = data.frame(prop = prop_cols, pvalue = col_pvalue,
                         pval_marginal = col_pvalue_small_model)

  df_pvalue$prop = gsub("domain_|_prop", "", df_pvalue$prop)
  df_pvalue$prop = factor(df_pvalue$prop, levels = as.character(seq(1, n_domains)))

  df_long <- df_pvalue %>%
    pivot_longer(cols = c(pvalue, pval_marginal), names_to = "type", values_to = "pval") %>%
    mutate(logp = -log10(pval))

  df_long$setting = setting_label

  return(as.data.frame(df_long))
}


region_ids = read.csv("../../data/Cords_data/region_list.csv", header = TRUE)$region_ID
cell_region_table = load_cell_region_table("../../data/Cords_data/raw_data", region_ids)

cat("cell to image map:", nrow(cell_region_table), "cells in", length(region_ids), "images\n")


cell_domain = fread(annotation_csv, select = c("CELL_ID", "domain"))
cell_domain[, domain := as.integer(domain)]

stopifnot(all(cell_domain$domain %in% 1:n_domains))

df_prop = compute_domain_prop_per_image(cell_region_table, cell_domain, n_domains)

combined = compute_domain_os_pvalue_long(df_prop, setting_label)


sig_threshold = -log10(0.05 / n_domains)

g1 = ggplot(combined, aes(x = prop, y = logp, fill = type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
  geom_hline(yintercept = sig_threshold, linetype = "dashed", color = "red") +
  scale_fill_manual(values = c("#D55E00", "#009E73")) +
  labs(x = "Domain", y = expression("-log"[10] * "(p)"), fill = "P-value type",
       title = gsub("\n", ", ", setting_label)) +
  theme_classic() +
  theme(plot.title = element_text(size = 10))

figure_file = file.path(figure_dir,
                        "Cords_domain_pvalue_OS_barplot_direct_domains_default.pdf")

ggsave(figure_file, plot = g1, width = 5.5, height = 3.5)

combined_to_write = combined
combined_to_write$setting = gsub("\n", ", ", combined_to_write$setting)

table_file = file.path(figure_dir, "domain_pvalue_OS_direct_domains_default.csv")

write.csv(combined_to_write, table_file, row.names = FALSE)

cat("\n==== domains passing Bonferroni 0.05 /", n_domains, ", adjusted model ====\n")

sig_table = combined_to_write[combined_to_write$type == "pvalue" &
                                !is.na(combined_to_write$pval) &
                                combined_to_write$pval < 0.05 / n_domains, ]

print(sig_table[order(sig_table$pval), c("setting", "prop", "pval")])

cat("\nsaved", figure_file, "\n")
cat("saved", table_file, "\n")
