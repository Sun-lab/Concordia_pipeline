library(data.table)

# one row per Cox model of overall survival on the patient-level tumor-domain
# share, for the three ways of combining a patient's images into that share:
# simple average, cell-weighted average and lmer BLUP

table_dir = "../revision_tables"
if (!dir.exists(table_dir)){
  dir.create(table_dir, recursive = TRUE)
}

table_file = file.path(table_dir, "domain_share_OS_cox_strata_age65_three_aggregations.csv")

figure_root = "../revision_figures"

aggregation_list = list(
  list(label = "simple average",
       dir = "figure3_domain_share_OS_and_cox_zph_three_methods_strata_age65"),
  list(label = "cell-weighted average",
       dir = "figure3_domain_share_OS_and_cox_zph_three_methods_strata_age65_cell_weighted"),
  list(label = "lmer BLUP",
       dir = "figure3_domain_share_OS_and_cox_zph_three_methods_strata_age65_lmer_blup")
)

method_levels = c("Concordia-Basic", "Concordia", "K-means")

needed_columns = c("setting", "n_patients_used", "n_events_used", "hazard_ratio",
                   "hazard_ratio_lower95", "hazard_ratio_upper95", "pvalue",
                   "zph_global_pvalue")

table_rows = list()

for (aggregation in aggregation_list){

  pvalue_csv = file.path(figure_root, aggregation$dir, "pvalues.csv")

  if (!file.exists(pvalue_csv)){
    stop("missing ", pvalue_csv)
  }

  df_cur = fread(pvalue_csv)

  missing_columns = setdiff(needed_columns, names(df_cur))
  if (length(missing_columns) > 0){
    stop("missing columns in ", pvalue_csv, ": ", paste(missing_columns, collapse = ", "))
  }

  stopifnot(setequal(df_cur$setting, method_levels))
  stopifnot(!any(duplicated(df_cur$setting)))

  table_rows[[length(table_rows) + 1]] = df_cur[, .(
    method = setting,
    aggregation = aggregation$label,
    pvalue, hazard_ratio, hazard_ratio_lower95, hazard_ratio_upper95,
    n_patients_used, n_events_used, zph_global_pvalue)]
}

df_table = rbindlist(table_rows)

df_table = df_table[order(match(method, method_levels),
                          match(aggregation, sapply(aggregation_list, function(x) x$label)))]

format_pvalue <- function(x){
  formatC(signif(x, 3), format = "g")
}

df_out = df_table[, .(
  "method" = method,
  "patient-level share" = aggregation,
  "p-value" = format_pvalue(pvalue),
  "hazard ratio" = sprintf("%.3f", hazard_ratio),
  "95% CI lower" = sprintf("%.3f", hazard_ratio_lower95),
  "95% CI upper" = sprintf("%.3f", hazard_ratio_upper95),
  "n patients" = n_patients_used,
  "n events" = n_events_used,
  "proportional hazards test p-value" = format_pvalue(zph_global_pvalue))]

fwrite(df_out, table_file)

options(width = 200)
print(df_out, row.names = FALSE)

cat("\nsaved ", table_file, "\n", sep = "")
