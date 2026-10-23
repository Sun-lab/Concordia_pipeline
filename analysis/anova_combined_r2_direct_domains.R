args = commandArgs(trailingOnly = TRUE)

if (length(args) != 1) {
  message("one argument is expected, method='<name>'\n")
  quit(save = "no")
}else{
  eval(parse(text = args[[1]]))
}

stopifnot(method %in% c("extended", "basic", "local", "spatial_lda", "spatopic"))

cat("method:", method, "\n")


library(data.table)

options(bitmapType = "cairo")

out_dir = file.path("figures", "anova_combined_r2_direct_domains", method)
if (!dir.exists(out_dir)){
  dir.create(out_dir, recursive = TRUE)
}

raw_dir = "../../data/Cords_data/raw_data"

model_run_id = "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20"
n_repeats = 10
n_domains = 10
epoch_id = 1000

ensemble_root = paste0("../code/results_rs_ensemble_scaled_direct_domains/",
                       model_run_id, "__ensemble_", n_repeats, "/")

spatial_lda_weight_file = "../../SpatialLDA_copy/results/cords/cords_topic_weights_radius_27_d_0.25.tsv"
spatial_lda_domain_file = "../../SpatialLDA_copy/results/cords/cords_results_radius_27_d_0.25.tsv"

spatopic_dir = "../../SpaTopic_copy/results/cords/radius_62_sigma_8"
spatopic_weight_file = file.path(spatopic_dir, "cell_topic_weights.csv")
spatopic_suffix = "_spatopic_results.csv"

n_images_to_choose = 100
random_seed = 1028


df_images = read.csv("../../data/Cords_data/patient_image_greq_1000_cells.csv",
                     header = TRUE)

full_subtype = c("Adenocarcinoma", "Squamous cell carcinoma")
df_images_subtype = df_images[which(df_images$DX.name %in% full_subtype), ]

region_ids = gsub(",", "_", df_images_subtype$RoiID)

region_list = read.csv("../../data/Cords_data/region_list.csv", header = TRUE)$region_ID
stopifnot(identical(as.character(region_list), as.character(region_ids)))

set.seed(random_seed)
chosen_region_ids = sample(region_ids, n_images_to_choose)

stopifnot(length(unique(chosen_region_ids)) == n_images_to_choose)

cat("chosen images:", length(chosen_region_ids), "of", length(region_ids), "\n")


cell_list = lapply(chosen_region_ids, function(region_id){
  cur = fread(file.path(raw_dir, paste0(region_id, ".csv")),
              select = c("CELL_ID", "X", "Y"))
  cur[, image_id := region_id]
  cur
})

df_cell = rbindlist(cell_list)
n_cells = nrow(df_cell)

stopifnot(!any(duplicated(df_cell$CELL_ID)))

cat("cells:", n_cells, "\n")


if (method %in% c("extended", "basic", "local")){

  graph_type = method

  df_runs = fread(file.path(ensemble_root, graph_type, "ensemble_runs.csv"))
  stopifnot(nrow(df_runs) == n_repeats)

  blocks = list()

  for (repeat_index in 1:n_repeats){

    embedding_dir = file.path("../code/results_rs",
                              paste0(model_run_id, "__repeat_", repeat_index),
                              graph_type, paste0("epoch_", epoch_id), "linear1")

    cur_block = do.call(rbind, lapply(chosen_region_ids, function(region_id){
      as.matrix(fread(file.path(embedding_dir, paste0("linear1_", region_id, ".csv")),
                      header = TRUE))
    }))

    stopifnot(nrow(cur_block) == n_cells)

    cur_sd = df_runs$overall_sd[df_runs$repeat_index == repeat_index]
    blocks[[repeat_index]] = cur_block / cur_sd

    cat("  read repeat", repeat_index, " overall_sd", sprintf("%.4f", cur_sd), "\n")
  }

  X = do.call(cbind, blocks)

  df_domain = fread(file.path(ensemble_root, graph_type,
                              paste0("domains_k", n_domains),
                              "domain_annotation_for_cells.csv"),
                    select = c("CELL_ID", "domain"))

  matched = df_domain$domain[match(df_cell$CELL_ID, df_domain$CELL_ID)]

}else if (method == "spatial_lda"){

  if (!file.exists(spatial_lda_weight_file)){
    stop("missing ", spatial_lda_weight_file)
  }

  df_weight = fread(spatial_lda_weight_file, sep = "\t", header = TRUE)
  setnames(df_weight, 1, "CELL_ID")

  weight_cols = setdiff(names(df_weight), "CELL_ID")

  keep = match(df_cell$CELL_ID, df_weight$CELL_ID)
  stopifnot(!any(is.na(keep)))

  X = as.matrix(df_weight[keep, ..weight_cols])

  df_lda = fread(spatial_lda_domain_file, sep = "\t")
  df_lda = df_lda[sample %in% chosen_region_ids]

  matched = df_lda$predicted_cluster[match(df_cell$CELL_ID, df_lda$cell_id)]

}else{

  if (!file.exists(spatopic_weight_file)){
    stop("missing ", spatopic_weight_file,
         "; that is the per-cell topic weight file, not",
         " topic_celltype_distribution.csv, which holds the topic definitions")
  }

  df_weight = fread(spatopic_weight_file, header = TRUE)
  setnames(df_weight, 1, "cell_label")

  weight_cols = setdiff(names(df_weight), "cell_label")

  df_weight[, CELL_ID := as.integer(sub("^[^.]*\\.(.*)_[0-9]+$", "\\1", cell_label))]
  stopifnot(!any(is.na(df_weight$CELL_ID)))
  stopifnot(!any(duplicated(df_weight$CELL_ID)))

  keep = match(df_cell$CELL_ID, df_weight$CELL_ID)
  stopifnot(!any(is.na(keep)))

  X = as.matrix(df_weight[keep, ..weight_cols])

  domain_list = lapply(chosen_region_ids, function(region_id){

    raw_cur = fread(file.path(raw_dir, paste0(region_id, ".csv")),
                    select = c("CELL_ID", "X", "Y"))

    method_csv = file.path(spatopic_dir, paste0(region_id, spatopic_suffix))
    if (!file.exists(method_csv)){
      stop("missing ", method_csv)
    }

    df_label = fread(method_csv)

    stopifnot(nrow(df_label) == nrow(raw_cur))
    stopifnot(isTRUE(all.equal(df_label$X, raw_cur$X)))
    stopifnot(isTRUE(all.equal(df_label$Y, raw_cur$Y)))

    data.table(CELL_ID = raw_cur$CELL_ID, domain = df_label$topic)
  })

  df_domain = rbindlist(domain_list)

  matched = df_domain$domain[match(df_cell$CELL_ID, df_domain$CELL_ID)]
}

stopifnot(!any(is.na(matched)))
stopifnot(nrow(X) == n_cells)

df_cell[, domain := factor(paste0("domain_", matched))]
df_cell[, image_id := factor(image_id, levels = chosen_region_ids)]

colnames(X) = paste0("dim_", seq_len(ncol(X)))

cat("feature matrix:", nrow(X), "x", ncol(X),
    " domains:", length(unique(matched)), "\n")


domain = df_cell$domain
image_id = df_cell$image_id

cat("fitting dim ~ domain\n")
fit_domain = lm(X ~ domain)
s_domain = summary(fit_domain)

cat("fitting dim ~ domain + image_id\n")
fit_both = lm(X ~ domain + image_id)
s_both = summary(fit_both)

r2_of <- function(s_list, field){
  sapply(s_list, function(s) s[[field]])
}

r2_df = data.frame(response = colnames(X),
                   r.squared = r2_of(s_domain, "r.squared"),
                   adj.r.squared = r2_of(s_domain, "adj.r.squared"),
                   r.squared.both = r2_of(s_both, "r.squared"),
                   adj.r.squared.both = r2_of(s_both, "adj.r.squared"),
                   stringsAsFactors = FALSE)

r2_df$method = method
r2_df$r.squared.image = r2_df$r.squared.both - r2_df$r.squared

out_file = file.path(out_dir, "r2_domain_and_both.csv")
write.csv(r2_df, file = out_file, row.names = FALSE)

cat("\nr.squared by domain: median", sprintf("%.3f", median(r2_df$r.squared)),
    " range", sprintf("%.3f", min(r2_df$r.squared)),
    "to", sprintf("%.3f", max(r2_df$r.squared)), "\n")

cat("saved", out_file, "\n")


gc()
sessionInfo()
