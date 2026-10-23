args = commandArgs(trailingOnly=TRUE)
args

if (!(length(args) %in% c(3, 4, 5, 6))) {
  message("four to six arguments are expected.\n")
  quit(save="no")
}else{
  for (arg in args){
    eval(parse(text=arg))
  }
}

if (!exists("model_run_id")){
  model_run_id = NULL
}

if (!exists("data_dir")){
  data_dir = NULL
}

if (!exists("n_repeats")){
  n_repeats = 10
}

n_domains = as.integer(n_domains)
n_repeats = as.integer(n_repeats)

model_run_id

data_name
graph_type
n_domains
n_repeats

options(bitmapType = "cairo")

source("data_utilities.R")

source("_get_n_cells_in_domains.R")
source("_save_summary_dist_matrices.R")


feature_list = data_features(data_name, model_run_id, data_dir)

df = feature_list$df
print(head(df))

region_ids = df$region_ID

print(length(unique(region_ids)))

result_subfolder = feature_list$result_subfolder
raw_dir = feature_list$raw_dir

n_cells_threshold = feature_list$n_cells_threshold
n_images_threshold = feature_list$n_images_threshold

ensemble_run_id = paste0(result_subfolder, "__ensemble_", n_repeats)

dir = file.path("./results_rs_ensemble_scaled_direct_domains", ensemble_run_id, graph_type)
domain_dir = file.path(dir, paste0("domains_k", n_domains))

print(paste("reading the domain annotation and per-image distances from", domain_dir))

stopifnot(dir.exists(domain_dir))

get_n_cells_in_domains(raw_dir, domain_dir, region_ids, n_domains)

save_summary_dist_matrices(domain_dir, region_ids, n_domains,
                           n_cells_threshold, n_images_threshold)




gc()
sessionInfo()

q(save="no")
