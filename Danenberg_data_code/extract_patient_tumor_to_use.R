# look into the data from 
# Danenberg et al. 2022 study
# Breast tumor microenvironment structures are associated with genomic features and clinical outcome

# this code file is for exploring and verifying certain aspects of the data
# narrowing down to the subset of images that are relevant (filter out images with n< 1000 cells)

# output the image, patient ID, subtype name (ER positive or not)

# split the survival outcome into two groups (may need to exclude certain patients with censored outcome)
# save the correspondance between image ID and patient ID, 
#   given that some patients have more than one tumor images

library(ggplot2)
library(dplyr) 
library(fst)
library(ggpubr)
theme_set(theme_classic())

ori_dir = "../Danenberg_2022"
  
df = read.csv(file.path(ori_dir, "MBTMEIMCPublic/SingleCells.csv"), header=TRUE)
dim(df)
colnames(df)

length(unique(df$metabric_id))
unique(df$metabric_id)

table(df$cellPhenotype)
length(unique(df$cellPhenotype))


# try identifying images that are of size different from majority of the images
length(unique(df$ImageNumber))

image_area = NULL
num_cells = NULL
sorted_ImageNumber = sort(unique(df$ImageNumber))

for (cur_image in sorted_ImageNumber){
  df_cur_image = df[which(df$ImageNumber==cur_image),]
  min_x = min(df_cur_image$Location_Center_X)
  min_y = min(df_cur_image$Location_Center_Y)
  max_x = max(df_cur_image$Location_Center_X)
  max_y = max(df_cur_image$Location_Center_Y)
  image_area = c(image_area, (max_x - min_x)*(max_y - min_y))
  num_cells = c(num_cells, nrow(df_cur_image))
}


df_image_area = data.frame(ImageNumber = sorted_ImageNumber, 
                           image_area = image_area,
                           num_cells = num_cells)

summary(df_image_area$num_cells)
table(df_image_area$num_cells<100)
table(df_image_area$num_cells<200)
table(df_image_area$num_cells<500)
table(df_image_area$num_cells<1000)

pdf(file = paste0("../figures/Danenberg_2022/image_area_histogram.pdf"), 
    width = 4, height =3)
ggplot(df_image_area, aes(x=image_area))+
  geom_histogram(color="darkblue", fill="lightblue") + theme_classic() +  
  geom_vline(xintercept=650000, color="grey", linetype="dashed")
dev.off()

pdf(file = paste0("../figures/Danenberg_2022/image_n_cells_histogram.pdf"), 
    width = 4, height =3)
ggplot(df_image_area, aes(x=num_cells))+
  geom_histogram(color="darkblue", fill="lightblue") + theme_classic() +  
  geom_vline(xintercept=100, color="grey", linetype="dashed") + 
  geom_vline(xintercept=200, color="grey", linetype="dashed") + 
  geom_vline(xintercept=300, color="grey", linetype="dashed") + 
  geom_vline(xintercept=500, color="grey", linetype="dashed") + 
  geom_vline(xintercept=1000, color="grey", linetype="dashed")  
dev.off()


table(df_image_area$image_area > 650000)
sum(df_image_area$image_area > 650000)/nrow(df_image_area)

large_images = df_image_area[which(df_image_area$image_area>650000),]$ImageNumber

table(df$is_tumour)
table(df$is_normal)

table(df$is_tumour, df$is_normal)

colSums(is.na(df))

df$row_index = rownames(df)

df_tumour = df[which(df$is_tumour==1),]
df_normal = df[which(df$is_normal==1),]

table(df_tumour$is_tumour)
table(df_normal$is_normal)

df_other = df[which((df$is_tumour!=1)&(df$is_normal!=1)), ]

# metabric_id seems to be patient ID
length(unique(df_tumour$metabric_id)) # there are 693 patients in the tumour subset
length(unique(df_normal$metabric_id))

intersect(unique(df_tumour$row_index), unique(df_normal$row_index))

intersect(unique(df_tumour$metabric_id), unique(df_normal$metabric_id))

intersect(unique(df_tumour$ImageNumber), large_images)

# ImageNumber seems to be unique for each image
# but according to the explanation in 
# DataAnnotation.pdf
# this number, assigned based on file order, is not the same across studies 
# so cannot be used to link images from other METABRIC data sets 
# e.g. Ali et al Nat Cancer 2020.
# Also from the explanation in 
# DataAnnotation.pdf
# Each observation is uniquely identified by the combination of ImageNumber and ObjectNumber
length(unique(df$ImageNumber))
length(unique(df_tumour$ImageNumber)) # there are 749 images in the tumour subset
length(unique(df_normal$ImageNumber)) 

intersect(unique(df_tumour$ImageNumber), unique(df_normal$ImageNumber))

intersect(unique(df_normal$ImageNumber), large_images)



df_tumour_image_count = df_tumour %>% 
                      group_by(metabric_id) %>% 
                      summarise(n_images=n_distinct(ImageNumber))
table(df_tumour_image_count$n_images)

# verify how many patients have sample containing epithelial cells
# the number 639 matches that mentioned in the method part of the paper
df_tumour_is_epithelial = filter(df_tumour, is_epithelial == 1)
dim(df_tumour_is_epithelial)
length(unique(df_tumour_is_epithelial$metabric_id))



# try extracting the survival information from another file

df_clinical = read_fst(file.path(ori_dir, 'MBTMEIMCPublic/IMCClinical.fst'))
dim(df_clinical)

colSums(is.na(df_clinical))

summary(df_clinical$yearsToStatus)

length(unique(df_clinical$metabric_id))
table(df_clinical$metabric_id%in%df_tumour$metabric_id)
table(df_tumour$metabric_id%in%df_clinical$metabric_id)

# there seem to be 682 individuals from df_tumour subset that have survival outcome data

setdiff(df_clinical$metabric_id, unique(df_tumour$metabric_id))
setdiff(unique(df_tumour$metabric_id), unique(df_clinical$metabric_id))

# there are 736 tumour images for these patients
patients_with_info = intersect(unique(df_tumour$metabric_id), 
                               unique(df_clinical$metabric_id))
length(unique(df_tumour[which(df_tumour$metabric_id%in%patients_with_info), ]$ImageNumber))

# there are 219 cases and 463 censored
table(df_clinical[which(df_clinical$metabric_id%in%patients_with_info),]$DeathBreast)

df_clinical_with_image = df_clinical[which(df_clinical$metabric_id%in%patients_with_info),]

df_clinical_with_image_cases = df_clinical_with_image[which(df_clinical_with_image$DeathBreast==1),]
df_clinical_with_image_censor = df_clinical_with_image[which(df_clinical_with_image$DeathBreast==0),]

table(df_clinical_with_image_cases$yearsToStatus > 5)
table(df_clinical_with_image_censor$yearsToStatus > 5)

table(df_clinical_with_image_cases$yearsToStatus > 8)
table(df_clinical_with_image_censor$yearsToStatus > 8)

table(df_clinical_with_image_cases$yearsToStatus > 10)
table(df_clinical_with_image_censor$yearsToStatus > 10)

# does the large area images have anything related to the survival outcome group
# there doesn't seem to be clear pattern for it
df_tumour_large_area = df_tumour[which(df_tumour$ImageNumber%in%large_images),]
dim(df_tumour_large_area)
length(unique(df_tumour_large_area$ImageNumber))

df_clinical_with_image_large_area = 
  df_clinical[which(df_clinical$metabric_id%in%df_tumour_large_area$metabric_id),]
dim(df_clinical_with_image_large_area)

table(df_clinical_with_image_large_area$DeathBreast)

df_clinical_with_image_large_area_cases = 
  df_clinical_with_image_large_area[which(df_clinical_with_image_large_area$DeathBreast==1),]
df_clinical_with_image_large_area_censor = 
  df_clinical_with_image_large_area[which(df_clinical_with_image_large_area$DeathBreast==0),]

table(df_clinical_with_image_large_area_cases$yearsToStatus > 5)
table(df_clinical_with_image_large_area_censor$yearsToStatus > 5)

table(df_clinical_with_image_large_area_cases$yearsToStatus > 8)
table(df_clinical_with_image_large_area_censor$yearsToStatus > 8)

table(df_clinical_with_image_large_area_cases$yearsToStatus > 10)
table(df_clinical_with_image_large_area_censor$yearsToStatus > 10)

# decide on which individuals to keep for supervised task later
# or use survival outcome
# for now, look through all images of these 682 patients

# only keep cells from these images
df_tumour_patients = df_tumour[which(df_tumour$metabric_id%in%patients_with_info),]
length(unique(df_tumour_patients$metabric_id))
length(unique(df_tumour_patients$ImageNumber))

# as of September 30, 2025, skip the part of writting out raw image files as it takes long
# --------------------------------------------------------------------------------
# to generate graphs for all of these images
# first need to check the degree of nodes in graphs at different cutoffs from 20 sampled images
# for now, generate separate file for each image with coordinate and cell type information

colnames(df_tumour_patients)

# columns to keep: 
# ['CELL_ID', 'X', 'Y', 'CELL_TYPE']

length(unique(df_tumour_patients$ObjectNumber))

df_kept = df_tumour_patients[, c("row_index", 
                                 "Location_Center_X", 
                                 "Location_Center_Y", 
                                 "cellPhenotype", 
                                 "ImageNumber",
                                 "metabric_id")]
colnames(df_kept) = c("CELL_ID", "X", "Y", "CELL_TYPE", "ImageNumber", "metabric_id")

length(unique(df_kept$ImageNumber))
length(unique(df_kept$metabric_id))

kept_images = unique(df_kept$ImageNumber)

df_kept$image_fullname = paste0(df_kept$metabric_id, "-", df_kept$ImageNumber)
length(unique(df_kept$metabric_id))
length(unique(df_kept$image_fullname))

# for (cur_fullname in df_kept$image_fullname){
#   df_kept_cur_image = df_kept[which(df_kept$image_fullname==cur_fullname), ]
#   write.csv(df_kept_cur_image, 
#             file = paste0("../data/Danenberg_data/raw_data/", cur_fullname, ".csv"), 
#             row.names=FALSE)
# }

df_kept_n_cells = df_kept %>% 
                    group_by(image_fullname) %>% 
                    summarise(n = n())
dim(df_kept_n_cells)
df_kept_n_cells = as.data.frame(df_kept_n_cells)
head(df_kept_n_cells)

df_kept_unique = unique(df_kept[, c("image_fullname", "metabric_id")])
dim(df_kept_unique)
length(unique(df_kept_unique$image_fullname))

stopifnot(setequal(df_kept_n_cells$image_fullname, 
                   df_kept_unique$image_fullname))

df_kept_n_cells_matched = df_kept_n_cells[match(df_kept_unique$image_fullname, 
                                                df_kept_n_cells$image_fullname),]
dim(df_kept_n_cells_matched)

stopifnot(all(df_kept_unique$image_fullname==df_kept_n_cells_matched$image_fullname))

df_kept_unique$n_cells = df_kept_n_cells_matched$n

# only keep regions with at least 1000 cells

table(df_kept_unique$n_cells>=1000)

df_kept_unique_1000 = df_kept_unique[which(df_kept_unique$n_cells>=1000),]
dim(df_kept_unique_1000)

table(table(df_kept_unique_1000$metabric_id))


# add ER positive or not status to the image, patient info data frame
dim(df_clinical_with_image)
table(df_clinical_with_image$er_sd, useNA="ifany")

table(df_kept_unique_1000$metabric_id%in%df_clinical_with_image$metabric_id)

df_clinical_matched = df_clinical_with_image[match(df_kept_unique_1000$metabric_id, 
                                                   df_clinical_with_image$metabric_id),]
dim(df_clinical_matched)

stopifnot(all(df_kept_unique_1000$metabric_id==df_clinical_matched$metabric_id))

df_kept_unique_1000$er_sd = df_clinical_matched$er_sd
dim(df_kept_unique_1000)
head(df_kept_unique_1000)
# there are 2 images with no ER positive or not status
# exclude these images
table(df_kept_unique_1000$er_sd, useNA="ifany")

df_kept_unique_1000 = df_kept_unique_1000[!is.na(df_kept_unique_1000$er_sd),]
dim(df_kept_unique_1000)
table(df_kept_unique_1000$er_sd, useNA="ifany")

# save image information table

df_kept_unique_1000_ordered <- df_kept_unique_1000[order(df_kept_unique_1000$image_fullname),]
dim(df_kept_unique_1000_ordered)
head(df_kept_unique_1000_ordered)

write.csv(df_kept_unique_1000_ordered, 
          file = paste0("../data/Danenberg_data/Metabric_images_geq_1000_cells.csv"), 
          row.names=FALSE)


# save clinical information for the patients

df_clinical_matched = 
  df_clinical_matched[which(df_clinical_matched$metabric_id%in%df_kept_unique_1000_ordered$metabric_id),]

df_clinical_matched_unique = unique(df_clinical_matched)
dim(df_clinical_matched_unique)
length(df_clinical_matched_unique$metabric_id)

write.csv(df_clinical_matched_unique, 
          file = paste0("../data/Danenberg_data/Metabric_clinical_w_images_geq_1000_cells.csv"), 
          row.names=FALSE)

# save the histogram of number of cells from kept images

p_list = list()

p_list[[1]] = ggplot(df_kept_unique_1000, aes(x=n_cells))+
  geom_histogram(color="darkblue", fill="lightblue") + theme_classic() +  
  ggtitle("number of cells")

p_list[[2]] = ggplot(df_kept_unique_1000, aes(x=log10(n_cells)))+
  geom_histogram(color="darkblue", fill="lightblue") + theme_classic() +  
  ggtitle("log10(number of cells)")

pdf(file = paste0("../figures/Danenberg_2022/kept_image_n_cells_histogram.pdf"), 
    width = 8, height =3)
print(ggarrange(plotlist = p_list, ncol = 2, nrow = 1))
dev.off()


sessionInfo()

q(save="no")
