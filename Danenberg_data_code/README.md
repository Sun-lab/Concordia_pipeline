## Processing the Danenberg et al. 2022 breast cancer dataset

Filter and export images:

extract_patient_tumor_to_use.R

Generate graph objects for each image (this step can be run parallely):

generate_graph_data_object_dist_20_Danenberg_prepare.py

generate_graph_data_object_dist_16_Cords_prepare.sh

Notes on random seed setting:

We used random seeds for different images for the graph generation process. The random seeds were obtained from [https://www.random.org/](https://www.random.org/). Although, even under the contol of random seed, the agumented graphs for the same image from two runs may have difference, due to the ranking difference caused by floating number accuracy issue in the step 1 graph extension step (local graph).

Copy the graph object files for different images to one common folder, for later model training:

copy_graph_files_for_danenberg_images.ipynb