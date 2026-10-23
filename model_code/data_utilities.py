import os
import pandas as pd
from collections import defaultdict

def _format_param(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace(".", "p")


def make_model_run_id(data_name, group_scheme, dist_cutoff,
                      expanded_edge_cutoff, top_k,
                      ctg_comp_dist_cutoff, degree_limit):
    return (
        f"{data_name}"
        f"__group_{group_scheme}"
        f"__base{_format_param(dist_cutoff)}"
        f"__rad{_format_param(expanded_edge_cutoff)}"
        f"__topk{top_k}"
        f"__ctg{_format_param(ctg_comp_dist_cutoff)}"
        f"__deg{_format_param(degree_limit)}"
    )


def get_cords_cell_type_mapping():
    return {'Bcell': 0,
            'Blood': 1,
            'CD4': 2,
            'CD4_Treg': 3,
            'CD8': 4,
            'Collagen_CAF': 5,
            'HEV': 6,
            'IDO_CAF': 7,
            'IDO_CD4': 8,
            'IDO_CD8': 9,
            'Lymphatic': 10,
            'Myeloid': 11,
            'Neutrophil': 12,
            'Other': 13,
            'PD1_CD4': 14,
            'PDPN_CAF': 15,
            'SMA_CAF': 16,
            'TCF1/7_CD4': 17,
            'TCF1/7_CD8': 18,
            'dCAF': 19,
            'hypoxic': 20,
            'hypoxic_CAF': 21,
            'hypoxic_tpCAF': 22,
            'iCAF': 23,
            'ki67_CD4': 24,
            'ki67_CD8': 25,
            'mCAF': 26,
            'normal': 27,
            'tpCAF': 28,
            'vCAF': 29}

def get_Danenberg_cell_type_mapping():

    return {'B cells': 0,
            'Basal': 1,
            'CD15^{+}': 2,
            'CD38^{+} lymphocytes': 3,
            'CD4^{+} T cells': 4,
            'CD4^{+} T cells & APCs': 5,
            'CD57^{+}': 6,
            'CD8^{+} T cells': 7,
            'CK^{+} CXCL12^{+}': 8,
            'CK^{lo}ER^{lo}': 9,
            'CK^{lo}ER^{med}': 10,
            'CK^{med}ER^{lo}': 11,
            'CK8-18^{+} ER^{hi}': 12,
            'CK8-18^{hi}CXCL12^{hi}': 13,
            'CK8-18^{hi}ER^{lo}': 14,
            'Endothelial': 15,
            'Ep CD57^{+}': 16,
            'Ep Ki67^{+}': 17,
            'ER^{hi}CXCL12^{+}': 18,
            'Fibroblasts': 19,
            'Fibroblasts FSP1^{+}': 20,
            'Granulocytes': 21,
            'HER2^{+}': 22,
            'Ki67^{+}': 23,
            'Macrophages': 24,
            'Macrophages & granulocytes': 25,
            'MHC I & II^{hi}': 26,
            'MHC I^{hi}CD57^{+}': 27,
            'MHC^{hi}CD15^{+}': 28,
            'Myofibroblasts': 29,
            'Myofibroblasts PDPN^{+}': 30,
            'T_{Reg} & T_{Ex}': 31
        }


def get_mpfc_cell_type_mapping():

    all_celltypes = ["Smc", "Endo", "Astro", "eL6-2", "Oligo",
                     "eL5-3", "Reln", "VIP", "eL2/3", "NPY", 
                     "eL5-2", "Lhx6", "L5-1", "SST", "eL6-1"]
    all_celltypes.sort()

    cell_type_mapping = {}

    for i, x in enumerate(all_celltypes):
        cell_type_mapping[x] = i

    return cell_type_mapping


def get_lermi_cell_type_mapping():

    all_celltypes = ["Alveolar macrophages", "AT2", "B cells", "CD4+ T cells",             
                     "CD8+ T cells", "Classical monocytes", "DC", "Fibroblasts",
                     "Interstitial macrophages", "Lymphatic EC", "Mast cells",
                     "Other", "Pericytes", "Plasma cells", "Smooth muscle cells",
                     "Tumor cells", "Venous systemic EC"]
    all_celltypes.sort()

    cell_type_mapping = {}

    for i, x in enumerate(all_celltypes):
        cell_type_mapping[x] = i

    return cell_type_mapping


def get_group_ct_mapping(data_name, group_scheme, cell_type_mapping):
    group_ct_mapping = defaultdict(set)

    if group_scheme == "fine":
        for cell_type in cell_type_mapping:
            group_ct_mapping[cell_type] = set([cell_type])

    elif ((data_name in ["cords_2024", "cords_permutate", "cords_permutate5pct"]) and (group_scheme == "default")):
        required = set(get_cords_cell_type_mapping().keys())
        if set(cell_type_mapping.keys()) != required:
            raise ValueError(
                "group_scheme='default' for data_name in ['cords_2024', 'cords_permutate', 'cords_permutate5pct'] requires Cords cell types. "
                "Use group_scheme='fine' or add a dataset-specific grouping in data_utilities.py."
            )
        group_ct_mapping["immune"] = set(['Bcell',
                                          'CD4',
                                          'CD4_Treg',
                                          'CD8',
                                          'IDO_CD4',
                                          'IDO_CD8',
                                          'ki67_CD4',
                                          'ki67_CD8',
                                          'Myeloid',
                                          'Neutrophil',
                                          'PD1_CD4',
                                          'TCF1/7_CD4',
                                          'TCF1/7_CD8'])

        group_ct_mapping["tumor"] = set(['hypoxic',
                                         'normal'])

        group_ct_mapping["Fibroblast"] = set(['Collagen_CAF',
                                              'dCAF',
                                              'hypoxic_CAF',
                                              'hypoxic_tpCAF',
                                              'iCAF',
                                              'IDO_CAF',
                                              'mCAF',
                                              'PDPN_CAF',
                                              'SMA_CAF',
                                              'tpCAF',
                                              'vCAF'])

        group_ct_mapping["vessel"] = set(['Blood',
                                          'HEV',
                                          'Lymphatic'])

        group_ct_mapping["Other"] = set(['Other'])

    elif data_name == "cords_2024" and group_scheme == "group8":
        required = set(get_cords_cell_type_mapping().keys())
        if set(cell_type_mapping.keys()) != required:
            raise ValueError(
                "group_scheme='group8' for data_name='cords_2024' requires Cords cell types. "
                "Use group_scheme='fine' or add a dataset-specific grouping in data_utilities.py."
            )
        group_ct_mapping["immune"] = set(['Bcell',
                                        'CD4',
                                        'CD4_Treg',
                                        'CD8',
                                        'IDO_CD4',
                                        'IDO_CD8',
                                        'ki67_CD4',
                                        'ki67_CD8',
                                        'Myeloid',
                                        'Neutrophil',
                                        'PD1_CD4',
                                        'TCF1/7_CD4',
                                        'TCF1/7_CD8'])

        group_ct_mapping["tumor"] = set(['hypoxic',
                                        'normal'])

        group_ct_mapping["caf1"] = set(['mCAF',
                                        'SMA_CAF'])
        group_ct_mapping["caf2"] = set(['Collagen_CAF',
                                        'PDPN_CAF',
                                        'hypoxic_CAF',
                                        'hypoxic_tpCAF'])
        group_ct_mapping["caf3"] = set(['iCAF',
                                        'IDO_CAF'])
        group_ct_mapping["caf4"] = set(['dCAF',
                                        'tpCAF',
                                        'vCAF'])

        group_ct_mapping["vessel"] = set(['Blood',
                                        'HEV',
                                        'Lymphatic'])

        group_ct_mapping["Other"] = set(['Other'])

    elif ((data_name in ["danenberg_2022"]) and (group_scheme == "default")):
        required = set(get_Danenberg_cell_type_mapping().keys())
        if set(cell_type_mapping.keys()) != required:
            raise ValueError(
                "group_scheme='default' for data_name in ['danenberg_2022'] requires Danenberg cell types. "
                "Use group_scheme='fine' or add a dataset-specific grouping in data_utilities.py."
            )

        group_ct_mapping["immune"] = set(['T_{Reg} & T_{Ex}',
                                            'CD4^{+} T cells & APCs',
                                            'CD4^{+} T cells',
                                            'CD8^{+} T cells',
                                            'B cells',
                                            'CD38^{+} lymphocytes',
                                            'Granulocytes',
                                            'Macrophages',
                                            'Macrophages & granulocytes',
                                            'Ki67^{+}',
                                            'CD57^{+}'])

        group_ct_mapping["tumor"] = set(['MHC I & II^{hi}',
                                            'MHC I^{hi}CD57^{+}',
                                            'MHC^{hi}CD15^{+}',
                                            'HER2^{+}',
                                            'CK8-18^{+} ER^{hi}',
                                            'CK^{lo}ER^{med}',
                                            'Basal',
                                            'ER^{hi}CXCL12^{+}',
                                            'CK8-18^{hi}CXCL12^{hi}',
                                            'CD15^{+}',
                                            'Ep CD57^{+}',
                                            'CK^{+} CXCL12^{+}',
                                            'Ep Ki67^{+}',
                                            'CK^{med}ER^{lo}',
                                            'CK^{lo}ER^{lo}',
                                            'CK8-18^{hi}ER^{lo}'])

        group_ct_mapping["stroma"] = set(['Fibroblasts',
                                            'Fibroblasts FSP1^{+}',
                                            'Myofibroblasts',
                                            'Myofibroblasts PDPN^{+}'])

        group_ct_mapping["endothelial"] = set(['Endothelial'])

    elif ((data_name in ["mpfc_2022"]) and (group_scheme == "default")):
        required = set(get_mpfc_cell_type_mapping().keys())
        if set(cell_type_mapping.keys()) != required:
            raise ValueError(
                "group_scheme='default' for data_name in ['mpfc_2022'] requires mpfc cell types. "
                "Use group_scheme='fine' or add a dataset-specific grouping in data_utilities.py."
            )

        all_celltypes = list(get_mpfc_cell_type_mapping().keys())

        for ct in all_celltypes:
            group_ct_mapping[ct] = set([ct])

    elif ((data_name in ["lermi_2025"]) and (group_scheme == "default")):
        required = set(get_lermi_cell_type_mapping().keys())
        if set(cell_type_mapping.keys()) != required:
            raise ValueError(
                "group_scheme='default' for data_name in ['lermi_2025'] requires Lermi cell types. "
                "Use group_scheme='fine' or add a dataset-specific grouping in data_utilities.py."
            )
        group_ct_mapping["immune"] = set(['CD4+ T cells',
                                          'CD8+ T cells',
                                          'B cells',
                                          'Plasma cells',
                                          'Alveolar macrophages',
                                          'Interstitial macrophages',
                                          'DC',
                                          'Classical monocytes',
                                          'Mast cells'])

        group_ct_mapping["tumor"] = set(['Tumor cells'])

        group_ct_mapping["Fibroblast"] = set(['Fibroblasts',
                                              'Smooth muscle cells'])

        group_ct_mapping["vessel"] = set(['Venous systemic EC',
                                          'Lymphatic EC',
                                          'Pericytes'])

        group_ct_mapping["Other"] = set(['AT2',
                                         'Other'])

    else:
        raise ValueError(
            f"Unknown group_scheme '{group_scheme}' for data_name '{data_name}'. "
            "Add a dataset-specific branch in get_group_ct_mapping() or use group_scheme='fine'."
        )

    grouped_cell_types = set().union(*group_ct_mapping.values())
    missing = set(cell_type_mapping.keys()) - grouped_cell_types
    extra = grouped_cell_types - set(cell_type_mapping.keys())
    if missing:
        raise ValueError(f"Cell types missing from group_scheme {group_scheme}: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown cell types in group_scheme {group_scheme}: {sorted(extra)}")

    return group_ct_mapping


class data_features(object):
    def __init__(self, data_name, graph_type, group_scheme="default",
                 dist_cutoff=16, expanded_edge_cutoff=48,
                 top_k=4, ctg_comp_dist_cutoff=0.176,
                 degree_limit=20, model_run_id=None, data_dir=None):

        if model_run_id is None:
            model_run_id = make_model_run_id(data_name, group_scheme, dist_cutoff,
                                             expanded_edge_cutoff, top_k,
                                             ctg_comp_dist_cutoff, degree_limit)

        self.model_run_id = model_run_id
        self.result_subfolder = model_run_id
        self.group_scheme = group_scheme

        if graph_type=="extended":
            self.processed_folder_name = "tg_graph_extended"
        elif graph_type=="basic":
            self.processed_folder_name = "tg_graph_basic"
        elif graph_type=="local":
            self.processed_folder_name = "tg_graph_local"

        if data_name in ["cords_2024", "cords_permutate", "cords_permutate5pct"]:

            if data_name == "cords_2024":
                self.data_dir = data_dir or "./data/Cords_data"
            elif data_name == "cords_permutate":
                self.data_dir = data_dir or "./data/Cords_data_permutate"
            elif data_name == "cords_permutate5pct":
                self.data_dir = data_dir or "./data/Cords_data_permutate5pct"

            self.raw_dir = os.path.join(self.data_dir, "raw_data")
            self.dataset_root = os.path.join(self.data_dir, "graph_objects", model_run_id)

            self.cell_type_mapping = get_cords_cell_type_mapping()
            self.group_ct_mapping = get_group_ct_mapping(data_name, group_scheme, self.cell_type_mapping)

            df_regions = pd.read_csv(os.path.join(self.data_dir, "region_list.csv"),
                                     header=0)

            self.train_images = df_regions["region_ID"].tolist()

            self.dist_cutoff = dist_cutoff
            self.expanded_edge_cutoff = expanded_edge_cutoff
            self.top_k = top_k
            self.ctg_comp_dist_cutoff = ctg_comp_dist_cutoff
            self.degree_limit = degree_limit
            self.path_purity_cutoff = 0.90
            self.n_cells_threshold = 30

        elif data_name in ["danenberg_2022"]:

            if data_name == "danenberg_2022":
                self.data_dir = data_dir or "./data/Danenberg_data"

            self.raw_dir = os.path.join(self.data_dir, "raw_data")
            self.dataset_root = os.path.join(self.data_dir, "graph_objects", model_run_id)

            self.cell_type_mapping = get_Danenberg_cell_type_mapping()
            self.group_ct_mapping = get_group_ct_mapping(data_name, group_scheme, self.cell_type_mapping)

            df_images = pd.read_csv(os.path.join(self.data_dir, "Metabric_images_geq_1000_cells.csv"),
                                    header=0)

            full_subtype = ["Positive", "Negative"]
            df_images_subtype = df_images[df_images['er_sd'].isin(full_subtype)]

            self.train_images = df_images_subtype["image_fullname"].tolist()

            self.dist_cutoff = dist_cutoff
            self.expanded_edge_cutoff = expanded_edge_cutoff
            self.top_k = top_k
            self.ctg_comp_dist_cutoff = ctg_comp_dist_cutoff
            self.degree_limit = degree_limit
            self.path_purity_cutoff = 0.90
            self.n_cells_threshold = 20

        elif data_name in ["mpfc_2022"]:

            if data_name == "mpfc_2022":
                self.data_dir = data_dir or "./data/mpfc_data"

            self.raw_dir = os.path.join(self.data_dir, "raw_data")
            self.dataset_root = os.path.join(self.data_dir, "graph_objects", model_run_id)

            self.cell_type_mapping = get_mpfc_cell_type_mapping()
            self.group_ct_mapping = get_group_ct_mapping(data_name, group_scheme, self.cell_type_mapping)

            df_images = pd.read_csv(os.path.join(self.data_dir, "image_info.csv"),
                                    header=0)
            full_subtype = ["all"]
            df_images_subtype = df_images[df_images['subtype'].isin(full_subtype)]

            self.train_images = df_images_subtype["image_fullname"].tolist()

            self.dist_cutoff = dist_cutoff
            self.expanded_edge_cutoff = expanded_edge_cutoff
            self.top_k = top_k
            self.ctg_comp_dist_cutoff = ctg_comp_dist_cutoff
            self.degree_limit = degree_limit
            self.path_purity_cutoff = 0.90
            self.n_cells_threshold = 10

        elif data_name in ["lermi_2025"]:

            if data_name == "lermi_2025":
                self.data_dir = data_dir or "../../data/Lermi_data"

            self.raw_dir = os.path.join(self.data_dir, "raw_data")
            self.dataset_root = os.path.join(self.data_dir, "graph_objects", model_run_id)

            self.cell_type_mapping = get_lermi_cell_type_mapping()
            self.group_ct_mapping = get_group_ct_mapping(data_name, group_scheme, self.cell_type_mapping)

            df_regions = pd.read_csv(os.path.join(self.data_dir, "region_list.csv"),
                                     header=0)

            self.train_images = df_regions["region_ID"].tolist()

            self.dist_cutoff = dist_cutoff
            self.expanded_edge_cutoff = expanded_edge_cutoff
            self.top_k = top_k
            self.ctg_comp_dist_cutoff = ctg_comp_dist_cutoff
            self.degree_limit = degree_limit
            self.path_purity_cutoff = 0.90
            self.n_cells_threshold = 30

        else:
            if data_dir is None:
                data_dir = os.path.join("./data", data_name)

            self.data_dir = data_dir
            self.raw_dir = os.path.join(self.data_dir, "raw_data")
            self.dataset_root = os.path.join(self.data_dir, "graph_objects", model_run_id)

            df_regions = pd.read_csv(os.path.join(self.data_dir, "region_list.csv"),
                                     header=0)
            self.train_images = df_regions["region_ID"].tolist()

            cell_types = set()
            for region_id in self.train_images:
                df_cur = pd.read_csv(os.path.join(self.raw_dir, f"{region_id}.csv"),
                                     header=0,
                                     usecols=["CELL_TYPE"])
                cell_types.update(df_cur["CELL_TYPE"].dropna().astype(str).unique())

            self.cell_type_mapping = {cell_type: i for i, cell_type in enumerate(sorted(cell_types))}
            self.group_ct_mapping = get_group_ct_mapping(data_name, group_scheme, self.cell_type_mapping)

            self.dist_cutoff = dist_cutoff
            self.expanded_edge_cutoff = expanded_edge_cutoff
            self.top_k = top_k
            self.ctg_comp_dist_cutoff = ctg_comp_dist_cutoff
            self.degree_limit = degree_limit
            self.path_purity_cutoff = 0.90
            self.n_cells_threshold = 30
