

import torch
from copy import deepcopy

class compute_ct_proportion(object):
    """Transformer to compute cell type proportion"""
    def __init__(self, n_cell_type):
        self.n_cell_type = n_cell_type

    def __call__(self, data):
        data = deepcopy(data)
        x = torch.nn.functional.one_hot(data.x[:, 0].long(),
                                num_classes = self.n_cell_type)
        x = torch.sum(x, 0)
        x = torch.div(x, torch.sum(x))
        x = torch.unsqueeze(x, 0)
        data.x = x
        return data

class add_num_of_cells(object):
    """Transformer to add the number of cells to each image"""
    def __call__(self, data):
        data = deepcopy(data)
        num_of_cells = data.x.size()[0]
        data.n_cells = num_of_cells
        return data
