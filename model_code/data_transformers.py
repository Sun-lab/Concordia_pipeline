
import torch
from copy import deepcopy


class add_num_of_cells(object):
    """Transformer to add the number of cells to each image"""
    def __call__(self, data):
        data = deepcopy(data)
        num_of_cells = data.x.size()[0]
        data.n_cells = num_of_cells
        return data
