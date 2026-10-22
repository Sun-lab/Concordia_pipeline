import torch
from typing import Tuple
from torch import Tensor


from torch_geometric.utils import to_torch_csr_tensor
from torch_geometric.utils import to_dense_batch, spmm, scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum('ijj->i', x)

def sparse_mincut_pool(
    x: Tensor,
    edge_index: Tensor,
    batch: Tensor,
    s: Tensor,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""a sparse version of the MinCut pooling operator from the `"Spectral Clustering in Graph
        Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
        paper, which can take sparse format of adjaceny matrix as input
        compatible with minibatch data object from torch_geometric.loader.DataLoader

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened and symmetrically
    normalized adjacency matrix and two auxiliary objectives: (1) The MinCut
    loss

    .. math::
        \mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A}
        \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}
        \mathbf{S})}

    where :math:`\mathbf{D}` is the degree matrix, and (2) the orthogonality
    loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F.


    Args:
        x (torch.Tensor): Node feature tensor, shape :math:`(|\mathcal{V}|, F_{in})`,
        edge_index (torch.Tensor): the edge index object in COO format, shape :math:`(2, |\mathcal{E}|)`,
                              where :math:`|\mathcal{E}|` is the number of edges after removing self edges, i.e., edges of the form :math:`(i,i)`.
        batch (torch.Tensor): Node-to-graph assignment vector for a batched graph,
                              shape :math:`(|\mathcal{V}|,)`, with entries in
                              :math:`\{0, 1, \dots, B-1\}`, where :math:`B` is the batch size.
                              Each element indicates which graph the corresponding node belongs to.
        s (torch.Tensor): assignment tensor before softmax, shape: math:`\mathbf{S} \in \mathbb{R}^{|\mathcal{V}| \times C}`
                          with number of clusters :math:`C`.
        temp (float, optional): Temperature parameter for softmax function.
            (default: :obj:`1.0`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`)
    """


    num_nodes = x.shape[0]

    norm_edge_index, norm_edge_weight = gcn_norm(
        edge_index, edge_weight=None, num_nodes=num_nodes,
        add_self_loops=False, dtype=x.dtype)

    adj_csr = to_torch_csr_tensor(norm_edge_index,
                                  norm_edge_weight,
                                  [num_nodes, num_nodes])

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    # matrix multiplication between unbatched matrices
    a_s = spmm(adj_csr.to(torch.float), s)

    # row sum of sparse adjacency matrix
    # in the format of unfolded vector
    row, col = norm_edge_index[0], norm_edge_index[1]
    idx = row
    deg = scatter(norm_edge_weight, idx, dim=0,
                  dim_size=num_nodes, reduce='sum')

    d_idx = torch.arange(deg.shape[0], device=deg.device)
    d_edge_index = torch.stack([d_idx, d_idx], dim=0)

    d_csr = to_torch_csr_tensor(
        d_edge_index,
        deg,
        size=(deg.shape[0], deg.shape[0]),
    )

    # matrix multiplication between unbatched matrices
    d_s = spmm(d_csr.to(torch.float).to(s.device), s)


    x_batched, mask = to_dense_batch(x, batch)
    s_batched, _ = to_dense_batch(s, batch)
    a_s_batched, _ = to_dense_batch(a_s, batch)
    d_s_batched, _ = to_dense_batch(d_s, batch)


    x_batched = x_batched.unsqueeze(0) if x_batched.dim() == 2 else x_batched
    #adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s_batched = s_batched.unsqueeze(0) if s_batched.dim() == 2 else s_batched
    a_s_batched = a_s_batched.unsqueeze(0) if a_s_batched.dim() == 2 else a_s_batched
    d_s_batched = d_s_batched.unsqueeze(0) if d_s_batched.dim() == 2 else d_s_batched


    (batch_size, batch_num_nodes, _), k = x_batched.size(), s_batched.size(-1)


    if mask is not None:
        mask = mask.view(batch_size, batch_num_nodes, 1).to(x_batched.dtype)
        x_batched, s_batched, a_s_batched, d_s_batched = \
        x_batched * mask, s_batched * mask, a_s_batched * mask, d_s_batched * mask

    out = torch.matmul(s_batched.transpose(1, 2), x_batched)
    out_adj = torch.matmul(s_batched.transpose(1, 2), a_s_batched)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    mincut_den = _rank3_trace(
        torch.matmul(s_batched.transpose(1, 2), d_s_batched))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s_batched.transpose(1, 2), s_batched)

    i_s = torch.eye(k).type_as(ss)

    ortho_loss = torch.linalg.matrix_norm(
        ss/torch.linalg.matrix_norm(ss, dim=(-2, -1), keepdim=True) -
        i_s/torch.linalg.matrix_norm(i_s), dim=(-2, -1))
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss

