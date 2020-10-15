import torch
import torch.nn.functional as F

from functools import partial

def project(W, p):
    """
    Projects weights W to S={w: card(w) <= p%}

    Parameters
    ----------
    W: torch.Tensor
        The weight matrix
    p: float
        Percent of weights to prune. 0<=p<=100

    Returns
    -------
    Z: torch.Tensor
        Pruned weight matrix (detached)
    mask: torch.Tensor
        A boolean mask of pruned weights
    """
    Z = W.clone().detach()  # we don't want to compute gradients with respect to Z
    Z = Z.view(-1)

    p = int(p * Z.numel()) // 100
    abs_Z = torch.abs(Z)
    v, _ = torch.kthvalue(abs_Z, p)
    mask = abs_Z <= v

    Z[mask] = 0
    Z = Z.view(W.shape)
    mask = mask.view(W.shape)
    return Z, mask

def project_column(W, p):
    """
    Projects weights W to S={w: #columns(w) <= p%}

    Parameters
    ----------
    W: torch.Tensor
        The weight matrix
    p: float
        Percent of weights to prune. 0<=p<=100

    Returns
    -------
    Z: torch.Tensor
        Pruned weight matrix (detached)
    mask: torch.Tensor
        A boolean mask of pruned weights
    """
    N = W.shape[0]
    Z = W.clone().detach() # we don't want to compute gradients with respect to Z
    Z = Z.view(N, -1)

    nz = torch.norm(Z, dim=0)
    p = int(p * nz.numel()) // 100
    v, _ = torch.kthvalue(nz, p)
    mask = (nz <= v).view(1, -1).repeat(N, 1)

    Z[mask] = 0
    Z = Z.view(W.shape)
    mask = mask.view(W.shape)
    return Z, mask

def project_filter(W, p):
    """
    Projects weights W to S={w: #filters(w) <= p%}

    Parameters
    ----------
    W: torch.Tensor
        The weight matrix
    p: float
        Percent of weights to prune. 0<=p<=100

    Returns
    -------
    Z: torch.Tensor
        Pruned weight matrix (detached)
    mask: torch.Tensor
        A boolean mask of pruned weights
    """
    N = W.shape[0]
    Z = W.clone().detach()
    Z = Z.view(N, -1)
    M = Z.shape[1]

    nz = torch.norm(Z, dim=1)
    p = int(p * nz.numel()) // 100
    v, _ = torch.kthvalue(nz, p)
    mask = (nz <= v).view(-1, 1).repeat(1, M)

    Z[mask] = 0
    Z = Z.view(W.shape)
    mask = mask.view(W.shape)
    return Z, mask

def admm_loss(aux):
    """
    Computes L2 regularization according to ADMM

    Parameters
    ----------
    aux: dict
        A map from weight name to tuple (W, Z, U, project_fun)

    Returns
    -------
    loss: torch.Tensor
        The resulting loss
    """

    loss = 0
    for weight_name, (W, Z, U, _) in aux.items():
        loss += F.mse_loss(W+U, Z, reduction='sum')
    return loss

def init_aux(model, p_array):
    """
    Returns auxiliary variables for filter pruning of the input network.

    Parameters
    ----------
    model: torch.nn.Module
        An input network

    Returns
    -------
    aux: dict
        A map from weight name to tuple (W, Z, U, project_fun)
    """
    aux = {}
    for weight_name, W in model.named_parameters():
        if len(aux) == len(p_array):
            break
        if not weight_name.endswith('weight'): # prune only weights
            continue
        project_fun = partial(project_filter, p=p_array[len(aux)])
        aux[weight_name] = (
                W,                                        # W
                project_fun(W)[0],                        # Z
                torch.zeros_like(W, requires_grad=False), # U
                project_fun,                              # project_fun
                )
    return aux

def init_mask(aux):
    """
    Initializes masks of weight to freeze

    Parameters
    ----------
    aux: dict
        A map from weight name to tuple (W, Z, U, project_fun)

    Returns
    -------
    mask: dict
        A map from weight name to tuple (W, mask)
    """
    with torch.no_grad():
        mask = {}
        for weight_name, (W, _, _, project_fun) in aux.items():
            _, m = project_fun(W)
            W[m] = 0
            mask[weight_name] = (W, m)
    return mask

def update_aux(aux, iteration=None):
    """
    Updates auxiliary variables

    Parameters
    ----------
    aux: dict
        A map from weight name to tuple (W, Z, U, project_fun)
    iteration: int, optional
        An index of the current iteration. Default: None
    """
    with torch.no_grad():
        for weight_name, (W, Z, U, project_fun) in aux.items():
            Z, _ = project_fun(W + U)
            diff = W - Z
            U += diff
            aux[weight_name] = (W, Z, U, project_fun)
            if not iteration is None:
                print(f'{iteration}th iteration: {weight_name} gap {torch.norm(diff).item()}')
