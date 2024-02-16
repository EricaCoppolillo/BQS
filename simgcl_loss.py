import torch.nn.functional as F
import torch


def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


def cal_cl_loss(model, idx):
    u_idx = torch.unique(idx[0]).cuda().long()
    i_idx = torch.unique(idx[1]).cuda().long()
    user_view_1, item_view_1 = model(perturbed=True)
    user_view_2, item_view_2 = model(perturbed=True)
    user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
    item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
    return user_cl_loss + item_cl_loss


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2) / emb.shape[0]
    return emb_loss * reg
