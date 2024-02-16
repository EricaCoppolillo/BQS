import torch.nn as nn
import torch.nn.functional as F
import torch

from simgcl_utils import convert_sparse_mat_to_tensor


class SimGCL(nn.Module):
    def __init__(self, norm_adj,
                 emb_size,
                 eps,
                 n_layers,
                 user_num,
                 item_num):
        super(SimGCL, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num])
        return user_all_embeddings, item_all_embeddings

    def predict(self, user, item_i, item_j=None):
        user_embeddings, item_embeddings = self.forward()
        user, item_i = user_embeddings[user], item_embeddings[item_i]
        prediction_i = (user * item_i).sum(dim=-1)
        if item_j is not None:
            item_j = item_embeddings[item_j]
            prediction_j = (user * item_j).sum(dim=-1)
            return prediction_i - prediction_j, user, item_i
        return prediction_i, user, item_i

    def score(self, user, items):
        user_embeddings, item_embeddings = self.forward()
        user, items_t = user_embeddings[user], item_embeddings[items]
        user = user.unsqueeze(-1)

        scores = torch.bmm(items_t, user).squeeze(-1)
        # scores = torch.mm(items_t, user).squeeze(-1)

        return scores
