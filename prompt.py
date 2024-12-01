import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt(nn.Module):
    def __init__(self, pool_size, keys_dim, prompt_dim, embed_dim, top_k):
        super(Prompt, self).__init__()
        self.pool_size = pool_size
        self.keys = nn.Parameter(torch.randn(pool_size, keys_dim)) # M x Embed_Dim
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_dim, embed_dim)) # M x Lp x Embed_Dim
        nn.init.uniform_(self.keys, -1, 1)
        nn.init.uniform_(self.prompts, -1, 1)
        self.k = top_k

    def select_prompt(self, queries): # Shape of Q (N x Embed_Dim)
        N, embed_dim = queries.shape[0], self.prompts.shape[2]

        queries_normalized = F.normalize(queries) # N x Embed_Dim
        keys_normalized = F.normalize(self.keys) # M x Embed_Dim

        cosine_similarity = torch.mm(queries_normalized, keys_normalized.T) # N x M

        _, indices = torch.topk(cosine_similarity, self.k, dim=1) # N x K

        prompt_idx, prompt_cnt = torch.unique(indices, sorted=True, return_counts=True) # 1D Tensor

        if prompt_idx.shape[0] < self.pool_size:
            min_value = torch.min(indices.flatten()).item()
            padding_size = self.pool_size - prompt_idx.shape[0]
            prompt_idx = torch.cat([prompt_idx, torch.full((padding_size,), min_value, device=prompt_idx.device)]) # 1D Tensor of size pool_size 
            prompt_cnt = torch.cat([prompt_cnt, torch.zeros((padding_size,), device=prompt_cnt.device)]) # 1D Tensor of size pool_size

        _, idx = torch.topk(prompt_cnt, self.k) # 1D Tensor of size to k

        final_indices = prompt_idx[idx].expand(N, -1) # N x K
        
        selected_prompts = self.prompts[final_indices].view(N, -1, embed_dim) # N x (K * Lp) x Embed_Dim

        selected_keys = keys_normalized[final_indices] # N x K x Embed_Dim
        queries_normalized = queries_normalized.unsqueeze(1) # N x 1 x Embed_Dim
        loss_term = selected_keys * queries_normalized # N x K x Emebd_Dim
        sum = torch.sum(loss_term) / N # 1 x 1

        return selected_prompts, sum # N x (K * Lp) x Embed_Dim and 1 x 1