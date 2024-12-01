import torch
import torch.nn as nn
import timm

from prompt import Prompt

class Model(nn.Module):
    def __init__(self, pool_size, prompt_dim, top_k, num_classes):
        super(Model, self).__init__()
        self.query_function = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        self.embed_dim = self.query_function.embed_dim # 768
        self.prompt_pool = Prompt(pool_size, self.embed_dim, prompt_dim, self.embed_dim, top_k)
        self.vit = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True, num_classes=num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.total_prompt_dim = top_k * prompt_dim

        # Freeze the ViT model (both query_function and vit)
        for param in self.query_function.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def forward(self, x): # N x channels x H x W 
        queries = self.query_function.forward_features(x)[:, 0] # CLS Token (N x Embed_Dim)
        prompts, distance_sum = self.prompt_pool.select_prompt(queries) # N x (K * Lp) x Embed_Dim

        patch_embeddings = self.vit.patch_embed(x) # N x Token_Len x Embed_Dim
        cls_token = self.vit.cls_token.expand(patch_embeddings.shape[0], -1, -1) # N x 1 x Embed_Dim
        patch_embeddings = torch.cat((cls_token, patch_embeddings), dim=1) # N x (1 + Token_Len) x Embed_Dim
        embeddings = patch_embeddings + self.vit.pos_embed # N x (1 + Token_Len) x Embed_Dim

        tokens = torch.cat((prompts, embeddings), dim=1) # N x (K * Lp + 1 + Token_Len) x Embed_Dim

        tokens = self.vit.blocks(tokens) # N x (K * Lp + 1 + Token_Len) x Embed_Dim
        tokens = self.vit.norm(tokens)

        extracted_prompts = tokens[:, 0:self.total_prompt_dim, :] # N x K * Lp x Embed_Dim
        extracted_prompts = extracted_prompts.permute(0, 2, 1) # N x Embed_Dim x K * Lp

        pooled_prompts = self.pool(extracted_prompts).squeeze(-1) # N x Embed_Dim

        outputs = self.vit.head(pooled_prompts) # N x num_classes

        return outputs, distance_sum