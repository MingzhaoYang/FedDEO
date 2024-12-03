import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]@self.text_projection
        return x


class PromptLearner_client(nn.Module):
    def __init__(self, n_ctx_num,classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = n_ctx_num
        self.n_ctx = n_ctx
        ctx_init = ''
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        CSC = True
        if CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        train_prompt = ' '+ 'sks' * n_ctx + ' style image'
        

        print(f'Initial context: "{train_prompt}"')
        print(f"Number of context words (tokens): {n_ctx}")

        #tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = clip.tokenize(train_prompt)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(dtype)
        ctx_vectors = embedding[0][1].repeat(1,n_ctx,1)
        #nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx_global = nn.Parameter(ctx_vectors)  # to be optimized
    
        # print(embedding.shape)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        #torch.set_printoptions(edgeitems=13)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        #self.name_lens = name_lens
        self.class_token_position = "end"
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        class_prompts = [' '+ '*' * n_ctx + ' style image' + " of " + name for name in classnames]
        
        tokenized_class_prompts = clip.tokenize(class_prompts)
        with torch.no_grad():
            self.class_embedding = clip_model.token_embedding(tokenized_class_prompts.cuda()).type(dtype)
        
        
    def get_train_feature(self):
        # print('---ctx:',ctx.shape)
        ctx_global = self.ctx_global
        #if ctx_global.dim() == 2:
        #    ctx_global = ctx_global.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx_global,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        #print(prompts)
        return prompts
    
    def get_class_feature(self,cate):
        # print('---ctx:',ctx.shape)
        ctx_global = self.ctx_global
        class_embedding = self.class_embedding[cate]
        #print(class_embedding)
        #if ctx_global.dim() == 2:
        #    ctx_global = ctx_global.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = class_embedding[:, :1, :]
        suffix = class_embedding[:, 1 + self.n_ctx:, :]
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx_global,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts
    
class descriptor(nn.Module):
    def __init__(self,init):
        super().__init__()
        ctx_vectors = init.detach().clone()
        #nn.init.normal_(ctx_vectors, std=0.02)
        self.feature = nn.Parameter(ctx_vectors)  # to be optimized

    def forward(self):
        return self.feature
    
''' 
class ceshi(nn.Module):
    def __init__(self,classnames, clip_model,n_ctx_num=0):
        super().__init__()
        self.prompt_learner = PromptLearner_client(n_ctx_num, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model) 
        self.dtype = clip_model.dtype


    def forward(self,cate,class_name=False):
        tokenized_prompts = self.tokenized_prompts
        
        if class_name==True :
            prompts = self.prompt_learner.get_class_feature(cate)
        else :
            prompts = self.prompt_learner.get_train_feature()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        return text_features
'''
