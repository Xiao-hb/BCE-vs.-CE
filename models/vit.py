# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat    # 分别用于重新指定张量矩阵的维度、增加张量矩阵的维度 → 还有 reduce() 函数，用来减少维度
from einops.layers.torch import Rearrange   # 功能与上述相似，区别在于：用于在神经网络模型中作为一个模块来处理张量，通常用于构建模型时，作为层的一部分

from models.linear import inner_product    # 自定义全连接层

# helpers

# 返回一个表示图像（块）高和宽的元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

# 实现 Transformer Encoder 结构中，在多头注意力机制和MLP前进行 Normalization（LN）的操作
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        
        # 获取自注意力的 Q K V
        # 注意：原始 Transformer 是单独使用线性映射得到 Q K V 的，而这里只使用了 1 个线性映射 → ∵ ViT中没有涉及到类似 Transformer 中的解码问题（即 Q K V 来自不同的输入）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)   # 将输入进行线性映射，注意有个 * 3

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    '''
    torch.chunk(input, chunks, dim=0) 函数会将输入张量沿指定维度 dim 均匀分割成特定数量的张量块 chunks，并返回元素为张量块的元组
        注意：1. PyTorch 的默认维度排列为 [batch_size, channel, height, width]，第 0 个维度即指 batch_size 维度
             2. 诸如 nn.Linear() 等函数，一般接受一个二维张量，大小为 [batch_size, in_features]，处理后变为 [batch_size, out_features]，即一般处理数据的最后一个维度'''
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)     # 对 tensor 张量分块，dim=-1 表示沿着最后一个维度进行分割（一般就是 width） → 得到一个长度为 3 的元组，元素的形状与 x 相同
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # 拆分，方便输入后续的多头，qkv的大小为 [batch_size, num_patches, heads*dim_head]

        '''
        torch.matmul() 用于执行矩阵乘法
        torch.transpose(input, dim0, dim1) 进行转置操作，交换维度 dim 和 dim1'''
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale    # 注意力公式中的：除以 sprt(d)

        attn = self.attend(dots)    # 经过 softmax() 转换

        out = torch.matmul(attn, v) # 计算的到注意力结果
        out = rearrange(out, 'b h n d -> b n (h d)')    # 将多头注意力的输出合并，注意不要遗漏
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        '''
        nn.ModuleList 是一个存储不同 module，并自动将每个 module 的 parameters 添加到网络中的容器
            注意：1. 加入到 nn.ModuleList 中的 module 及其参数会自动注册到整个网络上
                 2. 不同于 nn.Sequential 里的模块是按照顺序排列的，且必须保证前一个模块的输出大小等于下一个模块的输入大小，
                    nn.NoduleList 只是将不同的模块存储在一起，并无顺序可言
                 3. 使用 nn.ModuleList 可以方便的创建网络中重复/相似的层 → 用 for 循环、append() 等'''
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),    # 多头注意力 MSA
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))                              # 前馈神经网络 MLP
            ]))     # 注意这里也使用了 nn.ModuleList() 方法

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x     # 残差连接
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 gamma_WH=16, bias_init_mode=0, bias_init_mean=0., bias_init_var=0.):
        super().__init__()
        image_height, image_width = pair(image_size)    # e.g. 224x224
        patch_height, patch_width = pair(patch_size)    # e.g. 16 x 16

        # 断言语句：表达式结果为 False 时，触发异常
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)     # image patch 的数目
        patch_dim = channels * patch_height * patch_width   # flattened image patch 的长度

        # 整合输出结果用于分类的方式
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 处理 image patch 为embedding vector
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),      # 将 flattened image patch 的维度映射到 Transformer Encoder 的输入维度
        )

        # 生成可学习参数 → torch.randon 生成具有标准正态分布的随机数，接受一个形状参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))     # 位置编码，包括 [class] token 的位置编码，即 +1
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))       # [class] token 用于末层分类
        self.dropout = nn.Dropout(emb_dropout)      # patch embedding 后经过一个 dropout层

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)   # 输入 Transformer 结构

        self.pool = pool
        '''
        nn.Identity() 是一个层，本质是一个恒等映射，不对输入进行任何变换或操作，简单地将输入返回作为输出'''
        self.to_latent = nn.Identity()

        # 用于分类任务的 MLP 层，注意这里的输入维度 dim 和 Transformer Encoder 结构的输入维度 dim 相同 → 改为了下面两行代码
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        self.LN = nn.LayerNorm(dim)
        self.fc = inner_product(in_features=dim, out_features=num_classes,
                                bias_init_mode=bias_init_mode, bias_init_mean=bias_init_mean)

    def forward(self, img):
        x = self.to_patch_embedding(img)    # 输入img，大小为 [batch_size, 3, 224, 224]；输出大小为 [batch_size, 196, 1024]
        b, n, _ = x.shape   # 注意：这里的 n 与 repeat() 函数里的 n 没有直接关系 → 一个为 196，一个为 1

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)   # 生成 [class] token，并扩展到与输入 batch 大小相匹配的形状
        x = torch.cat((cls_tokens, x), dim=1)   # 将 [class] token 与 patch embedding 进行拼接
        x += self.pos_embedding[:, :(n + 1)]    # 加上位置编码
        x = self.dropout(x)     # 对应参数列表里的 emd_dropout

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # 提取用于分类的信息：[class] token 或 GAP，注意计算 mean 时的维度指定

        x = self.to_latent(x)

        x = self.LN(x)
        features = x    # 添加：获取 ViT 学习到的特征
        x = self.fc(x)

        return x, features
        # return self.mlp_head(x)
