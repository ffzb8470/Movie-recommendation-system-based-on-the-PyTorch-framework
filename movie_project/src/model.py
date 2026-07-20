import torch
import torch.nn as nn

class NeuMF(nn.Module):
    """神经协同过滤模型 (NeuMF: GMF + MLP 双通道)"""
    def __init__(self, n_users: int, n_movies: int, 
                 embedding_dim: int = 64, 
                 hidden_dims: list = None, 
                 dropout: float = 0.3,
                 gmf_embedding_dim: int = None):
        super(NeuMF, self).__init__()
        
        hidden_dims = hidden_dims or [128, 64, 32]
        gmf_embedding_dim = gmf_embedding_dim or embedding_dim
        
        # ─── GMF 分支 ───
        self.user_embedding_gmf = nn.Embedding(n_users, gmf_embedding_dim)
        self.movie_embedding_gmf = nn.Embedding(n_movies, gmf_embedding_dim)
        
        # ─── MLP 分支 ───
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding_mlp = nn.Embedding(n_movies, embedding_dim)
        
        mlp_modules = []
        input_dim = embedding_dim * 2
        for i, hidden_dim in enumerate(hidden_dims):
            mlp_modules.append(nn.Linear(input_dim, hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_modules)
        
        # ─── 融合输出层 ───
        gmf_output_dim = gmf_embedding_dim
        mlp_output_dim = hidden_dims[-1]
        self.fusion = nn.Linear(gmf_output_dim + mlp_output_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重 (GMF用较小标准差, MLP用xavier)"""
        for emb in [self.user_embedding_gmf, self.movie_embedding_gmf]:
            nn.init.normal_(emb.weight, std=0.01)
        for emb in [self.user_embedding_mlp, self.movie_embedding_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
    
    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        # GMF 分支: 逐元素相乘
        user_gmf = self.user_embedding_gmf(user_ids)
        movie_gmf = self.movie_embedding_gmf(movie_ids)
        gmf_output = user_gmf * movie_gmf
        
        # MLP 分支: 拼接 + MLP
        user_mlp = self.user_embedding_mlp(user_ids)
        movie_mlp = self.movie_embedding_mlp(movie_ids)
        mlp_input = torch.cat([user_mlp, movie_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # 融合输出
        fusion_input = torch.cat([gmf_output, mlp_output], dim=-1)
        rating = self.fusion(fusion_input)
        return rating.squeeze()


# 为了向后兼容，保留 NCF 别名指向 NeuMF
NCF = NeuMF