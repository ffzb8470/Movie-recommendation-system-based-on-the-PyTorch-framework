import torch
import torch.nn as nn

class NeuMF(nn.Module):
    """神经协同过滤模型 (NeuMF: GMF + MLP 双通道 + 特征增强)"""
    def __init__(self, n_users: int, n_movies: int, 
                 embedding_dim: int = 64, 
                 hidden_dims: list = None, 
                 dropout: float = 0.3,
                 gmf_embedding_dim: int = None,
                 n_genres: int = 20,
                 n_user_stats: int = 3,
                 use_genres: bool = True,
                 use_user_stats: bool = True):
        super(NeuMF, self).__init__()
        
        hidden_dims = hidden_dims or [128, 64, 32]
        gmf_embedding_dim = gmf_embedding_dim or embedding_dim
        self.use_genres = use_genres
        self.use_user_stats = use_user_stats
        
        # ─── GMF 分支 ───
        self.user_embedding_gmf = nn.Embedding(n_users, gmf_embedding_dim)
        self.movie_embedding_gmf = nn.Embedding(n_movies, gmf_embedding_dim)
        
        # ─── MLP 分支 ───
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding_mlp = nn.Embedding(n_movies, embedding_dim)
        
        # MLP 输入维度
        mlp_input_dim = embedding_dim * 2
        
        # ─── 电影类型特征 ───
        if use_genres:
            self.genre_embedding = nn.Linear(n_genres, embedding_dim // 2)
            mlp_input_dim += embedding_dim // 2
        
        # ─── 用户统计特征 ───
        if use_user_stats:
            self.user_stats_embedding = nn.Linear(n_user_stats, embedding_dim // 2)
            mlp_input_dim += embedding_dim // 2
        
        # MLP 层
        mlp_modules = []
        input_dim = mlp_input_dim
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
        """初始化权重"""
        for emb in [self.user_embedding_gmf, self.movie_embedding_gmf]:
            nn.init.normal_(emb.weight, std=0.01)
        for emb in [self.user_embedding_mlp, self.movie_embedding_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        if self.use_genres:
            nn.init.xavier_uniform_(self.genre_embedding.weight)
            nn.init.zeros_(self.genre_embedding.bias)
        if self.use_user_stats:
            nn.init.xavier_uniform_(self.user_stats_embedding.weight)
            nn.init.zeros_(self.user_stats_embedding.bias)
        
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
    
    def forward(self, user_ids, movie_ids, genres=None, user_stats=None):
        # GMF 分支
        user_gmf = self.user_embedding_gmf(user_ids)
        movie_gmf = self.movie_embedding_gmf(movie_ids)
        gmf_output = user_gmf * movie_gmf
        
        # MLP 分支
        user_mlp = self.user_embedding_mlp(user_ids)
        movie_mlp = self.movie_embedding_mlp(movie_ids)
        mlp_features = [user_mlp, movie_mlp]
        
        # 拼接电影类型特征
        if self.use_genres and genres is not None:
            genre_feat = self.genre_embedding(genres)
            mlp_features.append(genre_feat)
        
        # 拼接用户统计特征
        if self.use_user_stats and user_stats is not None:
            user_feat = self.user_stats_embedding(user_stats)
            mlp_features.append(user_feat)
        
        mlp_input = torch.cat(mlp_features, dim=-1)
        
        # 动态调整 MLP 输入维度（适配特征缺失情况）
        actual_input_dim = mlp_input.size(-1)
        expected_input_dim = self.mlp[0].in_features
        if actual_input_dim != expected_input_dim:
            # 当特征缺失时，跳过不匹配的层（降级为纯 user+movie 推荐）
            mlp_output = mlp_input
            for module in self.mlp:
                if isinstance(module, nn.Linear) and module.in_features != mlp_output.size(-1):
                    continue
                mlp_output = module(mlp_output)
        else:
            mlp_output = self.mlp(mlp_input)
        
        # 融合
        fusion_input = torch.cat([gmf_output, mlp_output], dim=-1)
        rating = self.fusion(fusion_input)
        return rating.squeeze()


# 向后兼容
NCF = NeuMF