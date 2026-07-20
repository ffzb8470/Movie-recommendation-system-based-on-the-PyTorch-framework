import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    """DeepFM: Factorization Machine + Deep Neural Network"""
    def __init__(self, n_users, n_movies, n_genres=18, n_user_stats=3,
                 n_gender=2, n_age=7, n_occupation=21,
                 embedding_dim=32, hidden_dims=None, dropout=0.5):
        super(DeepFM, self).__init__()
        
        hidden_dims = hidden_dims or [128, 64, 32]
        fm_dim = 8  # FM 交叉特征统一维度
        
        # ─── Embeddings for Deep MLP ───
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.movie_emb = nn.Embedding(n_movies, embedding_dim)
        self.gender_emb = nn.Embedding(n_gender, 4)
        self.age_emb = nn.Embedding(n_age, 4)
        self.occupation_emb = nn.Embedding(n_occupation, 8)
        
        # ─── Embeddings for FM 2nd-order (all same dim) ───
        self.fm_user = nn.Embedding(n_users, fm_dim)
        self.fm_movie = nn.Embedding(n_movies, fm_dim)
        self.fm_gender = nn.Embedding(n_gender, fm_dim)
        self.fm_age = nn.Embedding(n_age, fm_dim)
        self.fm_occupation = nn.Embedding(n_occupation, fm_dim)
        
        # ─── First-order linear weights ───
        self.user_linear = nn.Embedding(n_users, 1)
        self.movie_linear = nn.Embedding(n_movies, 1)
        self.gender_linear = nn.Embedding(n_gender, 1)
        self.age_linear = nn.Embedding(n_age, 1)
        self.occupation_linear = nn.Embedding(n_occupation, 1)
        self.genre_linear = nn.Linear(n_genres, 1, bias=False)
        self.stats_linear = nn.Linear(n_user_stats, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # ─── Deep MLP ───
        # Total embedding dimension for MLP input
        sparse_emb_dim = embedding_dim * 2 + 4 + 4 + 8  # user+movie+gender+age+occupation
        dense_dim = n_genres + n_user_stats
        mlp_input_dim = sparse_emb_dim + dense_dim
        
        mlp_modules = []
        input_dim = mlp_input_dim
        for h_dim in hidden_dims:
            mlp_modules.append(nn.Linear(input_dim, h_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_dim = h_dim
        mlp_modules.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_modules)
        
        self._init_weights()
    
    def _init_weights(self):
        for emb in [self.user_emb, self.movie_emb, self.fm_user, self.fm_movie]:
            nn.init.normal_(emb.weight, std=0.01)
        for emb in [self.gender_emb, self.age_emb, self.occupation_emb,
                    self.fm_gender, self.fm_age, self.fm_occupation]:
            nn.init.normal_(emb.weight, std=0.01)
        for lin in [self.user_linear, self.movie_linear, self.gender_linear, 
                     self.age_linear, self.occupation_linear]:
            nn.init.normal_(lin.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, movie_ids, genres=None, user_stats=None, user_features=None):
        batch_size = user_ids.size(0)
        device = user_ids.device
        
        # ─── Embeddings for FM 2nd-order ───
        user_emb = self.user_emb(user_ids)
        movie_emb = self.movie_emb(movie_ids)
        
        # User profile features
        if user_features is not None:
            gender = user_features[:, 0]
            age = user_features[:, 1]
            occupation = user_features[:, 2]
            gender_emb = self.gender_emb(gender)
            age_emb = self.age_emb(age)
            occ_emb = self.occupation_emb(occupation)
        else:
            gender_emb = torch.zeros(batch_size, 4, device=device)
            age_emb = torch.zeros(batch_size, 4, device=device)
            occ_emb = torch.zeros(batch_size, 8, device=device)
        
        # Dense features
        if genres is not None:
            genre_dense = genres
        else:
            genre_dense = torch.zeros(batch_size, 18, device=device)
        
        if user_stats is not None:
            stats_dense = user_stats
        else:
            stats_dense = torch.zeros(batch_size, 3, device=device)
        
        # ─── FM Part ───
        # 1st-order: linear
        first_order = (
            self.user_linear(user_ids).squeeze() +
            self.movie_linear(movie_ids).squeeze() +
            self.genre_linear(genre_dense).squeeze() +
            self.stats_linear(stats_dense).squeeze() +
            self.bias
        )
        if user_features is not None:
            first_order = (first_order +
                self.gender_linear(gender).squeeze() +
                self.age_linear(age).squeeze() +
                self.occupation_linear(occupation).squeeze()
            )
        
        # 2nd-order: pairwise FM
        # 使用统一维度的 FM embeddings
        fm_embeddings = [
            self.fm_user(user_ids),
            self.fm_movie(movie_ids),
        ]
        if user_features is not None:
            fm_embeddings.extend([
                self.fm_gender(gender),
                self.fm_age(age),
                self.fm_occupation(occupation),
            ])
        fm_emb = torch.stack(fm_embeddings, dim=1)
        
        # FM formula: 0.5 * sum( (sum(emb))^2 - sum(emb^2) )
        sum_emb = fm_emb.sum(dim=1)  # [batch, emb_dim]
        sum_sq = (fm_emb ** 2).sum(dim=1)  # [batch, emb_dim]
        second_order = 0.5 * (sum_emb ** 2 - sum_sq).sum(dim=1)
        
        fm_output = first_order + second_order
        
        # ─── Deep Part ───
        deep_input = torch.cat([
            user_emb, movie_emb, gender_emb, age_emb, occ_emb,
            genre_dense, stats_dense
        ], dim=1)
        
        deep_output = self.mlp(deep_input).squeeze()
        
        # ─── Combine FM + Deep ───
        rating = fm_output + deep_output
        return torch.sigmoid(rating) * 4 + 1


# 向后兼容
NCF = DeepFM