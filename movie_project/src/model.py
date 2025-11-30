import torch
import torch.nn as nn

class NCF(nn.Module):
    """神经协同过滤模型"""
    def __init__(self, n_users: int, n_movies: int, 
                 embedding_dim: int = 64, 
                 hidden_dims: list = None, 
                 dropout: float = 0.3):
        super(NCF, self).__init__()
        
        hidden_dims = hidden_dims or [128, 64, 32]
        
        # Embedding层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # MLP
        self.mlp = nn.Sequential()
        input_dim = embedding_dim * 2
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.mlp.add_module(f"fc_{i}", nn.Linear(input_dim, hidden_dim))
            self.mlp.add_module(f"relu_{i}", nn.ReLU())
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        # Embedding
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        
        # 拼接
        x = torch.cat([user_embedded, movie_embedded], dim=-1)
        
        # MLP
        x = self.mlp(x)
        
        # 输出
        rating = self.output_layer(x)
        return rating.squeeze()