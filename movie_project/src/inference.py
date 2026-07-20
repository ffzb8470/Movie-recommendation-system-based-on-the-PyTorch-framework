import torch
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from src.model import NCF
from src.config import Config, ModelConfig

# 注册安全全局变量
torch.serialization.add_safe_globals([Config, ModelConfig])

class InferenceEngine:
    """推理引擎（最终优化版：支持过滤已观看电影）"""
    def __init__(self, model_path: str, movies_path: str, encoders_path: str = "models"):
        self.encoders_path = encoders_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载依赖组件（顺序调整：先加载编码器，再加载模型）
        self.user_encoder, self.movie_encoder = self._load_encoders()
        self.movies = self._load_movies(movies_path)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> NCF:
        """加载模型（修复配置问题）"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        # 安全加载checkpoint
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )
        
        # 验证必要字段
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint缺少必要字段: {key}")
        
        # 处理模型配置（兼容多种保存方式）
        if 'config' in checkpoint and isinstance(checkpoint['config'], Config):
            model_config = checkpoint['config'].model
        elif 'model_config' in checkpoint:
            model_config = ModelConfig(**checkpoint['model_config'])
        else:
            model_config = ModelConfig()
        
        # 获取用户/电影数量（从编码器优先，最准确）
        try:
            n_users = len(self.user_encoder.classes_)
            n_movies = len(self.movie_encoder.classes_)
        except Exception as e:
            if 'stats' in checkpoint:
                n_users = checkpoint['stats']['n_users']
                n_movies = checkpoint['stats']['n_movies']
            else:
                raise ValueError(
                    "无法获取用户/电影数量！请确保：\n"
                    "1. 编码器文件（user_encoder.pkl/movie_encoder.pkl）存在且完整\n"
                    "或\n"
                    "2. 训练时将stats保存到checkpoint中"
                ) from e
        
        # 初始化并加载模型（DeepFM）
        model = NCF(
            n_users=n_users,
            n_movies=n_movies,
            embedding_dim=model_config.embedding_dim,
            hidden_dims=model_config.hidden_dims,
            dropout=model_config.dropout,
            n_genres=getattr(model_config, 'n_genres', 18),
            n_user_stats=getattr(model_config, 'n_user_stats', 3),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model

    def _load_encoders(self) -> tuple[pickle.Pickler, pickle.Pickler]:
        """加载用户/电影编码器（LabelEncoder）"""
        encoders_dir = Path(self.encoders_path)
        if not encoders_dir.exists():
            raise FileNotFoundError(f"编码器目录未找到: {encoders_dir}")
        
        # 加载用户编码器
        user_encoder_path = encoders_dir / "user_encoder.pkl"
        if not user_encoder_path.exists():
            raise FileNotFoundError(f"用户编码器未找到: {user_encoder_path}")
        with open(user_encoder_path, "rb") as f:
            user_encoder = pickle.load(f)
        
        # 加载电影编码器
        movie_encoder_path = encoders_dir / "movie_encoder.pkl"
        if not movie_encoder_path.exists():
            raise FileNotFoundError(f"电影编码器未找到: {movie_encoder_path}")
        with open(movie_encoder_path, "rb") as f:
            movie_encoder = pickle.load(f)
        
        return user_encoder, movie_encoder

    def _load_movies(self, movies_path: str) -> pd.DataFrame:
        """加载电影数据（含合法性校验）"""
        movies_path = Path(movies_path)
        if not movies_path.exists():
            raise FileNotFoundError(f"电影数据文件未找到: {movies_path}")
        
        movies = pd.read_csv(movies_path)
        movies['movieId'] = movies['movieId'].astype(int)
        
        # 校验必要列
        required_cols = ['movieId', 'title', 'genres']
        for col in required_cols:
            if col not in movies.columns:
                raise ValueError(f"电影数据文件缺少必要列: {col}")
        
        return movies

    def recommend(
        self, 
        user_id: int, 
        top_n: int = 10, 
        watched_movie_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        为指定用户生成推荐（支持过滤已观看电影）
        Args:
            user_id: 原始用户ID
            top_n: 推荐数量
            watched_movie_ids: 已观看电影的原始ID列表（可选）
        Returns:
            推荐结果列表（含电影ID、标题、类型、预测评分）
        """
        # 1. 验证用户ID有效性
        if user_id not in self.user_encoder.classes_:
            raise ValueError(f"用户ID {user_id} 不在训练数据中（有效用户ID需在编码器记录中）")
        
        # 2. 编码用户ID
        user_encoded = self.user_encoder.transform([user_id])[0]
        
        # 3. 处理已观看电影（默认空列表）
        if watched_movie_ids is None:
            watched_movie_ids = []
        # 过滤掉不在电影数据中的ID
        watched_movie_ids = [mid for mid in watched_movie_ids if mid in self.movies['movieId'].values]
        
        # 4. 获取待推荐电影（未观看 + 在训练集中出现过）
        all_movie_ids = self.movies['movieId'].tolist()
        candidate_movie_ids = [
            mid for mid in all_movie_ids
            if mid not in watched_movie_ids  # 过滤已观看
            and mid in self.movie_encoder.classes_  # 过滤训练集外的电影
        ]
        
        if not candidate_movie_ids:
            raise ValueError(f"用户 {user_id} 已观看所有训练集中的电影，无新推荐")
        
        # 5. 批处理预测（避免显存溢出）
        batch_size = 1024
        all_scores = []
        for i in range(0, len(candidate_movie_ids), batch_size):
            batch_mids = candidate_movie_ids[i:i+batch_size]
            # 编码电影ID
            batch_movies_encoded = self.movie_encoder.transform(batch_mids)
            # 转换为tensor
            user_tensor = torch.tensor([user_encoded] * len(batch_mids), dtype=torch.long).to(self.device)
            movie_tensor = torch.tensor(batch_movies_encoded, dtype=torch.long).to(self.device)
            # 预测评分
            with torch.no_grad():
                batch_scores = self.model(user_tensor, movie_tensor)
                all_scores.extend(batch_scores.cpu().numpy().flatten())
        
        # 6. 按评分排序，取Top-N
        movie_score_pairs = list(zip(candidate_movie_ids, all_scores))
        movie_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = movie_score_pairs[:top_n]
        
        # 7. 格式化推荐结果
        recommendations = []
        for movie_id, score in top_pairs:
            movie_info = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            recommendations.append({
                'movieId': int(movie_id),
                'title': movie_info['title'].strip(),
                'genres': movie_info['genres'].strip(),
                'predicted_rating': round(float(score), 2)  # 评分保留2位小数
            })
        
        return recommendations

    def batch_recommend(
        self, 
        user_ids: List[int], 
        top_n: int = 10, 
        watched_movie_ids_dict: Optional[Dict[int, List[int]]] = None
    ) -> Dict[int, List[Dict] | Dict[str, str]]:
        """
        批量生成推荐
        Args:
            user_ids: 原始用户ID列表
            top_n: 推荐数量
            watched_movie_ids_dict: 字典{user_id: [watched_mid1, ...]}（可选）
        Returns:
            批量推荐结果
        """
        if watched_movie_ids_dict is None:
            watched_movie_ids_dict = {}
        
        recommendations = {}
        for uid in user_ids:
            try:
                watched = watched_movie_ids_dict.get(uid, [])
                recommendations[uid] = self.recommend(uid, top_n, watched)
            except ValueError as e:
                recommendations[uid] = {'error': str(e)}
            except Exception as e:
                recommendations[uid] = {'error': f"未知错误: {str(e)}"}
        
        return recommendations