import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

# 电影类型列表（固定20种）
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
    '(no genres listed)'
]


class MovieRatingDataset(Dataset):
    """PyTorch数据集（含负样本采样）"""
    def __init__(self, users, movies, ratings, genres=None, user_stats=None,
                 is_train=False, n_movies=None, neg_ratio=3,
                 n_genres=20, n_user_stats=3):
        self.users = users.values if hasattr(users, 'values') else users
        self.movies = movies.values if hasattr(movies, 'values') else movies
        self.ratings = ratings.values if hasattr(ratings, 'values') else ratings
        self.genres = genres if isinstance(genres, np.ndarray) else (genres.values if genres is not None else None)
        self.user_stats = user_stats if isinstance(user_stats, np.ndarray) else (user_stats.values if user_stats is not None else None)
        self.is_train = is_train
        self.n_movies = n_movies
        self.neg_ratio = neg_ratio  # 负样本比例（正样本:负样本）
        self.n_genres = n_genres
        self.n_user_stats = n_user_stats
        
        if is_train and n_movies:
            # 扩展数据集：每条正样本配 neg_ratio 条负样本
            self._neg_users = np.repeat(self.users, self.neg_ratio)
            self._neg_movies = np.random.randint(0, n_movies, len(self._neg_users))
            self._total_size = len(self.users) + len(self._neg_users)
    
    def __len__(self):
        if self.is_train and self.n_movies:
            return self._total_size
        return len(self.users)
    
    def __getitem__(self, idx):
        if self.is_train and self.n_movies:
            if idx < len(self.users):
                # 正样本
                i = idx
                is_neg = 0
                user = self.users[i]
                movie = self.movies[i]
                rating = self.ratings[i]
                genres_i = self.genres[i] if self.genres is not None else None
                stats_i = self.user_stats[i] if self.user_stats is not None else None
            else:
                # 负样本
                i = idx - len(self.users)
                is_neg = 1
                user = self._neg_users[i]
                movie = self._neg_movies[i]
                rating = 0.0  # 负样本评分=0
                genres_i = None
                stats_i = None
        else:
            i = idx
            is_neg = 0
            user = self.users[i]
            movie = self.movies[i]
            rating = self.ratings[i]
            genres_i = self.genres[i] if self.genres is not None else None
            stats_i = self.user_stats[i] if self.user_stats is not None else None
        
        result = {
            'user': torch.tensor(int(user), dtype=torch.long),
            'movie': torch.tensor(int(movie), dtype=torch.long),
            'rating': torch.tensor(float(rating), dtype=torch.float32),
            'genres': torch.tensor(genres_i if genres_i is not None else np.zeros(self.n_genres), dtype=torch.float32),
            'user_stats': torch.tensor(stats_i if stats_i is not None else np.zeros(self.n_user_stats), dtype=torch.float32),
        }
        
        return result


class DataPreprocessor:
    """数据预处理器（含特征工程）"""
    def __init__(self, config):
        self.config = config  # DataConfig
        self.model_config = None  # 后面由 prepare_data 设置
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.genre_encoder = MultiLabelBinarizer(classes=ALL_GENRES)
        
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """加载原始数据"""
        data_path = Path(self.config.dataset_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据集不存在: {data_path}")
        
        movies = pd.read_csv(data_path / "movies.csv")
        ratings = pd.read_csv(data_path / "ratings.csv")
        
        print(f"📊 原始数据 - 电影: {len(movies):,}部, 评分: {len(ratings):,}条")
        return movies, ratings
    
    def filter_data(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """筛选活跃用户和热门电影"""
        movie_counts = ratings['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= self.config.min_movie_ratings].index
        ratings = ratings[ratings['movieId'].isin(valid_movies)]
        
        user_counts = ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.config.min_user_ratings].index
        ratings = ratings[ratings['userId'].isin(valid_users)]
        
        print(f"✅ 筛选后 - 用户: {ratings['userId'].nunique():,}人, "
              f"电影: {ratings['movieId'].nunique():,}部, 评分: {len(ratings):,}条")
        return ratings
    
    def encode_ids(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """编码用户和电影ID"""
        ratings['user'] = self.user_encoder.fit_transform(ratings['userId'])
        ratings['movie'] = self.movie_encoder.fit_transform(ratings['movieId'])
        return ratings
    
    def encode_genres(self, movies: pd.DataFrame) -> pd.DataFrame:
        """将电影类型编码为 multi-hot 矩阵"""
        genre_matrix = self.genre_encoder.fit_transform(movies['genres'].str.split('|'))
        genre_df = pd.DataFrame(genre_matrix, columns=ALL_GENRES, index=movies['movieId'])
        return genre_df
    
    def compute_user_stats(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """计算用户统计特征"""
        stats = ratings.groupby('userId').agg(
            user_rating_mean=('rating', 'mean'),
            user_rating_std=('rating', 'std'),
            user_rating_count=('rating', 'count'),
        ).reset_index()
        # 填充标准差为0的情况
        stats['user_rating_std'] = stats['user_rating_std'].fillna(0)
        return stats
    
    def split_data(self, ratings: pd.DataFrame) -> dict:
        """分割数据集"""
        ratings = ratings.drop('timestamp', axis=1)
        
        train_val, test = train_test_split(
            ratings, test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        val_size = self.config.val_size / (1 - self.config.test_size)
        train, val = train_test_split(
            train_val, test_size=val_size, 
            random_state=self.config.random_state
        )
        
        print(f"📈 数据集划分:")
        print(f"  训练集: {len(train):,}条")
        print(f"  验证集: {len(val):,}条")
        print(f"  测试集: {len(test):,}条")
        
        return {'train': train, 'val': val, 'test': test}
    
    def create_dataloaders(self, data_dict: dict, batch_size: int, n_movies: int, 
                          genre_df: pd.DataFrame = None, user_stats_df: pd.DataFrame = None,
                          model_config=None) -> dict:
        """创建DataLoader（含特征拼接）"""
        import os
        num_workers = 0 if os.name == 'nt' else 2
        dataloaders = {}
        
        for split, df in data_dict.items():
            # 拼接 genres 特征
            genres = None
            if genre_df is not None and model_config and model_config.use_genres:
                genres = df['movieId'].map(
                    lambda x: genre_df.loc[x].values if x in genre_df.index else np.zeros(len(ALL_GENRES))
                )
                genres = np.stack(genres.values)
            
            # 拼接用户统计特征
            user_stats = None
            if user_stats_df is not None and model_config and model_config.use_user_stats:
                stats_cols = ['user_rating_mean', 'user_rating_std', 'user_rating_count']
                user_stats = df['userId'].map(
                    lambda x: user_stats_df[user_stats_df['userId'] == x][stats_cols].values[0]
                    if x in user_stats_df['userId'].values else [3.5, 1.0, 0]
                )
                user_stats = np.stack(user_stats.values)
            
            is_train = (split == 'train')
            dataset = MovieRatingDataset(
                df['user'], df['movie'], df['rating'],
                genres=genres, user_stats=user_stats,
                is_train=False,  # 负采样只在 BPR 训练时启用
                n_movies=None,
                n_genres=len(ALL_GENRES),
                n_user_stats=3
            )
            dataloaders[split] = DataLoader(
                dataset, batch_size=batch_size, 
                shuffle=is_train, num_workers=num_workers, pin_memory=True
            )
        
        return dataloaders
    
    def save_encoders(self, path: str = "models"):
        """保存编码器"""
        Path(path).mkdir(exist_ok=True)
        with open(f"{path}/user_encoder.pkl", "wb") as f:
            pickle.dump(self.user_encoder, f)
        with open(f"{path}/movie_encoder.pkl", "wb") as f:
            pickle.dump(self.movie_encoder, f)
        print(f"💾 编码器已保存至 {path}/")
    
    def get_stats(self, ratings: pd.DataFrame) -> dict:
        """获取数据统计"""
        return {
            'n_users': ratings['user'].nunique(),
            'n_movies': ratings['movie'].nunique(),
            'n_ratings': len(ratings),
            'avg_rating': ratings['rating'].mean(),
            'sparsity': 1 - len(ratings) / (ratings['user'].nunique() * ratings['movie'].nunique())
        }


def prepare_data(config) -> tuple:
    """完整数据准备流程（含特征工程）"""
    preprocessor = DataPreprocessor(config.data)
    
    # 加载
    movies, ratings = preprocessor.load_data()
    
    # 筛选
    ratings = preprocessor.filter_data(ratings)
    
    # 编码电影类型（multi-hot）
    genre_df = preprocessor.encode_genres(movies)
    print(f"🎬 电影类型编码完成: {len(ALL_GENRES)} 种类型")
    
    # 计算用户统计特征
    user_stats_df = preprocessor.compute_user_stats(ratings)
    print(f"👤 用户统计特征计算完成: 均值/标准差/数量")
    
    # 编码用户和电影ID
    ratings = preprocessor.encode_ids(ratings)
    
    # 分割
    data_dict = preprocessor.split_data(ratings)
    
    # 获取电影总数（用于负采样）
    n_movies = preprocessor.movie_encoder.classes_.shape[0]
    
    # 创建loader（含特征拼接）
    loaders = preprocessor.create_dataloaders(
        data_dict, config.training.batch_size, n_movies,
        genre_df=genre_df, user_stats_df=user_stats_df,
        model_config=config.model
    )
    
    # 保存编码器
    preprocessor.save_encoders()
    
    # 统计信息
    stats = preprocessor.get_stats(ratings)
    stats['n_movies'] = n_movies
    
    # 传递特征信息到主函数
    extra_info = {
        'genre_df': genre_df,
        'user_stats_df': user_stats_df,
        'n_genres': len(ALL_GENRES),
        'n_user_stats': 3,  # 均值、标准差、数量
    }
    
    return movies, ratings, loaders, stats, preprocessor, extra_info