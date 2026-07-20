import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class MovieRatingDataset(Dataset):
    """PyTorch数据集"""
    def __init__(self, users: pd.Series, movies: pd.Series, ratings: pd.Series):
        self.users = users.values
        self.movies = movies.values
        self.ratings = ratings.values
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'movie': torch.tensor(self.movies[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float32)
        }

class DataPreprocessor:
    """数据预处理器"""
    def __init__(self, config):
        self.config = config
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
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
        # 筛选电影
        movie_counts = ratings['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= self.config.min_movie_ratings].index
        ratings = ratings[ratings['movieId'].isin(valid_movies)]
        
        # 筛选用户
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
    
    def split_data(self, ratings: pd.DataFrame) -> dict:
        """分割数据集"""
        # 删除时间戳列
        ratings = ratings.drop('timestamp', axis=1)
        
        # 分割
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
    
    def create_dataloaders(self, data_dict: dict, batch_size: int) -> dict:
        """创建DataLoader"""
        import os
        dataloaders = {}
        # Windows 下 num_workers>0 可能引发多进程问题，自动适配
        num_workers = 0 if os.name == 'nt' else 2
        
        for split, df in data_dict.items():
            dataset = MovieRatingDataset(df['user'], df['movie'], df['rating'])
            shuffle = (split == 'train')
            dataloaders[split] = DataLoader(
                dataset, batch_size=batch_size, 
                shuffle=shuffle, num_workers=num_workers, pin_memory=True
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
    """完整数据准备流程"""
    preprocessor = DataPreprocessor(config.data)
    
    # 加载
    movies, ratings = preprocessor.load_data()
    
    # 筛选
    ratings = preprocessor.filter_data(ratings)
    
    # 编码
    ratings = preprocessor.encode_ids(ratings)
    
    # 分割
    data_dict = preprocessor.split_data(ratings)
    
    # 创建loader
    loaders = preprocessor.create_dataloaders(data_dict, config.training.batch_size)
    
    # 保存编码器
    preprocessor.save_encoders()
    
    # 统计信息
    stats = preprocessor.get_stats(ratings)
    
    return movies, ratings, loaders, stats, preprocessor