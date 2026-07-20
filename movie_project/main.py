import argparse
import sys
import torch
import logging
import pandas as pd  # 新增：用于推荐时过滤已观看电影
from pathlib import Path
from src.config import Config, DataConfig, ModelConfig, TrainingConfig

# 注册安全全局变量
torch.serialization.add_safe_globals([Config, DataConfig, ModelConfig, TrainingConfig])

from src.data_preprocessing import prepare_data
from src.model import NCF
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.inference import InferenceEngine
from src.utils import set_seed


def setup_logger():
    logger = logging.getLogger("recommendation")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    #  添加控制台处理器（修复日志不输出问题）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Movie Recommendation System")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "eval", "recommend"],
                       help="运行模式: train/eval/recommend")
    parser.add_argument("--user-id", type=int, help="推荐模式下的用户ID")
    parser.add_argument("--top-n", type=int, default=10, help="推荐数量")
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "bpr"], help="损失函数类型")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()
    set_seed()
    
    # 初始化配置（完整的Config总对象）
    config = Config()
    config.training.loss_type = args.loss_type
    
    # 模式选择
    if args.mode == "train":
        train(config, logger)
    elif args.mode == "eval":
        evaluate(config, logger)
    elif args.mode == "recommend":
        if args.user_id is None:
            logger.error("推荐模式需要提供 --user-id")
            sys.exit(1)
        recommend(config, logger, args.user_id, args.top_n)

# -------------------------- 训练函数（核心修复） --------------------------
def train(config, logger):
    """训练模式"""
    logger.info("开始训练流程")
    
    # 数据准备（含特征工程）
    movies, ratings, loaders, stats, preprocessor, extra_info = prepare_data(config)
    logger.info(f"原始数据 - 电影: {stats['n_movies']}部, 评分: {len(ratings)}条")
    logger.info(f"筛选后 - 用户: {stats['n_users']}人, 电影: {stats['n_movies']}部, 评分: {len(ratings)}条")
    logger.info(f"数据集划分: 训练集: {len(loaders['train'].dataset)}条, 验证集: {len(loaders['val'].dataset)}条, 测试集: {len(loaders['test'].dataset)}条")
    
    # 模型初始化（DeepFM）
    model = NCF(
        n_users=stats['n_users'],
        n_movies=stats['n_movies'],
        embedding_dim=config.model.embedding_dim,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        n_genres=extra_info['n_genres'],
        n_user_stats=extra_info['n_user_stats'],
    )
    
    logger.info(f"模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练（Trainer会自动保存模型，无需手动保存）
    trainer = Trainer(model, loaders['train'], loaders['val'], config.training, n_movies=stats['n_movies'])
    history = trainer.train()
    
    # 绘制训练曲线
    trainer.plot_history()
    logger.info(f"训练完成，最佳验证损失: {history['best_val_loss']:.4f}")
    
    # 测试集评估
    logger.info("开始测试集评估")
    evaluator = Evaluator(model, loaders['test'], config.training.device)
    metrics = evaluator.evaluate()
    
    logger.info(f"测试集 RMSE: {metrics['RMSE']:.4f}")
    if 'MAE' in metrics:
        logger.info(f"测试集 MAE: {metrics['MAE']:.4f}")

# -------------------------- 评估函数（优化） --------------------------
def evaluate(config, logger):
    """评估模式"""
    logger.info("开始评估流程")
    
    # 加载模型路径
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        model_path = Path("models/latest_model.pth")
        if not model_path.exists():
            logger.error("未找到训练好的模型，请先运行训练模式")
            sys.exit(1)
    logger.info(f"加载模型: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(
        model_path,
        map_location=config.training.device,
        weights_only=True
    )
    
    # 验证checkpoint必要字段（完整验证）
    required_keys = ['model_state_dict', 'config', 'best_val_loss']
    for key in required_keys:
        if key not in checkpoint:
            logger.error(f"Checkpoint缺少必要字段: {key}")
            sys.exit(1)
    
    # 优先使用训练时的配置（确保数据预处理一致）
    train_config = checkpoint['config'] if isinstance(checkpoint['config'], Config) else config
    
    # 数据准备
    movies, ratings, loaders, stats, _, extra_info = prepare_data(train_config)
    logger.info(f"数据加载完成 - 用户: {stats['n_users']}人, 电影: {stats['n_movies']}部, 测试集: {len(loaders['test'].dataset)}条")
    
    # 提取模型配置（兼容Trainer保存的config结构）
    if hasattr(train_config, 'model'):
        model_config = train_config.model.__dict__
    elif 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        model_config = config.model.__dict__
    
    # 模型初始化
    model = NCF(
        n_users=stats['n_users'],
        n_movies=stats['n_movies'],
        **model_config
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.training.device)
    logger.info(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 评估
    evaluator = Evaluator(model, loaders['test'], config.training.device)
    metrics = evaluator.evaluate()
    
    logger.info("评估完成")
    logger.info(f"最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    logger.info(f"测试集 RMSE: {metrics['RMSE']:.4f}")
    if 'MAE' in metrics:
        logger.info(f"测试集 MAE: {metrics['MAE']:.4f}")

# -------------------------- 推荐函数（优化：过滤已观看电影） --------------------------
def recommend(config, logger, user_id: int, top_n: int):
    """推荐模式"""
    logger.info(f"为用户 {user_id} 生成Top-{top_n}推荐")
    
    # 加载模型路径
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        model_path = Path("models/latest_model.pth")
        if not model_path.exists():
            logger.error("未找到训练好的模型，请先运行训练模式")
            sys.exit(1)
    
    # 加载评分数据（过滤已观看电影）
    ratings_path = Path(config.data.dataset_path) / "ratings.csv"
    watched_movie_ids = []
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)
        watched_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].tolist()
        logger.info(f"用户 {user_id} 已观看 {len(watched_movie_ids)} 部电影")
    
    # 推理引擎
    try:
        engine = InferenceEngine(
            model_path=str(model_path),
            movies_path=f"{config.data.dataset_path}/movies.csv",
        )
        
        # 生成推荐（传入已观看电影ID进行过滤）
        recs = engine.recommend(user_id, top_n, watched_movie_ids)
        
        # 美化输出
        print("\n" + "="*60)
        print(f"用户 {user_id} 的Top-{top_n}电影推荐")
        print("="*60)
        
        for i, rec in enumerate(recs, 1):
            print(f"{i:2d}. {rec['title']:40s}")
            print(f"    类型: {rec.get('genres', '未知')}")
            print(f"    预测评分: {rec['predicted_rating']:.2f}")
            print()
            
    except ValueError as e:
        logger.error(f"推荐失败: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"推荐过程异常: {str(e)}")
        sys.exit(1)

# -------------------------- 入口 --------------------------
if __name__ == "__main__":
    main()