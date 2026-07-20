import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from pathlib import Path
from collections import defaultdict

class Evaluator:
    """评估器（含排序指标）"""
    def __init__(self, model, test_loader, device, n_movies=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.n_movies = n_movies
    
    def ndcg_at_k(self, scores, k):
        """计算 NDCG@K"""
        scores = np.array(scores)
        # 理想排序（降序）
        ideal = np.sort(scores)[::-1]
        if ideal.sum() == 0:
            return 0.0
        dcg = np.sum((2**scores[:k] - 1) / np.log2(np.arange(2, k + 2)))
        idcg = np.sum((2**ideal[:k] - 1) / np.log2(np.arange(2, k + 2)))
        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_ranking_metrics(self, k=10):
        """计算排序指标：NDCG@K, Recall@K, Precision@K"""
        self.model.eval()
        user_preds = defaultdict(list)
        user_true = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.test_loader:
                user = batch['user'].to(self.device, non_blocking=True)
                movie = batch['movie'].to(self.device, non_blocking=True)
                rating = batch['rating'].to(self.device, non_blocking=True)
                genres = batch.get('genres', None)
                if genres is not None:
                    genres = genres.to(self.device, non_blocking=True)
                user_stats = batch.get('user_stats', None)
                if user_stats is not None:
                    user_stats = user_stats.to(self.device, non_blocking=True)
                
                kwargs = {}
                if genres is not None:
                    kwargs['genres'] = genres
                if user_stats is not None:
                    kwargs['user_stats'] = user_stats
                
                predictions = self.model(user, movie, **kwargs)
                
                for u, p, r in zip(user.cpu().numpy(), predictions.cpu().numpy(), rating.cpu().numpy()):
                    user_preds[int(u)].append(float(p))
                    user_true[int(u)].append(float(r))
        
        ndcg_list, recall_list, precision_list = [], [], []
        
        for uid in user_preds:
            preds = np.array(user_preds[uid])
            true = np.array(user_true[uid])
            if len(preds) < k:
                continue
            
            # 按预测评分排序
            top_k_idx = np.argsort(preds)[-k:][::-1]
            top_k_true = true[top_k_idx]
            
            # NDCG@K
            ndcg = self.ndcg_at_k(top_k_true, k)
            ndcg_list.append(ndcg)
            
            # Precision@K: 预测为高分的项目中，实际高分的比例
            threshold = 4.0  # 评分>=4视为"喜欢"
            relevant = (top_k_true >= threshold).sum()
            precision_list.append(relevant / k)
            
            # Recall@K: 实际高分的项目中，被预测到的比例
            total_relevant = (true >= threshold).sum()
            recall_list.append(relevant / total_relevant if total_relevant > 0 else 0.0)
        
        return {
            'NDCG@10': np.mean(ndcg_list) if ndcg_list else 0.0,
            'Recall@10': np.mean(recall_list) if recall_list else 0.0,
            'Precision@10': np.mean(precision_list) if precision_list else 0.0,
        }
    
    def compute_metrics(self) -> dict:
        """计算回归指标（MSE, RMSE, MAE, R²）"""
        self.model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                user = batch['user'].to(self.device, non_blocking=True)
                movie = batch['movie'].to(self.device, non_blocking=True)
                rating = batch['rating'].to(self.device, non_blocking=True)
                genres = batch.get('genres', None)
                if genres is not None:
                    genres = genres.to(self.device, non_blocking=True)
                user_stats = batch.get('user_stats', None)
                if user_stats is not None:
                    user_stats = user_stats.to(self.device, non_blocking=True)
                
                kwargs = {}
                if genres is not None:
                    kwargs['genres'] = genres
                if user_stats is not None:
                    kwargs['user_stats'] = user_stats
                
                predictions = self.model(user, movie, **kwargs)
                
                all_preds.extend(predictions.cpu().numpy())
                all_true.extend(rating.cpu().numpy())
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        mse = mean_squared_error(all_true, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_true, all_preds)
        r2 = r2_score(all_true, all_preds)
        
        # 约束准确率（预测在±0.5分内的比例）
        acc_05 = np.mean(np.abs(all_preds - all_true) < 0.5)
        acc_1 = np.mean(np.abs(all_preds - all_true) < 1.0)
        
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'Acc@0.5': float(acc_05),
            'Acc@1.0': float(acc_1),
            'predictions': all_preds.tolist(),
            'ground_truth': all_true.tolist()
        }
        
        # 添加排序指标
        ranking = self.compute_ranking_metrics(k=10)
        metrics.update(ranking)
        
        return metrics
    
    def evaluate(self, save_path: str = "evaluation_results.json"):
        """执行评估并保存结果"""
        print("\n📊 开始评估测试集...")
        metrics = self.compute_metrics()
        
        print("\n" + "="*40)
        print("测试集评估结果:")
        print("="*40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:>10}: {value:.4f}")
        
        # 保存结果
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump({k: v for k, v in metrics.items() if k not in ['predictions', 'ground_truth']}, 
                     f, indent=2)
        
        print(f"\n✅ 评估结果已保存至: {save_path}")
        return metrics