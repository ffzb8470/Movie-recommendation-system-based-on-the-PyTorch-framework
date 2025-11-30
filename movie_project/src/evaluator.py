import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from pathlib import Path

class Evaluator:
    """评估器"""
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
    
    def compute_metrics(self) -> dict:
        """计算评估指标"""
        self.model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                user = batch['user'].to(self.device, non_blocking=True)
                movie = batch['movie'].to(self.device, non_blocking=True)
                rating = batch['rating'].to(self.device, non_blocking=True)
                
                predictions = self.model(user, movie)
                
                all_preds.extend(predictions.cpu().numpy())
                all_true.extend(rating.cpu().numpy())
        
        # 计算指标
        mse = mean_squared_error(all_true, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_true, all_preds)
        r2 = r2_score(all_true, all_preds)
        
        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'predictions': all_preds,
            'ground_truth': all_true
        }
    
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