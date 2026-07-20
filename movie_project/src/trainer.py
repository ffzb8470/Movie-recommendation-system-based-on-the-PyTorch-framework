import torch
from torch import nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ReduceLROnPlateau

class Trainer:
    """训练管理器"""
    def __init__(self, model, train_loader, val_loader, config, n_movies=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.n_movies = n_movies  # BPR loss 负采样需要
        
        # 损失函数
        self.loss_type = config.loss_type
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度
        self.warmup_epochs = 5
        if self.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
            main_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )
            self.scheduler = main_scheduler
            self.use_warmup = True
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )
            self.use_warmup = False
        self.warmup_scheduler = warmup_scheduler if self.warmup_epochs > 0 else None
        
        # 设备
        self.device = config.device
        self.model.to(self.device)
        
        # 早停
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []
        
        # 模型保存路径
        self.save_dir = Path("models")
        self.save_dir.mkdir(exist_ok=True)
    
    def _bpr_loss(self, user, pos_movie, neg_movie):
        """BPR (Bayesian Personalized Ranking) 损失"""
        pos_score = self.model(user, pos_movie)
        neg_score = self.model(user, neg_movie)
        # BPR loss: -log(sigmoid(pos - neg))
        diff = pos_score - neg_score
        loss = -torch.log(torch.sigmoid(diff) + 1e-10)
        return loss.mean()
    
    def _sample_negatives(self, pos_movie):
        """为 BPR 随机采样负样本"""
        neg_movie = torch.randint(1, self.n_movies, pos_movie.shape, device=self.device)
        return neg_movie

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="训练", leave=False)
        for batch in pbar:
            user = batch['user'].to(self.device, non_blocking=True)
            movie = batch['movie'].to(self.device, non_blocking=True)
            rating = batch['rating'].to(self.device, non_blocking=True)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.loss_type == 'bpr' and self.n_movies:
                neg_movie = self._sample_negatives(movie)
                loss = self._bpr_loss(user, movie, neg_movie)
            else:
                predictions = self.model(user, movie)
                loss = self.criterion(predictions, rating)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(user)
            total_samples += len(user)
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / total_samples
    
    def validate(self) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                user = batch['user'].to(self.device, non_blocking=True)
                movie = batch['movie'].to(self.device, non_blocking=True)
                rating = batch['rating'].to(self.device, non_blocking=True)
                
                predictions = self.model(user, movie)
                loss = self.criterion(predictions, rating)
                
                total_loss += loss.item() * len(user)
                total_samples += len(user)
        
        return total_loss / total_samples
    
    def train(self) -> dict:
        """完整训练流程"""
        print(f"\n 开始训练: {self.config.epochs} epochs, 设备: {self.device}")
        print("="*60)
        
        for epoch in range(self.config.epochs):
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 更新学习率
            if self.use_warmup and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step(val_loss)
            
            # 打印日志
            print(f"Epoch [{epoch+1:02d}/{self.config.epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print("  → 最佳模型已保存")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.config.patience:
                    print(f"\n 早停触发！最佳验证损失: {self.best_val_loss:.4f}")
                    break
            
            # 保存最新模型
            self.save_checkpoint(epoch, is_best=False)
        
        print("\n 训练完成！")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pth")
        else:
            torch.save(checkpoint, self.save_dir / "latest_model.pth")
    
    def plot_history(self, save_path: str = "training_curve.png"):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()