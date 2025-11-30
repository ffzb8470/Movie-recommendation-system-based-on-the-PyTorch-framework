import logging
import torch
import random
import numpy as np
from pathlib import Path

def setup_logger(name: str = "recommendation"):
    """设置日志"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
    
        logger.addHandler(ch)
        
        # 文件处理器
        Path("logs").mkdir(exist_ok=True)
        fh = logging.FileHandler("logs/training.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)