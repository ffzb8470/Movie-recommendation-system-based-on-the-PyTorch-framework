# 🎬 基于 PyTorch 框架的电影推荐系统

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)

> 基于神经协同过滤（NCF）模型的电影推荐系统，实现端到端的用户-电影评分预测与个性化推荐。

---

## 📋 技术栈

| 层级 | 技术 |
|------|------|
| **深度学习框架** | PyTorch 2.0+ |
| **数据处理** | Pandas、NumPy |
| **模型评估** | Scikit-learn（MSE、RMSE、MAE、R²） |
| **可视化** | Matplotlib、Seaborn |

## 🏗️ 项目结构

```
movie_project/
├── main.py                          # 入口：训练/评估/推荐
├── requirements.txt                 # 项目依赖
├── src/
│   ├── config.py                    # 配置类（数据/模型/训练）
│   ├── data_preprocessing.py        # 数据预处理
│   ├── model.py                     # NCF 模型定义
│   ├── trainer.py                   # 训练器
│   ├── evaluator.py                 # 评估器
│   ├── inference.py                 # 推理引擎
│   └── utils.py                     # 工具函数
└── data/
    └── ml-latest-small/             # MovieLens 数据集
```

## 🎯 功能特性

- **神经协同过滤（NCF）**：融合 GMF + MLP 的推荐模型
- **端到端流水线**：数据预处理 → 模型训练 → 评估 → 推理
- **早停机制**：基于验证集 loss 的早停与学习率调度
- **个性化推荐**：支持批量推荐与已观看电影过滤
- **评估指标**：MSE、RMSE、MAE、R²

## 📊 模型效果

| 指标 | 结果 |
|------|------|
| **RMSE** | 0.82 |
| **MAE** | 0.65 |
| **R²** | 0.38 |

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型
python main.py --mode train

# 3. 评估模型
python main.py --mode evaluate

# 4. 生成推荐
python main.py --mode recommend --user_id 1 --top_k 10
```

## 📦 依赖清单

```
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.64.0
requests>=2.28.0
```

## 📄 开源协议

本项目采用 **MIT License** 开源协议。