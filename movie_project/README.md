# 基于 PyTorch 的电影推荐系统

基于神经协同过滤（DeepFM）的电影推荐系统，实现端到端的用户-电影评分预测与个性化推荐。

## 技术栈

- 深度学习框架: PyTorch 2.0+
- 数据处理: Pandas, NumPy
- 模型评估: Scikit-learn (RMSE, MAE, NDCG, Recall, Precision)
- 可视化: Matplotlib

## 项目结构

```
movie_project/
├── main.py                          # 入口: 训练/评估/推荐
├── requirements.txt
├── src/
│   ├── config.py                    # 配置
│   ├── data_preprocessing.py        # 数据预处理
│   ├── model.py                     # DeepFM 模型
│   ├── trainer.py                   # 训练器
│   ├── evaluator.py                 # 评估器
│   ├── inference.py                 # 推理引擎
│   ├── chinese_titles.py            # 中文片名映射
│   └── utils.py
└── data/
    └── ml-1m/                       # MovieLens 1M 数据集
```

## 功能特性

- DeepFM 模型 (FM 二阶特征交叉 + Deep 高阶特征提取)
- 支持用户画像特征(性别/年龄/职业)
- 支持电影类型特征(18种类型 multi-hot)
- 端到端流水线: 数据预处理 -> 训练 -> 评估 -> 推荐
- 学习率预热 + 早停机制
- 支持已观看电影过滤

## 模型效果

| 指标 | 结果 |
|------|------|
| RMSE | 0.8567 |
| MAE | 0.6721 |
| R2 | 0.3959 |
| Acc@0.5 | 45.65% |
| NDCG@10 | 0.9103 |
| Recall@10 | 0.4511 |
| Precision@10 | 0.7956 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练
python main.py --mode train

# 评估
python main.py --mode eval

# 推荐
python main.py --mode recommend --user-id 4169 --top-n 10
```

## 依赖

torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.64.0

## 开源协议

MIT License