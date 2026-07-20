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

## 推荐示例

### 示例1: 用户10 (喜欢歌舞/爱情/奇幻片, 不喜欢犯罪/黑色电影)

| 排名 | 电影 | 类型 | 预测评分 |
|------|------|------|---------|
| 1 | Jumpin' Jack Flash (1986) | Action, Comedy, Romance, Thriller | 4.16 |
| 2 | Swing Kids (1993) | Drama, War | 3.99 |
| 3 | Where the Heart Is (2000) | Comedy, Drama | 3.95 |
| 4 | Home Alone 3 (1997) | Children's, Comedy | 3.95 |
| 5 | Cutthroat Island (1995) | Action, Adventure, Romance | 3.94 |
| 6 | Armageddon (1998) | Action, Adventure, Sci-Fi, Thriller | 3.91 |
| 7 | The Other Sister (1999) | Comedy, Drama, Romance | 3.90 |
| 8 | Dracula: Dead and Loving It (1995) | Comedy, Horror | 3.89 |
| 9 | The Three Musketeers (1993) | Action, Adventure, Comedy | 3.87 |
| 10 | The Spitfire Grill (1996) | Drama | 3.86 |

### 示例2: 用户300 (喜欢黑色电影/战争/恐怖片, 不喜欢科幻/奇幻片)

| 排名 | 电影 | 类型 | 预测评分 |
|------|------|------|---------|
| 1 | Star Wars: Episode IV - A New Hope (1977) | Action, Adventure, Fantasy, Sci-Fi | 3.73 |
| 2 | The Matrix (1999) | Action, Sci-Fi, Thriller | 3.70 |
| 3 | A Close Shave (1995) | Animation, Comedy, Thriller | 3.49 |
| 4 | The Wrong Trousers (1993) | Animation, Comedy | 3.46 |
| 5 | The Green Mile (1999) | Drama, Thriller | 3.44 |
| 6 | The Usual Suspects (1995) | Crime, Thriller | 3.41 |
| 7 | Forrest Gump (1994) | Comedy, Romance, War | 3.40 |
| 8 | Men in Black (1997) | Action, Adventure, Comedy, Sci-Fi | 3.32 |
| 9 | Young Frankenstein (1974) | Comedy, Horror | 3.32 |
| 10 | Schindler's List (1993) | Drama, War | 3.32 |

两个用户的推荐结果体现了模型对用户画像的区分能力: 用户10偏好浪漫喜剧和剧情片, 用户300偏向动作科幻和犯罪惊悚片, 差异明显。

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