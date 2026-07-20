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

### 示例1: 年轻女性用户 (userId=18, 年龄18岁, 历史均分3.65)

| 排名 | 电影 | 类型 | 预测评分 |
|------|------|------|---------|
| 1 | The Godfather (1972) | Action, Crime, Drama | 4.14 |
| 2 | The Shawshank Redemption (1994) | Drama | 4.06 |
| 3 | A Close Shave (1995) | Animation, Comedy, Thriller | 3.95 |
| 4 | The Wrong Trousers (1993) | Animation, Comedy | 3.90 |
| 5 | The Matrix (1999) | Action, Sci-Fi, Thriller | 3.79 |
| 6 | Monty Python and the Holy Grail (1974) | Comedy | 3.75 |
| 7 | Wallace & Gromit: Aardman Animation (1996) | Animation | 3.73 |
| 8 | Rear Window (1954) | Mystery, Thriller | 3.68 |
| 9 | The Palm Beach Story (1942) | Comedy | 3.65 |
| 10 | North by Northwest (1959) | Drama, Thriller | 3.65 |

### 示例2: 年长男性用户 (userId=2, 年龄56岁, 历史均分3.71)

| 排名 | 电影 | 类型 | 预测评分 |
|------|------|------|---------|
| 1 | The Sixth Sense (1999) | Thriller | 3.68 |
| 2 | A Close Shave (1995) | Animation, Comedy, Thriller | 3.47 |
| 3 | Star Wars: Episode IV - A New Hope (1977) | Action, Adventure, Fantasy, Sci-Fi | 3.44 |
| 4 | Toy Story (1995) | Animation, Children's, Comedy | 3.38 |
| 5 | The Princess Bride (1987) | Action, Adventure, Comedy, Romance | 3.38 |
| 6 | The Wrong Trousers (1993) | Animation, Comedy | 3.37 |
| 7 | It's a Wonderful Life (1946) | Drama | 3.32 |
| 8 | Toy Story 2 (1999) | Animation, Children's, Comedy | 3.32 |
| 9 | Life Is Beautiful (1997) | Comedy, Drama | 3.31 |
| 10 | Indiana Jones and the Last Crusade (1989) | Action, Adventure | 3.31 |

两个用户的推荐结果差异体现了模型对用户画像的区分能力: 年轻女性用户偏向经典犯罪剧情片, 年长男性用户偏向合家欢和科幻冒险片。预测评分接近用户历史均分, 符合实际评分习惯。

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