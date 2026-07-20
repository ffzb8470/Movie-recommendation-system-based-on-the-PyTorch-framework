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

### 示例1: 年轻女性用户 (userId=18, 年龄18岁)

| 排名 | 电影 | 类型 | 预测评分 |
|------|------|------|---------|
| 1 | General, The (1927) | Comedy | 2.56 |
| 2 | 美丽人生 | Comedy, Drama | 2.26 |
| 3 | 生活多美好 | Drama | 2.24 |
| 4 | Palm Beach Story, The (1942) | Comedy | 2.21 |
| 5 | Yojimbo (1961) | Comedy, Drama, Western | 2.20 |
| 6 | Wallace & Gromit: Aardman Animation | Animation | 2.15 |
| 7 | It Happened One Night (1934) | Comedy | 2.13 |
| 8 | City Lights (1931) | Comedy, Drama, Romance | 2.11 |
| 9 | 人鬼情未了 | Comedy, Horror | 2.08 |
| 10 | Gold Rush, The (1925) | Comedy | 2.06 |

### 示例2: 年长男性用户 (userId=2, 年龄56岁)

| 排名 | 电影 | 类型 | 预测评分 |
|------|------|------|---------|
| 1 | 生活多美好 | Drama | 1.68 |
| 2 | 玩具总动员 | Animation, Children's, Comedy | 1.67 |
| 3 | 星球大战4：新希望 | Action, Adventure, Fantasy, Sci-Fi | 1.60 |
| 4 | 第六感 | Thriller | 1.59 |
| 5 | 美女与野兽 | Animation, Children's, Musical | 1.54 |
| 6 | My Fair Lady (1964) | Musical, Romance | 1.50 |
| 7 | 欢乐满人间 | Children's, Comedy, Musical | 1.49 |
| 8 | 美丽人生 | Comedy, Drama | 1.47 |
| 9 | 音乐之声 | Musical | 1.47 |
| 10 | E.T.外星人 | Children's, Drama, Fantasy, Sci-Fi | 1.42 |

两个用户的推荐结果差异体现了模型对用户画像的区分能力: 年轻女性用户偏向经典喜剧和剧情片, 年长男性用户偏向合家欢和科幻冒险片。评分范围在1-5分之间, 输出值受用户评分习惯影响。

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