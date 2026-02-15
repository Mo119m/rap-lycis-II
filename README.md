# rap-lycis-II

面向**计算语言学（Computational Linguistics）**与**数字人文（Digital Humanities）**研究者的中文说唱命名实体研究项目。

## 研究目标

本项目聚焦两个层面：

1. **方法论层面**：构建可复用的中文说唱歌词 NER（Named Entity Recognition）流水线，缓解正式文本模型在口语/俚语域上的失配问题。
2. **文化分析层面**：通过实体分布与聚类结果，观察中文说唱歌手的身份表达、地理想象和文化符号使用。

## 当前数据

- 语料文件：`lyrics_chunks_enriched.csv`
- 核心字段：`artist`, `song_id`, `song_title`, `chunk_id`, `text`
- 覆盖 250+ 艺人，约 300 万行歌词

## 项目结构

```
├── main.ipynb                      # 主 notebook — 直接 Run All 查看完整结果
├── lyrics_chunks_enriched.csv      # 原始语料
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py            # 数据加载与清洗（过滤 Live/伴奏、剥离制作信息）
│   ├── ner.py                      # NER 实体识别（spaCy + EntityRuler）
│   ├── clustering.py               # Bag-of-Entities + K-Means 聚类
│   ├── io_utils.py                 # 输出保存工具
│   └── rap_ner_pipeline.py         # CLI 入口（调用上述模块）
├── configs/
│   └── rap_lexicon_seed.jsonl      # 种子词典（84 条：ARTIST/CITY/BRAND/CULTURE/CREW/PLATFORM）
├── ner.ipynb                       # 早期探索 notebook
└── outputs/                        # 运行后的输出目录
```

## 快速开始

### 1) 安装依赖

建议 Python 3.10+。

```bash
pip install pandas spacy scikit-learn
# 可选（推荐）: 下载中文模型
python -m spacy download zh_core_web_lg
```

> 如果未安装 `zh_core_web_lg`，脚本会自动回退到 `spacy.blank("zh")` + 词典规则。

### 2) 使用 Notebook（推荐）

打开 `main.ipynb`，直接 **Run All** 即可看到：
- 数据清洗报告（移除了多少 Live 版本、制作信息行等）
- NER 实体提取结果与分布
- K-Means 聚类结果与各聚类代表实体
- 质量检查（可疑实体、噪声分析）

### 3) 使用命令行

```bash
python src/rap_ner_pipeline.py \
  --input lyrics_chunks_enriched.csv \
  --lexicon configs/rap_lexicon_seed.jsonl \
  --output-dir outputs \
  --n-clusters 6 \
  --top-k 25
```

小样本调试（建议先跑）：

```bash
python src/rap_ner_pipeline.py --max-rows 5000 --n-clusters 4
```

## 数据清洗说明

数据清洗模块（`src/data_cleaning.py`）自动处理以下噪声：

| 类型 | 处理方式 | 影响范围 |
|------|---------|---------|
| Live 版本 | 按 song_title 过滤 | ~294 首歌, ~1500 行 |
| 伴奏/Instrumental | 按 song_title 过滤 | ~7 首歌 |
| 制作信息行（出品/Prod./混音/母带等） | 按文本行级过滤 | ~1000+ 行 |
| 结构标记（Verse/Hook/Chorus 等独立标签行） | 按文本行级过滤 | 仅移除独立标签行 |

## 输出文件说明

- `outputs/artist_lyrics.csv`：按艺人聚合后的长文本
- `outputs/entities_long.csv`：长表格式实体结果（artist/entity/label）
- `outputs/bag_of_entities.csv`：艺人 × 实体频次矩阵
- `outputs/artist_clusters.csv`：艺人聚类归属
- `outputs/cluster_entity_summary.csv`：每个聚类的代表实体（按质心权重排序）

## 建议的研究工作流（下一步）

1. **词典扩充**：继续增加 `ARTIST/CITY/BRAND/PLATFORM/CULTURE/CREW` 等类型的实体，逐步加入细粒度类型（如 `SLANG`, `LUXURY`, `PLACE_LOCAL`）。
2. **误差分析**：对每类实体抽样 100 条，标注 precision/recall，记录常见误识别模式。
3. **聚类解释**：结合歌手背景（地域/年代/厂牌）进行 cluster labeling。
4. **比较研究**：
   - 主流 vs 地下说唱
   - 不同地域（川渝/华北/华东）
   - 时间切片（2017 前后）
5. **论文写作**：方法（pipeline + bias）、结果（实体生态）、讨论（身份构建与数据偏差）分层展开。
