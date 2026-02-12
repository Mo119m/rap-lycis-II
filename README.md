# rap-lycis-II

面向**计算语言学（Computational Linguistics）**与**数字人文（Digital Humanities）**研究者的中文说唱命名实体研究项目。

## 研究目标

本项目聚焦两个层面：

1. **方法论层面**：构建可复用的中文说唱歌词 NER（Named Entity Recognition）流水线，缓解正式文本模型在口语/俚语域上的失配问题。
2. **文化分析层面**：通过实体分布与聚类结果，观察中文说唱歌手的身份表达、地理想象和文化符号使用。

## 当前数据

- 语料文件：`lyrics_chunks_enriched.csv`
- 核心字段：`artist`, `song_id`, `song_title`, `chunk_id`, `text`
- 覆盖 250+ 艺人，约 300 万行歌词（按你的 proposal 描述）

## 项目结构

- `src/rap_ner_pipeline.py`：端到端实验脚本（NER → bag-of-entities → k-means）
- `configs/rap_lexicon_seed.jsonl`：可扩展的种子词典（给 `EntityRuler` 用）
- `ner.ipynb`：早期探索 notebook
- `outputs/`：运行脚本后的输出目录

## 快速开始

### 1) 安装依赖

建议 Python 3.10+。

```bash
pip install pandas spacy scikit-learn
# 可选（推荐）: 下载中文模型
python -m spacy download zh_core_web_lg
```

> 如果未安装 `zh_core_web_lg`，脚本会自动回退到 `spacy.blank("zh")` + 词典规则。

### 2) 运行流水线

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

## 输出文件说明

- `outputs/artist_lyrics.csv`：按艺人聚合后的长文本
- `outputs/entities_long.csv`：长表格式实体结果（artist/entity/label）
- `outputs/bag_of_entities.csv`：艺人 × 实体频次矩阵
- `outputs/artist_clusters.csv`：艺人聚类归属
- `outputs/cluster_entity_summary.csv`：每个聚类的代表实体（按质心权重排序）

## 建议的研究工作流（下一步）

1. **词典扩充**：先构建 `ARTIST/CITY/BRAND/PLATFORM/CULTURE` 五大类，逐步增加细粒度类型（如 `CREW`, `SLANG`, `LUXURY`, `PLACE_LOCAL`）。
2. **误差分析**：对每类实体抽样 100 条，标注 precision/recall，记录常见误识别模式。
3. **聚类解释**：结合歌手背景（地域/年代/厂牌）进行 cluster labeling。
4. **比较研究**：
   - 主流 vs 地下说唱
   - 不同地域（川渝/华北/华东）
   - 时间切片（2017 前后）
5. **论文写作**：方法（pipeline + bias）、结果（实体生态）、讨论（身份构建与数据偏差）分层展开。

## 你可以马上做的三件事

1. 给我 30~100 个你最关心的实体词（艺人名、地名、品牌、黑话）加入 `configs/rap_lexicon_seed.jsonl`。
2. 选一个子语料（如 20 位头部艺人）先跑一轮，检查 `cluster_entity_summary.csv` 是否可解释。
3. 在 `ner.ipynb` 里建立一个误差分析表格，记录 false positive / false negative 示例。

---

如果你愿意，我们下一步可以直接一起做：
- **V1 标注规范（annotation guideline）**
- **小规模 gold set 构建**
- **实体类型体系迭代**
- **论文方法部分初稿**
