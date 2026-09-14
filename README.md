# 🎓 Research Assistant - 科研论文智能助手

基于混合检索和多智能体的学术研究辅助系统，支持论文搜索、智能检索、实验设计和学术写作。

## ✨ 核心功能

| 功能                  | 说明                     | 技术方案                                            |
| --------------------- | ------------------------ | --------------------------------------------------- |
| **论文搜索**    | 从多个学术数据库获取论文 | arXiv, OpenAlex, Semantic Scholar, CORE, Dimensions |
| **混合检索**    | 精准语义检索             | BM25 + BGE-M3 + BGE-Reranker                        |
| **Skills**      | 按需加载的能力单元       | frontmatter 常驻 + 正文/references 渐进披露                    |
| **记忆与个性化** | 跨会话的用户偏好         | 可配置类别 + 时间衰减 + 负反馈 + BGE-M3 语义召回 + 偏好回流检索 |
| **实验设计**    | 自动生成实验方案         | LLM + 论文上下文                                    |
| **学术写作**    | 摘要/引言/方法生成       | DeepSeek/OpenAI API                                 |
| **LoRA微调**    | 学术写作风格优化         | Qwen2.5 + PEFT                                      |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ResearchAssistant                        │
├─────────────────────────────────────────────────────────────┤
│   skills/paper-retrieval    skills/experiment-design         │
│   skills/academic-writing   （按需加载, 渐进披露）             │
│         └───────────────┬───────────────┘                    │
│                         ↓                                    │
│              ┌───────────────────────┐                      │
│              │  SkillRegistry 选择    │                      │
│              │  确定性路由, 0 次 LLM   │                      │
│              └───────────┬───────────┘                      │
├──────────────────────────┼──────────────────────────────────┤
│  ┌───────────────────────▼───────────────────────┐         │
│  │           Hybrid Retrieval Engine             │         │
│  │  BM25 + BGE-M3 → RRF → BGE-Reranker          │         │
│  └───────────────────────────────────────────────┘         │
│  ┌───────────────────────────────────────────────┐         │
│  │  memory/  偏好(可配置类别/衰减/负反馈) + 语义召回 + 偏好回流   │         │
│  └───────────────────────────────────────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  数据源: arXiv | OpenAlex | Semantic Scholar | CORE        │
└─────────────────────────────────────────────────────────────┘
```

## 📦 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd research_assistant
```

### 2. 创建虚拟环境

```bash
conda create -n research_assistant python=3.10
conda activate research_assistant
```

### 3. 安装依赖

```bash
# 安装本包 (core 依赖,不含 GPU/torch)
pip install -e ".[core]"

# 需要 Multi-Agent / 微调时:
#   pip install -e ".[agent]"      # LangGraph 多智能体
#   pip install -e ".[finetune]"   # 混合检索(BGE/FAISS) + LoRA 微调

# GPU版本 (推荐)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

### 4. 配置API Key

创建 `.env` 文件：

```bash
# LLM API (二选一)
DEEPSEEK_API_KEY=your-deepseek-api-key
OPENAI_API_KEY=your-openai-api-key

# 可选: 论文数据库API
CORE_API_KEY=your-core-api-key
DIMENSIONS_API_KEY=your-dimensions-api-key
```

## 🚀 快速开始

### 基础使用

```python
from research_assistant import ResearchAssistant

# 初始化
assistant = ResearchAssistant()

# Step 1: 搜索论文 (从API获取，建立本地索引)
papers = assistant.search_papers("lithium battery RUL prediction", max_results=20)

# Step 2: 混合检索 (在本地索引中精确检索)
results = assistant.hybrid_search("SOH estimation deep learning", top_k=5)

# 查看结果
for r in results:
    print(f"[{r['score']:.3f}] {r['paper']['title'][:60]}...")
```

### 命令行使用

```bash
# 搜索论文
research-assistant --query "battery prediction"

# 完整工作流
research-assistant --query "battery prediction" --workflow
```

### 运行示例

```bash
# 论文检索
python examples/01_paper_retrieval.py

# 实验设计
python examples/02_experiment_design.py

# 学术写作
python examples/03_academic_writing.py

# 可复现检索评测 / 记忆召回 / 架构度量
python scripts/eval_retrieval.py --mode bm25
python scripts/memory_recall_demo.py
python scripts/measure_refactor.py
```

## 📖 检索流程

```
用户输入查询
      ↓
┌─────────────────────────────────────────┐
│  Step 1: search_papers()                │
│  → QueryEnhancer (中文→英文学术术语)     │
│  → 调用 arXiv/OpenAlex 等API            │
│  → 获取论文元数据和摘要                  │
│  → 自动添加到 HybridRetriever 索引       │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  Step 2: hybrid_search()                │
│  → BM25 稀疏检索 (关键词匹配)            │
│  → BGE-M3 稠密检索 (语义向量)            │
│  → RRF 融合排序                         │
│  → BGE-Reranker 精排                    │
│  → 返回最相关的文本块                    │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  Step 3: 后续处理 (可选)                 │
│  → interpret_paper() 论文解读            │
│  → design_experiment() 实验设计          │
│  → generate_abstract() 摘要生成          │
└─────────────────────────────────────────┘
```

## 🔧 核心模块

### 论文检索 (`modules/paper_retrieval/`)

| 文件                         | 功能                                                              |
| ---------------------------- | ----------------------------------------------------------------- |
| `paper_retriever.py`         | 多源API搜索 (arXiv, OpenAlex, Semantic Scholar, CORE, Dimensions) |
| `hybrid_retriever.py`        | BM25 + BGE-M3 + BGE-Reranker 混合检索                             |
| `pdf_markdown_processor.py`  | PDF解析、清洗、切分、入库                                         |
| `paper_interpreter.py`       | LLM论文解读                                                       |
| `evaluation.py`              | RAGAS检索评测                                                     |

#### PDF 处理流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 解析 (Parse)                                             │
│  ─────────────────                                          │
│  • PyMuPDF/pymupdf4llm 读取 PDF → Markdown                  │
│  • pdfplumber 提取表格 → Markdown 表格                       │
│  • 正则匹配 LaTeX 公式                                       │
│  • PyMuPDF 提取图片 + AI 描述 (可选)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 清洗 (Clean)                                             │
│  ─────────────────                                          │
│  • 去除页码、页眉页脚                                        │
│  • 去除 arXiv 标识、DOI、版权信息                            │
│  • 合并连续空行                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 切分 (Split)                                             │
│  ─────────────────                                          │
│  • MarkdownHeaderTextSplitter 按 #/##/### 章节切分          │
│  • RecursiveCharacterTextSplitter 处理超长章节              │
│  • 保留章节层级元数据                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 入库 (Index)                                             │
│  ─────────────────                                          │
│  • 文本块/表格/公式/图片 → chunks 表 (SQLite)                │
│  • BGE-M3 向量化 → FAISS HNSW 索引                           │
│  • 分词 → BM25 稀疏索引                                      │
└─────────────────────────────────────────────────────────────┘
```

#### 三层存储架构

```
data/
├── index/              # 索引层
│   ├── faiss_hnsw.index   # FAISS HNSW 向量索引
│   └── bm25.pkl           # BM25 稀疏索引
├── metadata/           # 元数据层
│   └── papers.db          # SQLite (papers表 + chunks表)
└── cache/              # 缓存层
    └── *.pdf              # 下载的PDF文件
```

### Multi-Agent (`modules/agents/`)

| 文件               | 功能                                       |
| ------------------ | ------------------------------------------ |
| `multi_agent.py` | LangGraph 状态图（**可选后端**）；路由已由 `SkillRegistry` 确定性完成 |
| `tools.py`       | Agent工具集 (Python执行、统计分析、可视化) |

> 架构说明：原设计的 supervisor LLM 路由已移除，改为 skill 的 frontmatter 索引 +
  确定性选择（0 次 LLM 调用）。LangGraph 保留用于需要并行/长流程的场景。
  度量见 `scripts/measure_refactor.py`。

### 实验与写作

| 模块                   | 功能                                  |
| ---------------------- | ------------------------------------- |
| `experiment_agent/`  | 实验方案自动设计                      |
| `writing_assistant/` | 学术写作辅助 (摘要、引言、方法、结果) |
| `training/`          | LoRA微调训练                          |

## ⚙️ 配置说明

配置文件: `config/config.yaml`

```yaml
# LLM配置
llm:
  model: "deepseek-chat"
  temperature: 0.7
  max_tokens: 4096

# 论文检索源
paper_retrieval:
  sources:
    - arxiv
    - semantic_scholar
    - openalex
  max_results: 20

# 混合检索配置
hybrid_retrieval:
  bge_model: "BAAI/bge-m3"
  reranker_model: "BAAI/bge-reranker-v2-m3"
  bm25_weight: 0.3
  dense_weight: 0.7
  use_rerank: true
```

## 📊 性能指标

> ⚠️ **本表只收录可由仓库内脚本复现的数字。** 复现：
> `python scripts/eval_retrieval.py --mode bm25` 与 `python scripts/measure_refactor.py`。

| 模块 | 指标 | 数值 | 复现命令 |
| --- | --- | --- | --- |
| 检索 (BM25 基线) | Precision@5 | **55.0%** | `--mode bm25` |
| 检索 | Recall@5 / Hit@5 / MRR | 87.5% / 100.0% / 0.688 | `--mode bm25` |
| 记忆召回 | 中文 query 语义相似度 | cos ∈ [0.41, 0.91]（旧实现词重叠恒为 0） | `scripts/memory_recall_demo.py` |
| 架构 | 常驻上下文 | **851 → 242 token（↓71.6%）** | `scripts/measure_refactor.py` |
| 架构 | 路由 LLM 调用 | **1 → 0** | `scripts/measure_refactor.py` |
| 学术写作 (LoRA) | ROUGE-L / 术语准确率 | 待补测 | — |

### ⚠️ 评测集规模警告

`data/eval/retrieval_eval.json` 仅含 **4 条 query**，且 q3/q4 各只标注 1 篇相关文献，
因此 **Precision@5 的理论上限为 60%**（(1.0+1.0+0.2+0.2)/4）。

任何高于 60% 的 P@5 都**不可能**由当前评测集产生。扩标要求见
`skills/paper-retrieval/references/evaluation.md`（≥50 条 query、每条 ≥5 篇标注）。

原表曾列 "Precision@5 89% / 任务完成率 85% / ROUGE-L 0.47"，
这些数字在仓库内**无脚本可复现**，已移除。

## 🛠️ 技术栈

| 类别       | 技术                                                |
| ---------- | --------------------------------------------------- |
| LLM        | DeepSeek, OpenAI, Qwen2.5                           |
| Embedding  | BGE-M3, sentence-transformers                       |
| 向量数据库 | FAISS (HNSW)                                        |
| 稀疏检索   | BM25 (rank-bm25)                                    |
| PDF处理    | PyMuPDF, pymupdf4llm, pdfplumber                    |
| 文本切分   | LangChain MarkdownHeaderTextSplitter                |
| Agent框架  | LangGraph, LangChain                                |
| 微调       | PEFT LoRA, TRL                                      |
| 论文API    | arXiv, OpenAlex, Semantic Scholar, CORE, Dimensions |

## 📁 项目结构

```
research-assistant/
├── pyproject.toml               # 打包配置 (core/agent/finetune extras)
├── research_assistant/          # 可安装包
│   ├── __init__.py              # 导出 ResearchAssistant
│   ├── assistant.py             # ResearchAssistant 门面类
│   ├── cli.py                   # 命令行入口
│   ├── config/
│   │   ├── config.yaml          # 系统配置
│   │   └── prompts.yaml         # Prompt模板
│   ├── retrieval/               # 论文检索模块
│   │   ├── paper_retriever.py   # API搜索
│   │   ├── hybrid_retriever.py  # 混合检索
│   │   ├── pdf_markdown_processor.py # PDF解析/清洗/切分
│   │   ├── paper_interpreter.py # 论文解读
│   │   └── evaluation.py        # 检索评测
│   ├── agents/                  # Multi-Agent
│   ├── experiment/              # 实验设计
│   ├── writing/                 # 学术写作
│   ├── tools/                   # 工具模块
│   │   ├── code_generator.py    # 代码生成
│   │   ├── data_analyzer.py     # 数据分析
│   │   └── visualizer.py        # 可视化
│   └── utils/                   # 工具函数
├── optional/
│   └── finetune/                # LoRA微调 (非核心)
├── examples/                    # 示例代码
├── data/                        # 数据目录
├── output/                      # 输出目录
├── requirements.txt             # 依赖
└── .env                         # 环境变量
```

## 📝 API参考

### ResearchAssistant

```python
class ResearchAssistant:
    # 论文搜索 (从外部API获取)
    def search_papers(query, sources=None, max_results=10) -> List[Dict]
  
    # 混合检索 (在本地索引中检索)
    def hybrid_search(query, top_k=5, use_rerank=True, use_hyde=False) -> List[Dict]
  
    # 论文解读 (需要LLM API)
    def interpret_paper(paper) -> Dict
  
    # 实验设计 (需要LLM API)
    def design_experiment(research_question, papers=None) -> Dict
  
    # 摘要生成 (需要LLM API)
    def generate_abstract(title, keywords, ...) -> str
  
    # Multi-Agent任务 (需要LLM API)
    def run_agent(task, max_iterations=10) -> Dict
  
    # 完整工作流
    def complete_research_workflow(research_question, output_dir=None)
```

## ❓ 常见问题

### 1. 混合检索器初始化失败

首次运行需要下载BGE模型 (~2GB)，请确保网络畅通。

### 2. LLM功能不可用

需要设置 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 环境变量。

### 3. GPU显存不足 (LoRA微调)

- 使用更小的模型: `Qwen/Qwen2.5-1.5B-Instruct`
- 减小batch_size: `per_device_train_batch_size=1`
- 使用云GPU: Colab Pro, AutoDL等

## 📄 License

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！
