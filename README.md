# 🎓 Research Assistant - 科研论文智能助手

基于混合检索和多智能体的学术研究辅助系统，支持论文搜索、智能检索、实验设计和学术写作。

## ✨ 核心功能

| 功能                  | 说明                     | 技术方案                                            |
| --------------------- | ------------------------ | --------------------------------------------------- |
| **论文搜索**    | 从多个学术数据库获取论文 | arXiv, OpenAlex, Semantic Scholar                   |
| **混合检索**    | 精准语义检索             | BM25 + BGE-M3 + BGE-Reranker                        |
| **Multi-Agent** | 复杂任务自动化           | LangGraph + ReAct 策略                              |
| **实验设计**    | 自动生成实验方案         | LLM + 论文上下文                                    |
| **学术写作**    | 摘要/引言/方法生成       | DeepSeek/OpenAI API                                 |
| **LoRA微调**    | 学术写作风格优化         | Qwen2.5 + PEFT (见 `optional/finetune/`)            |
| **记忆与个性化** | 记住用户偏好与检索历史   | SQLite + 词法召回 (`research_assistant/retrieval/memory.py`) |
| **HTTP 服务**   | 把能力暴露为 REST 接口   | FastAPI (`api.py` / `scripts/serve.py`)             |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ResearchAssistant                        │
├─────────────────────────────────────────────────────────────┤
│              ┌───────────────────────┐                      │
│              │  Supervisor (Router)  │                      │
│              │  LangGraph 状态图      │                      │
│              └───────────┬───────────┘                      │
│  ┌─────────────┐  ┌──────▼──────┐  ┌─────────────┐          │
│  │  Retrieval  │  │ Experiment  │  │   Writing   │          │
│  │    Agent    │  │    Agent    │  │    Agent    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────┐  ┌─────────────────────────┐  │
│  │    Hybrid Retrieval      │  │  Memory (偏好/检索历史)  │  │
│  │ BM25 + BGE-M3 → RRF →    │  │  SQLite                  │  │
│  │        BGE-Reranker      │  │  retrieval/memory.py     │  │
│  └──────────────────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  数据源: arXiv | OpenAlex | Semantic Scholar                │
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
# 安装本包 (基础依赖，不含 torch/GPU)
pip install -e .

# 需要 Multi-Agent / 微调时:
#   pip install -e ".[agent]"      # LangGraph 多智能体
#   pip install -e ".[finetune]"   # 混合检索(BGE/FAISS) + LoRA 微调
#   pip install -e ".[serve]"      # FastAPI 服务
#   pip install -e ".[dev]"        # 测试

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
# 论文检索 (多源API搜索 → 建索引 → 混合检索)
python examples/01_paper_retrieval.py

# 实验设计 (LLM 生成实验方案)
python examples/02_experiment_design.py

# 学术写作 (摘要/引言/方法生成)
python examples/03_academic_writing.py

# 可复现的检索评测 (用仓库内评测集,无需 LLM)
python scripts/eval_retrieval.py --mode bm25

# 启动 HTTP 服务 (需先 pip install -e ".[serve]")
python scripts/serve.py
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

### 论文检索 (`research_assistant/retrieval/`)

| 文件                         | 功能                                                              |
| ---------------------------- | ----------------------------------------------------------------- |
| `paper_retriever.py`         | 多源API搜索 (arXiv, OpenAlex, Semantic Scholar；CORE/Dimensions 需 Key，默认关闭) |
| `hybrid_retriever.py`        | BM25 + BGE-M3 + BGE-Reranker 混合检索                             |
| `pdf_markdown_processor.py`  | PDF解析、清洗、切分、入库                                         |
| `paper_interpreter.py`       | LLM论文解读                                                       |
| `evaluation.py`              | 检索评测 (Precision@k / Recall@k / MRR；可选 RAGAS 指标)          |
| `memory.py`                  | 用户偏好与检索历史记忆 (SQLite)                                   |

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

#### 数据目录结构

```
data/
├── papers/             # PDF 原件（可复用资产，勿删）
├── cache/search/       # 搜索响应缓存（可再生，24h TTL）
├── embeddings/         # documents.pkl + faiss.index
├── index/              # bm25_data.pkl / faiss_hnsw.index / faiss_chunk_ids.pkl
├── metadata/           # papers.db  (SQLite: papers 表 + chunks 表)
├── eval/               # 评测集 (corpus.json / retrieval_eval.json)
└── memory.db           # 用户偏好与检索历史
```

> 除 `data/eval/` 外，`data/` 下内容均不入 git（见 `.gitignore`）。
> PDF 与搜索缓存**分开存放**：前者是可复用资产，后者是可再生数据。

### Multi-Agent (`research_assistant/agents/`)

| 文件               | 功能                                       |
| ------------------ | ------------------------------------------ |
| `multi_agent.py` | LangGraph状态图：Supervisor 路由 + 三个 Agent (检索/实验/写作) |
| `tools.py`       | Agent工具集 (Python执行、统计分析、可视化) |

### 实验 / 写作 / 微调

| 模块                             | 功能                                  |
| -------------------------------- | ------------------------------------- |
| `research_assistant/experiment/` | 实验方案自动设计 (`experiment_planner.py`) |
| `research_assistant/writing/`    | 学术写作 + 引用管理 (`academic_writer.py` / `citation_manager.py`) |
| `optional/finetune/`             | LoRA 微调 (`lora_trainer.py` / `data_processor.py` / `evaluator.py`) |
| `research_assistant/api.py`      | FastAPI 服务层 (`scripts/serve.py` 启动) |

## ⚙️ 配置说明

配置文件: `research_assistant/config/config.yaml`

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

**本仓库只收录能由仓库内脚本复现的数字。**

```bash
python scripts/eval_retrieval.py --mode bm25   # 无需 LLM API Key、无需下载模型
```

| 模块            | 指标        | 数值        | 说明     |
| --------------- | ----------- | ----------- | -------- |
| 检索 (BM25 基线) | Precision@5 | **55.00%**  | 可复现   |
| 检索 (BM25 基线) | Recall@5    | **87.50%**  | 可复现   |
| 检索 (BM25 基线) | Hit@5       | **100.00%** | 可复现   |
| 检索 (BM25 基线) | MRR         | **68.75%**  | 可复现   |

> ⚠️ **当前评测集太小，不足以支撑结论。**
> `data/eval/retrieval_eval.json` 只有 **4 条 query**（语料 23 篇），且 q3/q4 各只标注
> 1 篇相关文献 —— 因此 **Precision@5 的理论上限只有 60%**，
> 任何高于 60% 的数值都不可能由该评测集产生。
> **扩标到 ≥50 条且跨域之前，请勿对外引用检索指标。**

**待补测**（仓库内暂无复现脚本，引用前请先补齐）：

| 模块            | 指标        | 缺少什么                                  |
| --------------- | ----------- | ----------------------------------------- |
| Multi-Agent     | 任务完成率  | 需 LLM API Key + 明确的任务成功判定标准    |
| 学术写作 (LoRA) | ROUGE-L     | 需 `optional/finetune` 的评测脚本          |
| 学术写作 (LoRA) | 术语准确率  | 需术语表与判定脚本                        |

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
| 论文API    | arXiv, OpenAlex, Semantic Scholar (CORE/Dimensions 需 Key，默认关闭) |

## 📁 项目结构

```
research-assistant/
├── pyproject.toml               # 打包配置 (extras: agent/finetune/dev/serve)
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
├── scripts/                     # 入口脚本 (arxiv_ingest / eval_retrieval / serve)
├── tests/                       # 单元测试
├── data/                        # 数据目录
│   ├── papers/                  #   PDF 原件（可复用资产，勿删）
│   ├── cache/search/            #   搜索响应缓存（可再生数据，24h TTL）
│   ├── index/ embeddings/       #   检索索引（可重建）
│   ├── metadata/papers.db       #   SQLite 元数据
│   ├── eval/                    #   评测集（唯一入库 git 的文件）
│   └── memory.db                #   用户偏好记忆
├── pyproject.toml               # 依赖与打包 (唯一来源)
├── README.md                    # 本文档
└── .env                         # 环境变量 (含 API key，已 gitignore)
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

### 1. 混合检索器初始化失败 / 模型下载不了

首次运行需下载 BGE 模型（BGE-M3 约 2.3GB，BGE-Reranker 约 2.1GB）。

**国内网络下 `huggingface.co` 通常不可达**（表现：`WinError 10054` 连接被强制关闭），
请走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # Windows PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
python -c "
import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3')
snapshot_download('BAAI/bge-reranker-v2-m3')"
```

若本地已有权重，可设 `HF_HUB_OFFLINE=1` 完全离线运行。

模型缺失时的行为是**如实降级而非静默失败**：`HybridRetriever.last_search_meta` 会给出
`backend`（`bm25+dense` / `bm25` / `none`）与 `reranker_skipped`，据此可判断当前跑到哪一步。

### 2. LLM功能不可用

需要设置 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 环境变量。
未配置时实验设计/学术写作会走内置模板回退，可离线跑通但输出非 LLM 产物。

### 3. 论文检索全部超时

arXiv API 与 Semantic Scholar 在国内常见 5xx / 429。`paper_retrieval.sources` 里
**OpenAlex 无需 API Key 且通常可用**，建议优先：

```yaml
paper_retrieval:
  sources: [openalex, arxiv, semantic_scholar]
```

### 4. GPU显存不足 (LoRA微调)

- 使用更小的模型: `Qwen/Qwen2.5-1.5B-Instruct`
- 减小batch_size: `per_device_train_batch_size=1`
- 使用云GPU: Colab Pro, AutoDL等

## 📄 License

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！
