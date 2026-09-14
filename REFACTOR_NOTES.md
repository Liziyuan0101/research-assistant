# 重构说明：4-Agent supervisor 编排 → Single-Agent + Skills + 独立记忆层

> 本文记录 2026-09 的重构，以及**所有实测数字的复现方式**。
> 原则：文档里出现的每个数字都必须能由仓库内脚本重跑出来。

## 1. 动机

| 原设计的问题 | 说明 |
|---|---|
| 路由消耗 LLM 调用且不可靠 | ``_supervisor_node`` 用 LLM 生成 JSON 执行计划，解析失败即静默回退到关键词匹配 |
| 4 份 system prompt 常驻 | SUPERVISOR + 检索 + 实验 + 写作，无论任务只需要哪一个 |
| 上下文重复搬运 | ``AgentState`` 在 agent 间反复传递 ``retrieved_papers`` / ``tool_outputs`` |
| 记忆是检索的子模块 | ``retrieval/memory.py``，架构上撑不起"个性化"这一核心能力 |
| 记忆召回对中文失效 | ``query.lower().split()`` 词重叠，中文无空格 ⇒ 重叠恒为 0 |
| 偏好不影响检索 | 只把偏好拼进 prompt（"软个性化"），检索器看不到偏好 |

## 2. 改了什么

### 2.1 记忆层提升为顶层模块

```
research_assistant/memory/
├── schema.py      偏好类别的可配置 schema（类别/半衰期/权重上限/严格模式）
├── encoder.py     BGE-M3 懒加载编码器（离线可用，不可用时显式降级）
├── store.py       MemoryStore v2（衰减 / 负反馈 / 语义召回 / 成功轨迹 / 自动迁移老库）
└── personalize.py 偏好回流：preference_prior() 把偏好混合进检索打分
```

`retrieval/memory.py` 保留为 shim 转发，老导入不破。

### 2.2 能力 skill 化

```
skills/
├── paper-retrieval/SKILL.md      + references/{data-sources,scoring-and-personalization,evaluation}.md
├── experiment-design/SKILL.md    + references/{prompt-templates,plan-schema}.md
└── academic-writing/SKILL.md     + references/{prompt-templates,style-guide}.md
```

`research_assistant/skills/{registry,loader}.py` 实现渐进披露三级加载：
L0 frontmatter 常驻 → L1 SKILL.md 正文按需 → L2 references 二次按需。

### 2.3 路由去 LLM 化

删除 `SUPERVISOR_PROMPT`、`AgentType.SUPERVISOR`、`_supervisor_node`、`_simple_task_classification`；
新增 `_dispatch_node`（`SkillRegistry` 词元覆盖率打分）与 `_SKILL_TO_TASK_TYPE` / `_SKILL_TO_AGENT_KEY` 映射。
多 skill 链式能力（原 `task_type == 'complex'`）由 `skill_queue` 保留。

`ReActAgent.get_prompt()` 改为优先返回按需注入的 `skill_prompt`。

## 3. 实测结果

复现：

```bash
python scripts/measure_refactor.py --save      # M1 / M2 / M4
python scripts/memory_recall_demo.py --save    # 记忆召回对照
python scripts/eval_retrieval.py --mode bm25   # 检索基线
python scripts/eval_retrieval.py --mode personalized --lam 0 0.1 0.2 0.3 --prior-mode boost expand
python -m pytest -q                            # 74 passed
```

### 3.1 M1 常驻上下文（tiktoken cl100k_base）

| | tokens |
|---|---|
| 重构前：SUPERVISOR_PROMPT | 224 |
| 重构前：RETRIEVAL_AGENT_PROMPT | 167 |
| 重构前：EXPERIMENT_AGENT_PROMPT | 228 |
| 重构前：WRITING_AGENT_PROMPT | 232 |
| **重构前合计（每请求都付）** | **851** |
| 重构后：3 个 skill 的 frontmatter 索引 | **242** |
| **常驻下降** | **↓71.6%** |
| 命中单个 skill 后的最坏情况 | 1011 |

各 skill 正文（按需，不进常驻）：paper-retrieval 769 / experiment-design 626 / academic-writing 593。

### 3.2 M2 路由

| | 重构前 | 重构后 |
|---|---|---|
| 路由函数 | 4 个 | 3 个 |
| **路由 LLM 调用** | **1 次** | **0 次** |
| 每请求决策 | 1(supervisor) + 1(_route_task) + N(_check_next_agent) | 1(registry.select) |

skill 选择冒烟测试（5 条中英任务，0 次 LLM 调用）全部命中预期 skill。

### 3.3 M4 代码行数

| | 重构前 | 重构后 |
|---|---|---|
| `multi_agent.py` 文件总行数 | 820 | 855 (+35) |
| 其中**路由层**行数 | 60 | **50 (↓16.7%)** |

> 文件总行数上升是因为度量埋点与 docstring 写在同一文件；路由层本身是净减少。

### 3.4 记忆召回（修复中文失效）

| query | 旧实现（词重叠） | 新实现（BGE-M3 cos） |
|---|---|---|
| 电池剩余寿命预测 | 5/5 条历史重叠为 0（排序退化） | top1 = 锂离子电池剩余寿命预测方法，cos 0.908 |
| remaining useful life prediction... | 词重叠全为 0 | top1 = 锂离子电池剩余寿命预测方法，cos 0.764 |

扩标后新增覆盖：可配置类别（含自动登记）、时间衰减（半衰期处减半，已单测）、权重上限、负反馈、
成功轨迹读取、老库自动迁移、降级时 `backend='lexical'` 如实标注。

### 3.5 ⚠️ 偏好回流检索：**未测出增益（负面结果）**

同一 corpus / query，仅变先验强度与机制：

| 机制:画像 | λ=0 | 0.1 | 0.2 | 0.3 | 结论 |
|---|---|---|---|---|---|
| boost:battery_research | 55.0% | 55.0% | 50.0% | 50.0% | 无害或略有害 |
| boost:off_domain_control | 55.0% | 55.0% | 55.0% | 55.0% | P@1 50%→75% |
| expand:battery_research | 55.0% | 35.0% | 35.0% | 35.0% | **显著有害** |
| expand:off_domain_control | 55.0% | 35.0% | 35.0% | 35.0% | 同幅变差 |

**归因**：评测语料 23 篇**全为锂电池**，用户画像在"领域"这一维对全部文档近似等权，
没有区分度；且对照组（离域画像）变化幅度与同域画像几乎一致，
说明当前评测集**无法分辨**个性化信号。这是统计功效问题，不是机制问题。

**结论：在评测集扩标（≥50 query）且**跨域**之前，不得宣称个性化带来的检索增益。**

## 4. 第二阶段：把新架构接到**产品路径**（本次）

**动机**：上一阶段（§2–§5）只落地了组件与测试，产品入口 `ResearchAssistant` **仍然走
`ResearchAgentGraph`**，而且"偏好回流检索打分先验"**只存在于评测脚本** ——
`HybridRetriever` 完全没有接收偏好的入口。也就是说：组件是真的，接线是假的。

### 6.1 本次改动

| 位置 | 内容 |
|---|---|
| `agents/single_agent.py`（新增） | 单 Agent 运行时：路由一跳 → 按需加载 skill → 执行器分发 → 记忆落库；无 LLM 时如实标注 `llm_used=False` |
| `retrieval/hybrid_retriever.py` | `search()` 新增 `preference_profile / prior_lambda / memory / user_id`，并在 RRF 之后增加**个性化步骤**（偏好先验重算打分）；`last_search_meta` 如实上报 `backend / reranker_skipped / personalized` |
| `_encode_dense()`（新增） | 稠密编码统一入口：FlagEmbedding 优先，缺失时回退 sentence-transformers（**同一份** bge-m3 权重 → 与记忆层同向量空间） |
| `assistant.py` | 新增 `backend='single'\|'graph'`、`prior_lambda`、`data_dir`；`run_agent()` 按后端分发；`hybrid_search(personalize=...)`；`add_preference / get_preferences / memory_report`；`health_report` 增加 skills 与 dense_available |
| `cli.py` | `--user-id / --backend / --lambda / --pref CAT=VAL（! 前缀=负面）/ --search / --agent / --chain / --skills / --memory` |
| `api.py` | 新增 `/agent`、`/skills`、`/local-search`、`/memory/stats`、`/memory/traces` |
| `config/config.yaml` | 新增 `personalization:` 段（lambda / 偏好类别 / 半衰期 / 权重上限） |
| `scripts/demo_e2e.py`（新增） | 端到端演示，**不需要 API key** 即可跑通全链路 |

### 6.2 修掉的两个**真 bug**（都是本次实测才暴露的）

1. **链式编排顺序错误**：按**相关度**顺序执行，导致 `experiment-design` 先于
   `paper-retrieval` 跑，下游 skill 拿不到检索结果。
   改为按**依赖顺序**（`_SKILL_DEPENDENCY_ORDER`：检索 → 实验 → 写作）。
   —— 相关度决定"要不要执行"，依赖决定"按什么次序执行"，两者必须分开。
2. **Reranker 守卫写错层级**：原守卫是"FlagEmbedding 能不能 import"。但装了
   FlagEmbedding 而本地**没有** `BAAI/bge-reranker-v2-m3` 权重时，`FlagReranker(...)`
   会抛 `OSError`（离线无网可下），直接把整个检索打挂。
   改为"**权重能不能加载**"（try/except + 失败后不再重试），跳过时记
   `reranker_skipped: True` 并保持 RRF 顺序。

### 6.3 环境与外部依赖（已解决）

| 事实 | 状态 |
|---|---|
| `FlagEmbedding` | **1.3.5 已安装**（`pip show` 确认）→ 此前"FlagEmbedding 缺失"的记录已过时 |
| `BAAI/bge-m3`（稠密编码） | 已缓存 8.5G ✓ |
| `BAAI/bge-reranker-v2-m3`（精排） | **已从 hf-mirror 下载补齐（2.14G）** ✓ `reranker_skipped: False` / `reranked: True` 实测通过 |

**下载要点**（本机 `huggingface.co` 被墙：`WinError 10054`；`hf-mirror.com` 可达）：
```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0 <python> -c "
import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-reranker-v2-m3', allow_patterns=[
    'config.json','model.safetensors','sentencepiece.bpe.model',
    'special_tokens_map.json','tokenizer.json','tokenizer_config.json'])"
```
用 `allow_patterns` 排除 `onnx/` 与 `assets/`：只需 6 个文件、2.14G（全量会多出数百 MB 无用体积）。
实测耗时 725s。注意 Windows 无开发者模式时 HF 缓存退回**复制**而非符号链接（会多占空间）。

### 6.3b 顺带修的第三个坑：CPU 上的 fp16

`FlagReranker` / `BGEM3FlagModel` 原本硬编码 `use_fp16=True`。本机 **CPU-only**
（`torch.cuda.is_available() == False`，`resolve_device('auto') → 'cpu'`），
fp16 在 CPU 上算子缺失会直接报错。已改为**随设备决定**：
`use_fp16 = str(self.device).startswith('cuda')`。与 §6.2 的第 2 个 bug 是同一类
——"只在真跑时才暴露"。

### 6.4 复现

```bash
PYTHONPATH= HF_HUB_OFFLINE=1 <conda research_assistant>/python.exe scripts/demo_e2e.py
PYTHONPATH= HF_HUB_OFFLINE=1 <...>/python.exe -m research_assistant.cli --health
PYTHONPATH= HF_HUB_OFFLINE=1 <...>/python.exe -m research_assistant.cli \
    --user-id lzy --lambda 0.25 --pref "method=Bayesian deep learning" \
    --pref "style=!purely empirical curve fitting" \
    --search "remaining useful life prediction of lithium-ion batteries"
PYTHONPATH= HF_HUB_OFFLINE=1 <...>/python.exe -m pytest -q     # 103 passed
```

实测（demo 输出）：
- 检索后端 `bm25+dense`（sentence-transformers/FlagEmbedding），`personalized: True`
- 个性化**确实改变排序**（如偏好画像把 `Two-stage Early Prediction` 从 #2 顶到 #1）
- 链式编排按依赖序：`['paper-retrieval', 'experiment-design', 'academic-writing']`，
  实验方案拿到 **5 篇**上游检索结果
- 路由跳数 1、**LLM 调用 0**；常驻 290 字符 / 加载 1016–1178 字符
- `test_*` 共 **103 passed**

---

## 5. 跨域评测集与个性化消融（结论：机制被证伪）

### 5.1 为什么要重建评测集

`data/eval/retrieval_eval.json` 只有 **4 条 query / 23 篇同域语料**，两个结构性问题：

1. `Precision@5` 的理论上限只有 **60%**（q3/q4 各只标 1 篇相关）→ 任何更高数字都不可能；
2. 语料**全为锂电池** → 用户画像在"领域"维度对全部文档近似等权，**没有区分度**。

原假设是"换跨域语料就能测出增益"。**这个假设已被实验证伪（见 5.3）。**

### 5.2 新评测集（自动构造 + 可复现）

`scripts/build_crossdomain_eval.py`：

| 项 | 值 |
|---|---|
| 语料 | **360 篇**（OpenAlex，8 个领域 / 30 个主题 × 12 篇） |
| 查询 | **30 条**，每条 **12 篇**相关 → **P@5 上限 100%** |
| 领域覆盖 | 电池/能源、时序预测、NLP/LLM、检索推荐、图模型、视觉、材料/健康、工业/供应链、安全/隐私、语音/机器人 |
| 相关性标签 | 论文的 OpenAlex **`primary_topic.id` 是否等于该查询的目标主题** —— 用**分类学元数据**而非"文本里是否含查询词"，避免自我实现 |
| 质量过滤 | `type:article` + `language:en` + 按被引排序取池 + 丢弃 `topic_score<0.55` + 摘要≥300 字符 |
| 复现 | 固定种子 `20260914`；`python scripts/build_crossdomain_eval.py` |

**第一批未加过滤时标签噪声很大**（OpenAlex 把 "Air Conditioning Repairs Mosman" 归入光伏故障检测、
混入乌克兰语/西语标题）。加上质量过滤后抽查正常（见 `data/eval/crossdomain_spotcheck.md`），
残余噪声约 1/8。**这是弱标注集，需人工抽查后才能对外使用。**

### 5.3 消融结果（360 篇 / 30 条查询，按域分组）

BM25 基线：**P@5 64.67%**（电池域 73.33% / 其他域 62.50%），MRR 0.772 —— 现在有增长空间了。

| 机制 | 画像 | 查询组 | λ=0 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|
| boost | 对齐（电池） | 电池域(6) | 73.33% | 70.00% | 70.00% | 70.00% |
| boost | 对齐 | 其他域(24) | 62.50% | 62.50% | 60.83% | 60.00% |
| boost | 离域对照 | 电池域 | 73.33% | 73.33% | 73.33% | 73.33% |
| boost | 离域对照 | 其他域 | 62.50% | 62.50% | 63.33% | 60.83% |
| **tiebreak** | 任意 | 任意 | **完全不变（空操作）** | | | |
| **expand** | 任意 | 任意 | **灾难性**：1.67% – 29.17% | | | |

**结论（三条）**：

1. **偏好回流打分先验没有正向增益**，即使在跨域语料上、即使画像与查询域对齐（对齐画像反而 −3.3%）。
   → 我上一轮"跨域就能测出增益"的假设**被证伪**；问题不在语料，在机制本身：
   它把**与查询无关的用户先验混进相关性打分**，本质是用先验污染相关性信号。
2. **`tiebreak`（只在近似并列时用先验）是空操作** —— 重排幅度（±0.011）小于真实检索分差，
   BM25 归一化分数上不存在可操作的"并列"。该机制形式上合理，实测无效果。
3. **`expand`（查询侧扩展）应弃用** —— 在任何设置下都有害，跨域时崩得更彻底。

**因此：不再调 λ，机制本身作废。** 若还想做"个性化"，方向应转向
**不改相关性打分**的用法（结果过滤/多样化重排、或让用户显式确认查询意图），
而不是继续往文档分数里混先验。

### 5.4 对简历的影响

* **不能**再写"个性化带来检索增益"（现在有两套语料、三种机制的负面证据）。
* **可以**写的：建立了**可复现的跨域检索评测集**（360 篇 / 30 查询 / 10 领域，主题标签做相关性、附人工抽查），
  并通过 **3 机制 × λ 扫描 × 离域对照组**的消融**证伪**了"偏好回流打分"这一机制。
  —— 这是"会做实验、敢报负面结果"的能力信号，比一个漂亮的假数字有说服力。
* 检索这一项现在有**可复现的真实数字**：BM25 跨域 P@5 **64.67%**（30 条查询）。


---

## 6. 未做 / 待办

- [x] **M9 评测集扩标** → 已由 §5 完成（自动构造 360 篇 / 30 查询，跨域）。
      遗留：**人工抽查** `data/eval/crossdomain_spotcheck.md`，剔除残余噪声（约 1/8）
- [x] 跨域语料上的偏好回流消融 → 已完成，**机制被证伪**（§5.3）
- [ ] 若仍要做个性化：换**不改相关性打分**的用法（结果过滤 / 多样化重排 / 查询意图显式确认）
- [ ] 端到端延迟 P95 与任务完成率（需 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`；
      注意首调用含模型加载：实测检索类任务 34.5s 里大部分是 BGE-M3 首次加载，
      精排首次调用还要再加 reranker 的 2.14G 权重加载）
- [ ] LoRA 写作指标的复现脚本（当前 ROUGE-L 0.47 / 术语 89% 无脚本可复现）


## 7. 对既有测试的影响

`tests/test_memory.py::test_invalid_preference_category_raises` 的断言已随契约变更而更新：
未知类别默认**告警并自动登记**（`PreferenceSchema(strict=True)` 可恢复旧的抛错行为）。
该测试被替换为两个测试（默认行为 + 严格模式）。

`74 passed`。
