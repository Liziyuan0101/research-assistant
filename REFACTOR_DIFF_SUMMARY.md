# 改动摘要（可读版）—— `refactor/skills-and-memory`

> **范围说明**：本文覆盖**第一阶段**（组件层：记忆层 / skills / 路由去 LLM 化）。
> **第二阶段**（把新架构接到产品路径：单 Agent 运行时、`HybridRetriever` 的偏好入口、
> CLI/API、端到端 demo，以及实测暴露的 2 个真 bug）见 `REFACTOR_NOTES.md` §4。
> 完整实现与复现命令以 `REFACTOR_NOTES.md` 为准。

> 对照对象：`main`（9046b12）。本文按**行为变化**组织，不按文件。
> 每条给出「改前 / 改后 / 关键片段 / 为什么」。
> 原始 diff 自取：`git diff main --stat`、`git show main:research_assistant/agents/multi_agent.py`

---

## 0. 一页速览

| 类别 | 文件 | 说明 |
|---|---|---|
| **新增包** | `research_assistant/memory/{__init__,schema,encoder,store,personalize}.py` | 独立记忆层（5 个文件，约 840 行） |
| **新增包** | `research_assistant/skills/{__init__,registry,loader}.py` | skill 注册与渐进披露加载（3 个文件，约 320 行） |
| **新增资源** | `skills/{paper-retrieval,experiment-design,academic-writing}/SKILL.md` + 8 个 `references/*.md` | 能力 skill 化 |
| **新增脚本** | `scripts/measure_refactor.py`、`scripts/memory_recall_demo.py` | 可复现度量 |
| **新增测试** | `tests/test_memory_layer.py`（217 行）、`tests/test_skills_layer.py`（115 行） | 覆盖语义召回/衰减/负反馈/迁移 + 渐进披露/确定性路由 |
| **修改** | `agents/multi_agent.py`（+217/−…）、`retrieval/evaluation.py`（+94）、`scripts/eval_retrieval.py`、`assistant.py`、`README.md` | 见下 |
| **降级为 shim** | `retrieval/memory.py`（186 行 → 31 行） | 向后兼容，老导入不破 |
| **契约变更** | `tests/test_memory.py` | 1 条测试的断言随行为变更更新 |

**刻意未动**：LangGraph 状态图结构、`agents/tools.py` 的 5 个工具、`experiment/`、`writing/`、`retrieval/` 下其余模块（`hybrid_retriever.py` 一行未改）、`examples/`、`pyproject.toml` 依赖分组。

---

## 1. 路由：从「LLM 当调度器」到「确定性查表」

### 改前

`multi_agent.py` 有 4 个路由函数 + 1 段独立提示词：

```python
SUPERVISOR_PROMPT = """你是一个科研助手的任务调度器。根据用户的请求，决定应该由哪个专业Agent来处理：
1. **retrieval** - 论文检索Agent ...
2. **experiment** - 实验设计Agent ...
3. **writing** - 写作辅助Agent ...
请分析任务并返回JSON格式的执行计划：
{{ "task_type": "retrieval|experiment|writing|complex", "agents_sequence": [...], "reasoning": "..." }}"""

class AgentType(Enum):
    RETRIEVAL = "retrieval"; EXPERIMENT = "experiment"; WRITING = "writing"
    SUPERVISOR = "supervisor"        # ← 删除

    def _supervisor_node(self, state):
        if self.llm:
            prompt = SUPERVISOR_PROMPT.format(task=task)
            response = self.llm.invoke([HumanMessage(content=prompt)])   # ← 1 次 LLM 调用
            try:
                plan = json.loads(response_text[json_start:json_end])    # ← 靠解析 JSON
                state['task_type'] = plan.get('task_type', 'retrieval')
            except:
                state['task_type'] = self._simple_task_classification(task)  # ← 静默回退硬编码关键词
```

### 改后

```python
#: skill 名 → 图节点类型
_SKILL_TO_TASK_TYPE = {'paper-retrieval': 'retrieval',
                       'experiment-design': 'experiment',
                       'academic-writing': 'writing'}

    def _dispatch_node(self, state):
        """确定性分发节点（取代原 supervisor）。"""
        task = state['task']
        metas = self.registry.select(task, top_k=3)          # ← 词元覆盖率打分, 0 次 LLM 调用
        state['skills'] = [m.name for m in metas]
        state['task_type'] = _SKILL_TO_TASK_TYPE.get(metas[0].name, 'retrieval')
        state['skill_queue'] = list(state['skills'][1:])     # ← 多 skill 链式编排
        self.hops += 1
        state['route_hops'] = self.hops
        state['resident_chars'] = self.registry.resident_chars()
        for m in metas:                                      # ← 渐进披露：只加载命中的
            body = self.loader.load(m.name)
            if body and (k := _SKILL_TO_AGENT_KEY.get(m.name)) in self.agents:
                self.agents[k].skill_prompt = body.body
        return state
```

**行为差异**：

| | 改前 | 改后 |
|---|---|---|
| 路由 LLM 调用 | 1 次 | **0 次** |
| 失败模式 | JSON 解析失败 → 静默退化为关键词表 | 无解析环节 |
| 可解释性 | 依赖 LLM 的 `reasoning` 字段 | 可复现的覆盖率分数（`registry.score()`） |
| 多能力链式 | `task_type == 'complex'` 硬编码 retrieval→experiment→writing 顺序 | `skill_queue` 按相关度降序 |

`_simple_task_classification`（硬编码中英关键词表，54 行）与 `SUPERVISOR_PROMPT`、`AgentType.SUPERVISOR` **整体删除**。
`ReActAgent.get_prompt()` 改为优先返回按需注入的 `skill_prompt`，未注入才回退内置模板。

---

## 2. 提示词：从「4 份常驻」到「frontmatter 常驻 + 正文按需」

### 改前

4 份 system prompt 常量写在 `multi_agent.py` 里，无论任务只需要哪一个都在上下文里：

| 常量 | tokens |
|---|---|
| `SUPERVISOR_PROMPT` | 224 |
| `RETRIEVAL_AGENT_PROMPT` | 167 |
| `EXPERIMENT_AGENT_PROMPT` | 228 |
| `WRITING_AGENT_PROMPT` | 232 |
| **合计** | **851** |

### 改后

`skills/<name>/SKILL.md`，frontmatter 进常驻索引，正文命中才加载：

```markdown
---
name: paper-retrieval
description: 当用户要检索学术论文、查找相关工作、构建文献库，或解读单篇论文时使用。触发词：检索/找论文/…
tools: [PaperRetrievalTool, PythonExecutorTool]
version: 1.0.0
---

# paper-retrieval（论文检索与解读）
## 何时用我 / ## 何时不用我 / ## 输入输出契约 / ## 执行步骤
## 细节参考（按需加载）
- references/data-sources.md
- references/scoring-and-personalization.md
- references/evaluation.md
```

| | tokens |
|---|---|
| 常驻：3 个 skill 的 frontmatter 索引 | **242** |
| 命中 paper-retrieval 时的正文 | 769（**不进常驻**） |
| 命中 experiment-design 时的正文 | 626 |
| 命中 academic-writing 时的正文 | 593 |

三级加载的实现：`SkillRegistry`（L0 扫描 frontmatter）+ `SkillLoader`（L1 正文、L2 references，带缓存与计量）。

> ⚠️ 注意：命中单个 skill 后是 242+769 = **1011 token，高于旧的 851**。收益在"多能力任务"——
> 旧设计 4 份提示全常驻，新设计只为命中的 skill 付费。别对外说"每请求都省了 71.6%"。

---

## 3. 记忆：从「关键词重叠」到「语义召回」

### 改前（`retrieval/memory.py`，原 163 行）

```python
PREFERENCE_CATEGORIES = ('journal', 'keyword', 'method')       # ← 硬编码

    def add_preference(self, user_id, category, value, weight=1.0):
        if category not in PREFERENCE_CATEGORIES:
            raise ValueError(f"Unsupported preference category: {category}")   # ← 扩展要改源码

    def add_preference(...):   # 只增不减
        ON CONFLICT(user_id, category, value)
        DO UPDATE SET weight = weight + excluded.weight            # ← 无衰减、无上限

    @staticmethod
    def _rank_qa(query, qa):
        """按 query 与历史 query 的 token 重叠度排序（简单关键词匹配，GPU 无关）"""
        query_tokens = set(query.lower().split())                  # ← 中文无空格
        hist_tokens  = set(item['query'].lower().split())
        overlap = len(query_tokens & hist_tokens)                  # ← 恒为 0
```

实测：`"电池剩余寿命预测"` 与 `"锂离子电池剩余寿命预测方法"` 的重叠 **= 0**；5/5 条历史全为 0，
排序退化成"按时间取最新 N 条"。而简历写的是"**向量化知识库**"。

### 改后（`memory/`，顶层包）

```python
# schema.py —— 类别可配置（默认 5 类），未知类别告警+自动登记，可 strict 恢复旧行为
DEFAULT_CATEGORIES = ('journal', 'keyword', 'method', 'dataset', 'style')

@dataclass
class PreferenceSchema:
    categories: Tuple[str, ...] = DEFAULT_CATEGORIES
    strict: bool = False
    half_life_days: float = 90.0     # 衰减半衰期
    max_weight: float = 10.0         # 权重上限
    min_effective_weight: float = 0.05

# store.py —— 语义召回 + 衰减 + 负反馈 + 成功轨迹
    def _decayed(self, weight, updated_at, as_of):
        decay = math.pow(0.5, age_days / self.schema.half_life_days)
        return min(weight * decay, self.schema.max_weight)

    def _rank_qa_semantic(self, query, qa):
        vecs = self.encoder.encode([query] + [i['query'] for i in qa])   # ← BGE-M3
        sims = hist @ vecs[0]
        return [dict(qa[i], similarity=float(sims[i])) for i in order]

    def get_successful_traces(self, user_id, limit=5):   # ← 旧实现存了 success 却从不读
        ... WHERE success = 1 AND answer IS NOT NULL ...
```

实测对照（`scripts/memory_recall_demo.py`）：

| query | 旧（重叠） | 新（cos） |
|---|---|---|
| 电池剩余寿命预测 | 5/5 条为 0，排序退化 | top1 = 锂离子电池剩余寿命预测方法，**0.908** |
| remaining useful life prediction of lithium-ion batteries | 全 0 | top1 = 锂离子电池剩余寿命预测方法，**0.764** |

其它：`ALTER TABLE` 自动迁移老库（补 `polarity` / `trace` 列）；编码器不可用时 `recall()` 返回
`backend='lexical'` **如实标注降级**，不假装语义可用。

`retrieval/memory.py` 缩为 31 行 shim，`from research_assistant.retrieval.memory import MemoryStore` 仍可用。

---

## 4. 新增：偏好回流检索（机制落地，但**实测无增益**）

```python
# memory/personalize.py
def preference_prior(profile, doc_texts, encoder, lam=0.2, base_scores=None, neg_lambda=None):
    """final = (1-λ)·minmax(base) + λ·minmax(偏好相似度)"""
```

`lam=0` 时**不加载编码器**，保证基线可纯离线复现。新增 `--mode personalized` 与 `--prior-mode {boost,expand}`。

**实测结果（同一 corpus/query，只变 λ 与画像）**：

| 机制:画像 | λ=0 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|
| boost:同域画像 | 55.0% | 55.0% | **50.0%** | **50.0%** |
| boost:离域对照 | 55.0% | 55.0% | 55.0% | 55.0% |
| expand:同域画像 | 55.0% | **35.0%** | **35.0%** | **35.0%** |
| expand:离域对照 | 55.0% | **35.0%** | **35.0%** | **35.0%** |

**归因**：语料 23 篇全为锂电池 → 画像在"领域"维度对所有文档近似等权，无区分度；
离域对照组与同域画像几乎同幅变化 → 指标分辨不出个性化信号；4 条 query → 统计功效不足。
**结论：评测集扩标且跨域前，不得宣称个性化增益。**（详见 `REFACTOR_NOTES.md` §3.5）

---

## 5. 修正对外声明（README / assistant.py）

| 位置 | 改前 | 改后 |
|---|---|---|
| README 架构图 | `Retrieval/Experiment/Writing Agent` + `LangGraph Router (ReAct Strategy)` | `skills/...` + `SkillRegistry 选择（确定性路由, 0 次 LLM）` + `memory/` 层 |
| README 核心功能表 | `Multi-Agent ｜ LangGraph + ReAct 策略` | `Skills ｜ frontmatter 常驻 + 正文/references 渐进披露`；新增 `记忆与个性化` 行 |
| README 性能指标 | `Precision@5 89%`、`任务完成率 85%`、`ROUGE-L 0.47`、`术语准确率 89%`（**均无脚本可复现**） | 改为 **P@5 55.0%** + 记忆召回 cos + 常驻 token + 路由 LLM 调用数，每行附复现命令 |
| README | （无） | 新增 **「评测集规模警告」**：仅 4 条 query / 23 篇语料，**P@5 理论上限 60%** |
| README 示例 | `01_quick_start.py` 等 3 个**不存在的文件名** | 改为实际的 `01_paper_retrieval.py` / `02_experiment_design.py` / `03_academic_writing.py` |
| `assistant.py` 文件头 | 同上 4 个指标写死在 docstring | 改为功能描述 + 「所有对外声明必须可由仓库内脚本复现」 |

---

## 6. 需要你知道的三处「非纯增量」改动

| # | 改动 | 影响 | 可否回退 |
|---|---|---|---|
| 1 | `tests/test_memory.py::test_invalid_preference_category_raises` 断言被替换 | 未知偏好类别从 `raise ValueError` 改为**告警 + 自动登记**（`PreferenceSchema(strict=True)` 可恢复旧行为） | 可，设 `strict=True` |
| 2 | 命中单 skill 的上下文（1011）> 旧常驻（851） | 单能力任务上下文变大，多能力任务变小 | 结构性，无法"一半回退" |
| 3 | 偏好回流检索已实现但**无正向证据** | 简历/README 中不得写个性化增益 | 机制保留（λ=0 即关闭） |

---

## 7. 建议的阅读顺序

1. `REFACTOR_NOTES.md` —— 动机、实测数字、复现命令（3 分钟）
2. 本文 §1 §2 —— 架构主干的两处改动（5 分钟）
3. `research_assistant/skills/registry.py::select()` + `loader.py` —— 渐进披露的实际实现（5 分钟）
4. `research_assistant/memory/store.py` —— 对照本文 §3 的「改前」看（8 分钟）
5. `research_assistant/agents/multi_agent.py` 搜 `_dispatch_node`、`skill_queue`（3 分钟）
6. 想跑一遍：`python -m pytest -q` → `scripts/measure_refactor.py` → `scripts/memory_recall_demo.py`

---

## 8. 回退

分支未 push，`main` 完好。回退即：

```bash
cd /d/dev/research-assistant
git checkout main          # 或 git branch -D refactor/skills-and-memory 丢弃
```
