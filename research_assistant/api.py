"""
FastAPI 服务：把 ResearchAssistant 暴露为 HTTP 接口。

启动：python scripts/serve.py（默认 http://0.0.0.0:8000）

默认运行时后端是单 Agent + Skills（渐进披露，路由 0 次 LLM 调用）；
``/search`` 与 ``/agent`` 都会带上个性化（偏好回流检索先验）。
"""

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .assistant import ResearchAssistant

app = FastAPI(title="Research Assistant API", version="0.2.0")

# 懒加载单例（首次请求才初始化，避免 import 时加载重模块）
_assistant: Optional[ResearchAssistant] = None


def get_assistant() -> ResearchAssistant:
    global _assistant
    if _assistant is None:
        _assistant = ResearchAssistant(verbose=False)
    return _assistant


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    personalize: Optional[bool] = None       # None = 按 λ 自动决定
    prior_lambda: Optional[float] = None


class InterpretRequest(BaseModel):
    paper: dict


class PreferenceRequest(BaseModel):
    user_id: str
    category: str
    value: str
    weight: float = 1.0
    polarity: int = 1                        # -1 = 负面偏好


class AgentRequest(BaseModel):
    task: str
    user_id: str = "default"
    chain: bool = False


@app.get("/health")
def health():
    return get_assistant().health_report()


@app.get("/skills")
def skills():
    """列出 skills 与上下文占用（渐进披露的 L0 常驻 vs L1/L2 已加载）。"""
    a = get_assistant()
    if a.agent is None:
        return {"error": "single-agent runtime unavailable"}
    reg, loader = a.agent.registry, a.agent.loader
    return {
        "skills": [s.to_dict() for s in reg.skills],
        "resident_index_chars": reg.resident_chars(),
        "loaded_chars": loader.loaded_chars,
        "routing": "skill-frontmatter (0 LLM calls)",
    }


@app.post("/search")
def search(req: SearchRequest):
    """外部源检索（arxiv 等）。"""
    papers = get_assistant().search_papers(req.query, max_results=req.max_results,
                                           auto_enhance=False)
    return {"count": len(papers), "papers": papers}


@app.post("/local-search")
def local_search(req: SearchRequest):
    """本地索引的混合检索（BM25 + BGE-M3），可选个性化偏好回流。"""
    a = get_assistant()
    results = a.hybrid_search(req.query, top_k=req.max_results, use_rerank=False,
                              personalize=req.personalize,
                              prior_lambda=req.prior_lambda)
    meta = getattr(a.hybrid_retriever, "last_search_meta", {}) if a.hybrid_retriever else {}
    return {"count": len(results), "meta": meta, "results": results}


@app.post("/agent")
def run_agent(req: AgentRequest):
    """单 Agent + Skills 执行一条任务（sqlite 记忆落库 + 路由元信息）。"""
    a = get_assistant()
    if req.user_id != a.user_id:
        a.user_id = req.user_id
        if a.agent is not None:
            a.agent.user_id = req.user_id
    return a.run_agent(req.task, chain=req.chain)


@app.post("/interpret")
def interpret(req: InterpretRequest):
    return get_assistant().interpret_paper(req.paper)


@app.get("/memory/preferences/{user_id}")
def get_preferences(user_id: str, include_negative: bool = False):
    a = get_assistant()
    return a.memory.get_preferences(user_id, include_negative=include_negative)


@app.post("/memory/preferences")
def add_preference(req: PreferenceRequest):
    ok = get_assistant().memory.add_preference(
        req.user_id, req.category, req.value,
        weight=req.weight, polarity=req.polarity,
    )
    return {"ok": ok}


@app.get("/memory/stats/{user_id}")
def memory_stats(user_id: str):
    """记忆画像统计（用于诊断个性化为何生效/不生效）。"""
    return get_assistant().memory.stats(user_id)


@app.get("/memory/traces/{user_id}")
def memory_traces(user_id: str, limit: int = 5):
    """成功轨迹（旧实现写入 success 却从不读取，这里开放出来供 skill 复用）。"""
    return get_assistant().memory.get_successful_traces(user_id, limit=limit)
