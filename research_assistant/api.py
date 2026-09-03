"""
FastAPI 服务：把 ResearchAssistant 暴露为 HTTP 接口。

启动：python scripts/serve.py（默认 http://0.0.0.0:8000）
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .assistant import ResearchAssistant

app = FastAPI(title="Research Assistant API", version="0.1.0")

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


class InterpretRequest(BaseModel):
    paper: dict


class PreferenceRequest(BaseModel):
    user_id: str
    category: str
    value: str


@app.get("/health")
def health():
    return get_assistant().health_report()


@app.post("/search")
def search(req: SearchRequest):
    papers = get_assistant().search_papers(req.query, max_results=req.max_results, auto_enhance=False)
    return {"count": len(papers), "papers": papers}


@app.post("/interpret")
def interpret(req: InterpretRequest):
    return get_assistant().interpret_paper(req.paper)


@app.get("/memory/preferences/{user_id}")
def get_preferences(user_id: str):
    return get_assistant().memory.get_preferences(user_id)


@app.post("/memory/preferences")
def add_preference(req: PreferenceRequest):
    ok = get_assistant().memory.add_preference(req.user_id, req.category, req.value)
    return {"ok": ok}
