"""可复现检索评测的测试:验证 BM25 评测产出真实、合理的指标。"""

from pathlib import Path

from research_assistant.retrieval.evaluation import evaluate_bm25_retrieval

ROOT = Path(__file__).resolve().parent.parent


def test_bm25_eval_produces_meaningful_metrics():
    corpus = ROOT / 'data' / 'eval' / 'corpus.json'
    eval_path = ROOT / 'data' / 'eval' / 'retrieval_eval.json'

    summary = evaluate_bm25_retrieval(str(corpus), str(eval_path), k_values=(1, 5))

    metrics = summary['metrics']
    assert summary['total_queries'] == 4
    # 指标在合法区间
    assert 0 <= metrics['Precision@5'] <= 1
    assert 0 <= metrics['MRR'] <= 1
    # 评测必须"有意义":确实能召回相关论文(非 0 分)
    assert metrics['Precision@5'] >= 0.3
    assert metrics['MRR'] >= 0.5
