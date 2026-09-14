"""
Retrieval Evaluation Module
检索评测模块：支持 Precision@K、RAGAS 自动生成评测集 + 人工修正

主要功能：
1. Precision@K 评测（需要标注的 ground truth）
2. RAGAS 自动生成评测数据
3. 人工修正接口
4. 完整评测报告生成
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

try:
    from ragas import evaluate
    # 新版 RAGAS 导入路径
    try:
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        
        # 初始化指标，禁用可能导致 n>1 的功能
        context_precision = ContextPrecision()
        context_recall = ContextRecall()
        # faithfulness 和 answer_relevancy 可能使用 n>1，暂时禁用
        faithfulness = None
        answer_relevancy = None
    except ImportError:
        # 兼容旧版
        try:
            from ragas.metrics import (
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            )
            # 旧版也可能有问题，禁用 faithfulness 和 answer_relevancy
            faithfulness = None
            answer_relevancy = None
        except:
            context_precision = None
            context_recall = None
            faithfulness = None
            answer_relevancy = None
    from ragas.testset import TestsetGenerator
    HAS_RAGAS = True
except (ImportError, TypeError, ModuleNotFoundError) as e:
    HAS_RAGAS = False
    evaluate = None
    context_precision = None
    context_recall = None
    faithfulness = None
    answer_relevancy = None
    TestsetGenerator = None

try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


import logging
logger = logging.getLogger(__name__)

@dataclass
class EvalSample:
    """评测样本"""
    query_id: str
    query: str
    relevant_paper_ids: List[str]  # ground truth
    relevant_chunk_ids: Optional[List[int]] = None
    source: str = "manual"  # manual, ragas_generated, ragas_modified
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Optional[Dict] = None


@dataclass
class EvalResult:
    """评测结果"""
    query_id: str
    query: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    hit_at_k: Dict[int, bool]


class RetrievalEvaluator:
    """检索评测器"""
    
    def __init__(self, eval_data_path: Optional[str] = None):
        """
        初始化评测器
        
        Args:
            eval_data_path: 评测数据文件路径
        """
        self.eval_samples: List[EvalSample] = []
        self.eval_results: List[EvalResult] = []
        
        if eval_data_path:
            self.load_eval_data(eval_data_path)
    
    def load_eval_data(self, path: str):
        """加载评测数据"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.eval_samples = [
                EvalSample(**sample) for sample in data.get('samples', [])
            ]
            logger.info(f"📂 Loaded {len(self.eval_samples)} evaluation samples")
        except Exception as e:
            logger.error(f"❌ Error loading eval data: {e}")
    
    def save_eval_data(self, path: str):
        """保存评测数据"""
        try:
            data = {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(self.eval_samples),
                'samples': [asdict(s) for s in self.eval_samples]
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Saved {len(self.eval_samples)} samples to {path}")
        except Exception as e:
            logger.error(f"❌ Error saving eval data: {e}")
    
    def add_sample(self, sample: EvalSample):
        """添加评测样本"""
        self.eval_samples.append(sample)
    
    def evaluate_retriever(
        self,
        retriever,
        k_values: List[int] = [1, 3, 5, 10],
        verbose: bool = True
    ) -> Dict:
        """
        评测检索器
        
        Args:
            retriever: 检索器实例（需要有 search 方法）
            k_values: 要评测的 K 值列表
            verbose: 是否打印详细信息
            
        Returns:
            评测结果汇总
        """
        if not self.eval_samples:
            logger.warning("⚠️ No evaluation samples loaded")
            return {}
        
        self.eval_results = []
        
        all_precisions = {k: [] for k in k_values}
        all_recalls = {k: [] for k in k_values}
        all_mrrs = []
        all_hits = {k: [] for k in k_values}
        
        for sample in self.eval_samples:
            # 执行检索
            max_k = max(k_values)
            results = retriever.search(sample.query, top_k=max_k)
            
            # 提取检索到的paper_ids
            retrieved_ids = []
            for r in results:
                if r.get('paper') and r['paper'].get('paper_id'):
                    paper_id = r['paper']['paper_id']
                    if paper_id not in retrieved_ids:
                        retrieved_ids.append(paper_id)
            
            # 计算指标
            precision_at_k = {}
            recall_at_k = {}
            hit_at_k = {}
            
            for k in k_values:
                top_k_ids = retrieved_ids[:k]
                
                # Precision@K
                relevant_in_top_k = len(set(top_k_ids) & set(sample.relevant_paper_ids))
                precision = relevant_in_top_k / k if k > 0 else 0
                precision_at_k[k] = precision
                all_precisions[k].append(precision)
                
                # Recall@K
                recall = relevant_in_top_k / len(sample.relevant_paper_ids) if sample.relevant_paper_ids else 0
                recall_at_k[k] = recall
                all_recalls[k].append(recall)
                
                # Hit@K
                hit = relevant_in_top_k > 0
                hit_at_k[k] = hit
                all_hits[k].append(hit)
            
            # MRR (Mean Reciprocal Rank)
            mrr = 0
            for i, pid in enumerate(retrieved_ids):
                if pid in sample.relevant_paper_ids:
                    mrr = 1 / (i + 1)
                    break
            all_mrrs.append(mrr)
            
            # 保存结果
            result = EvalResult(
                query_id=sample.query_id,
                query=sample.query,
                retrieved_ids=retrieved_ids,
                relevant_ids=sample.relevant_paper_ids,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                mrr=mrr,
                hit_at_k=hit_at_k
            )
            self.eval_results.append(result)
            
            if verbose:
                logger.info(f"Query: {sample.query[:50]}...")
                logger.info(f"  P@5: {precision_at_k.get(5, 0):.2%}, MRR: {mrr:.4f}")
        
        # 汇总结果
        summary = {
            'total_queries': len(self.eval_samples),
            'metrics': {}
        }
        
        for k in k_values:
            summary['metrics'][f'Precision@{k}'] = np.mean(all_precisions[k])
            summary['metrics'][f'Recall@{k}'] = np.mean(all_recalls[k])
            summary['metrics'][f'Hit@{k}'] = np.mean(all_hits[k])
        
        summary['metrics']['MRR'] = np.mean(all_mrrs)
        
        if verbose:
            logger.info("\n" + "="*60)
            logger.info("📊 EVALUATION SUMMARY")
            logger.info("="*60)
            for metric, value in summary['metrics'].items():
                logger.info(f"  {metric}: {value:.4f} ({value:.2%})")
        
        return summary
    
    def generate_report(self, output_path: str):
        """生成详细评测报告"""
        if not self.eval_results:
            logger.warning("⚠️ No evaluation results to report")
            return
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_queries': len(self.eval_results),
                'avg_precision_at_5': np.mean([r.precision_at_k.get(5, 0) for r in self.eval_results]),
                'avg_mrr': np.mean([r.mrr for r in self.eval_results])
            },
            'detailed_results': [
                {
                    'query_id': r.query_id,
                    'query': r.query,
                    'precision_at_k': r.precision_at_k,
                    'recall_at_k': r.recall_at_k,
                    'mrr': r.mrr,
                    'retrieved_ids': r.retrieved_ids[:10],
                    'relevant_ids': r.relevant_ids
                }
                for r in self.eval_results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 Report saved to {output_path}")


class RAGASTestsetGenerator:
    """
    RAGAS 评测集生成器
    
    使用 RAGAS 自动生成评测数据，支持人工修正
    """
    
    def __init__(self, llm=None, embeddings=None):
        """
        初始化生成器
        
        Args:
            llm: LangChain LLM 实例
            embeddings: LangChain Embeddings 实例
        """
        self.llm = llm
        self.embeddings = embeddings
        self.generated_samples: List[Dict] = []
    
    def generate_from_papers(
        self,
        papers: List[Dict],
        num_samples: int = 50,
        distribution: Dict[str, float] = None
    ) -> List[EvalSample]:
        """
        从论文生成评测样本
        
        Args:
            papers: 论文列表
            num_samples: 生成样本数量
            distribution: 难度分布 {'simple': 0.5, 'reasoning': 0.3, 'multi_context': 0.2}
            
        Returns:
            生成的评测样本列表
        """
        if not HAS_RAGAS:
            logger.warning("⚠️ RAGAS not available, using fallback generation")
            return self._fallback_generate(papers, num_samples)
        
        if distribution is None:
            distribution = {'simple': 0.5, 'reasoning': 0.3, 'multi_context': 0.2}
        
        # 准备文档
        documents = []
        for paper in papers:
            doc_text = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}"
            documents.append({
                'page_content': doc_text,
                'metadata': {
                    'paper_id': paper.get('id', ''),
                    'title': paper.get('title', ''),
                    'source': paper.get('source', '')
                }
            })
        
        try:
            # 使用 RAGAS TestsetGenerator
            generator = TestsetGenerator.from_langchain(
                generator_llm=self.llm,
                critic_llm=self.llm,
                embeddings=self.embeddings
            )
            
            # 生成测试集
            testset = generator.generate_with_langchain_docs(
                documents,
                test_size=num_samples,
                distributions={
                    simple: distribution.get('simple', 0.5),
                    reasoning: distribution.get('reasoning', 0.3),
                    multi_context: distribution.get('multi_context', 0.2)
                }
            )
            
            # 转换为 EvalSample
            samples = []
            for i, row in enumerate(testset.to_pandas().itertuples()):
                sample = EvalSample(
                    query_id=f"ragas_{i}",
                    query=row.question,
                    relevant_paper_ids=[],  # 需要人工标注
                    source="ragas_generated",
                    difficulty=self._infer_difficulty(row),
                    metadata={
                        'ground_truth': row.ground_truth if hasattr(row, 'ground_truth') else None,
                        'contexts': row.contexts if hasattr(row, 'contexts') else None
                    }
                )
                samples.append(sample)
            
            self.generated_samples = samples
            return samples
            
        except Exception as e:
            logger.error(f"❌ RAGAS generation failed: {e}")
            return self._fallback_generate(papers, num_samples)
    
    def _fallback_generate(self, papers: List[Dict], num_samples: int) -> List[EvalSample]:
        """备用生成方法（不依赖 RAGAS）"""
        samples = []
        
        # 基于论文标题和摘要生成查询模板
        query_templates = [
            "What are the main contributions of research on {topic}?",
            "How does {method} work for {task}?",
            "What datasets are used for {task}?",
            "What are the limitations of {method}?",
            "Compare different approaches for {task}",
            "{topic}的主要研究方法有哪些？",
            "如何使用{method}解决{task}问题？",
            "{topic}领域的最新进展是什么？"
        ]
        
        for i in range(min(num_samples, len(papers))):
            paper = papers[i % len(papers)]
            template = random.choice(query_templates)
            
            # 从标题提取关键词
            title = paper.get('title', '')
            words = title.split()
            topic = ' '.join(words[:3]) if len(words) >= 3 else title
            method = words[0] if words else 'the method'
            task = ' '.join(words[-3:]) if len(words) >= 3 else 'the task'
            
            query = template.format(topic=topic, method=method, task=task)
            
            sample = EvalSample(
                query_id=f"fallback_{i}",
                query=query,
                relevant_paper_ids=[paper.get('id', '')],
                source="fallback_generated",
                difficulty="medium"
            )
            samples.append(sample)
        
        self.generated_samples = samples
        return samples
    
    def _infer_difficulty(self, row) -> str:
        """推断样本难度"""
        # 简单启发式：根据问题长度和上下文数量
        question_len = len(row.question) if hasattr(row, 'question') else 0
        
        if question_len < 50:
            return "easy"
        elif question_len < 100:
            return "medium"
        else:
            return "hard"
    
    def export_for_annotation(self, output_path: str):
        """
        导出生成的样本供人工标注
        
        生成一个 JSON 文件，包含需要人工填写的字段
        """
        annotation_data = {
            'instructions': """
人工标注说明：
1. 检查每个 query 是否合理，如不合理请修改或删除
2. 为每个 query 填写 relevant_paper_ids（相关论文ID列表）
3. 可以调整 difficulty 字段
4. 标注完成后，将 source 改为 "ragas_modified"
5. 保存文件后，使用 load_annotated_data() 加载
            """,
            'generated_at': datetime.now().isoformat(),
            'total_samples': len(self.generated_samples),
            'samples': [asdict(s) for s in self.generated_samples]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📝 Exported {len(self.generated_samples)} samples for annotation to {output_path}")
    
    def load_annotated_data(self, path: str) -> List[EvalSample]:
        """加载人工标注后的数据"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = [EvalSample(**s) for s in data.get('samples', [])]
            
            # 过滤掉没有标注 relevant_paper_ids 的样本
            valid_samples = [s for s in samples if s.relevant_paper_ids]
            
            logger.info(f"📂 Loaded {len(valid_samples)} annotated samples (out of {len(samples)} total)")
            return valid_samples
            
        except Exception as e:
            logger.error(f"❌ Error loading annotated data: {e}")
            return []


def create_deepseek_llm():
    """创建 DeepSeek LLM 实例用于 RAGAS 评测"""
    import os
    try:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0
        )
    except ImportError:
        return None


def create_openai_llm():
    """创建 OpenAI LLM 实例用于 RAGAS 评测"""
    import os
    try:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0
        )
    except ImportError:
        return None


def get_default_llm():
    """获取默认 LLM，优先使用 DeepSeek，其次 OpenAI"""
    llm = create_deepseek_llm()
    if llm:
        logger.info("📌 使用 DeepSeek 作为 RAGAS 评测 LLM")
        return llm
    
    llm = create_openai_llm()
    if llm:
        logger.info("📌 使用 OpenAI 作为 RAGAS 评测 LLM")
        return llm
    
    return None


class RAGEvaluator:
    """
    RAG 系统评测器
    
    使用 RAGAS 框架评测 RAG 系统的整体质量
    支持 DeepSeek 和 OpenAI 作为评测 LLM
    """
    
    def __init__(self, llm=None, embeddings=None, auto_init_llm: bool = True):
        """
        初始化评测器
        
        Args:
            llm: LangChain LLM 实例，如果为 None 且 auto_init_llm=True，则自动创建
            embeddings: LangChain Embeddings 实例
            auto_init_llm: 是否自动初始化 LLM（优先 DeepSeek，其次 OpenAI）
        """
        if llm is None and auto_init_llm:
            self.llm = get_default_llm()
        else:
            self.llm = llm
        self.embeddings = embeddings
    
    def evaluate_rag_system(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict:
        """
        评测 RAG 系统
        
        Args:
            questions: 问题列表
            answers: RAG 生成的答案列表
            contexts: 检索到的上下文列表
            ground_truths: 标准答案列表（可选）
            
        Returns:
            评测结果
        """
        if not HAS_RAGAS or not HAS_DATASETS:
            logger.warning("⚠️ RAGAS or datasets not available")
            return self._fallback_evaluate(questions, answers, contexts)
        
        if self.llm is None:
            logger.warning("⚠️ LLM not configured, using fallback evaluation")
            logger.info("   设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量")
            return self._fallback_evaluate(questions, answers, contexts)
        
        # 准备数据集
        data = {
            'user_input': questions,
            'response': answers,
            'retrieved_contexts': contexts
        }
        
        # RAGAS 需要 reference 字段用于 context_precision
        if ground_truths:
            data['reference'] = ground_truths
        else:
            # 如果没有 ground truth，使用答案作为 reference
            data['reference'] = answers
        
        dataset = Dataset.from_dict(data)
        
        # 选择评测指标
        # 只使用与 DeepSeek API 兼容的指标
        metrics = []
        if context_precision:
            metrics.append(context_precision)
        if context_recall:
            metrics.append(context_recall)
        
        # faithfulness 和 answer_relevancy 与 DeepSeek API 不兼容（需要 n>1）
        # 暂时禁用这两个指标
        # if ground_truths and faithfulness:
        #     metrics.append(faithfulness)
        # if ground_truths and answer_relevancy:
        #     metrics.append(answer_relevancy)
        
        if not metrics:
            logger.warning("⚠️ 没有可用的评测指标")
            return self._fallback_evaluate(questions, answers, contexts)
        
        try:
            # 运行评测
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            return {
                'metrics': result,
                'num_samples': len(questions)
            }
            
        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {e}")
            return self._fallback_evaluate(questions, answers, contexts)
    
    def _fallback_evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]]
    ) -> Dict:
        """备用评测方法"""
        # 简单的启发式评测
        results = {
            'num_samples': len(questions),
            'avg_answer_length': np.mean([len(a) for a in answers]),
            'avg_context_count': np.mean([len(c) for c in contexts]),
            'avg_context_length': np.mean([
                np.mean([len(ctx) for ctx in c]) if c else 0 
                for c in contexts
            ])
        }
        
        return results


def create_sample_eval_data(output_path: str, num_samples: int = 20):
    """
    创建示例评测数据（用于演示）
    """
    samples = [
        EvalSample(
            query_id="demo_1",
            query="graph neural networks for battery prediction",
            relevant_paper_ids=["paper_gnn_battery_1", "paper_gnn_battery_2"],
            source="manual",
            difficulty="medium"
        ),
        EvalSample(
            query_id="demo_2",
            query="transformer models for scientific text understanding",
            relevant_paper_ids=["paper_transformer_sci_1"],
            source="manual",
            difficulty="easy"
        ),
        EvalSample(
            query_id="demo_3",
            query="如何使用深度学习预测锂电池剩余寿命",
            relevant_paper_ids=["paper_dl_battery_1", "paper_dl_battery_2", "paper_dl_battery_3"],
            source="manual",
            difficulty="medium"
        ),
        EvalSample(
            query_id="demo_4",
            query="compare LSTM and GNN for time series prediction in battery systems",
            relevant_paper_ids=["paper_lstm_ts_1", "paper_gnn_ts_1"],
            source="manual",
            difficulty="hard"
        ),
        EvalSample(
            query_id="demo_5",
            query="attention mechanism in scientific document retrieval",
            relevant_paper_ids=["paper_attention_retrieval_1"],
            source="manual",
            difficulty="medium"
        )
    ]
    
    # 扩展到指定数量
    while len(samples) < num_samples:
        base = samples[len(samples) % 5]
        new_sample = EvalSample(
            query_id=f"demo_{len(samples) + 1}",
            query=base.query + f" (variant {len(samples)})",
            relevant_paper_ids=base.relevant_paper_ids,
            source="manual",
            difficulty=base.difficulty
        )
        samples.append(new_sample)
    
    # 保存
    data = {
        'created_at': datetime.now().isoformat(),
        'description': 'Sample evaluation data for demonstration',
        'total_samples': len(samples),
        'samples': [asdict(s) for s in samples]
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📝 Created sample evaluation data with {len(samples)} samples at {output_path}")


def evaluate_bm25_retrieval(corpus_path: str, eval_path: str, k_values=(1, 3, 5)) -> Dict:
    """用 BM25 对真实语料做可复现检索评测(无网络、无 GPU)。

    corpus_path: JSON 论文语料,每篇含 id/title/abstract。
    eval_path: RetrievalEvaluator 格式的评测数据。
    """
    from rank_bm25 import BM25Okapi
    from ..utils.helpers import tokenize

    corpus = json.loads(Path(corpus_path).read_text(encoding='utf-8'))

    class _BM25Retriever:
        def __init__(self, papers):
            self.paper_ids = [p['id'] for p in papers]
            texts = [f"Title: {p['title']}\n\nAbstract: {p['abstract']}" for p in papers]
            self.index = BM25Okapi([tokenize(t) for t in texts])

        def search(self, query, top_k=5):
            scores = self.index.get_scores(tokenize(query))
            top = np.argsort(scores)[::-1][:top_k]
            return [{'paper': {'paper_id': self.paper_ids[i]}, 'score': float(scores[i])}
                    for i in top if scores[i] > 0]

    evaluator = RetrievalEvaluator(eval_path)
    return evaluator.evaluate_retriever(_BM25Retriever(corpus), k_values=list(k_values), verbose=False)


def evaluate_hybrid_retrieval(corpus_path: str, eval_path: str, k_values=(1, 3, 5), data_dir=None) -> Dict:
    """用混合检索(BM25 + BGE 稠密,不含精排)对真实语料做评测(需安装 FlagEmbedding)。"""
    import tempfile

    from .hybrid_retriever import HybridRetriever

    corpus = json.loads(Path(corpus_path).read_text(encoding='utf-8'))
    papers = [
        {'id': p['id'], 'title': p['title'], 'authors': p.get('authors', []),
         'abstract': p.get('abstract', ''), 'source': p.get('source', 'arxiv')}
        for p in corpus
    ]

    retriever = HybridRetriever(config={}, data_dir=data_dir or tempfile.mkdtemp())
    retriever.add_papers(papers, verbose=False)

    class _Wrapper:
        def __init__(self, r):
            self.r = r

        def search(self, query, top_k=5):
            return self.r.search(query, top_k=top_k, use_rerank=False, verbose=False)

    evaluator = RetrievalEvaluator(eval_path)
    return evaluator.evaluate_retriever(_Wrapper(retriever), k_values=list(k_values), verbose=False)


def evaluate_personalized_retrieval(
    corpus_path: str,
    eval_path: str,
    preferences: Optional[List[str]] = None,
    negative_preferences: Optional[List[str]] = None,
    lam: float = 0.2,
    k_values=(1, 3, 5),
    model_name: str = 'BAAI/bge-m3',
    encoder=None,
    prior_mode: str = 'boost',
) -> Dict:
    """个性化检索评测：BM25 基线 + 偏好先验回流（消融用）。

    与 :func:`evaluate_bm25_retrieval` 使用**同一套 corpus / query / 指标**，
    唯一变量是偏好先验，因此 ``lam=0`` 与 ``lam>0`` 的差值就是"记忆带来的净增益"。

    ``lam = 0`` 时**不会加载编码器**，因此基线可完全离线复现。

    Args:
        preferences: 正向偏好文本（如 ``['Battery RUL prediction', 'Bayesian deep learning']``）。
        negative_preferences: 负向偏好文本，从分数中扣除。
        lam: 先验混合权重 ``λ``。
        encoder: 复用外部编码器（避免重复加载模型）。
    """
    from rank_bm25 import BM25Okapi
    from ..utils.helpers import tokenize
    from ..memory.encoder import SemanticEncoder
    from ..memory.personalize import PreferenceProfile, preference_prior

    corpus = json.loads(Path(corpus_path).read_text(encoding='utf-8'))
    paper_ids = [p['id'] for p in corpus]
    texts = [f"Title: {p['title']}\n\nAbstract: {p.get('abstract', '')}" for p in corpus]
    index = BM25Okapi([tokenize(t) for t in texts])

    # lam=0 时跳过编码器加载，保证纯离线基线可复现
    profile = PreferenceProfile(user_id='eval')
    active_encoder = None
    if lam > 0 and preferences:
        active_encoder = encoder or SemanticEncoder(model_name=model_name)
        if not active_encoder.available:
            raise RuntimeError(
                'lam > 0 需要语义编码器，但 sentence-transformers / BAAI/bge-m3 不可用；'
                '请改用 --lam 0 或先安装依赖。'
            )
        pos = list(preferences)
        neg = list(negative_preferences or [])
        pv = active_encoder.encode(pos)
        if pv is not None and len(pv):
            v = pv.mean(axis=0)
            profile.pos_vector = v / (np.linalg.norm(v) + 1e-12)
            profile.positive_texts = pos
        if neg:
            nv = active_encoder.encode(neg)
            if nv is not None and len(nv):
                v = nv.mean(axis=0)
                profile.neg_vector = v / (np.linalg.norm(v) + 1e-12)
                profile.negative_texts = neg
        profile.backend = 'semantic' if not profile.is_empty else 'empty'

    class _PersonalizedRetriever:
        def __init__(self, lam, profile, encoder, prior_mode):
            self.lam = lam
            self.profile = profile
            self.encoder = encoder
            self.prior_mode = prior_mode
            self.last_scores = None

        def search(self, query, top_k=5):
            effective_query = query
            if self.prior_mode == 'expand' and self.lam > 0 and self.profile.positive_texts:
                # 查询侧个性化：把偏好并入查询，使"同域但方法不同"的文档被区分开。
                effective_query = query + ' ' + ' '.join(self.profile.positive_texts)
            base = np.asarray(index.get_scores(tokenize(effective_query)), dtype='float32')
            if self.prior_mode == 'boost' and self.lam > 0 and not self.profile.is_empty:
                scores = preference_prior(self.profile, texts, self.encoder, lam=self.lam,
                                          base_scores=base)
            else:
                scores = base
            self.last_scores = scores
            top = np.argsort(scores)[::-1][:top_k]
            return [{'paper': {'paper_id': paper_ids[i]}, 'score': float(scores[i])}
                    for i in top]

    evaluator = RetrievalEvaluator(eval_path)
    summary = evaluator.evaluate_retriever(
        _PersonalizedRetriever(lam, profile, active_encoder, prior_mode),
        k_values=list(k_values), verbose=False,
    )
    summary['lam'] = lam
    summary['prior_mode'] = prior_mode
    summary['profile'] = profile.summary()
    return summary


if __name__ == "__main__":
    # 创建示例评测数据
    eval_data_path = Path(__file__).parent.parent.parent / 'data' / 'eval' / 'sample_eval_data.json'
    create_sample_eval_data(str(eval_data_path))
    
    # 测试评测器
    evaluator = RetrievalEvaluator(str(eval_data_path))
    print(f"\n📊 Loaded {len(evaluator.eval_samples)} evaluation samples")
    
    # 测试 RAGAS 生成器
    generator = RAGASTestsetGenerator()
    test_papers = [
        {
            'id': 'test1',
            'title': 'Graph Neural Networks for Battery Life Prediction',
            'abstract': 'This paper presents a novel approach...'
        }
    ]
    samples = generator._fallback_generate(test_papers, 5)
    print(f"\n📝 Generated {len(samples)} test samples")
