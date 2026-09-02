"""
Writing Evaluator
学术写作评测器：ROUGE-L、术语准确率等指标

支持：
- ROUGE-L 分数计算
- 术语准确率评测
- 学术写作质量综合评估
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("⚠️ rouge_score not installed. Run: pip install rouge-score")

try:
    import evaluate
    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False


@dataclass
class EvaluationResult:
    """评测结果"""
    rouge_l: float
    rouge_1: float
    rouge_2: float
    terminology_accuracy: float
    terminology_recall: float
    terminology_f1: float
    avg_length: float
    num_samples: int
    detailed_scores: Optional[List[Dict]] = None


class TerminologyDatabase:
    """术语数据库"""
    
    # 默认学术术语表（可扩展）
    DEFAULT_TERMS = {
        # 机器学习术语
        'machine_learning': [
            'neural network', 'deep learning', 'convolutional', 'recurrent',
            'transformer', 'attention mechanism', 'embedding', 'gradient descent',
            'backpropagation', 'overfitting', 'regularization', 'dropout',
            'batch normalization', 'learning rate', 'optimizer', 'loss function',
            'activation function', 'softmax', 'sigmoid', 'relu',
            'encoder', 'decoder', 'autoencoder', 'generative', 'discriminative',
            'supervised', 'unsupervised', 'reinforcement learning', 'fine-tuning',
            'pre-training', 'transfer learning', 'few-shot', 'zero-shot'
        ],
        # 自然语言处理术语
        'nlp': [
            'tokenization', 'word embedding', 'language model', 'bert', 'gpt',
            'named entity recognition', 'sentiment analysis', 'text classification',
            'sequence labeling', 'machine translation', 'question answering',
            'information extraction', 'text generation', 'summarization',
            'semantic similarity', 'coreference resolution'
        ],
        # 统计术语
        'statistics': [
            'mean', 'variance', 'standard deviation', 'correlation',
            'regression', 'hypothesis testing', 'p-value', 'confidence interval',
            'statistical significance', 'anova', 't-test', 'chi-square',
            'precision', 'recall', 'f1 score', 'accuracy', 'auc', 'roc'
        ],
        # 学术写作术语
        'academic_writing': [
            'methodology', 'hypothesis', 'experiment', 'evaluation',
            'baseline', 'benchmark', 'state-of-the-art', 'ablation study',
            'qualitative', 'quantitative', 'empirical', 'theoretical',
            'contribution', 'limitation', 'future work', 'related work',
            'literature review', 'citation', 'reference'
        ],
        # 中文学术术语
        'chinese_academic': [
            '神经网络', '深度学习', '卷积', '循环', '注意力机制',
            '嵌入', '梯度下降', '反向传播', '过拟合', '正则化',
            '损失函数', '优化器', '激活函数', '编码器', '解码器',
            '预训练', '微调', '迁移学习', '消融实验', '基线',
            '评估指标', '准确率', '召回率', '精确率', 'F1分数',
            '方法论', '假设', '实验', '贡献', '局限性', '未来工作'
        ]
    }
    
    def __init__(self, custom_terms: Optional[Dict[str, List[str]]] = None):
        self.terms = self.DEFAULT_TERMS.copy()
        if custom_terms:
            for category, term_list in custom_terms.items():
                if category in self.terms:
                    self.terms[category].extend(term_list)
                else:
                    self.terms[category] = term_list
        
        # 构建扁平化术语集合
        self.all_terms: Set[str] = set()
        for term_list in self.terms.values():
            self.all_terms.update(t.lower() for t in term_list)
    
    def add_terms(self, category: str, terms: List[str]):
        """添加术语"""
        if category not in self.terms:
            self.terms[category] = []
        self.terms[category].extend(terms)
        self.all_terms.update(t.lower() for t in terms)
    
    def load_from_file(self, path: str):
        """从文件加载术语"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for category, terms in data.items():
            self.add_terms(category, terms)
    
    def extract_terms_from_text(self, text: str) -> List[str]:
        """从文本中提取术语"""
        text_lower = text.lower()
        found_terms = []
        
        for term in self.all_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def get_term_count(self) -> int:
        """获取术语总数"""
        return len(self.all_terms)


class WritingEvaluator:
    """
    学术写作评测器
    
    评测指标：
    - ROUGE-L: 衡量生成文本与参考文本的重叠程度
    - 术语准确率: 衡量生成文本中学术术语的使用准确性
    - 术语召回率: 衡量参考文本中术语在生成文本中的覆盖程度
    """
    
    def __init__(self, terminology_db: Optional[TerminologyDatabase] = None):
        self.terminology_db = terminology_db or TerminologyDatabase()
        
        # 初始化 ROUGE scorer
        if HAS_ROUGE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """计算 ROUGE 分数"""
        if not HAS_ROUGE:
            return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(scores['rouge1']),
            'rouge2': np.mean(scores['rouge2']),
            'rougeL': np.mean(scores['rougeL'])
        }
    
    def compute_terminology_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算术语相关指标
        
        - 术语准确率 (Precision): 生成文本中正确使用的术语比例
        - 术语召回率 (Recall): 参考文本中术语在生成文本中的覆盖比例
        - 术语 F1
        """
        total_pred_terms = 0
        total_ref_terms = 0
        total_correct_terms = 0
        
        for pred, ref in zip(predictions, references):
            pred_terms = set(self.terminology_db.extract_terms_from_text(pred))
            ref_terms = set(self.terminology_db.extract_terms_from_text(ref))
            
            correct_terms = pred_terms & ref_terms
            
            total_pred_terms += len(pred_terms)
            total_ref_terms += len(ref_terms)
            total_correct_terms += len(correct_terms)
        
        # 计算指标
        precision = total_correct_terms / total_pred_terms if total_pred_terms > 0 else 0
        recall = total_correct_terms / total_ref_terms if total_ref_terms > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'terminology_precision': precision,
            'terminology_recall': recall,
            'terminology_f1': f1,
            'total_pred_terms': total_pred_terms,
            'total_ref_terms': total_ref_terms,
            'total_correct_terms': total_correct_terms
        }
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        compute_detailed: bool = False
    ) -> EvaluationResult:
        """
        综合评测
        
        Args:
            predictions: 模型生成的文本列表
            references: 参考文本列表
            compute_detailed: 是否计算每个样本的详细分数
            
        Returns:
            EvaluationResult
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have same length")
        
        # 计算 ROUGE
        rouge_scores = self.compute_rouge(predictions, references)
        
        # 计算术语指标
        term_metrics = self.compute_terminology_metrics(predictions, references)
        
        # 计算平均长度
        avg_length = np.mean([len(p) for p in predictions])
        
        # 详细分数（可选）
        detailed_scores = None
        if compute_detailed:
            detailed_scores = []
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                if self.rouge_scorer:
                    score = self.rouge_scorer.score(ref, pred)
                    rouge_l = score['rougeL'].fmeasure
                else:
                    rouge_l = 0
                
                pred_terms = set(self.terminology_db.extract_terms_from_text(pred))
                ref_terms = set(self.terminology_db.extract_terms_from_text(ref))
                correct = len(pred_terms & ref_terms)
                
                detailed_scores.append({
                    'index': i,
                    'rouge_l': rouge_l,
                    'pred_terms': len(pred_terms),
                    'ref_terms': len(ref_terms),
                    'correct_terms': correct,
                    'pred_length': len(pred)
                })
        
        return EvaluationResult(
            rouge_l=rouge_scores['rougeL'],
            rouge_1=rouge_scores['rouge1'],
            rouge_2=rouge_scores['rouge2'],
            terminology_accuracy=term_metrics['terminology_precision'],
            terminology_recall=term_metrics['terminology_recall'],
            terminology_f1=term_metrics['terminology_f1'],
            avg_length=avg_length,
            num_samples=len(predictions),
            detailed_scores=detailed_scores
        )
    
    def evaluate_from_file(
        self,
        predictions_path: str,
        references_path: str
    ) -> EvaluationResult:
        """从文件加载并评测"""
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        with open(references_path, 'r', encoding='utf-8') as f:
            references = json.load(f)
        
        # 处理不同格式
        if isinstance(predictions[0], dict):
            predictions = [p.get('output', p.get('text', '')) for p in predictions]
        if isinstance(references[0], dict):
            references = [r.get('output', r.get('text', '')) for r in references]
        
        return self.evaluate(predictions, references)
    
    def generate_report(
        self,
        result: EvaluationResult,
        output_path: Optional[str] = None
    ) -> str:
        """生成评测报告"""
        report = f"""
# 学术写作评测报告

## 概览
- 评测样本数: {result.num_samples}
- 平均生成长度: {result.avg_length:.1f} 字符

## ROUGE 分数
- **ROUGE-L**: {result.rouge_l:.4f} ({result.rouge_l*100:.2f}%)
- ROUGE-1: {result.rouge_1:.4f}
- ROUGE-2: {result.rouge_2:.4f}

## 术语指标
- **术语准确率 (Precision)**: {result.terminology_accuracy:.4f} ({result.terminology_accuracy*100:.2f}%)
- **术语召回率 (Recall)**: {result.terminology_recall:.4f} ({result.terminology_recall*100:.2f}%)
- 术语 F1: {result.terminology_f1:.4f}

## 评估说明
- ROUGE-L 衡量生成文本与参考文本的最长公共子序列重叠
- 术语准确率衡量生成文本中学术术语使用的正确性
- 术语召回率衡量参考文本中术语在生成文本中的覆盖程度
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Report saved to {output_path}")
        
        return report
    
    def compare_models(
        self,
        model_results: Dict[str, EvaluationResult]
    ) -> str:
        """比较多个模型的评测结果"""
        report = "# 模型对比报告\n\n"
        report += "| 模型 | ROUGE-L | 术语准确率 | 术语召回率 | 术语F1 |\n"
        report += "|------|---------|-----------|-----------|--------|\n"
        
        for model_name, result in model_results.items():
            report += f"| {model_name} | {result.rouge_l:.4f} | {result.terminology_accuracy:.4f} | {result.terminology_recall:.4f} | {result.terminology_f1:.4f} |\n"
        
        return report


def run_evaluation_pipeline(
    model_output_path: str,
    reference_path: str,
    output_dir: str,
    custom_terms_path: Optional[str] = None
) -> EvaluationResult:
    """
    运行完整评测流水线
    
    Args:
        model_output_path: 模型输出文件路径
        reference_path: 参考答案文件路径
        output_dir: 输出目录
        custom_terms_path: 自定义术语表路径
        
    Returns:
        评测结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化术语库
    term_db = TerminologyDatabase()
    if custom_terms_path:
        term_db.load_from_file(custom_terms_path)
    
    # 初始化评测器
    evaluator = WritingEvaluator(term_db)
    
    # 执行评测
    result = evaluator.evaluate_from_file(model_output_path, reference_path)
    
    # 生成报告
    report = evaluator.generate_report(
        result,
        str(output_dir / "evaluation_report.md")
    )
    
    # 保存详细结果
    result_dict = {
        'rouge_l': result.rouge_l,
        'rouge_1': result.rouge_1,
        'rouge_2': result.rouge_2,
        'terminology_accuracy': result.terminology_accuracy,
        'terminology_recall': result.terminology_recall,
        'terminology_f1': result.terminology_f1,
        'avg_length': result.avg_length,
        'num_samples': result.num_samples
    }
    
    with open(output_dir / "evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    print(report)
    return result


if __name__ == "__main__":
    # 测试评测器
    evaluator = WritingEvaluator()
    
    # 测试数据
    predictions = [
        "This paper proposes a novel deep learning approach for battery life prediction using graph neural networks. Our method achieves state-of-the-art performance on benchmark datasets.",
        "我们提出了一种基于注意力机制的神经网络模型，用于文本分类任务。实验结果表明，该方法在准确率和召回率上均优于基线方法。"
    ]
    
    references = [
        "This paper presents a deep learning method for predicting battery remaining useful life. We use graph neural networks to capture spatial-temporal patterns. Experiments show our approach outperforms existing baselines.",
        "本文提出了一种新的注意力机制神经网络，应用于文本分类。实验评估表明，我们的方法在精确率、召回率和F1分数上都取得了显著提升。"
    ]
    
    result = evaluator.evaluate(predictions, references, compute_detailed=True)
    
    print("\n📊 Evaluation Results:")
    print(f"  ROUGE-L: {result.rouge_l:.4f}")
    print(f"  Terminology Accuracy: {result.terminology_accuracy:.4f}")
    print(f"  Terminology Recall: {result.terminology_recall:.4f}")
    print(f"  Terminology F1: {result.terminology_f1:.4f}")
    
    # 生成报告
    report = evaluator.generate_report(result)
    print(report)
