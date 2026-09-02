"""
Academic Data Processor
学术语料数据处理器：将论文/学术文本转换为微调训练格式
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from datasets import Dataset, DatasetDict
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class TrainingSample:
    """训练样本"""
    instruction: str
    input: str
    output: str
    task_type: str  # abstract, introduction, methodology, results, polish, etc.
    source: Optional[str] = None
    metadata: Optional[Dict] = None


class AcademicDataProcessor:
    """
    学术语料数据处理器
    
    支持多种任务类型：
    - 摘要生成 (abstract)
    - 引言写作 (introduction)
    - 方法论写作 (methodology)
    - 结果描述 (results)
    - 文本润色 (polish)
    - 术语解释 (terminology)
    """
    
    # 任务模板
    TASK_TEMPLATES = {
        'abstract': {
            'instructions': [
                "请根据以下研究信息生成学术论文摘要：",
                "基于给定的研究内容，撰写一篇学术摘要：",
                "Generate an academic abstract based on the following research:",
                "请为以下研究撰写200-300字的学术摘要："
            ],
            'input_format': "标题：{title}\n关键词：{keywords}\n研究背景：{background}\n方法：{methods}\n结果：{results}\n结论：{conclusion}"
        },
        'introduction': {
            'instructions': [
                "请撰写论文引言部分：",
                "根据以下信息撰写学术论文的引言：",
                "Write an introduction section for the following research:"
            ],
            'input_format': "研究主题：{topic}\n研究问题：{research_question}\n相关工作：{related_work}\n本文贡献：{contributions}"
        },
        'methodology': {
            'instructions': [
                "请撰写方法论章节：",
                "描述以下研究方法的技术细节：",
                "Write a methodology section describing:"
            ],
            'input_format': "方法名称：{method_name}\n技术细节：{technical_details}\n算法流程：{algorithm}"
        },
        'results': {
            'instructions': [
                "请撰写实验结果章节：",
                "根据以下实验数据撰写结果分析：",
                "Write a results section based on the following data:"
            ],
            'input_format': "实验设置：{setup}\n数据集：{datasets}\n评估指标：{metrics}\n结果数据：{data}"
        },
        'polish': {
            'instructions': [
                "请润色以下学术文本，提升其专业性和流畅性：",
                "对以下学术文本进行润色和改进：",
                "Polish the following academic text to improve its professionalism:"
            ],
            'input_format': "{text}"
        },
        'terminology': {
            'instructions': [
                "请解释以下学术术语：",
                "用学术语言解释以下概念：",
                "Explain the following academic term:"
            ],
            'input_format': "术语：{term}\n领域：{field}"
        }
    }
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path('data/training')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[TrainingSample] = []
    
    def load_papers(self, papers_path: str) -> List[Dict]:
        """加载论文数据"""
        with open(papers_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        return papers if isinstance(papers, list) else papers.get('papers', [])
    
    def generate_abstract_samples(self, papers: List[Dict]) -> List[TrainingSample]:
        """从论文生成摘要写作样本"""
        samples = []
        
        for paper in papers:
            if not paper.get('abstract'):
                continue
            
            # 从摘要中提取结构化信息（简化处理）
            abstract = paper['abstract']
            title = paper.get('title', '')
            
            # 构建输入
            input_text = f"标题：{title}\n关键词：{', '.join(paper.get('keywords', []))}"
            
            # 随机选择指令
            instruction = random.choice(self.TASK_TEMPLATES['abstract']['instructions'])
            
            sample = TrainingSample(
                instruction=instruction,
                input=input_text,
                output=abstract,
                task_type='abstract',
                source=paper.get('source', 'unknown'),
                metadata={'paper_id': paper.get('id', '')}
            )
            samples.append(sample)
        
        return samples
    
    def generate_polish_samples(self, texts: List[Dict]) -> List[TrainingSample]:
        """生成文本润色样本"""
        samples = []
        
        for item in texts:
            original = item.get('original', '')
            polished = item.get('polished', '')
            
            if not original or not polished:
                continue
            
            instruction = random.choice(self.TASK_TEMPLATES['polish']['instructions'])
            
            sample = TrainingSample(
                instruction=instruction,
                input=original,
                output=polished,
                task_type='polish',
                source=item.get('source', 'manual')
            )
            samples.append(sample)
        
        return samples
    
    def generate_terminology_samples(self, terms: List[Dict]) -> List[TrainingSample]:
        """生成术语解释样本"""
        samples = []
        
        for item in terms:
            term = item.get('term', '')
            definition = item.get('definition', '')
            field = item.get('field', '计算机科学')
            
            if not term or not definition:
                continue
            
            instruction = random.choice(self.TASK_TEMPLATES['terminology']['instructions'])
            input_text = f"术语：{term}\n领域：{field}"
            
            sample = TrainingSample(
                instruction=instruction,
                input=input_text,
                output=definition,
                task_type='terminology',
                source='terminology_db'
            )
            samples.append(sample)
        
        return samples
    
    def add_samples(self, samples: List[TrainingSample]):
        """添加样本"""
        self.samples.extend(samples)
    
    def convert_to_chat_format(self, sample: TrainingSample) -> Dict:
        """转换为对话格式（适用于Qwen等模型）"""
        return {
            'messages': [
                {'role': 'system', 'content': '你是一个专业的学术写作助手，擅长撰写高质量的学术论文。'},
                {'role': 'user', 'content': f"{sample.instruction}\n\n{sample.input}"},
                {'role': 'assistant', 'content': sample.output}
            ],
            'task_type': sample.task_type
        }
    
    def convert_to_instruction_format(self, sample: TrainingSample) -> Dict:
        """转换为指令格式"""
        return {
            'instruction': sample.instruction,
            'input': sample.input,
            'output': sample.output,
            'task_type': sample.task_type
        }
    
    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List, List, List]:
        """划分数据集"""
        random.seed(seed)
        samples = self.samples.copy()
        random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        return train_samples, val_samples, test_samples
    
    def save_dataset(
        self,
        format: str = 'chat',  # 'chat' or 'instruction'
        split: bool = True
    ) -> Dict[str, str]:
        """
        保存数据集
        
        Args:
            format: 输出格式
            split: 是否划分训练/验证/测试集
            
        Returns:
            保存的文件路径
        """
        paths = {}
        
        if split:
            train, val, test = self.split_dataset()
            splits = {'train': train, 'val': val, 'test': test}
        else:
            splits = {'full': self.samples}
        
        convert_fn = self.convert_to_chat_format if format == 'chat' else self.convert_to_instruction_format
        
        for split_name, samples in splits.items():
            data = [convert_fn(s) for s in samples]
            
            output_path = self.output_dir / f'{split_name}_{format}.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            paths[split_name] = str(output_path)
            print(f"💾 Saved {len(data)} samples to {output_path}")
        
        # 保存数据集统计信息
        stats = {
            'total_samples': len(self.samples),
            'task_distribution': {},
            'format': format,
            'created_at': datetime.now().isoformat()
        }
        
        for sample in self.samples:
            task = sample.task_type
            stats['task_distribution'][task] = stats['task_distribution'].get(task, 0) + 1
        
        stats_path = self.output_dir / 'dataset_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return paths
    
    def to_hf_dataset(self, format: str = 'chat') -> Optional['DatasetDict']:
        """转换为 HuggingFace Dataset 格式"""
        if not HAS_DATASETS:
            print("⚠️ datasets library not installed")
            return None
        
        train, val, test = self.split_dataset()
        convert_fn = self.convert_to_chat_format if format == 'chat' else self.convert_to_instruction_format
        
        def samples_to_dict(samples):
            if format == 'chat':
                return {'messages': [convert_fn(s)['messages'] for s in samples]}
            else:
                data = [convert_fn(s) for s in samples]
                return {
                    'instruction': [d['instruction'] for d in data],
                    'input': [d['input'] for d in data],
                    'output': [d['output'] for d in data]
                }
        
        dataset_dict = DatasetDict({
            'train': Dataset.from_dict(samples_to_dict(train)),
            'validation': Dataset.from_dict(samples_to_dict(val)),
            'test': Dataset.from_dict(samples_to_dict(test))
        })
        
        return dataset_dict
    
    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.samples),
            'task_distribution': {},
            'avg_input_length': 0,
            'avg_output_length': 0
        }
        
        total_input_len = 0
        total_output_len = 0
        
        for sample in self.samples:
            task = sample.task_type
            stats['task_distribution'][task] = stats['task_distribution'].get(task, 0) + 1
            total_input_len += len(sample.input)
            total_output_len += len(sample.output)
        
        if self.samples:
            stats['avg_input_length'] = total_input_len / len(self.samples)
            stats['avg_output_length'] = total_output_len / len(self.samples)
        
        return stats


def create_sample_training_data(output_path: str, num_samples: int = 100):
    """创建示例训练数据"""
    samples = []
    
    # 摘要生成样本
    abstract_templates = [
        {
            'title': 'Graph Neural Networks for Battery Life Prediction',
            'abstract': 'This paper proposes a novel graph neural network (GNN) approach for predicting lithium-ion battery remaining useful life (RUL). We introduce a temporal graph attention mechanism that captures both spatial correlations between battery cells and temporal degradation patterns. Experiments on NASA battery dataset demonstrate that our method achieves 15% improvement in prediction accuracy compared to existing LSTM-based approaches.',
            'keywords': ['graph neural network', 'battery prediction', 'remaining useful life']
        },
        {
            'title': 'Transformer-based Scientific Document Understanding',
            'abstract': 'We present SciTransformer, a pre-trained language model specifically designed for scientific document understanding. Our model is trained on 10 million scientific papers and incorporates domain-specific tokenization and positional encoding. Evaluation on SciDocs benchmark shows state-of-the-art performance on citation prediction, document classification, and information extraction tasks.',
            'keywords': ['transformer', 'scientific NLP', 'document understanding']
        }
    ]
    
    for i in range(num_samples // 2):
        template = abstract_templates[i % len(abstract_templates)]
        samples.append(TrainingSample(
            instruction="请根据以下研究信息生成学术论文摘要：",
            input=f"标题：{template['title']}\n关键词：{', '.join(template['keywords'])}",
            output=template['abstract'],
            task_type='abstract',
            source='synthetic'
        ))
    
    # 润色样本
    polish_pairs = [
        {
            'original': '我们的方法比其他方法好很多。实验结果表明效果很好。',
            'polished': '实验结果表明，本文提出的方法在多个评估指标上均显著优于现有基线方法，验证了所提方法的有效性。'
        },
        {
            'original': 'This paper uses deep learning to solve the problem. The results are good.',
            'polished': 'This paper proposes a deep learning-based approach to address the aforementioned challenge. Experimental results demonstrate the effectiveness of our method, achieving significant improvements over baseline approaches.'
        }
    ]
    
    for i in range(num_samples // 2):
        pair = polish_pairs[i % len(polish_pairs)]
        samples.append(TrainingSample(
            instruction="请润色以下学术文本，提升其专业性和流畅性：",
            input=pair['original'],
            output=pair['polished'],
            task_type='polish',
            source='synthetic'
        ))
    
    # 保存
    processor = AcademicDataProcessor(output_path)
    processor.add_samples(samples)
    paths = processor.save_dataset(format='chat')
    
    print(f"📝 Created sample training data with {len(samples)} samples")
    return paths


if __name__ == "__main__":
    # 创建示例数据
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'training'
    create_sample_training_data(str(output_dir), num_samples=50)
