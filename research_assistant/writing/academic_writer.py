"""
Academic Writer Module
学术写作辅助模块
"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path
import yaml

from ..utils.llm import create_openai_client, chat_completion


import logging
logger = logging.getLogger(__name__)

class AcademicWriter:
    """学术写作助手"""
    
    def __init__(self, config: dict, prompts: dict):
        self.config = config
        self.prompts = prompts
        self.writing_config = config.get('writing_assistant', {})
        
        # 初始化LLM
        llm_config = config.get('llm', {})
        
        self.client = create_openai_client(config)
        
        self.model = self.writing_config.get('model') or llm_config.get('model', 'deepseek-chat')
        self.temperature = self.writing_config.get('temperature', 0.8)
        self.max_length = self.writing_config.get('max_length', 2048)
    
    def generate_abstract(
        self,
        title: str,
        keywords: List[str],
        background: str,
        methods: str,
        results: str,
        conclusion: str,
        language: str = 'zh'
    ) -> str:
        """
        生成论文摘要
        
        Args:
            title: 论文标题
            keywords: 关键词
            background: 研究背景
            methods: 方法
            results: 结果
            conclusion: 结论
            language: 语言 ('zh' 或 'en')
            
        Returns:
            生成的摘要
        """
        logger.info(f"✍️ Generating abstract in {language}")
        
        prompt_key = 'abstract' if language == 'zh' else 'abstract_en'
        prompt_template = self.prompts.get('academic_writing', {}).get(prompt_key, '')
        
        prompt = prompt_template.format(
            title=title,
            keywords=', '.join(keywords),
            background=background,
            methods=methods,
            results=results,
            conclusion=conclusion
        )
        
        if not self.client:
            return self._fallback_abstract(title, keywords, background, methods, results, conclusion)
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的学术论文写作专家。" if language == 'zh' 
                                 else "You are a professional academic writer."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_length
            )
            
            abstract = response.choices[0].message.content.strip()
            return abstract
            
        except Exception as e:
            logger.error(f"❌ Error generating abstract: {e}")
            return self._fallback_abstract(title, keywords, background, methods, results, conclusion)
    
    def generate_introduction(
        self,
        topic: str,
        research_question: str,
        related_work: str,
        contributions: List[str],
        language: str = 'zh'
    ) -> str:
        """
        生成引言部分
        
        Args:
            topic: 研究主题
            research_question: 研究问题
            related_work: 相关工作
            contributions: 本文贡献
            language: 语言
            
        Returns:
            生成的引言
        """
        logger.info(f"✍️ Generating introduction in {language}")
        
        prompt_template = self.prompts.get('academic_writing', {}).get('introduction', '')
        
        prompt = prompt_template.format(
            topic=topic,
            research_question=research_question,
            related_work=related_work,
            contributions='\n'.join([f"- {c}" for c in contributions])
        )
        
        if not self.client:
            return f"# Introduction\n\n{topic}\n\n{research_question}"
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的学术论文写作专家。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_length * 2
            )
            
            introduction = response.choices[0].message.content.strip()
            return introduction
            
        except Exception as e:
            logger.error(f"❌ Error generating introduction: {e}")
            return f"# Introduction\n\n{topic}\n\n{research_question}"
    
    def generate_methodology(
        self,
        method_name: str,
        technical_details: str,
        algorithm: Optional[str] = None,
        language: str = 'zh'
    ) -> str:
        """
        生成方法论部分
        
        Args:
            method_name: 方法名称
            technical_details: 技术细节
            algorithm: 算法描述
            language: 语言
            
        Returns:
            生成的方法论
        """
        logger.info(f"✍️ Generating methodology in {language}")
        
        prompt_template = self.prompts.get('academic_writing', {}).get('methodology', '')
        
        prompt = prompt_template.format(
            method_name=method_name,
            technical_details=technical_details,
            algorithm=algorithm or "详见技术细节"
        )
        
        if not self.client:
            return f"# Methodology\n\n## {method_name}\n\n{technical_details}"
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的学术论文写作专家，擅长撰写方法论部分。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_length * 2
            )
            
            methodology = response.choices[0].message.content.strip()
            return methodology
            
        except Exception as e:
            logger.error(f"❌ Error generating methodology: {e}")
            return f"# Methodology\n\n## {method_name}\n\n{technical_details}"
    
    def generate_results(
        self,
        experimental_setup: str,
        datasets: List[str],
        metrics: List[str],
        results_data: Dict,
        language: str = 'zh'
    ) -> str:
        """
        生成实验结果部分
        
        Args:
            experimental_setup: 实验设置
            datasets: 数据集
            metrics: 评估指标
            results_data: 结果数据
            language: 语言
            
        Returns:
            生成的结果部分
        """
        logger.info(f"✍️ Generating results section in {language}")
        
        prompt_template = self.prompts.get('academic_writing', {}).get('results', '')
        
        prompt = prompt_template.format(
            experimental_setup=experimental_setup,
            datasets=', '.join(datasets),
            metrics=', '.join(metrics),
            results_data=json.dumps(results_data, ensure_ascii=False, indent=2)
        )
        
        if not self.client:
            return f"# Results\n\n{experimental_setup}\n\nDatasets: {', '.join(datasets)}"
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的学术论文写作专家，擅长撰写实验结果部分。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_length * 2
            )
            
            results = response.choices[0].message.content.strip()
            return results
            
        except Exception as e:
            logger.error(f"❌ Error generating results: {e}")
            return f"# Results\n\n{experimental_setup}"
    
    def generate_discussion(
        self,
        findings: List[str],
        interpretation: str,
        limitations: List[str],
        future_work: List[str],
        language: str = 'zh'
    ) -> str:
        """
        生成讨论部分
        
        Args:
            findings: 主要发现
            interpretation: 结果解释
            limitations: 局限性
            future_work: 未来工作
            language: 语言
            
        Returns:
            生成的讨论部分
        """
        logger.info(f"✍️ Generating discussion in {language}")
        
        prompt_template = self.prompts.get('academic_writing', {}).get('discussion', '')
        
        prompt = prompt_template.format(
            findings='\n'.join([f"- {f}" for f in findings]),
            interpretation=interpretation,
            limitations='\n'.join([f"- {l}" for l in limitations]),
            future_work='\n'.join([f"- {f}" for f in future_work])
        )
        
        if not self.client:
            return f"# Discussion\n\n{interpretation}"
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的学术论文写作专家，擅长撰写讨论部分。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_length * 2
            )
            
            discussion = response.choices[0].message.content.strip()
            return discussion
            
        except Exception as e:
            logger.error(f"❌ Error generating discussion: {e}")
            return f"# Discussion\n\n{interpretation}"
    
    def polish_text(
        self,
        original_text: str,
        language: str = 'zh'
    ) -> str:
        """
        润色学术文本
        
        Args:
            original_text: 原始文本
            language: 语言
            
        Returns:
            润色后的文本
        """
        logger.info(f"✨ Polishing text in {language}")
        
        prompt_template = self.prompts.get('academic_writing', {}).get('polish', '')
        
        prompt = prompt_template.format(original_text=original_text)
        
        if not self.client:
            return original_text
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的学术文本润色专家。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=self.max_length * 2
            )
            
            polished = response.choices[0].message.content.strip()
            return polished
            
        except Exception as e:
            logger.error(f"❌ Error polishing text: {e}")
            return original_text
    
    def translate_academic_text(
        self,
        text: str,
        source_lang: str = 'zh',
        target_lang: str = 'en'
    ) -> str:
        """
        翻译学术文本
        
        Args:
            text: 原文
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            翻译后的文本
        """
        logger.info(f"🌐 Translating from {source_lang} to {target_lang}")
        
        prompt = f"""
请将以下学术文本从{source_lang}翻译为{target_lang}，保持学术性和专业性：

{text}

要求：
1. 使用标准学术术语
2. 保持原文的逻辑结构
3. 确保翻译准确、流畅
"""
        
        if not self.client:
            return text
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的学术翻译专家。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=self.max_length * 2
            )
            
            translated = response.choices[0].message.content.strip()
            return translated
            
        except Exception as e:
            logger.error(f"❌ Error translating text: {e}")
            return text
    
    def _fallback_abstract(
        self,
        title: str,
        keywords: List[str],
        background: str,
        methods: str,
        results: str,
        conclusion: str
    ) -> str:
        """备用摘要生成"""
        return f"""
{background} {methods} {results} {conclusion}

关键词: {', '.join(keywords)}
"""
    
    def save_document(self, content: str, output_path: str, format: str = 'md'):
        """
        保存文档
        
        Args:
            content: 文档内容
            output_path: 输出路径
            format: 格式 ('md', 'txt', 'docx')
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'docx':
                try:
                    from docx import Document
                    doc = Document()
                    doc.add_paragraph(content)
                    doc.save(output_file)
                except ImportError:
                    logger.warning("⚠️ python-docx not installed. Saving as text instead.")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            logger.info(f"💾 Document saved to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving document: {e}")


if __name__ == "__main__":
    config = {
        'llm': {'model': 'gpt-4-turbo-preview'},
        'writing_assistant': {
            'model': 'gpt-4-turbo-preview',
            'temperature': 0.8,
            'max_length': 2048
        }
    }
    
    prompts_path = Path(__file__).parent.parent / 'config' / 'prompts.yaml'
    if prompts_path.exists():
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
    else:
        prompts = {}
    
    writer = AcademicWriter(config, prompts)
    
    # 测试生成摘要
    abstract = writer.generate_abstract(
        title="基于图神经网络的电池寿命预测",
        keywords=["图神经网络", "电池", "寿命预测"],
        background="电池寿命预测对于电动汽车和储能系统至关重要",
        methods="本文提出了一种基于图神经网络的预测方法",
        results="实验结果表明该方法优于传统方法",
        conclusion="本文为电池寿命预测提供了新的思路"
    )
    
    print(abstract)
