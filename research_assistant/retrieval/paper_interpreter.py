"""
Paper Interpreter Module
使用LLM解读和分析学术论文
"""

import os
import yaml
from typing import Dict, List, Optional
from pathlib import Path
import json

from ..utils.llm import create_openai_client


class PaperInterpreter:
    """论文解读器，使用LLM分析论文内容"""
    
    def __init__(self, config: dict, prompts: dict):
        self.config = config
        self.prompts = prompts
        
        # 初始化LLM客户端
        llm_config = config.get('llm', {})
        
        self.client = create_openai_client(config)
        
        self.model = llm_config.get('model', 'deepseek-chat')
        self.temperature = llm_config.get('temperature', 0.7)
        self.max_tokens = llm_config.get('max_tokens', 4096)
    
    def interpret_paper(self, paper: Dict) -> Dict:
        """
        全面解读论文
        
        Args:
            paper: 论文信息字典
            
        Returns:
            解读结果
        """
        print(f"📖 Interpreting paper: {paper.get('title', 'Unknown')}")
        
        interpretation = {
            'paper_id': paper.get('id', ''),
            'title': paper.get('title', ''),
            'summary': self.summarize_paper(paper),
            'key_findings': self.extract_key_findings(paper),
            'methodology': self.extract_methodology(paper),
            'contributions': self.extract_contributions(paper),
            'limitations': self.extract_limitations(paper)
        }
        
        return interpretation
    
    def summarize_paper(self, paper: Dict) -> str:
        """生成论文摘要"""
        if not self.client:
            return self._fallback_summary(paper)
        
        prompt_template = self.prompts.get('paper_interpretation', {}).get('summary', '')
        
        prompt = prompt_template.format(
            title=paper.get('title', ''),
            authors=', '.join(paper.get('authors', [])[:5]),
            abstract=paper.get('abstract', '')
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的学术论文分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return self._fallback_summary(paper)
    
    def extract_key_findings(self, paper: Dict) -> List[str]:
        """提取关键发现"""
        if not self.client:
            return []
        
        prompt_template = self.prompts.get('paper_interpretation', {}).get('key_findings', '')
        
        content = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}"
        prompt = prompt_template.format(content=content)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的学术论文分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2048
            )
            
            # 解析返回的列表
            findings_text = response.choices[0].message.content.strip()
            findings = [line.strip('- ').strip() for line in findings_text.split('\n') if line.strip()]
            
            return findings
            
        except Exception as e:
            print(f"❌ Error extracting key findings: {e}")
            return []
    
    def extract_methodology(self, paper: Dict) -> Dict:
        """提取方法论"""
        if not self.client:
            return {}
        
        prompt_template = self.prompts.get('paper_interpretation', {}).get('methodology_extract', '')
        
        content = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}"
        prompt = prompt_template.format(content=content)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的学术论文分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2048
            )
            
            methodology_text = response.choices[0].message.content.strip()
            
            return {
                'description': methodology_text,
                'extracted_from': 'abstract'
            }
            
        except Exception as e:
            print(f"❌ Error extracting methodology: {e}")
            return {}
    
    def extract_contributions(self, paper: Dict) -> List[str]:
        """提取主要贡献"""
        # 简单实现：从摘要中提取
        abstract = paper.get('abstract', '')
        
        contributions = []
        
        # 查找常见的贡献关键词
        keywords = ['propose', 'present', 'introduce', 'develop', 'demonstrate', 
                   'show', 'achieve', 'improve', 'novel', 'new']
        
        sentences = abstract.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                contributions.append(sentence.strip())
        
        return contributions[:3]  # 返回前3个
    
    def extract_limitations(self, paper: Dict) -> List[str]:
        """提取局限性"""
        # 简单实现：查找常见的局限性关键词
        abstract = paper.get('abstract', '')
        
        limitations = []
        
        keywords = ['limitation', 'challenge', 'future work', 'however', 'but',
                   'drawback', 'constraint', 'restrict']
        
        sentences = abstract.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                limitations.append(sentence.strip())
        
        return limitations
    
    def compare_papers(self, papers: List[Dict]) -> str:
        """比较多篇论文"""
        if not self.client:
            return "LLM client not available for paper comparison."
        
        prompt_template = self.prompts.get('literature_review', {}).get('comparison', '')
        
        # 构建论文列表文本
        papers_text = []
        for i, paper in enumerate(papers, 1):
            papers_text.append(
                f"论文 {i}:\n"
                f"标题: {paper.get('title', '')}\n"
                f"作者: {', '.join(paper.get('authors', [])[:3])}\n"
                f"摘要: {paper.get('abstract', '')}\n"
            )
        
        prompt = prompt_template.format(papers_to_compare='\n---\n'.join(papers_text))
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的学术论文分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error comparing papers: {e}")
            return "Error occurred during paper comparison."
    
    def generate_literature_review(self, papers: List[Dict]) -> str:
        """生成文献综述"""
        if not self.client:
            return "LLM client not available for literature review."
        
        prompt_template = self.prompts.get('literature_review', {}).get('synthesis', '')
        
        # 构建论文列表
        papers_text = []
        for i, paper in enumerate(papers, 1):
            papers_text.append(
                f"{i}. {paper.get('title', '')} ({paper.get('published', '')[:4]})\n"
                f"   作者: {', '.join(paper.get('authors', [])[:3])}\n"
                f"   摘要: {paper.get('abstract', '')[:300]}...\n"
            )
        
        prompt = prompt_template.format(papers_list='\n'.join(papers_text))
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的学术论文分析专家，擅长撰写文献综述。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating literature review: {e}")
            return "Error occurred during literature review generation."
    
    def _fallback_summary(self, paper: Dict) -> str:
        """备用摘要生成（不使用LLM）"""
        title = paper.get('title', 'Unknown')
        authors = ', '.join(paper.get('authors', [])[:3])
        abstract = paper.get('abstract', '')[:500]
        
        return f"论文标题: {title}\n作者: {authors}\n\n摘要:\n{abstract}..."
    
    def save_interpretation(self, interpretation: Dict, output_path: str):
        """保存解读结果"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(interpretation, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Saved interpretation to {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving interpretation: {e}")


if __name__ == "__main__":
    # 测试代码
    config = {
        'llm': {
            'model': 'gpt-4-turbo-preview',
            'temperature': 0.7,
            'max_tokens': 4096
        }
    }
    
    # 加载prompts
    prompts_path = Path(__file__).parent.parent / 'config' / 'prompts.yaml'
    if prompts_path.exists():
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
    else:
        prompts = {}
    
    interpreter = PaperInterpreter(config, prompts)
    
    # 测试论文
    test_paper = {
        'id': 'test1',
        'title': 'Graph Neural Networks for Battery Life Prediction',
        'authors': ['Author A', 'Author B', 'Author C'],
        'abstract': 'This paper proposes a novel graph neural network approach for predicting battery lifetime...',
        'published': '2024-01-01'
    }
    
    interpretation = interpreter.interpret_paper(test_paper)
    print(json.dumps(interpretation, ensure_ascii=False, indent=2))
