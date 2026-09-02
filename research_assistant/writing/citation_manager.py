"""
Citation Manager Module
引用和参考文献管理
"""

from typing import Dict, List, Optional
import re
from pathlib import Path
import json


class CitationManager:
    """引用管理器"""
    
    def __init__(self, style: str = 'ieee'):
        """
        初始化引用管理器
        
        Args:
            style: 引用格式 ('ieee', 'apa', 'mla', 'chicago')
        """
        self.style = style
        self.citations = []
        self.citation_counter = 1
    
    def add_citation(self, paper: Dict) -> str:
        """
        添加引用
        
        Args:
            paper: 论文信息
            
        Returns:
            引用标记
        """
        # 检查是否已存在
        for i, cite in enumerate(self.citations, 1):
            if cite.get('id') == paper.get('id'):
                return self._format_citation_mark(i)
        
        # 添加新引用
        self.citations.append(paper)
        citation_mark = self._format_citation_mark(self.citation_counter)
        self.citation_counter += 1
        
        return citation_mark
    
    def _format_citation_mark(self, number: int) -> str:
        """格式化引用标记"""
        if self.style == 'ieee':
            return f"[{number}]"
        elif self.style == 'apa':
            return f"({number})"
        elif self.style == 'mla':
            return f"({number})"
        else:
            return f"[{number}]"
    
    def format_reference(self, paper: Dict) -> str:
        """
        格式化单个参考文献
        
        Args:
            paper: 论文信息
            
        Returns:
            格式化的参考文献
        """
        if self.style == 'ieee':
            return self._format_ieee(paper)
        elif self.style == 'apa':
            return self._format_apa(paper)
        elif self.style == 'mla':
            return self._format_mla(paper)
        elif self.style == 'chicago':
            return self._format_chicago(paper)
        else:
            return self._format_ieee(paper)
    
    def _format_ieee(self, paper: Dict) -> str:
        """IEEE格式"""
        authors = paper.get('authors', [])
        
        # 格式化作者
        if len(authors) == 0:
            author_str = "Anonymous"
        elif len(authors) <= 3:
            author_str = ', '.join([self._format_author_ieee(a) for a in authors])
        else:
            author_str = f"{self._format_author_ieee(authors[0])} et al."
        
        title = paper.get('title', 'Untitled')
        year = paper.get('published', '')[:4] if paper.get('published') else 'n.d.'
        
        # 期刊或会议
        venue = paper.get('journal_ref') or paper.get('venue', '')
        
        if venue:
            return f'{author_str}, "{title}," {venue}, {year}.'
        else:
            return f'{author_str}, "{title}," {year}.'
    
    def _format_apa(self, paper: Dict) -> str:
        """APA格式"""
        authors = paper.get('authors', [])
        
        # 格式化作者
        if len(authors) == 0:
            author_str = "Anonymous"
        elif len(authors) <= 7:
            author_str = ', '.join([self._format_author_apa(a) for a in authors[:-1]])
            author_str += f", & {self._format_author_apa(authors[-1])}"
        else:
            author_str = ', '.join([self._format_author_apa(a) for a in authors[:6]])
            author_str += ", ... " + self._format_author_apa(authors[-1])
        
        title = paper.get('title', 'Untitled')
        year = paper.get('published', '')[:4] if paper.get('published') else 'n.d.'
        venue = paper.get('journal_ref') or paper.get('venue', 'Unpublished')
        
        return f'{author_str} ({year}). {title}. {venue}.'
    
    def _format_mla(self, paper: Dict) -> str:
        """MLA格式"""
        authors = paper.get('authors', [])
        
        if len(authors) == 0:
            author_str = "Anonymous"
        elif len(authors) == 1:
            author_str = self._format_author_mla(authors[0])
        elif len(authors) == 2:
            author_str = f"{self._format_author_mla(authors[0])} and {authors[1]}"
        else:
            author_str = f"{self._format_author_mla(authors[0])} et al."
        
        title = paper.get('title', 'Untitled')
        venue = paper.get('journal_ref') or paper.get('venue', '')
        year = paper.get('published', '')[:4] if paper.get('published') else 'n.d.'
        
        if venue:
            return f'{author_str}. "{title}." {venue}, {year}.'
        else:
            return f'{author_str}. "{title}." {year}.'
    
    def _format_chicago(self, paper: Dict) -> str:
        """Chicago格式"""
        authors = paper.get('authors', [])
        
        if len(authors) == 0:
            author_str = "Anonymous"
        elif len(authors) <= 3:
            author_str = ', '.join(authors)
        else:
            author_str = f"{authors[0]} et al."
        
        title = paper.get('title', 'Untitled')
        year = paper.get('published', '')[:4] if paper.get('published') else 'n.d.'
        venue = paper.get('journal_ref') or paper.get('venue', '')
        
        if venue:
            return f'{author_str}. {year}. "{title}." {venue}.'
        else:
            return f'{author_str}. {year}. "{title}."'
    
    def _format_author_ieee(self, author: str) -> str:
        """IEEE作者格式: F. Lastname"""
        parts = author.split()
        if len(parts) == 1:
            return parts[0]
        else:
            initials = '. '.join([p[0] for p in parts[:-1]]) + '.'
            return f"{initials} {parts[-1]}"
    
    def _format_author_apa(self, author: str) -> str:
        """APA作者格式: Lastname, F."""
        parts = author.split()
        if len(parts) == 1:
            return parts[0]
        else:
            initials = '. '.join([p[0] for p in parts[:-1]]) + '.'
            return f"{parts[-1]}, {initials}"
    
    def _format_author_mla(self, author: str) -> str:
        """MLA作者格式: Lastname, Firstname"""
        parts = author.split()
        if len(parts) == 1:
            return parts[0]
        else:
            return f"{parts[-1]}, {' '.join(parts[:-1])}"
    
    def generate_bibliography(self) -> str:
        """
        生成参考文献列表
        
        Returns:
            格式化的参考文献列表
        """
        if not self.citations:
            return "No citations added."
        
        bibliography = "# References\n\n" if self.style != 'ieee' else "# REFERENCES\n\n"
        
        for i, paper in enumerate(self.citations, 1):
            ref = self.format_reference(paper)
            if self.style == 'ieee':
                bibliography += f"[{i}] {ref}\n\n"
            else:
                bibliography += f"{ref}\n\n"
        
        return bibliography
    
    def insert_citations_in_text(self, text: str, citation_map: Dict[str, str]) -> str:
        """
        在文本中插入引用
        
        Args:
            text: 原始文本
            citation_map: 引用映射 {placeholder: paper_id}
            
        Returns:
            插入引用后的文本
        """
        for placeholder, paper_id in citation_map.items():
            # 查找对应的引用编号
            for i, cite in enumerate(self.citations, 1):
                if cite.get('id') == paper_id:
                    citation_mark = self._format_citation_mark(i)
                    text = text.replace(placeholder, citation_mark)
                    break
        
        return text
    
    def export_bibtex(self, output_path: str):
        """
        导出为BibTeX格式
        
        Args:
            output_path: 输出文件路径
        """
        bibtex_entries = []
        
        for i, paper in enumerate(self.citations, 1):
            entry_type = "article"
            cite_key = f"ref{i}"
            
            authors = ' and '.join(paper.get('authors', ['Anonymous']))
            title = paper.get('title', 'Untitled')
            year = paper.get('published', '')[:4] if paper.get('published') else ''
            journal = paper.get('journal_ref', '')
            
            entry = f"""@{entry_type}{{{cite_key},
  author = {{{authors}}},
  title = {{{title}}},
  year = {{{year}}},
  journal = {{{journal}}}
}}"""
            
            bibtex_entries.append(entry)
        
        bibtex_content = '\n\n'.join(bibtex_entries)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(bibtex_content)
            print(f"📚 BibTeX exported to {output_path}")
        except Exception as e:
            print(f"❌ Error exporting BibTeX: {e}")
    
    def save_citations(self, output_path: str):
        """
        保存引用数据
        
        Args:
            output_path: 输出文件路径
        """
        try:
            data = {
                'style': self.style,
                'citations': self.citations
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Citations saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving citations: {e}")
    
    def load_citations(self, input_path: str):
        """
        加载引用数据
        
        Args:
            input_path: 输入文件路径
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.style = data.get('style', 'ieee')
            self.citations = data.get('citations', [])
            self.citation_counter = len(self.citations) + 1
            
            print(f"📂 Loaded {len(self.citations)} citations")
            
        except Exception as e:
            print(f"❌ Error loading citations: {e}")
    
    def clear_citations(self):
        """清空所有引用"""
        self.citations = []
        self.citation_counter = 1
        print("🗑️ Citations cleared")


if __name__ == "__main__":
    # 测试代码
    manager = CitationManager(style='ieee')
    
    # 添加引用
    paper1 = {
        'id': 'paper1',
        'title': 'Graph Neural Networks for Battery Prediction',
        'authors': ['John Doe', 'Jane Smith'],
        'published': '2024-01-01',
        'journal_ref': 'IEEE Transactions on Neural Networks'
    }
    
    paper2 = {
        'id': 'paper2',
        'title': 'Deep Learning in Energy Systems',
        'authors': ['Alice Johnson', 'Bob Williams', 'Charlie Brown'],
        'published': '2023-06-15',
        'venue': 'NeurIPS 2023'
    }
    
    cite1 = manager.add_citation(paper1)
    cite2 = manager.add_citation(paper2)
    
    print(f"Citation 1: {cite1}")
    print(f"Citation 2: {cite2}")
    
    # 生成参考文献
    bibliography = manager.generate_bibliography()
    print("\n" + bibliography)
