"""
Paper Retrieval Module
支持从多个学术数据库检索论文：
- arXiv (免费)
- Semantic Scholar (免费)
- OpenAlex (免费，推荐)
- CORE (需要API Key)
- Dimensions (需要API Key)
- Google Scholar (不稳定)

集成 KeyBERT 进行关键短语提取，保持多词术语完整性
"""

import os
import json
import arxiv
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

# 导入 QueryEnhancer 用于 KeyBERT 关键短语提取
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.query_enhancer import QueryEnhancer
    HAS_QUERY_ENHANCER = True
except ImportError:
    HAS_QUERY_ENHANCER = False
    QueryEnhancer = None


class PaperRetriever:
    """论文检索器，支持多个学术数据库"""
    
    def __init__(self, config: dict):
        self.config = config
        cache_path = config.get('cache_directory', 'data/papers')
        # 如果是相对路径，则相对于项目根目录
        if not Path(cache_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.cache_dir = project_root / cache_path
        else:
            self.cache_dir = Path(cache_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_results = config.get('max_results', 10)
        
        # 初始化 QueryEnhancer 用于关键短语提取
        self._query_enhancer = None
        
    def search_papers(
        self, 
        query: str, 
        sources: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        搜索论文
        
        Args:
            query: 搜索查询
            sources: 数据源列表 ['arxiv', 'google_scholar']
            max_results: 最大结果数
            verbose: 是否输出详细信息
            
        Returns:
            论文列表
        """
        if sources is None:
            sources = self.config.get('sources', ['arxiv'])
        
        if max_results is None:
            max_results = self.max_results
            
        all_papers = []
        
        if 'arxiv' in sources:
            arxiv_papers = self._search_arxiv(query, max_results, verbose)
            all_papers.extend(arxiv_papers)
            
        if 'google_scholar' in sources:
            scholar_papers = self._search_google_scholar(query, max_results, verbose)
            all_papers.extend(scholar_papers)
        
        if 'semantic_scholar' in sources:
            semantic_papers = self._search_semantic_scholar(query, max_results, verbose)
            all_papers.extend(semantic_papers)
        
        if 'openalex' in sources:
            openalex_papers = self._search_openalex(query, max_results, verbose)
            all_papers.extend(openalex_papers)
        
        if 'core' in sources:
            core_papers = self._search_core(query, max_results, verbose)
            all_papers.extend(core_papers)
        
        if 'dimensions' in sources:
            dimensions_papers = self._search_dimensions(query, max_results, verbose)
            all_papers.extend(dimensions_papers)
            
        # 去重
        all_papers = self._deduplicate_papers(all_papers)
        
        # 保存到缓存
        self._save_to_cache(query, all_papers)
        
        return all_papers
    
    @property
    def query_enhancer(self):
        """延迟加载 QueryEnhancer"""
        if self._query_enhancer is None and HAS_QUERY_ENHANCER:
            self._query_enhancer = QueryEnhancer()
        return self._query_enhancer
    
    # 通用学术术语模式（可从配置文件扩展）
    # 格式：常见的多词学术短语模式
    DEFAULT_DOMAIN_TERMS = [
        # 电池领域
        'remaining useful life', 'state of health', 'state of charge',
        'lithium ion battery', 'lithium-ion battery', 'li-ion battery',
        'battery degradation', 'capacity fade', 'cycle life',
        # 机器学习通用
        'neural network', 'deep learning', 'machine learning',
        'graph neural network', 'convolutional neural network',
        'recurrent neural network', 'long short term memory',
        'physics informed', 'transfer learning', 'attention mechanism',
        'reinforcement learning', 'natural language processing',
        'computer vision', 'object detection', 'image classification',
        'time series', 'anomaly detection', 'feature extraction',
        # 其他常见学术短语
        'case study', 'literature review', 'systematic review',
        'experimental results', 'comparative analysis',
    ]
    
    def _extract_keyphrases_for_search(self, query: str) -> List[str]:
        """
        提取关键短语用于搜索（通用方法，适用于任何领域）
        
        策略：
        1. 匹配预定义的常见学术短语
        2. 使用 KeyBERT 提取语义关键短语
        3. 智能合并，优先保留多词短语
        """
        query_lower = query.lower()
        result = []
        
        # 获取用户自定义术语（从配置文件）
        custom_terms = self.config.get('domain_terms', [])
        all_terms = custom_terms + self.DEFAULT_DOMAIN_TERMS
        
        # 1. 匹配预定义的多词短语
        for term in all_terms:
            if term in query_lower:
                result.append(term)
        
        # 2. 使用 KeyBERT 提取关键短语
        if self.query_enhancer:
            keyphrases = self.query_enhancer.extract_keyphrases(
                query,
                top_n=6,
                keyphrase_ngram_range=(1, 3),  # 支持1-3词
                use_mmr=True,
                diversity=0.4
            )
            
            for phrase, score in keyphrases:
                phrase_lower = phrase.lower()
                # 检查是否与已有结果重复或是子集
                is_duplicate = any(
                    phrase_lower in r.lower() or r.lower() in phrase_lower 
                    for r in result
                )
                if not is_duplicate and score > 0.25:
                    # 优先保留多词短语（更精确）
                    word_count = len(phrase.split())
                    if word_count >= 2 or score > 0.4:
                        result.append(phrase)
                
                if len(result) >= 4:
                    break
        
        # 3. 如果结果太少，从查询中提取重要单词
        if len(result) < 2:
            # 停用词列表
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are',
                         'was', 'were', 'be', 'using', 'based', 'how', 'what'}
            words = [t for t in query.split() if len(t) > 3 and t.lower() not in stop_words]
            for w in words:
                if w.lower() not in [r.lower() for r in result]:
                    result.append(w)
                if len(result) >= 3:
                    break
        
        # 按词数排序，多词短语优先
        result.sort(key=lambda x: len(x.split()), reverse=True)
        
        return result[:4]
    
    def _search_arxiv(self, query: str, max_results: int, verbose: bool = True) -> List[Dict]:
        """搜索arXiv，使用 KeyBERT 提取的关键短语"""
        papers = []
        
        try:
            # 配置arXiv客户端
            client = arxiv.Client()
            
            # 使用 KeyBERT 提取关键短语（保持多词术语完整性）
            keyphrases = self._extract_keyphrases_for_search(query)
            
            if keyphrases:
                # 构建 arXiv 查询策略：
                # - 第一个短语（最重要）必须匹配
                # - 其他短语用 OR 连接，提高召回率
                query_parts = []
                for i, phrase in enumerate(keyphrases[:4]):
                    if ' ' in phrase:
                        # 多词短语用引号包裹
                        query_parts.append(f'all:"{phrase}"')
                    else:
                        query_parts.append(f'all:{phrase}')
                
                if len(query_parts) == 1:
                    arxiv_query = query_parts[0]
                elif len(query_parts) >= 2:
                    # 第一个短语 AND (其他短语 OR 连接)
                    primary = query_parts[0]
                    secondary = ' OR '.join(query_parts[1:])
                    arxiv_query = f'{primary} AND ({secondary})'
                else:
                    arxiv_query = query
            else:
                arxiv_query = query
            
            search = arxiv.Search(
                query=arxiv_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = client.results(search)
            
            for result in tqdm(results, desc="Fetching arXiv papers", total=max_results, disable=not verbose):
                paper = {
                    'id': result.entry_id,
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'published': result.published.isoformat(),
                    'updated': result.updated.isoformat(),
                    'pdf_url': result.pdf_url,
                    'categories': result.categories,
                    'primary_category': result.primary_category,
                    'source': 'arxiv',
                    'doi': result.doi,
                    'journal_ref': result.journal_ref,
                    'comment': result.comment
                }
                
                # 下载PDF（如果配置启用）
                if self.config.get('download_pdf', False):
                    pdf_path = self._download_pdf(result, paper['id'])
                    paper['local_pdf_path'] = str(pdf_path) if pdf_path else None
                    
                papers.append(paper)
                
        except Exception as e:
            pass
            
        return papers
    
    def _search_google_scholar(self, query: str, max_results: int, verbose: bool = True) -> List[Dict]:
        """
        搜索Google Scholar
        注意：Google Scholar没有官方API，这里提供基础实现
        实际使用建议使用scholarly库或SerpAPI
        """
        papers = []
        
        try:
            # 使用scholarly库（需要安装）
            from scholarly import scholarly
            
            search_query = scholarly.search_pubs(query)
            
            for i, result in enumerate(tqdm(search_query, desc="Fetching Google Scholar papers", total=max_results, disable=not verbose)):
                if i >= max_results:
                    break
                    
                paper = {
                    'id': result.get('pub_url', f"scholar_{i}"),
                    'title': result.get('bib', {}).get('title', ''),
                    'authors': result.get('bib', {}).get('author', []),
                    'abstract': result.get('bib', {}).get('abstract', ''),
                    'published': result.get('bib', {}).get('pub_year', ''),
                    'venue': result.get('bib', {}).get('venue', ''),
                    'citations': result.get('num_citations', 0),
                    'url': result.get('pub_url', ''),
                    'source': 'google_scholar'
                }
                
                papers.append(paper)
                time.sleep(1)  # 避免请求过快
                
        except ImportError:
            pass
        except Exception as e:
            pass
            
        return papers
    
    def _search_semantic_scholar(self, query: str, max_results: int, verbose: bool = True) -> List[Dict]:
        """
        搜索Semantic Scholar
        使用官方API，免费且稳定
        """
        papers = []
        
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar()
            
            # 搜索论文
            results = sch.search_paper(query, limit=max_results)
            
            for i, result in enumerate(tqdm(results, desc="Fetching Semantic Scholar papers", total=max_results, disable=not verbose)):
                if i >= max_results:
                    break
                
                # 提取论文信息
                paper = {
                    'id': result.paperId or f"s2_{i}",
                    'title': result.title or '',
                    'authors': [author.name for author in (result.authors or [])],
                    'abstract': result.abstract or '',
                    'published': str(result.year) if result.year else '',
                    'venue': result.venue or '',
                    'citations': result.citationCount or 0,
                    'url': result.url or '',
                    'doi': result.externalIds.get('DOI', '') if result.externalIds else '',
                    'source': 'semantic_scholar',
                    'influential_citation_count': result.influentialCitationCount or 0,
                    'is_open_access': result.isOpenAccess or False,
                    'fields_of_study': result.fieldsOfStudy or []
                }
                
                papers.append(paper)
                time.sleep(0.1)  # 避免请求过快
                
        except ImportError:
            pass
        except Exception as e:
            pass
            
        return papers
    
    def _search_openalex(self, query: str, max_results: int, verbose: bool = True) -> List[Dict]:
        """
        搜索 OpenAlex (免费，无需API Key，推荐)
        API文档: https://docs.openalex.org/api-entities/works/search-works
        """
        papers = []
        
        try:
            # OpenAlex API endpoint
            base_url = "https://api.openalex.org/works"
            
            # 构建请求参数
            params = {
                'search': query,
                'per_page': min(max_results, 50),  # OpenAlex 每页最多50条
                'sort': 'relevance_score:desc',
                'mailto': self.config.get('openalex', {}).get('email', 'research@example.com')
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            
            for i, result in enumerate(tqdm(results[:max_results], desc="Fetching OpenAlex papers", disable=not verbose)):
                # 提取作者
                authors = []
                for authorship in result.get('authorships', []):
                    author = authorship.get('author', {})
                    if author.get('display_name'):
                        authors.append(author['display_name'])
                
                # 提取论文信息
                paper = {
                    'id': result.get('id', f"openalex_{i}"),
                    'title': result.get('title', ''),
                    'authors': authors,
                    'abstract': self._reconstruct_abstract(result.get('abstract_inverted_index', {})),
                    'published': str(result.get('publication_year', '')),
                    'venue': result.get('primary_location', {}).get('source', {}).get('display_name', '') if result.get('primary_location') else '',
                    'citations': result.get('cited_by_count', 0),
                    'url': result.get('doi', '') or result.get('id', ''),
                    'doi': result.get('doi', '').replace('https://doi.org/', '') if result.get('doi') else '',
                    'source': 'openalex',
                    'is_open_access': result.get('open_access', {}).get('is_oa', False),
                    'pdf_url': result.get('open_access', {}).get('oa_url', ''),
                    'type': result.get('type', ''),
                    'concepts': [c.get('display_name', '') for c in result.get('concepts', [])[:5]]
                }
                
                papers.append(paper)
                
        except requests.exceptions.RequestException as e:
            pass
        except Exception as e:
            pass
            
        return papers
    
    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        """从 OpenAlex 的倒排索引重建摘要文本"""
        if not inverted_index:
            return ''
        
        # 重建文本
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        word_positions.sort(key=lambda x: x[0])
        return ' '.join([word for _, word in word_positions])
    
    def _search_core(self, query: str, max_results: int, verbose: bool = True) -> List[Dict]:
        """
        搜索 CORE (需要API Key)
        API文档: https://core.ac.uk/documentation/api
        注册获取API Key: https://core.ac.uk/services/api
        """
        papers = []
        
        # 获取 API Key
        api_key = self.config.get('core', {}).get('api_key') or os.environ.get('CORE_API_KEY')
        
        if not api_key:
            return papers
        
        try:
            # CORE API v3 endpoint
            base_url = "https://api.core.ac.uk/v3/search/works"
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'q': query,
                'limit': min(max_results, 100)
            }
            
            response = requests.get(base_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            
            for i, result in enumerate(tqdm(results[:max_results], desc="Fetching CORE papers", disable=not verbose)):
                # 提取作者
                authors = []
                for author in result.get('authors', []):
                    if isinstance(author, dict):
                        authors.append(author.get('name', ''))
                    elif isinstance(author, str):
                        authors.append(author)
                
                paper = {
                    'id': result.get('id', f"core_{i}"),
                    'title': result.get('title', ''),
                    'authors': authors,
                    'abstract': result.get('abstract', ''),
                    'published': str(result.get('yearPublished', '')),
                    'venue': result.get('publisher', '') or result.get('journals', [{}])[0].get('title', '') if result.get('journals') else '',
                    'citations': result.get('citationCount', 0),
                    'url': result.get('downloadUrl', '') or result.get('sourceFulltextUrls', [''])[0] if result.get('sourceFulltextUrls') else '',
                    'doi': result.get('doi', ''),
                    'source': 'core',
                    'is_open_access': True,  # CORE 专注于开放获取
                    'pdf_url': result.get('downloadUrl', ''),
                    'language': result.get('language', {}).get('code', '') if isinstance(result.get('language'), dict) else result.get('language', '')
                }
                
                papers.append(paper)
                time.sleep(0.1)  # 遵守速率限制
                
        except requests.exceptions.RequestException as e:
            pass
        except Exception as e:
            pass
            
        return papers
    
    def _search_dimensions(self, query: str, max_results: int, verbose: bool = True) -> List[Dict]:
        """
        搜索 Dimensions (需要API Key)
        API文档: https://docs.dimensions.ai/dsl/
        注册: https://app.dimensions.ai/
        """
        papers = []
        
        # 获取 API Key 或 Token
        api_key = self.config.get('dimensions', {}).get('api_key') or os.environ.get('DIMENSIONS_API_KEY')
        
        if not api_key:
            return papers
        
        try:
            # Dimensions DSL API
            base_url = "https://app.dimensions.ai/api/dsl.json"
            
            headers = {
                'Authorization': f'JWT {api_key}',
                'Content-Type': 'application/json'
            }
            
            # DSL 查询语法
            dsl_query = f'''
            search publications for "{query}"
            return publications[id, title, authors, abstract, year, journal, doi, open_access, times_cited]
            limit {min(max_results, 50)}
            '''
            
            response = requests.post(
                base_url,
                headers=headers,
                json={'query': dsl_query},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get('publications', [])
            
            for i, result in enumerate(tqdm(results[:max_results], desc="Fetching Dimensions papers", disable=not verbose)):
                # 提取作者
                authors = []
                for author in result.get('authors', []):
                    if isinstance(author, dict):
                        first = author.get('first_name', '')
                        last = author.get('last_name', '')
                        authors.append(f"{first} {last}".strip())
                    elif isinstance(author, str):
                        authors.append(author)
                
                paper = {
                    'id': result.get('id', f"dimensions_{i}"),
                    'title': result.get('title', ''),
                    'authors': authors,
                    'abstract': result.get('abstract', ''),
                    'published': str(result.get('year', '')),
                    'venue': result.get('journal', {}).get('title', '') if isinstance(result.get('journal'), dict) else result.get('journal', ''),
                    'citations': result.get('times_cited', 0),
                    'url': f"https://doi.org/{result.get('doi', '')}" if result.get('doi') else '',
                    'doi': result.get('doi', ''),
                    'source': 'dimensions',
                    'is_open_access': result.get('open_access', []) != [],
                }
                
                papers.append(paper)
                
        except requests.exceptions.RequestException as e:
            pass
        except Exception as e:
            pass
            
        return papers
    
    def _download_pdf(self, result: arxiv.Result, paper_id: str) -> Optional[Path]:
        """下载PDF文件"""
        try:
            # 清理文件名
            safe_id = paper_id.split('/')[-1].replace(':', '_')
            pdf_path = self.cache_dir / f"{safe_id}.pdf"
            
            if pdf_path.exists():
                return pdf_path
                
            result.download_pdf(dirpath=str(self.cache_dir), filename=f"{safe_id}.pdf")
            return pdf_path
            
        except Exception as e:
            return None
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """根据标题去重"""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
                
        return unique_papers
    
    def _save_to_cache(self, query: str, papers: List[Dict]):
        """保存搜索结果到缓存"""
        try:
            cache_file = self.cache_dir / f"search_{hash(query)}.json"
            cache_data = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'papers': papers
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            pass
    
    def load_from_cache(self, query: str) -> Optional[List[Dict]]:
        """从缓存加载搜索结果"""
        try:
            cache_file = self.cache_dir / f"search_{hash(query)}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # 检查缓存是否过期（24小时）
                timestamp = datetime.fromisoformat(cache_data['timestamp'])
                if (datetime.now() - timestamp).total_seconds() < 86400:
                    return cache_data['papers']
                    
        except Exception as e:
            pass
            
        return None
    
    def get_paper_details(self, paper_id: str, source: str = 'arxiv') -> Optional[Dict]:
        """获取论文详细信息"""
        if source == 'arxiv':
            return self._get_arxiv_details(paper_id)
        else:
            return None
    
    def _get_arxiv_details(self, paper_id: str) -> Optional[Dict]:
        """获取arXiv论文详细信息"""
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            result = next(client.results(search))
            
            return {
                'id': result.entry_id,
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'published': result.published.isoformat(),
                'updated': result.updated.isoformat(),
                'pdf_url': result.pdf_url,
                'categories': result.categories,
                'primary_category': result.primary_category,
                'source': 'arxiv'
            }
            
        except Exception as e:
            return None


if __name__ == "__main__":
    # 测试代码
    config = {
        'cache_directory': './data/papers',
        'max_results': 5,
        'download_pdf': False,
        'sources': ['arxiv']
    }
    
    retriever = PaperRetriever(config)
    papers = retriever.search_papers("graph neural networks battery prediction")
    
    print(f"\n✅ Found {len(papers)} papers")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Published: {paper['published']}")
