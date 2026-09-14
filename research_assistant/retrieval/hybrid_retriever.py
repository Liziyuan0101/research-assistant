"""
Hybrid Retrieval System
混合检索系统：BM25稀疏检索 + BGE-M3稠密向量检索 + BGE-Reranker二阶段精排

三层存储架构：
- 索引层 (Index): FAISS HNSW索引存储BGE-M3向量 (~5-10GB for 100k papers)
- 元数据层 (Metadata): SQLite存储标题、摘要、DOI、PDF远程链接 (~1GB)
- 缓存层 (Cache): 本地临时文件夹存储被检索到的PDF全文 (动态变化)
"""

import os
import json
import sqlite3
import pickle
import hashlib
import tempfile
import shutil
import warnings
import logging
from pathlib import Path

# 抑制警告和日志
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from ..utils.helpers import resolve_device, tokenize
from tqdm import tqdm

# FAISS
import faiss

# BM25
from rank_bm25 import BM25Okapi

# BGE Models
try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    HAS_BGE = True
except ImportError:
    HAS_BGE = False
    logger.warning("⚠️ FlagEmbedding not installed. Run: pip install FlagEmbedding")

# PDF Processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# 导入PDF处理器
try:
    from .pdf_markdown_processor import PDFMarkdownProcessor
    HAS_PDF_MARKDOWN = True
except ImportError:
    HAS_PDF_MARKDOWN = False


@dataclass
class PaperMetadata:
    """论文元数据结构"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    source: str
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_remote_path: Optional[str] = None  # 云端存储路径
    categories: Optional[List[str]] = None
    citations: int = 0
    chunk_ids: Optional[List[int]] = None  # 关联的chunk索引


@dataclass
class TextChunk:
    """文本块结构"""
    chunk_id: int
    paper_id: str
    content: str
    chunk_type: str  # 'abstract', 'section', 'paragraph'
    section_name: Optional[str] = None
    page_num: Optional[int] = None


class MetadataStore:
    """元数据层：SQLite存储论文元数据"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 论文元数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                published TEXT,
                source TEXT,
                doi TEXT,
                pdf_url TEXT,
                pdf_remote_path TEXT,
                categories TEXT,
                citations INTEGER DEFAULT 0,
                chunk_ids TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 文本块表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_type TEXT,
                section_name TEXT,
                page_num INTEGER,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_title ON papers(title)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_paper ON chunks(paper_id)')
        
        conn.commit()
        conn.close()
    
    def add_paper(self, paper: PaperMetadata) -> bool:
        """添加论文元数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO papers 
                (paper_id, title, authors, abstract, published, source, doi, 
                 pdf_url, pdf_remote_path, categories, citations, chunk_ids, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper.paper_id,
                paper.title,
                json.dumps(paper.authors),
                paper.abstract,
                paper.published,
                paper.source,
                paper.doi,
                paper.pdf_url,
                paper.pdf_remote_path,
                json.dumps(paper.categories) if paper.categories else None,
                paper.citations,
                json.dumps(paper.chunk_ids) if paper.chunk_ids else None,
                datetime.now().isoformat()
            ))
            conn.commit()
            return True
        except Exception as e:
            return False
        finally:
            conn.close()
    
    def add_chunk(self, chunk: TextChunk) -> int:
        """添加文本块，返回chunk_id。如果已存在相同内容则返回已有的chunk_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 先检查是否已存在相同的 paper_id + chunk_type 组合
            cursor.execute('''
                SELECT chunk_id FROM chunks 
                WHERE paper_id = ? AND chunk_type = ? AND content = ?
            ''', (chunk.paper_id, chunk.chunk_type, chunk.content))
            existing = cursor.fetchone()
            
            if existing:
                # 已存在，返回 0 表示跳过（不是新添加的）
                return 0
            
            cursor.execute('''
                INSERT INTO chunks (paper_id, content, chunk_type, section_name, page_num)
                VALUES (?, ?, ?, ?, ?)
            ''', (chunk.paper_id, chunk.content, chunk.chunk_type, 
                  chunk.section_name, chunk.page_num))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            return -1
        finally:
            conn.close()
    
    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """获取论文元数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM papers WHERE paper_id = ?', (paper_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return PaperMetadata(
                paper_id=row[0],
                title=row[1],
                authors=json.loads(row[2]) if row[2] else [],
                abstract=row[3],
                published=row[4],
                source=row[5],
                doi=row[6],
                pdf_url=row[7],
                pdf_remote_path=row[8],
                categories=json.loads(row[9]) if row[9] else None,
                citations=row[10],
                chunk_ids=json.loads(row[11]) if row[11] else None
            )
        return None
    
    def get_chunks_by_paper(self, paper_id: str) -> List[TextChunk]:
        """获取论文的所有文本块"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks WHERE paper_id = ?', (paper_id,))
        rows = cursor.fetchall()
        conn.close()
        
        return [TextChunk(
            chunk_id=row[0],
            paper_id=row[1],
            content=row[2],
            chunk_type=row[3],
            section_name=row[4],
            page_num=row[5]
        ) for row in rows]
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[TextChunk]:
        """根据chunk_id获取文本块"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return TextChunk(
                chunk_id=row[0],
                paper_id=row[1],
                content=row[2],
                chunk_type=row[3],
                section_name=row[4],
                page_num=row[5]
            )
        return None
    
    def remove_duplicate_chunks(self) -> int:
        """删除重复的文本块，保留最早的一个"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 找出重复的 chunks（相同 paper_id + content）
            cursor.execute('''
                DELETE FROM chunks 
                WHERE chunk_id NOT IN (
                    SELECT MIN(chunk_id) 
                    FROM chunks 
                    GROUP BY paper_id, content
                )
            ''')
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
        except Exception as e:
            return 0
        finally:
            conn.close()
    
    def search_papers_by_title(self, query: str, limit: int = 10) -> List[PaperMetadata]:
        """按标题搜索论文"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM papers WHERE title LIKE ? LIMIT ?',
            (f'%{query}%', limit)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_paper(row) for row in rows]
    
    def _row_to_paper(self, row) -> PaperMetadata:
        """将数据库行转换为PaperMetadata"""
        return PaperMetadata(
            paper_id=row[0],
            title=row[1],
            authors=json.loads(row[2]) if row[2] else [],
            abstract=row[3],
            published=row[4],
            source=row[5],
            doi=row[6],
            pdf_url=row[7],
            pdf_remote_path=row[8],
            categories=json.loads(row[9]) if row[9] else None,
            citations=row[10],
            chunk_ids=json.loads(row[11]) if row[11] else None
        )
    
    def get_all_chunks(self) -> List[TextChunk]:
        """获取所有文本块"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM chunks')
        rows = cursor.fetchall()
        conn.close()
        
        return [TextChunk(
            chunk_id=row[0],
            paper_id=row[1],
            content=row[2],
            chunk_type=row[3],
            section_name=row[4],
            page_num=row[5]
        ) for row in rows]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM papers')
        paper_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM chunks')
        chunk_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_papers': paper_count,
            'total_chunks': chunk_count
        }


class HybridRetriever:
    """
    混合检索器
    
    融合BM25稀疏检索与BGE-M3稠密向量检索，
    引入BGE-Reranker二阶段精排
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.config = config
        self._verbose = verbose
        
        # 数据目录
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data'
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 三层存储路径
        self.index_dir = self.data_dir / 'index'
        self.metadata_db_path = self.data_dir / 'metadata' / 'papers.db'
        self.cache_dir = self.data_dir / 'cache'
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化元数据存储
        self.metadata_store = MetadataStore(str(self.metadata_db_path))
        
        # 初始化PDF处理器
        if HAS_PDF_MARKDOWN:
            self.pdf_processor = PDFMarkdownProcessor(
                chunk_size=config.get('chunk_size', 1000),
                chunk_overlap=config.get('chunk_overlap', 200),
                extract_tables=config.get('extract_tables', True),
                extract_formulas=config.get('extract_formulas', True),
                extract_images=config.get('extract_images', True),
                describe_images=config.get('describe_images', False),
                temp_dir=str(self.cache_dir)
            )
        else:
            self.pdf_processor = None
        
        # BGE-M3 模型配置
        self.bge_model_name = config.get('bge_model', 'BAAI/bge-m3')
        self.reranker_model_name = config.get('reranker_model', 'BAAI/bge-reranker-v2-m3')
        self.device = resolve_device(config.get('device', 'auto'))
        
        # 延迟加载模型
        self._bge_model = None
        self._reranker = None
        #: sentence-transformers 回退编码器（FlagEmbedding 不可用时使用同一份 BGE-M3 权重）
        self._st_encoder_obj = None
        #: Reranker 权重加载失败后不再重试（离线环境下会反复抛 OSError）
        self._reranker_failed = False
        #: 最近一次检索是否跳过了精排（供 last_search_meta 如实上报）
        self._last_rerank_skipped = False
        #: 最近一次检索的后端/个性化元信息（供上层如实汇报降级情况）
        self.last_search_meta: Dict = {}

        # BM25索引
        self._bm25_index = None
        self._bm25_corpus = []  # 分词后的语料
        self._chunk_ids = []  # 与BM25索引对应的chunk_id列表
        
        # FAISS索引
        self._faiss_index = None
        self._faiss_chunk_ids = []  # 与FAISS索引对应的chunk_id列表
        
        # 加载已有索引
        self._load_indices()
    
    @property
    def bge_model(self):
        """延迟加载BGE-M3模型"""
        if self._bge_model is None:
            if not HAS_BGE:
                raise ImportError("FlagEmbedding not installed")
            self._bge_model = BGEM3FlagModel(
                self.bge_model_name,
                use_fp16=True,
                device=self.device
            )
        return self._bge_model
    
    @property
    def reranker(self):
        """延迟加载BGE-Reranker模型"""
        if self._reranker is None:
            if not HAS_BGE:
                raise ImportError("FlagEmbedding not installed")
            self._reranker = FlagReranker(
                self.reranker_model_name,
                use_fp16=True,
                device=self.device
            )
        return self._reranker

    # ------------------------------------------------------------ 稠密编码
    @property
    def dense_available(self) -> bool:
        """稠密检索是否可用（FlagEmbedding 或 sentence-transformers 任一）。不触发模型加载。"""
        if HAS_BGE:
            return True
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _st_encoder(self):
        """sentence-transformers 编码器（懒加载，与 memory/ 共用同一份 BGE-M3 权重）。"""
        if self._st_encoder_obj is None:
            from ..memory.encoder import SemanticEncoder
            use_gpu = self.device not in ('cpu',)
            self._st_encoder_obj = SemanticEncoder(
                self.bge_model_name,
                device='cuda' if use_gpu else None,
                offline=True,
            )
        return self._st_encoder_obj

    def _encode_dense(self, texts: List[str]) -> np.ndarray:
        """统一的稠密编码入口。

        优先 FlagEmbedding（项目原始依赖）；缺失时回退 sentence-transformers ——
        两者用**同一份** BAAI/bge-m3 权重，因此检索向量与 memory/ 的偏好向量处于
        同一空间，偏好先验可以直接作用到检索打分上（这正是个性化能生效的前提）。
        """
        if HAS_BGE:
            out = self.bge_model.encode(texts, batch_size=32, max_length=512)['dense_vecs']
            return np.asarray(out, dtype='float32')
        vecs = self._st_encoder().encode(texts, batch_size=32)
        if vecs is None:
            raise RuntimeError('dense encoder unavailable: 需 FlagEmbedding，或 '
                               'sentence-transformers + 本地 BAAI/bge-m3')
        return np.asarray(vecs, dtype='float32')

    def cleanup_and_rebuild_index(self) -> int:
        """清理重复数据并重建索引"""
        deleted = self.metadata_store.remove_duplicate_chunks()
        return self.rebuild_index_from_db()
    
    def rebuild_index_from_db(self) -> int:
        """从数据库重建 BM25 和 FAISS 索引"""
        conn = sqlite3.connect(self.metadata_store.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT chunk_id, content FROM chunks ORDER BY chunk_id')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return 0
        
        chunk_ids = [row[0] for row in rows]
        contents = [row[1] for row in rows]
        
        # 重置索引
        self._bm25_corpus = []
        self._chunk_ids = []
        self._faiss_index = None
        self._faiss_chunk_ids = []
        
        # 重建 BM25
        self._build_bm25_index(contents, chunk_ids)
        
        # 重建 FAISS
        self._build_faiss_index(contents, chunk_ids)
        
        # 保存
        self._save_indices()
        
        return len(chunk_ids)
    
    def add_papers(self, papers: List[Dict], process_pdf: bool = False, verbose: bool = True) -> int:
        """
        添加论文到索引
        
        Args:
            papers: 论文列表
            process_pdf: 是否处理PDF全文
            verbose: 是否输出详细信息
            
        Returns:
            添加的chunk数量
        """
        
        all_chunks = []
        all_chunk_ids = []
        
        for paper in tqdm(papers, desc="Processing papers", disable=not verbose):
            # 创建元数据
            paper_meta = PaperMetadata(
                paper_id=paper.get('id', hashlib.md5(paper.get('title', '').encode()).hexdigest()),
                title=paper.get('title', ''),
                authors=paper.get('authors', []),
                abstract=paper.get('abstract', ''),
                published=paper.get('published', ''),
                source=paper.get('source', ''),
                doi=paper.get('doi'),
                pdf_url=paper.get('pdf_url'),
                pdf_remote_path=paper.get('pdf_remote_path'),
                categories=paper.get('categories'),
                citations=paper.get('citations', 0)
            )
            
            # 添加摘要作为chunk
            if paper_meta.abstract:
                chunk = TextChunk(
                    chunk_id=-1,  # 将由数据库分配
                    paper_id=paper_meta.paper_id,
                    content=f"Title: {paper_meta.title}\n\nAbstract: {paper_meta.abstract}",
                    chunk_type='abstract'
                )
                chunk_id = self.metadata_store.add_chunk(chunk)
                if chunk_id > 0:
                    all_chunks.append(chunk.content)
                    all_chunk_ids.append(chunk_id)
            
            # 处理PDF全文（如果启用且有本地PDF）
            if process_pdf and paper.get('local_pdf_path') and self.pdf_processor:
                # PDFMarkdownProcessor 使用 process() 方法
                if hasattr(self.pdf_processor, 'process'):
                    pdf_result = self.pdf_processor.process(paper['local_pdf_path'])
                    pdf_chunks = pdf_result.get('chunks', [])
                else:
                    # 旧版处理器使用 parse_pdf()
                    pdf_result = self.pdf_processor.parse_pdf(paper['local_pdf_path'])
                    if isinstance(pdf_result, dict):
                        pdf_chunks = pdf_result.get('chunks', [])
                    else:
                        pdf_chunks = pdf_result
                
                for pc in pdf_chunks:
                    chunk = TextChunk(
                        chunk_id=-1,
                        paper_id=paper_meta.paper_id,
                        content=pc.get('content', ''),
                        chunk_type=pc.get('chunk_type', 'paragraph'),
                        section_name=pc.get('section_name'),
                        page_num=pc.get('page_num')
                    )
                    chunk_id = self.metadata_store.add_chunk(chunk)
                    if chunk_id > 0:
                        all_chunks.append(chunk.content)
                        all_chunk_ids.append(chunk_id)
            
            # 保存论文元数据
            self.metadata_store.add_paper(paper_meta)
        
        if not all_chunks:
            return 0
        
        # 构建BM25索引
        self._build_bm25_index(all_chunks, all_chunk_ids)
        
        # 构建FAISS索引
        self._build_faiss_index(all_chunks, all_chunk_ids)
        
        # 保存索引
        self._save_indices()
        
        return len(all_chunks)
    
    def _build_bm25_index(self, texts: List[str], chunk_ids: List[int]):
        """构建BM25索引"""
        # 简单分词（可以替换为更好的分词器）
        tokenized_corpus = [self._tokenize(text) for text in texts]
        
        # 合并到现有索引
        self._bm25_corpus.extend(tokenized_corpus)
        self._chunk_ids.extend(chunk_ids)
        
        # 重建BM25索引
        self._bm25_index = BM25Okapi(self._bm25_corpus)
    
    def _build_faiss_index(self, texts: List[str], chunk_ids: List[int]):
        """构建FAISS HNSW索引"""
        # 生成BGE-M3 embeddings（FlagEmbedding 或 sentence-transformers 回退）
        embeddings = self._encode_dense(texts)
        
        embeddings = np.array(embeddings).astype('float32')
        
        # 创建或更新FAISS索引
        if self._faiss_index is None:
            dimension = embeddings.shape[1]
            # 使用HNSW索引，适合大规模检索
            self._faiss_index = faiss.IndexHNSWFlat(dimension, 32)  # M=32
            self._faiss_index.hnsw.efConstruction = 200
            self._faiss_index.hnsw.efSearch = 128
        
        self._faiss_index.add(embeddings)
        self._faiss_chunk_ids.extend(chunk_ids)
    
    def _tokenize(self, text: str) -> List[str]:
        """中英文混合分词"""
        return tokenize(text)
    
    def _generate_hypothetical_document(self, query: str) -> str:
        """
        HyDE: 使用 LLM 生成假设性文档
        
        根据用户的模糊问题，让 LLM 生成一段看起来专业的"虚假答案"，
        用这个答案的向量去检索真实论文。
        """
        try:
            import openai
            
            # 从环境变量获取 API 配置
            api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')
            base_url = os.getenv('OPENAI_BASE_URL', 'https://api.deepseek.com')
            
            if not api_key:
                return query
            
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            
            # HyDE Prompt: 生成假设性学术文档
            prompt = f"""You are a scientific writing assistant. Given a research question, write a short hypothetical abstract (100-150 words) that would answer this question. 
The abstract should be written in academic style, as if it were from a real research paper.
Do NOT say "This paper" or "We propose" - just write the content directly.

Research Question: {query}

Hypothetical Abstract:"""
            
            response = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'deepseek-chat'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            return hypothetical_doc
            
        except Exception as e:
            return query
    
    def search_with_hyde(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        verbose: bool = True
    ) -> List[Dict]:
        """
        HyDE (Hypothetical Document Embeddings) 检索
        
        流程：
        1. 用 LLM 根据查询生成假设性文档（虚假但专业的答案）
        2. 用假设文档的向量进行稠密检索
        3. 可选：使用 Reranker 精排
        
        Args:
            query: 用户的模糊查询
            top_k: 返回结果数
            use_rerank: 是否使用 reranker 精排
            verbose: 是否输出详细信息
            
        Returns:
            检索结果列表
        """
        if not self._faiss_index:
            return []
        
        # 1. 生成假设性文档
        hypothetical_doc = self._generate_hypothetical_document(query)
        
        # 2. 用假设文档进行稠密检索
        dense_results = self._dense_search(hypothetical_doc, top_k=20)
        
        # 3. 可选：Reranker 精排（用原始查询）
        if use_rerank and dense_results:
            results = self._rerank(query, dense_results, top_k)
        else:
            results = dense_results[:top_k]
        
        # 4. 补充元数据
        results = self._enrich_results(results)
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        use_rerank: bool = True,
        rerank_top_k: int = 20,
        use_hyde: bool = False,
        verbose: bool = True,
        preference_profile=None,
        prior_lambda: Optional[float] = None,
        memory=None,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            bm25_weight: BM25权重
            dense_weight: 稠密检索权重
            use_rerank: 是否使用reranker精排
            rerank_top_k: 精排前的候选数量
            use_hyde: 是否使用 HyDE (Hypothetical Document Embeddings)
            verbose: 是否输出详细信息
            
        Returns:
            检索结果列表
        """
        # 如果启用 HyDE，使用专门的 HyDE 检索方法
        if use_hyde:
            return self.search_with_hyde(query, top_k, use_rerank, verbose=verbose)

        bm25_ok = self._bm25_index is not None
        dense_ok = self._faiss_index is not None
        if not bm25_ok and not dense_ok:
            logger.warning('检索不可用：BM25 与 FAISS 索引均为空（先用 add_papers() 建索引）')
            self.last_search_meta = {'backend': 'none', 'reason': 'no-index'}
            return []

        # 1. BM25稀疏检索
        bm25_results = self._bm25_search(query, top_k=rerank_top_k) if bm25_ok else []

        # 2. BGE-M3稠密检索（编码器不可用时降级为纯 BM25，并如实记录）
        dense_results = []
        if dense_ok:
            try:
                dense_results = self._dense_search(query, top_k=rerank_top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning('稠密检索不可用，本次降级为 BM25：%s', exc)

        # 3. 融合结果（RRF - Reciprocal Rank Fusion）
        fused_results = self._rrf_fusion(
            bm25_results, dense_results,
            bm25_weight, dense_weight
        )

        # 4. 个性化：把长期记忆中的偏好回流为**检索打分先验**
        lam = prior_lambda
        if lam is None:
            lam = float(self.config.get('personalization', {}).get('lambda', 0.0))
        personalized = False
        profile = preference_profile
        if profile is None and memory is not None:
            try:
                from ..memory.personalize import build_profile
                profile = build_profile(memory, user_id or 'default')
            except Exception as exc:  # noqa: BLE001
                logger.warning('构建偏好画像失败：%s', exc)
        if profile is not None and float(lam) > 0 and not profile.is_empty and fused_results:
            from ..memory.personalize import preference_prior
            enc = getattr(memory, 'encoder', None) or self._st_encoder()
            pairs = []
            for cand in fused_results:
                chunk = self.metadata_store.get_chunk_by_id(cand['chunk_id'])
                if chunk is not None:
                    pairs.append((cand, chunk))
            if pairs:
                base = np.array([c['score'] for c, _ in pairs], dtype='float32')
                boosted = preference_prior(
                    profile, [ch.content for _, ch in pairs], enc,
                    lam=float(lam), base_scores=base,
                )
                for (cand, _), score in zip(pairs, boosted):
                    cand['score'] = float(score)
                    cand['personalized'] = True   # 该条分数已被偏好先验重算
                fused_results = sorted([c for c, _ in pairs], key=lambda x: -x['score'])
                personalized = True
                if verbose:
                    logger.info('个性化检索生效：λ=%.2f，正向偏好 %d 条',
                                float(lam), profile.n_positive)

        # 5. BGE-Reranker精排
        if use_rerank and fused_results:
            fused_results = self._rerank(query, fused_results, top_k)
        else:
            fused_results = fused_results[:top_k]

        self.last_search_meta = {
            'backend': ('bm25+dense' if (bm25_results and dense_results)
                        else 'bm25' if bm25_results else 'dense'),
            'reranker_skipped': bool(self._last_rerank_skipped),
            'personalized': personalized,
            'prior_lambda': float(lam),
            'n_bm25': len(bm25_results),
            'n_dense': len(dense_results),
        }

        # 6. 补充元数据
        results = self._enrich_results(fused_results)
        
        return results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25检索"""
        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)
        
        # 获取top_k结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'chunk_id': self._chunk_ids[idx],
                    'score': float(scores[idx]),
                    'source': 'bm25'
                })
        
        return results
    
    def _dense_search(self, query: str, top_k: int) -> List[Dict]:
        """BGE-M3稠密检索"""
        # 生成查询embedding（FlagEmbedding 或 sentence-transformers 回退）
        query_embedding = self._encode_dense([query])
        
        query_embedding = np.array(query_embedding).astype('float32')
        
        # FAISS检索
        distances, indices = self._faiss_index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._faiss_chunk_ids):
                # 将L2距离转换为相似度分数
                similarity = 1 / (1 + dist)
                results.append({
                    'chunk_id': self._faiss_chunk_ids[idx],
                    'score': float(similarity),
                    'source': 'dense'
                })
        
        return results
    
    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        dense_results: List[Dict],
        bm25_weight: float,
        dense_weight: float,
        k: int = 60
    ) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合
        
        RRF_score = sum(weight / (k + rank))
        """
        chunk_scores = {}
        
        # BM25结果
        for rank, result in enumerate(bm25_results):
            chunk_id = result['chunk_id']
            rrf_score = bm25_weight / (k + rank + 1)
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'score': 0, 'sources': []}
            chunk_scores[chunk_id]['score'] += rrf_score
            chunk_scores[chunk_id]['sources'].append('bm25')
        
        # Dense结果
        for rank, result in enumerate(dense_results):
            chunk_id = result['chunk_id']
            rrf_score = dense_weight / (k + rank + 1)
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'score': 0, 'sources': []}
            chunk_scores[chunk_id]['score'] += rrf_score
            chunk_scores[chunk_id]['sources'].append('dense')
        
        # 排序
        sorted_results = sorted(
            chunk_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        return [
            {'chunk_id': chunk_id, 'score': data['score'], 'sources': data['sources']}
            for chunk_id, data in sorted_results
        ]
    
    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """BGE-Reranker精排

        ⚠️ 可用性守卫看的是**reranker 权重能不能加载**，而不是"FlagEmbedding 能不能
        import"。装了 FlagEmbedding 但本地没有 ``BAAI/bge-reranker-v2-m3`` 权重时，
        加载会抛 OSError（离线环境无网可下），因此必须捕获并优雅跳过，
        同时把 ``reranker_skipped`` 记进 ``last_search_meta`` —— 不静默假装精排过。
        """
        self._last_rerank_skipped = False
        if not candidates:
            return []
        try:
            reranker = self.reranker
        except Exception as exc:  # noqa: BLE001
            self._reranker_failed = True
            self._last_rerank_skipped = True
            logger.warning('Reranker 权重不可用，跳过精排并保持 RRF 融合顺序：%s', exc)
            return candidates[:top_k]
        # 获取候选文本
        pairs = []
        valid_candidates = []
        
        for cand in candidates:
            chunk = self.metadata_store.get_chunk_by_id(cand['chunk_id'])
            if chunk:
                pairs.append([query, chunk.content])
                valid_candidates.append(cand)
        
        if not pairs:
            self._last_rerank_skipped = True
            return candidates[:top_k]
        
        # 计算rerank分数
        rerank_scores = reranker.compute_score(pairs, normalize=True)
        
        # 如果只有一个结果，compute_score返回float而不是list
        if isinstance(rerank_scores, float):
            rerank_scores = [rerank_scores]
        
        # 更新分数并排序
        for i, score in enumerate(rerank_scores):
            valid_candidates[i]['rerank_score'] = float(score)
        
        # 按rerank分数排序
        reranked = sorted(valid_candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return reranked[:top_k]
    
    def _enrich_results(self, results: List[Dict]) -> List[Dict]:
        """补充结果的元数据"""
        enriched = []
        
        for result in results:
            chunk = self.metadata_store.get_chunk_by_id(result['chunk_id'])
            if chunk:
                paper = self.metadata_store.get_paper(chunk.paper_id)
                enriched.append({
                    'chunk_id': result['chunk_id'],
                    'content': chunk.content,
                    'chunk_type': chunk.chunk_type,
                    'score': result.get('rerank_score', result.get('score', 0)),
                    'sources': result.get('sources', []),
                    # 观测字段：让上层能判断这一条是否经过精排 / 是否被偏好先验影响
                    'reranked': 'rerank_score' in result,
                    'personalized': bool(result.get('personalized', False)),
                    'paper': {
                        'paper_id': paper.paper_id if paper else chunk.paper_id,
                        'title': paper.title if paper else '',
                        'authors': paper.authors if paper else [],
                        'published': paper.published if paper else '',
                        'doi': paper.doi if paper else None,
                        'pdf_url': paper.pdf_url if paper else None
                    } if paper else None
                })
        
        return enriched
    
    def _save_indices(self):
        """保存索引到磁盘"""
        # 保存BM25相关数据
        bm25_path = self.index_dir / 'bm25_data.pkl'
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'corpus': self._bm25_corpus,
                'chunk_ids': self._chunk_ids
            }, f)
        
        # 保存FAISS索引
        faiss_path = self.index_dir / 'faiss_hnsw.index'
        faiss.write_index(self._faiss_index, str(faiss_path))
        
        # 保存FAISS chunk_ids映射
        faiss_ids_path = self.index_dir / 'faiss_chunk_ids.pkl'
        with open(faiss_ids_path, 'wb') as f:
            pickle.dump(self._faiss_chunk_ids, f)
        
    
    def _load_indices(self):
        """从磁盘加载索引"""
        # 加载BM25数据
        bm25_path = self.index_dir / 'bm25_data.pkl'
        if bm25_path.exists():
            try:
                with open(bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    self._bm25_corpus = data['corpus']
                    self._chunk_ids = data['chunk_ids']
                    if self._bm25_corpus:
                        self._bm25_index = BM25Okapi(self._bm25_corpus)
            except Exception as e:
                logger.warning("Index load failed: %s", e)
        
        # 加载FAISS索引
        faiss_path = self.index_dir / 'faiss_hnsw.index'
        faiss_ids_path = self.index_dir / 'faiss_chunk_ids.pkl'
        
        if faiss_path.exists() and faiss_ids_path.exists():
            try:
                self._faiss_index = faiss.read_index(str(faiss_path))
                with open(faiss_ids_path, 'rb') as f:
                    self._faiss_chunk_ids = pickle.load(f)
            except Exception as e:
                logger.warning("Index load failed: %s", e)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        db_stats = self.metadata_store.get_stats()
        
        return {
            'total_papers': db_stats['total_papers'],
            'total_chunks': db_stats['total_chunks'],
            'bm25_documents': len(self._bm25_corpus),
            'faiss_vectors': self._faiss_index.ntotal if self._faiss_index else 0,
            'index_dir': str(self.index_dir),
            'metadata_db': str(self.metadata_db_path)
        }
    
    def clear_cache(self):
        """清理缓存层"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            pass


if __name__ == "__main__":
    # 测试代码
    config = {
        'bge_model': 'BAAI/bge-m3',
        'reranker_model': 'BAAI/bge-reranker-v2-m3',
        'device': 'cuda',
        'chunk_size': 512,
        'chunk_overlap': 50
    }
    
    retriever = HybridRetriever(config)
    
    # 测试添加论文
    test_papers = [
        {
            'id': 'test1',
            'title': 'Graph Neural Networks for Battery Life Prediction',
            'authors': ['Author A', 'Author B'],
            'abstract': 'This paper presents a novel approach using graph neural networks (GNNs) for predicting battery remaining useful life. We propose a temporal graph attention mechanism that captures both spatial and temporal dependencies in battery degradation data.',
            'published': '2024-01-01',
            'source': 'arxiv'
        },
        {
            'id': 'test2',
            'title': 'Transformer-based Models for Scientific Text Understanding',
            'authors': ['Author C', 'Author D'],
            'abstract': 'We introduce a transformer-based model specifically designed for understanding scientific literature. Our approach achieves state-of-the-art results on multiple scientific NLP benchmarks.',
            'published': '2024-02-01',
            'source': 'arxiv'
        }
    ]
    
    retriever.add_papers(test_papers)
    
    # 测试检索
    results = retriever.search("battery prediction using neural networks", top_k=5)
    
    print(f"\n✅ Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Title: {result['paper']['title'] if result['paper'] else 'N/A'}")
        print(f"   Sources: {result['sources']}")
        print(f"   Content: {result['content'][:100]}...")
    
    # 打印统计信息
    print(f"\n📊 Stats: {retriever.get_stats()}")
