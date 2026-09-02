"""
PDF Markdown Processor
PDF 转 Markdown 处理器

处理流程:
1. 解析: PyMuPDF 读取 PDF -> 转为 Markdown 字符串
2. 清洗: 去掉页码、页眉页脚
3. 切分: 使用 MarkdownHeaderTextSplitter 按章节切开
4. 入库: 存入 chunks 表 + 向量化存入 FAISS

功能模块分布:
- 解析阶段: 提取表格、公式、图片
- 清洗阶段: 去除页眉页脚、页码
- 切分阶段: 按章节结构切分，保留元数据
- 入库阶段: 文本块 + 表格块 + 公式块 + 图片描述块
"""

import os
import re
import base64
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# PyMuPDF
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# pymupdf4llm - PDF to Markdown (更好的格式转换)
try:
    import pymupdf4llm
    HAS_PYMUPDF4LLM = True
except ImportError:
    HAS_PYMUPDF4LLM = False

# pdfplumber for table extraction
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# LangChain Text Splitters
try:
    from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    HAS_LANGCHAIN_SPLITTER = True
except ImportError:
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        HAS_LANGCHAIN_SPLITTER = True
    except ImportError:
        HAS_LANGCHAIN_SPLITTER = False

# OpenAI for image description
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class ChunkType(Enum):
    """分块类型"""
    SECTION = "section"           # 章节文本
    TABLE = "table"               # 表格
    FORMULA = "formula"           # 公式
    IMAGE = "image"               # 图片描述
    ABSTRACT = "abstract"         # 摘要
    REFERENCE = "reference"       # 参考文献


@dataclass
class PDFChunk:
    """PDF 分块结构"""
    content: str                          # 文本内容
    chunk_type: ChunkType                 # 分块类型
    section_name: str = ""                # 所属章节
    section_hierarchy: List[str] = field(default_factory=list)  # 章节层级
    page_num: Optional[int] = None        # 页码
    metadata: Dict = field(default_factory=dict)  # 额外元数据
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'content': self.content,
            'chunk_type': self.chunk_type.value,
            'section_name': self.section_name or " > ".join(self.section_hierarchy),
            'page_num': self.page_num,
            'metadata': self.metadata
        }


@dataclass
class ExtractedTable:
    """提取的表格"""
    table_id: int
    page_num: int
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    
    def to_markdown(self) -> str:
        """转换为 Markdown 表格"""
        if not self.headers and not self.rows:
            return ""
        
        lines = []
        if self.caption:
            lines.append(f"**{self.caption}**\n")
        
        if self.headers:
            lines.append("| " + " | ".join(str(h) for h in self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")
        
        return "\n".join(lines)


@dataclass
class ExtractedFormula:
    """提取的公式"""
    formula_id: int
    page_num: int
    latex: str
    context: str = ""
    is_inline: bool = False


@dataclass
class ExtractedImage:
    """提取的图片"""
    image_id: int
    page_num: int
    image_bytes: bytes
    image_ext: str
    caption: Optional[str] = None
    description: Optional[str] = None


class PDFMarkdownProcessor:
    """
    PDF Markdown 处理器
    
    完整处理流程:
    1. 解析 (Parse): PDF -> Markdown + 提取表格/公式/图片
    2. 清洗 (Clean): 去除页眉页脚、页码
    3. 切分 (Split): MarkdownHeaderTextSplitter 按章节切分
    4. 入库 (Index): 返回可入库的分块列表
    """
    
    # 页眉页脚模式
    HEADER_FOOTER_PATTERNS = [
        r'^\s*\d+\s*$',                          # 单独页码
        r'^\s*-\s*\d+\s*-\s*$',                  # -1-
        r'^\s*Page\s+\d+\s*(of\s+\d+)?\s*$',    # Page 1 of 10
        r'^\s*第\s*\d+\s*页\s*$',                # 第1页
        r'^arXiv:\d+\.\d+v?\d*\s*\[.*\].*$',    # arXiv 页眉
        r'^\s*©.*\d{4}.*$',                      # 版权
        r'^\s*Copyright.*$',
        r'^\s*https?://doi\.org/.*$',            # DOI
        r'^\s*DOI:.*$',
    ]
    
    # LaTeX 公式模式
    FORMULA_PATTERNS = [
        (r'\$\$(.+?)\$\$', False),               # $$...$$
        (r'\\\[(.+?)\\\]', False),               # \[...\]
        (r'\\begin\{equation\}(.+?)\\end\{equation\}', False),
        (r'\\begin\{align\}(.+?)\\end\{align\}', False),
        (r'\$([^\$\n]+?)\$', True),              # $...$
    ]
    
    # Markdown 标题分割配置
    HEADERS_TO_SPLIT = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extract_tables: bool = True,
        extract_formulas: bool = True,
        extract_images: bool = True,
        describe_images: bool = False,
        image_model: str = "gpt-4o-mini",
        temp_dir: Optional[str] = None
    ):
        """
        初始化处理器
        
        Args:
            chunk_size: 最大分块大小
            chunk_overlap: 分块重叠
            extract_tables: 是否提取表格
            extract_formulas: 是否提取公式
            extract_images: 是否提取图片
            describe_images: 是否用 AI 描述图片
            image_model: 图片描述模型
            temp_dir: 临时目录
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_tables = extract_tables and HAS_PDFPLUMBER
        self.extract_formulas = extract_formulas
        self.extract_images = extract_images
        self.describe_images = describe_images and HAS_OPENAI
        self.image_model = image_model
        
        # 临时目录
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "pdf_processor"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 编译正则
        self._header_footer_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in self.HEADER_FOOTER_PATTERNS
        ]
        
        # LLM 客户端
        self._llm_client = None
        
        # 初始化分割器
        self._init_splitters()
    
    def _init_splitters(self):
        """初始化文本分割器"""
        if HAS_LANGCHAIN_SPLITTER:
            self._md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.HEADERS_TO_SPLIT,
                strip_headers=False
            )
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            self._md_splitter = None
            self._text_splitter = None
    
    @property
    def llm_client(self):
        """延迟初始化 LLM 客户端"""
        if self._llm_client is None and HAS_OPENAI:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('DEEPSEEK_API_KEY')
            base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            if api_key:
                self._llm_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return self._llm_client
    
    # ==================== 主处理流程 ====================
    
    def process(self, pdf_path: str) -> Dict[str, Any]:
        """
        处理 PDF 文件
        
        流程:
        1. 解析: PDF -> Markdown + 表格/公式/图片
        2. 清洗: 去除页眉页脚
        3. 切分: 按章节切分
        4. 整合: 返回可入库的分块
        
        Returns:
            {
                'chunks': List[Dict],      # 可入库的分块
                'markdown': str,           # 清洗后的 Markdown
                'metadata': Dict,          # PDF 元数据
                'tables': List[Dict],      # 提取的表格
                'formulas': List[Dict],    # 提取的公式
                'images': List[Dict],      # 提取的图片
            }
        """
        if not HAS_PYMUPDF:
            return {'error': 'PyMuPDF not installed'}
        
        result = {
            'chunks': [],
            'markdown': '',
            'metadata': {},
            'tables': [],
            'formulas': [],
            'images': []
        }
        
        try:
            # ========== 1. 解析阶段 ==========
            parse_result = self._parse(pdf_path)
            result['metadata'] = parse_result['metadata']
            result['tables'] = parse_result['tables']
            result['formulas'] = parse_result['formulas']
            result['images'] = parse_result['images']
            
            # ========== 2. 清洗阶段 ==========
            cleaned_markdown = self._clean(parse_result['markdown'])
            result['markdown'] = cleaned_markdown
            
            # ========== 3. 切分阶段 ==========
            text_chunks = self._split(cleaned_markdown)
            
            # ========== 4. 整合阶段 ==========
            all_chunks = self._integrate(
                text_chunks,
                parse_result['tables'],
                parse_result['formulas'],
                parse_result['images']
            )
            result['chunks'] = [chunk.to_dict() for chunk in all_chunks]
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # ==================== 1. 解析阶段 ====================
    
    def _parse(self, pdf_path: str) -> Dict:
        """
        解析 PDF
        
        - PDF -> Markdown (使用 pymupdf4llm 或 PyMuPDF)
        - 提取表格 (pdfplumber)
        - 提取公式 (正则匹配 LaTeX)
        - 提取图片 (PyMuPDF)
        """
        result = {
            'markdown': '',
            'metadata': {},
            'tables': [],
            'formulas': [],
            'images': []
        }
        
        doc = fitz.open(pdf_path)
        
        # 提取元数据
        result['metadata'] = self._extract_metadata(doc)
        
        # PDF -> Markdown
        if HAS_PYMUPDF4LLM:
            result['markdown'] = pymupdf4llm.to_markdown(pdf_path)
        else:
            result['markdown'] = self._pdf_to_markdown_fallback(doc)
        
        # 提取表格
        if self.extract_tables:
            result['tables'] = self._extract_tables(pdf_path)
        
        # 提取公式
        if self.extract_formulas:
            result['formulas'] = self._extract_formulas(result['markdown'])
        
        # 提取图片
        if self.extract_images:
            result['images'] = self._extract_images(doc)
        
        doc.close()
        
        return result
    
    def _extract_metadata(self, doc: 'fitz.Document') -> Dict:
        """提取 PDF 元数据"""
        meta = doc.metadata or {}
        return {
            'title': meta.get('title', ''),
            'author': meta.get('author', ''),
            'subject': meta.get('subject', ''),
            'keywords': meta.get('keywords', ''),
            'page_count': len(doc),
            'creator': meta.get('creator', ''),
            'producer': meta.get('producer', '')
        }
    
    def _pdf_to_markdown_fallback(self, doc: 'fitz.Document') -> str:
        """PyMuPDF 原生方法转 Markdown (降级方案)"""
        markdown_parts = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_lines = []
            
            for block in blocks:
                if block["type"] != 0:
                    continue
                
                for line in block.get("lines", []):
                    text = ""
                    max_size = 0
                    
                    for span in line.get("spans", []):
                        text += span.get("text", "")
                        max_size = max(max_size, span.get("size", 12))
                    
                    text = text.strip()
                    if not text:
                        continue
                    
                    # 根据字体大小推断标题
                    if max_size > 16:
                        text = f"# {text}"
                    elif max_size > 14:
                        text = f"## {text}"
                    elif max_size > 12.5:
                        text = f"### {text}"
                    
                    page_lines.append(text)
            
            markdown_parts.append("\n".join(page_lines))
        
        return "\n\n".join(markdown_parts)
    
    def _extract_tables(self, pdf_path: str) -> List[ExtractedTable]:
        """使用 pdfplumber 提取表格"""
        tables = []
        table_id = 0
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_data in page_tables:
                        if not table_data or len(table_data) < 2:
                            continue
                        
                        headers = [str(c) if c else '' for c in table_data[0]]
                        rows = [[str(c) if c else '' for c in row] for row in table_data[1:]]
                        
                        # 查找表格标题
                        caption = self._find_table_caption(page)
                        
                        tables.append(ExtractedTable(
                            table_id=table_id,
                            page_num=page_num + 1,
                            headers=headers,
                            rows=rows,
                            caption=caption
                        ))
                        table_id += 1
        except Exception as e:
            logger.warning("Table extraction failed: %s", e)
        
        return tables
    
    def _find_table_caption(self, page) -> Optional[str]:
        """查找表格标题"""
        text = page.extract_text() or ''
        match = re.search(r'Table\s+\d+[.:]\s*([^\n]+)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _extract_formulas(self, markdown_text: str) -> List[ExtractedFormula]:
        """提取 LaTeX 公式"""
        formulas = []
        formula_id = 0
        
        for pattern, is_inline in self.FORMULA_PATTERNS:
            for match in re.finditer(pattern, markdown_text, re.DOTALL):
                latex = match.group(1).strip()
                if len(latex) < 2:
                    continue
                
                # 获取上下文
                start = max(0, match.start() - 100)
                end = min(len(markdown_text), match.end() + 100)
                context = markdown_text[start:end]
                
                formulas.append(ExtractedFormula(
                    formula_id=formula_id,
                    page_num=0,  # Markdown 中无法确定页码
                    latex=latex,
                    context=context,
                    is_inline=is_inline
                ))
                formula_id += 1
        
        return formulas
    
    def _extract_images(self, doc: 'fitz.Document') -> List[ExtractedImage]:
        """提取图片"""
        images = []
        image_id = 0
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img in image_list:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    if base_image:
                        # 查找图片标题
                        caption = self._find_image_caption(page, image_id)
                        
                        # AI 描述图片
                        description = None
                        if self.describe_images:
                            description = self._describe_image(base_image["image"])
                        
                        images.append(ExtractedImage(
                            image_id=image_id,
                            page_num=page_num + 1,
                            image_bytes=base_image["image"],
                            image_ext=base_image["ext"],
                            caption=caption,
                            description=description
                        ))
                        image_id += 1
                except Exception:
                    continue
        
        return images
    
    def _find_image_caption(self, page, img_index: int) -> Optional[str]:
        """查找图片标题"""
        text = page.get_text()
        matches = re.findall(r'(?:Figure|Fig\.?)\s+\d+[.:]\s*([^\n]+)', text, re.IGNORECASE)
        return matches[img_index].strip() if img_index < len(matches) else None
    
    def _describe_image(self, image_bytes: bytes) -> Optional[str]:
        """使用多模态模型描述图片"""
        if not self.llm_client:
            return None
        
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            response = self.llm_client.chat.completions.create(
                model=self.image_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this scientific figure in 2-3 sentences. Focus on the type of visualization, axes, and key findings."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None
    
    # ==================== 2. 清洗阶段 ====================
    
    def _clean(self, markdown_text: str) -> str:
        """
        清洗 Markdown 文本
        
        - 去除页眉页脚
        - 去除页码
        - 合并连续空行
        """
        lines = markdown_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            is_noise = False
            for pattern in self._header_footer_patterns:
                if pattern.match(line.strip()):
                    is_noise = True
                    break
            
            if not is_noise:
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    # ==================== 3. 切分阶段 ====================
    
    def _split(self, markdown_text: str) -> List[PDFChunk]:
        """
        使用 MarkdownHeaderTextSplitter 按章节切分
        """
        chunks = []
        
        if self._md_splitter:
            # 使用 LangChain 分割器
            md_splits = self._md_splitter.split_text(markdown_text)
            
            for doc in md_splits:
                content = doc.page_content
                metadata = doc.metadata
                
                # 提取章节层级
                hierarchy = []
                for key in ['h1', 'h2', 'h3', 'h4']:
                    if key in metadata:
                        hierarchy.append(metadata[key])
                
                # 如果内容太长，进一步切分
                if len(content) > self.chunk_size:
                    sub_texts = self._text_splitter.split_text(content)
                    for i, sub_text in enumerate(sub_texts):
                        chunks.append(PDFChunk(
                            content=sub_text,
                            chunk_type=ChunkType.SECTION,
                            section_hierarchy=hierarchy,
                            metadata={'part': i + 1, 'total_parts': len(sub_texts)}
                        ))
                else:
                    chunks.append(PDFChunk(
                        content=content,
                        chunk_type=ChunkType.SECTION,
                        section_hierarchy=hierarchy
                    ))
        else:
            # 降级：简单分块
            chunks = self._simple_split(markdown_text)
        
        return chunks
    
    def _simple_split(self, text: str) -> List[PDFChunk]:
        """简单分块 (降级方案)"""
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_hierarchy = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 检测标题
            if para.startswith('#'):
                match = re.match(r'^(#+)\s*(.+)$', para)
                if match:
                    level = len(match.group(1))
                    title = match.group(2)
                    
                    if current_chunk.strip():
                        chunks.append(PDFChunk(
                            content=current_chunk.strip(),
                            chunk_type=ChunkType.SECTION,
                            section_hierarchy=current_hierarchy.copy()
                        ))
                        current_chunk = ""
                    
                    current_hierarchy = current_hierarchy[:level-1]
                    current_hierarchy.append(title)
            
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(PDFChunk(
                    content=current_chunk.strip(),
                    chunk_type=ChunkType.SECTION,
                    section_hierarchy=current_hierarchy.copy()
                ))
                current_chunk = current_chunk[-self.chunk_overlap:]
            
            current_chunk += para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(PDFChunk(
                content=current_chunk.strip(),
                chunk_type=ChunkType.SECTION,
                section_hierarchy=current_hierarchy
            ))
        
        return chunks
    
    # ==================== 4. 整合阶段 ====================
    
    def _integrate(
        self,
        text_chunks: List[PDFChunk],
        tables: List[ExtractedTable],
        formulas: List[ExtractedFormula],
        images: List[ExtractedImage]
    ) -> List[PDFChunk]:
        """
        整合所有分块
        
        将文本块、表格、公式、图片描述合并为统一的分块列表
        """
        all_chunks = []
        
        # 1. 添加文本块
        all_chunks.extend(text_chunks)
        
        # 2. 添加表格块
        for table in tables:
            md_table = table.to_markdown()
            if md_table:
                all_chunks.append(PDFChunk(
                    content=md_table,
                    chunk_type=ChunkType.TABLE,
                    section_name=f"Table {table.table_id + 1}",
                    page_num=table.page_num,
                    metadata={'caption': table.caption}
                ))
        
        # 3. 添加公式块 (只添加行间公式)
        for formula in formulas:
            if not formula.is_inline and len(formula.latex) > 5:
                all_chunks.append(PDFChunk(
                    content=f"Formula: ${formula.latex}$\n\nContext: {formula.context[:200]}",
                    chunk_type=ChunkType.FORMULA,
                    section_name="Equations",
                    metadata={'latex': formula.latex}
                ))
        
        # 4. 添加图片描述块
        for image in images:
            content_parts = []
            if image.caption:
                content_parts.append(f"Figure: {image.caption}")
            if image.description:
                content_parts.append(f"Description: {image.description}")
            
            if content_parts:
                all_chunks.append(PDFChunk(
                    content="\n".join(content_parts),
                    chunk_type=ChunkType.IMAGE,
                    section_name=f"Figure {image.image_id + 1}",
                    page_num=image.page_num,
                    metadata={'has_description': image.description is not None}
                ))
        
        return all_chunks
    
    # ==================== 便捷方法 ====================
    
    def get_chunks_for_db(self, pdf_path: str) -> List[Dict]:
        """
        处理 PDF 并返回可直接入库的分块列表
        
        Returns:
            List of dicts with 'content', 'chunk_type', 'section_name', etc.
        """
        result = self.process(pdf_path)
        return result.get('chunks', [])


# 便捷函数
def process_pdf(pdf_path: str, **kwargs) -> Dict:
    """处理 PDF 文件"""
    processor = PDFMarkdownProcessor(**kwargs)
    return processor.process(pdf_path)


def get_pdf_chunks(pdf_path: str, **kwargs) -> List[Dict]:
    """获取 PDF 分块"""
    processor = PDFMarkdownProcessor(**kwargs)
    return processor.get_chunks_for_db(pdf_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_markdown_processor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    print(f"Processing: {pdf_path}")
    print("=" * 60)
    
    processor = PDFMarkdownProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        extract_tables=True,
        extract_formulas=True,
        extract_images=True,
        describe_images=False
    )
    
    result = processor.process(pdf_path)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print(f"\n📄 Metadata:")
    print(f"   Title: {result['metadata'].get('title', 'N/A')}")
    print(f"   Pages: {result['metadata'].get('page_count', 'N/A')}")
    
    print(f"\n📊 Extracted:")
    print(f"   Tables: {len(result['tables'])}")
    print(f"   Formulas: {len(result['formulas'])}")
    print(f"   Images: {len(result['images'])}")
    
    print(f"\n📦 Chunks: {len(result['chunks'])}")
    
    # 按类型统计
    type_counts = {}
    for chunk in result['chunks']:
        ct = chunk['chunk_type']
        type_counts[ct] = type_counts.get(ct, 0) + 1
    
    for ct, count in type_counts.items():
        print(f"   - {ct}: {count}")
    
    print(f"\n📝 Markdown preview (first 500 chars):")
    print(result['markdown'][:500])
