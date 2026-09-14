"""
Research Assistant - Main Application
科研论文智能助手主程序

核心功能：
1. 混合检索RAG：BM25 + BGE-M3 + BGE-Reranker
2. Skills 化能力层：paper-retrieval / experiment-design / academic-writing（渐进披露, 0 次路由 LLM 调用）
3. 记忆与个性化：可配置偏好类别 + 时间衰减 + 负反馈 + BGE-M3 语义召回 + 偏好回流检索
4. Qwen2.5-7B LoRA微调：学术写作优化

注：所有对外声明的指标必须可由仓库内脚本复现（见 README「性能指标」小节）。
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
load_dotenv()

from .retrieval import PaperRetriever, PaperInterpreter
from .retrieval import HybridRetriever, RetrievalEvaluator
from .retrieval.memory import MemoryStore
from .experiment import ExperimentPlanner
from .writing import AcademicWriter, CitationManager
from .tools import CodeGenerator, DataAnalyzer, Visualizer
from .utils.query_enhancer import QueryEnhancer
from .utils.helpers import resolve_env_placeholders
from .utils.llm import set_llm_cache

# 可选导入
try:
    from .agents import ResearchAgentGraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

try:
    from optional.finetune import LoRATrainer, TrainingConfig, WritingEvaluator
    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False

# 路径锚点:config 随包迁移(research_assistant/config/),运行时数据仍在项目根
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = Path(__file__).resolve().parent / 'config'

logger = logging.getLogger(__name__)


class ResearchAssistant:
    """科研论文智能助手主类"""
    
    def __init__(self, config_path: Optional[str] = None, verbose: bool = True, user_id: str = "default"):
        """
        初始化研究助手

        Args:
            config_path: 配置文件路径
            verbose: 是否输出初始化信息
            user_id: 用户标识,用于个性化记忆
        """
        self._verbose = verbose
        self.user_id = user_id
        self.status = {}  # 各模块加载状态,用于 health_report()
        
        # 加载配置
        if config_path is None:
            config_path = _CONFIG_DIR / 'config.yaml'

        self.config = self._load_config(config_path)

        # 加载prompts
        prompts_path = _CONFIG_DIR / 'prompts.yaml'
        self.prompts = self._load_config(prompts_path)

        # 用户记忆(个性化:偏好 + 历史问答)
        self.memory = MemoryStore(str(_PROJECT_ROOT / 'data' / 'memory.db'))

        # LLM 结果缓存(重复 query 不再重复调用 LLM)
        set_llm_cache(str(_PROJECT_ROOT / 'data' / 'cache' / 'llm_cache.json'))

        # 论文检索模块（API搜索）
        self.paper_retriever = PaperRetriever(
            self.config.get('paper_retrieval', {})
        )
        self.paper_interpreter = PaperInterpreter(
            self.config,
            self.prompts
        )
        self.paper_interpreter.preferences = self.memory.get_preferences(self.user_id)
        
        # 混合检索模块 - BM25 + BGE-M3 + BGE-Reranker
        hybrid_config = self.config.get('hybrid_retrieval', {})
        if hybrid_config:
            try:
                self.hybrid_retriever = HybridRetriever(
                    config=hybrid_config,
                    data_dir=str(_PROJECT_ROOT / 'data'),
                    verbose=verbose
                )
                self.status['hybrid_retriever'] = 'ok'
            except Exception as e:
                self.hybrid_retriever = None
                self.status['hybrid_retriever'] = f'error: {e}'
        else:
            self.hybrid_retriever = None
            self.status['hybrid_retriever'] = 'disabled (no config)'
        
        # Multi-Agent 模块（新）- LangGraph
        if HAS_LANGGRAPH:
            try:
                agent_config = {
                    'llm': self.config.get('llm', {}),
                    'max_iterations': self.config.get('multi_agent', {}).get('max_iterations', 10),
                    'output_dir': self.config.get('multi_agent', {}).get('output_dir', 'output/agent_outputs'),
                    'citation_style': self.config.get('multi_agent', {}).get('citation_style', 'ieee')
                }
                self.agent_graph = ResearchAgentGraph(agent_config, self.hybrid_retriever, memory=self.memory, user_id=self.user_id)
                self.status['agent_graph'] = 'ok'
            except Exception as e:
                self.agent_graph = None
                self.status['agent_graph'] = f'error: {e}'
        else:
            self.agent_graph = None
            self.status['agent_graph'] = 'unavailable (langgraph not installed)'
        
        # 实验设计模块
        self.experiment_planner = ExperimentPlanner(
            self.config,
            self.prompts
        )
        
        # 写作辅助模块
        self.academic_writer = AcademicWriter(
            self.config,
            self.prompts
        )
        self.citation_manager = CitationManager(
            style=self.config.get('writing_assistant', {}).get('citation_style', 'ieee')
        )
        
        # 工具模块
        self.code_generator = CodeGenerator(self.config)
        self.data_analyzer = DataAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)
        
        # 查询增强器
        self.query_enhancer = QueryEnhancer(self.config)
    
    def health_report(self) -> Dict:
        """返回各模块加载状态,便于诊断(而非静默降级)。"""
        return dict(self.status)

    def _load_config(self, config_path: Path) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return resolve_env_placeholders(config)
        except Exception as e:
            logger.warning(f"⚠️ Error loading config from {config_path}: {e}")
            return {}
    
    # ==================== 论文检索与解读 ====================
    
    def search_papers(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: int = 10,
        use_cache: bool = True,
        auto_enhance: bool = True,
        broad_search: bool = False,
        verbose: bool = True
    ) -> List[Dict]:
        """
        搜索论文
        
        Args:
            query: 搜索查询（支持中文，会自动转换为英文学术术语）
            sources: 数据源
            max_results: 最大结果数
            use_cache: 是否使用缓存
            auto_enhance: 是否自动增强查询（中文转英文）
            broad_search: 是否使用广泛搜索（使用更通用的关键词，提高召回率）
            
        Returns:
            论文列表
        """
        # 自动增强查询
        if auto_enhance:
            enhanced_result = self.query_enhancer.suggest_search_keywords(query, broad_search=broad_search)
            search_query = enhanced_result['suggested_query']
            
            if verbose:
                logger.info(f"原始查询: {query}")
                logger.info(f"增强查询: {search_query}")
        else:
            search_query = query

        # 个性化:合并用户关键词偏好到查询
        keyword_prefs = self.memory.get_preferences(self.user_id).get('keyword', [])
        if keyword_prefs:
            search_query = f"{search_query} {' '.join(keyword_prefs)}"
            if verbose:
                logger.info(f"个性化查询(含关键词偏好): {search_query}")

        # 尝试从缓存加载
        if use_cache:
            cached_papers = self.paper_retriever.load_from_cache(search_query)
            if cached_papers:
                if verbose:
                    logger.info(f"✅ 从缓存加载 {len(cached_papers)} 篇论文")
                return cached_papers[:max_results]
        
        # 搜索论文
        papers = self.paper_retriever.search_papers(
            query=search_query,
            sources=sources,
            max_results=max_results,
            verbose=verbose
        )
        
        # 自动添加到混合检索索引
        if papers and self.hybrid_retriever:
            try:
                self.hybrid_retriever.add_papers(papers, verbose=verbose)
            except Exception as e:
                if verbose:
                    logger.warning(f"⚠️ 索引失败: {e}")
        
        return papers
    
    def evaluate_retrieval(self, eval_data_path: Optional[str] = None, k_values: List[int] = None) -> Dict:
        """运行可复现的检索评测(BM25,真实语料)并返回摘要。"""
        from .retrieval.evaluation import evaluate_bm25_retrieval

        if k_values is None:
            k_values = [1, 3, 5]

        corpus_path = _PROJECT_ROOT / 'data' / 'eval' / 'corpus.json'
        if eval_data_path is None:
            eval_data_path = str(_PROJECT_ROOT / 'data' / 'eval' / 'retrieval_eval.json')

        if not Path(corpus_path).exists() or not Path(eval_data_path).exists():
            logger.warning("⚠️ 缺少评测语料或评测数据(data/eval/corpus.json + retrieval_eval.json)")
            return {}

        try:
            return evaluate_bm25_retrieval(str(corpus_path), eval_data_path, k_values=k_values)
        except Exception as e:
            logger.warning("⚠️ 评测失败: %s", e)
            return {}

    def interpret_paper(self, paper: Dict, verbose: bool = True) -> Dict:
        """
        解读论文
        
        Args:
            paper: 论文信息
            verbose: 是否输出详细信息
            
        Returns:
            解读结果
        """
        interpretation = self.paper_interpreter.interpret_paper(paper)
        return interpretation
    
    # ==================== 混合检索 ====================
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        use_hyde: bool = False,
        verbose: bool = True
    ) -> List[Dict]:
        """
        混合检索：BM25 + BGE-M3 + BGE-Reranker
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            use_rerank: 是否使用精排
            use_hyde: 是否使用 HyDE (Hypothetical Document Embeddings)
            verbose: 是否输出详细信息
            
        Returns:
            检索结果列表
        """
        if self.hybrid_retriever is None:
            if verbose:
                logger.warning("⚠️ 混合检索器未初始化")
            return []
        
        results = self.hybrid_retriever.search(
            query=query,
            top_k=top_k,
            use_rerank=use_rerank,
            use_hyde=use_hyde,
            verbose=verbose
        )
        
        return results
    
    def index_papers(self, papers: List[Dict], process_pdf: bool = False) -> int:
        """
        将论文添加到混合检索索引
        
        Args:
            papers: 论文列表
            process_pdf: 是否处理PDF全文
            
        Returns:
            添加的chunk数量
        """
        if self.hybrid_retriever is None:
            logger.warning("⚠️ Hybrid retriever not available")
            return 0
        
        return self.hybrid_retriever.add_papers(papers, process_pdf)
    
    # ==================== Multi-Agent (NEW) ====================
    
    def run_agent(self, task: str, max_iterations: int = 10) -> Dict:
        """
        运行 Multi-Agent 系统处理复杂任务
        
        Args:
            task: 任务描述
            max_iterations: 最大迭代次数
            
        Returns:
            执行结果
        """
        if self.agent_graph is None:
            logger.warning("⚠️ Multi-Agent system not available")
            return {'success': False, 'error': 'Agent system not initialized'}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🤖 MULTI-AGENT TASK")
        logger.info(f"{'='*60}")
        logger.info(f"Task: {task}")
        logger.info(f"{'='*60}\n")
        
        result = self.agent_graph.run(task, max_iterations)
        
        if result.get('success'):
            logger.info(f"✅ Task completed successfully")
        else:
            logger.error(f"❌ Task failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def run_retrieval_agent(self, query: str) -> Dict:
        """运行检索Agent"""
        if self.agent_graph is None:
            return {'success': False, 'error': 'Agent system not initialized'}
        return self.agent_graph.run_retrieval(query)
    
    def run_experiment_agent(self, task: str, papers: List[Dict] = None) -> Dict:
        """运行实验设计Agent"""
        if self.agent_graph is None:
            return {'success': False, 'error': 'Agent system not initialized'}
        return self.agent_graph.run_experiment(task, papers)
    
    def run_writing_agent(self, task: str, papers: List[Dict] = None, draft: str = '') -> Dict:
        """运行写作Agent"""
        if self.agent_graph is None:
            return {'success': False, 'error': 'Agent system not initialized'}
        return self.agent_graph.run_writing(task, papers, draft)
    
    # ==================== 实验设计 ====================
    
    def design_experiment(
        self,
        research_question: str,
        papers: Optional[List[Dict]] = None,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        设计实验方案
        
        Args:
            research_question: 研究问题
            papers: 相关论文
            constraints: 约束条件
            
        Returns:
            实验方案
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 EXPERIMENT DESIGN")
        logger.info(f"{'='*60}\n")
        
        # 如果没有提供论文，尝试从混合检索获取
        if papers is None and self.hybrid_retriever:
            results = self.hybrid_search(research_question, top_k=3)
            papers = [r.get('paper', r) for r in results]
        
        experiment_plan = self.experiment_planner.design_experiment(
            research_question=research_question,
            papers_context=papers,
            constraints=constraints
        )
        
        return experiment_plan
    
    def generate_experiment_code(
        self,
        experiment_plan: Dict,
        framework: str = "pytorch"
    ) -> str:
        """
        生成实验代码
        
        Args:
            experiment_plan: 实验方案
            framework: 框架
            
        Returns:
            生成的代码
        """
        code = self.experiment_planner.generate_experiment_code(
            experiment_plan,
            framework
        )
        return code
    
    # ==================== 学术写作 ====================
    
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
            title: 标题
            keywords: 关键词
            background: 背景
            methods: 方法
            results: 结果
            conclusion: 结论
            language: 语言
            
        Returns:
            生成的摘要
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"✍️ GENERATING ABSTRACT")
        logger.info(f"{'='*60}\n")
        
        abstract = self.academic_writer.generate_abstract(
            title=title,
            keywords=keywords,
            background=background,
            methods=methods,
            results=results,
            conclusion=conclusion,
            language=language
        )
        
        return abstract
    
    def generate_full_paper(
        self,
        paper_info: Dict,
        output_path: str
    ):
        """
        生成完整论文
        
        Args:
            paper_info: 论文信息
            output_path: 输出路径
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📝 GENERATING FULL PAPER")
        logger.info(f"{'='*60}\n")
        
        sections = []
        
        # 摘要
        if 'abstract' in paper_info:
            sections.append("# Abstract\n\n" + paper_info['abstract'])
        
        # 引言
        if 'introduction' in paper_info:
            intro = self.academic_writer.generate_introduction(
                topic=paper_info.get('topic', ''),
                research_question=paper_info.get('research_question', ''),
                related_work=paper_info.get('related_work', ''),
                contributions=paper_info.get('contributions', [])
            )
            sections.append(intro)
        
        # 方法
        if 'methodology' in paper_info:
            method = self.academic_writer.generate_methodology(
                method_name=paper_info['methodology'].get('name', ''),
                technical_details=paper_info['methodology'].get('details', '')
            )
            sections.append(method)
        
        # 结果
        if 'results' in paper_info:
            results = self.academic_writer.generate_results(
                experimental_setup=paper_info['results'].get('setup', ''),
                datasets=paper_info['results'].get('datasets', []),
                metrics=paper_info['results'].get('metrics', []),
                results_data=paper_info['results'].get('data', {})
            )
            sections.append(results)
        
        # 讨论
        if 'discussion' in paper_info:
            discussion = self.academic_writer.generate_discussion(
                findings=paper_info['discussion'].get('findings', []),
                interpretation=paper_info['discussion'].get('interpretation', ''),
                limitations=paper_info['discussion'].get('limitations', []),
                future_work=paper_info['discussion'].get('future_work', [])
            )
            sections.append(discussion)
        
        # 参考文献
        bibliography = self.citation_manager.generate_bibliography()
        sections.append(bibliography)
        
        # 合并并保存
        full_paper = '\n\n---\n\n'.join(sections)
        self.academic_writer.save_document(full_paper, output_path)
        
        logger.info(f"✅ Full paper generated and saved to {output_path}")
    
    # ==================== 工作流示例 ====================
    
    def complete_research_workflow(
        self,
        research_question: str,
        output_dir: str = None
    ):
        """
        完整的研究工作流示例
        
        Args:
            research_question: 研究问题
            output_dir: 输出目录（默认为项目根目录下的output）
        """
        if output_dir is None:
            output_dir = _PROJECT_ROOT / "output"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 COMPLETE RESEARCH WORKFLOW")
        logger.info(f"Research Question: {research_question}")
        logger.info(f"{'='*60}\n")
        
        # 1. 搜索相关论文
        papers = self.search_papers(research_question, max_results=5)
        
        # 2. 解读关键论文
        if papers:
            interpretation = self.interpret_paper(papers[0])
            
            # 保存解读结果
            import json
            with open(output_path / 'paper_interpretation.json', 'w', encoding='utf-8') as f:
                json.dump(interpretation, f, ensure_ascii=False, indent=2)
        
        # 3. 设计实验
        experiment_plan = self.design_experiment(research_question, papers)
        
        # 保存实验方案
        self.experiment_planner.save_plan(
            experiment_plan,
            str(output_path / 'experiment_plan.json')
        )
        
        # 4. 生成实验代码
        code = self.generate_experiment_code(experiment_plan)
        
        with open(output_path / 'experiment_code.py', 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"\n✅ Complete workflow finished! Check {output_dir} for results.")
