"""
Agent Tools
LangGraph Agent 工具集：Python执行器、统计分析、数据可视化、论文检索
"""

import os
import sys
import json
import tempfile
import traceback
from io import StringIO
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    output: Any
    error: Optional[str] = None
    artifacts: Optional[Dict] = None  # 生成的文件路径等


class PythonExecutorTool:
    """
    Python 代码执行器
    
    安全地执行 Python 代码，支持数据分析和科学计算
    """
    
    name = "python_executor"
    description = """Execute Python code for data analysis, computation, or file operations.
    Input should be valid Python code as a string.
    The code has access to: numpy (np), pandas (pd), matplotlib.pyplot (plt), scipy, sklearn.
    Use print() to output results. Generated plots will be saved automatically."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "agent_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.execution_count = 0
    
    def execute(self, code: str) -> ToolResult:
        """执行 Python 代码"""
        self.execution_count += 1
        
        # 准备执行环境
        local_vars = {
            'np': np,
            'pd': pd,
            '__output_dir__': str(self.output_dir),
            '__exec_id__': self.execution_count
        }
        
        # 尝试导入可选库
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            local_vars['plt'] = plt
        except ImportError:
            pass
        
        try:
            from scipy import stats as scipy_stats
            import scipy
            local_vars['scipy'] = scipy
            local_vars['scipy_stats'] = scipy_stats
        except ImportError:
            pass
        
        try:
            import sklearn
            local_vars['sklearn'] = sklearn
        except ImportError:
            pass
        
        # 捕获输出
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        artifacts = {}
        
        try:
            # 执行代码
            exec(code, local_vars)
            
            # 获取输出
            output = sys.stdout.getvalue()
            
            # 检查是否有图表需要保存
            if 'plt' in local_vars:
                plt = local_vars['plt']
                if plt.get_fignums():
                    fig_path = self.output_dir / f"figure_{self.execution_count}.png"
                    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close('all')
                    artifacts['figure'] = str(fig_path)
            
            return ToolResult(
                success=True,
                output=output if output else "Code executed successfully (no output)",
                artifacts=artifacts if artifacts else None
            )
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return ToolResult(
                success=False,
                output=None,
                error=error_msg
            )
        finally:
            sys.stdout = old_stdout


class StatisticalAnalysisTool:
    """
    统计分析工具
    
    提供常用的统计分析功能
    """
    
    name = "statistical_analysis"
    description = """Perform statistical analysis on data.
    Supports: descriptive statistics, hypothesis testing, correlation analysis, regression.
    Input: JSON with 'analysis_type' and 'data' fields."""
    
    def __init__(self):
        pass
    
    def execute(self, params: Dict) -> ToolResult:
        """执行统计分析"""
        analysis_type = params.get('analysis_type', 'descriptive')
        data = params.get('data')
        
        if data is None:
            return ToolResult(success=False, output=None, error="No data provided")
        
        try:
            if analysis_type == 'descriptive':
                return self._descriptive_stats(data)
            elif analysis_type == 'correlation':
                return self._correlation_analysis(data)
            elif analysis_type == 't_test':
                return self._t_test(data, params)
            elif analysis_type == 'anova':
                return self._anova(data, params)
            elif analysis_type == 'regression':
                return self._regression(data, params)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown analysis type: {analysis_type}"
                )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def _descriptive_stats(self, data) -> ToolResult:
        """描述性统计"""
        df = pd.DataFrame(data)
        stats = df.describe().to_dict()
        
        # 添加额外统计量
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col]['skewness'] = float(df[col].skew())
            stats[col]['kurtosis'] = float(df[col].kurtosis())
        
        return ToolResult(success=True, output=stats)
    
    def _correlation_analysis(self, data) -> ToolResult:
        """相关性分析"""
        df = pd.DataFrame(data)
        corr_matrix = df.corr().to_dict()
        
        return ToolResult(success=True, output={
            'correlation_matrix': corr_matrix,
            'method': 'pearson'
        })
    
    def _t_test(self, data, params) -> ToolResult:
        """T检验"""
        try:
            from scipy import stats
        except ImportError:
            return ToolResult(success=False, output=None, error="scipy not installed")
        
        group1 = data.get('group1', [])
        group2 = data.get('group2', [])
        
        if not group1 or not group2:
            return ToolResult(success=False, output=None, error="Need group1 and group2 data")
        
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        return ToolResult(success=True, output={
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        })
    
    def _anova(self, data, params) -> ToolResult:
        """方差分析"""
        try:
            from scipy import stats
        except ImportError:
            return ToolResult(success=False, output=None, error="scipy not installed")
        
        groups = data.get('groups', [])
        if len(groups) < 2:
            return ToolResult(success=False, output=None, error="Need at least 2 groups")
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        return ToolResult(success=True, output={
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        })
    
    def _regression(self, data, params) -> ToolResult:
        """回归分析"""
        try:
            from scipy import stats
        except ImportError:
            return ToolResult(success=False, output=None, error="scipy not installed")
        
        x = data.get('x', [])
        y = data.get('y', [])
        
        if not x or not y:
            return ToolResult(success=False, output=None, error="Need x and y data")
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return ToolResult(success=True, output={
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_error': float(std_err)
        })


class DataVisualizationTool:
    """
    数据可视化工具
    
    生成各类图表
    """
    
    name = "data_visualization"
    description = """Create data visualizations.
    Supports: line, bar, scatter, histogram, heatmap, box plots.
    Input: JSON with 'plot_type', 'data', and optional 'config' fields."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "agent_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_count = 0
    
    def execute(self, params: Dict) -> ToolResult:
        """生成可视化"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return ToolResult(success=False, output=None, error="matplotlib/seaborn not installed")
        
        plot_type = params.get('plot_type', 'line')
        data = params.get('data')
        config = params.get('config', {})
        
        if data is None:
            return ToolResult(success=False, output=None, error="No data provided")
        
        self.plot_count += 1
        fig_path = self.output_dir / f"plot_{self.plot_count}.png"
        
        try:
            fig, ax = plt.subplots(figsize=config.get('figsize', (10, 6)))
            
            if plot_type == 'line':
                self._line_plot(ax, data, config)
            elif plot_type == 'bar':
                self._bar_plot(ax, data, config)
            elif plot_type == 'scatter':
                self._scatter_plot(ax, data, config)
            elif plot_type == 'histogram':
                self._histogram(ax, data, config)
            elif plot_type == 'heatmap':
                self._heatmap(fig, data, config)
            elif plot_type == 'box':
                self._box_plot(ax, data, config)
            else:
                plt.close(fig)
                return ToolResult(success=False, output=None, error=f"Unknown plot type: {plot_type}")
            
            # 设置标题和标签
            if config.get('title'):
                ax.set_title(config['title'])
            if config.get('xlabel'):
                ax.set_xlabel(config['xlabel'])
            if config.get('ylabel'):
                ax.set_ylabel(config['ylabel'])
            
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return ToolResult(
                success=True,
                output=f"Plot saved to {fig_path}",
                artifacts={'figure': str(fig_path)}
            )
            
        except Exception as e:
            plt.close('all')
            return ToolResult(success=False, output=None, error=str(e))
    
    def _line_plot(self, ax, data, config):
        """折线图"""
        if isinstance(data, dict):
            for label, values in data.items():
                ax.plot(values, label=label)
            ax.legend()
        else:
            ax.plot(data)
    
    def _bar_plot(self, ax, data, config):
        """柱状图"""
        if isinstance(data, dict):
            x = list(data.keys())
            y = list(data.values())
            ax.bar(x, y)
        else:
            ax.bar(range(len(data)), data)
    
    def _scatter_plot(self, ax, data, config):
        """散点图"""
        x = data.get('x', [])
        y = data.get('y', [])
        ax.scatter(x, y, alpha=0.6)
    
    def _histogram(self, ax, data, config):
        """直方图"""
        bins = config.get('bins', 30)
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    
    def _heatmap(self, fig, data, config):
        """热力图"""
        import seaborn as sns
        ax = fig.gca()
        df = pd.DataFrame(data)
        sns.heatmap(df, annot=config.get('annot', True), cmap='coolwarm', ax=ax)
    
    def _box_plot(self, ax, data, config):
        """箱线图"""
        if isinstance(data, dict):
            ax.boxplot(list(data.values()), labels=list(data.keys()))
        else:
            ax.boxplot(data)


class PaperRetrievalTool:
    """
    论文检索工具
    
    封装混合检索器供 Agent 调用
    """
    
    name = "paper_retrieval"
    description = """Search academic papers using hybrid retrieval (BM25 + BGE-M3 + Reranker).
    Input: JSON with 'query' (required) and optional 'top_k', 'use_rerank' fields.
    Returns relevant papers with titles, abstracts, and relevance scores."""
    
    def __init__(self, retriever=None):
        """
        初始化
        
        Args:
            retriever: HybridRetriever 实例
        """
        self.retriever = retriever
    
    def set_retriever(self, retriever):
        """设置检索器"""
        self.retriever = retriever
    
    def execute(self, params: Dict) -> ToolResult:
        """执行检索"""
        if self.retriever is None:
            return ToolResult(
                success=False,
                output=None,
                error="Retriever not initialized"
            )
        
        query = params.get('query')
        if not query:
            return ToolResult(success=False, output=None, error="No query provided")
        
        top_k = params.get('top_k', 5)
        use_rerank = params.get('use_rerank', True)
        
        try:
            results = self.retriever.search(
                query=query,
                top_k=top_k,
                use_rerank=use_rerank
            )
            
            # 格式化结果
            formatted_results = []
            for r in results:
                paper = r.get('paper', {})
                formatted_results.append({
                    'title': paper.get('title', 'N/A'),
                    'authors': paper.get('authors', [])[:3],
                    'score': r.get('score', 0),
                    'content_snippet': r.get('content', '')[:300] + '...' if r.get('content') else '',
                    'paper_id': paper.get('paper_id', ''),
                    'doi': paper.get('doi')
                })
            
            return ToolResult(
                success=True,
                output={
                    'query': query,
                    'num_results': len(formatted_results),
                    'results': formatted_results
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class CitationTool:
    """
    引用管理工具
    
    格式化引用和生成参考文献
    """
    
    name = "citation_manager"
    description = """Manage citations and generate bibliography.
    Input: JSON with 'action' (format_citation, generate_bibliography) and 'papers' data."""
    
    def __init__(self, style: str = "ieee"):
        self.style = style
    
    def execute(self, params: Dict) -> ToolResult:
        """执行引用操作"""
        action = params.get('action', 'format_citation')
        papers = params.get('papers', [])
        
        try:
            if action == 'format_citation':
                return self._format_citations(papers)
            elif action == 'generate_bibliography':
                return self._generate_bibliography(papers)
            else:
                return ToolResult(success=False, output=None, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def _format_citations(self, papers: List[Dict]) -> ToolResult:
        """格式化引用"""
        citations = []
        for i, paper in enumerate(papers, 1):
            citation = self._format_single_citation(paper, i)
            citations.append(citation)
        
        return ToolResult(success=True, output={'citations': citations})
    
    def _format_single_citation(self, paper: Dict, index: int) -> str:
        """格式化单个引用"""
        authors = paper.get('authors', ['Unknown'])
        title = paper.get('title', 'Untitled')
        year = paper.get('published', '')[:4] if paper.get('published') else 'n.d.'
        
        if self.style == 'ieee':
            author_str = ', '.join(authors[:3])
            if len(authors) > 3:
                author_str += ' et al.'
            return f"[{index}] {author_str}, \"{title},\" {year}."
        else:
            # APA style
            author_str = ', '.join(authors[:3])
            if len(authors) > 3:
                author_str += ', et al.'
            return f"{author_str} ({year}). {title}."
    
    def _generate_bibliography(self, papers: List[Dict]) -> ToolResult:
        """生成参考文献列表"""
        bibliography = []
        for i, paper in enumerate(papers, 1):
            entry = self._format_single_citation(paper, i)
            bibliography.append(entry)
        
        return ToolResult(
            success=True,
            output={
                'bibliography': '\n'.join(bibliography),
                'count': len(bibliography)
            }
        )


# 工具注册表
AVAILABLE_TOOLS = {
    'python_executor': PythonExecutorTool,
    'statistical_analysis': StatisticalAnalysisTool,
    'data_visualization': DataVisualizationTool,
    'paper_retrieval': PaperRetrievalTool,
    'citation_manager': CitationTool
}


def get_tool(tool_name: str, **kwargs):
    """获取工具实例"""
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    return AVAILABLE_TOOLS[tool_name](**kwargs)


def get_all_tool_descriptions() -> List[Dict]:
    """获取所有工具的描述"""
    descriptions = []
    for name, tool_class in AVAILABLE_TOOLS.items():
        descriptions.append({
            'name': name,
            'description': tool_class.description
        })
    return descriptions
