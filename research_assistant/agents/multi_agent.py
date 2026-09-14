"""
LangGraph Multi-Agent Framework
基于 LangGraph 的多智能体协作框架

实现三大功能模块：
1. 论文检索 Agent (RetrievalAgent)
2. 实验设计 Agent (ExperimentAgent)
3. 写作辅助 Agent (WritingAgent)

使用 ReAct 规划策略进行工具调用
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    StateGraph = None
    END = None
    logger.warning("⚠️ LangGraph not installed. Run: pip install langgraph")

try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseMessage = None
    HumanMessage = None
    AIMessage = None
    SystemMessage = None
    ToolMessage = None
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    ChatOpenAI = None
    logger.warning("⚠️ LangChain not installed")

from .tools import (
    PythonExecutorTool,
    StatisticalAnalysisTool,
    DataVisualizationTool,
    PaperRetrievalTool,
    CitationTool,
    ToolResult
)
from ..retrieval.memory import format_preferences


class AgentType(Enum):
    """Agent 类型

    注意：``SUPERVISOR`` 已移除。路由不再由 LLM 扮演 supervisor 完成，
    改为 ``SkillRegistry`` 的确定性选择（0 次 LLM 调用）。
    """
    RETRIEVAL = "retrieval"
    EXPERIMENT = "experiment"
    WRITING = "writing"


if HAS_LANGCHAIN:
    class AgentState(TypedDict):
        """Agent 状态定义"""
        messages: Annotated[Sequence[BaseMessage], "对话历史"]
        current_agent: str
        task: str
        task_type: str  # retrieval, experiment, writing, complex
        retrieved_papers: List[Dict]
        experiment_plan: Optional[Dict]
        draft_content: str
        tool_outputs: List[Dict]
        iteration: int
        max_iterations: int
        final_output: Optional[str]
        error: Optional[str]
else:
    # 当 LangChain 未安装时使用简化版本
    class AgentState(TypedDict):
        """Agent 状态定义"""
        messages: List[Dict]
        current_agent: str
        task: str
        task_type: str
        retrieved_papers: List[Dict]
        experiment_plan: Optional[Dict]
        draft_content: str
        tool_outputs: List[Dict]
        iteration: int
        max_iterations: int
        final_output: Optional[str]
        error: Optional[str]


# ReAct Prompts
# 说明：原 SUPERVISOR_PROMPT 已删除 —— 路由改由 SkillRegistry 确定性完成，
# 不再消耗一次 LLM 调用来"决定派给谁"。三类 Agent 的系统提示保留，
# 但在 skill 化架构中它们由 skills/*/SKILL.md 的正文按需提供（见 skills/）。
RETRIEVAL_AGENT_PROMPT = """你是一个专业的学术论文检索专家。你的任务是帮助用户找到相关的学术论文。

你可以使用以下工具：
- paper_retrieval: 搜索学术论文（支持混合检索：BM25 + BGE-M3 + Reranker）

使用 ReAct 格式进行推理和行动：

Thought: 分析用户需求，思考如何检索
Action: 选择工具和参数
Observation: 观察工具返回结果
... (可以重复多次)
Final Answer: 总结检索结果

当前任务: {task}
已有上下文: {context}

请开始你的检索工作。"""

EXPERIMENT_AGENT_PROMPT = """你是一个专业的科研实验设计专家。你的任务是帮助用户设计实验方案、生成代码、进行数据分析。

你可以使用以下工具：
- python_executor: 执行Python代码进行计算和分析
- statistical_analysis: 进行统计分析（描述统计、假设检验、回归分析等）
- data_visualization: 生成数据可视化图表
- paper_retrieval: 检索相关论文作为参考

使用 ReAct 格式进行推理和行动：

Thought: 分析实验需求，规划实验步骤
Action: 选择工具和参数
Observation: 观察工具返回结果
... (可以重复多次)
Final Answer: 总结实验方案和结果

当前任务: {task}
相关论文: {papers}
已有上下文: {context}

请开始你的实验设计工作。"""

WRITING_AGENT_PROMPT = """你是一个专业的学术写作专家。你的任务是帮助用户进行论文写作、文本润色、引用管理。

你可以使用以下工具：
- citation_manager: 管理引用和生成参考文献
- paper_retrieval: 检索相关论文作为引用来源

使用 ReAct 格式进行推理和行动：

Thought: 分析写作需求，规划写作结构
Action: 选择工具和参数（如需要）
Observation: 观察工具返回结果
... (可以重复多次)
Final Answer: 输出写作内容

写作要求：
1. 使用学术规范的语言
2. 逻辑清晰，结构完整
3. 适当引用相关文献

当前任务: {task}
相关论文: {papers}
已有草稿: {draft}

请开始你的写作工作。"""


#: skill 名 → 图节点类型（dispatch 用）
_SKILL_TO_TASK_TYPE = {
    'paper-retrieval': 'retrieval',
    'experiment-design': 'experiment',
    'academic-writing': 'writing',
}

#: skill 名 → ReActAgent 的键（dispatch 把 skill 正文注入对应 Agent）
_SKILL_TO_AGENT_KEY = {
    'paper-retrieval': 'retrieval',
    'experiment-design': 'experiment',
    'academic-writing': 'writing',
}


class ReActAgent:
    """
    ReAct Agent 基类
    
    实现 Thought -> Action -> Observation 循环
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        llm,
        tools: Dict[str, Any],
        max_iterations: int = 5
    ):
        self.agent_type = agent_type
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        #: skill 化后由 SkillLoader 注入的按需上下文（取代固定 system prompt）
        self.skill_prompt: Optional[str] = None
    
    def get_prompt(self) -> str:
        """获取Agent的系统提示。

        优先使用按需加载的 skill 正文（渐进披露）；未注入时回退到内置模板。
        """
        if self.skill_prompt:
            return self.skill_prompt
        prompts = {
            AgentType.RETRIEVAL: RETRIEVAL_AGENT_PROMPT,
            AgentType.EXPERIMENT: EXPERIMENT_AGENT_PROMPT,
            AgentType.WRITING: WRITING_AGENT_PROMPT,
        }
        return prompts.get(self.agent_type, "")
    
    def parse_action(self, response: str) -> Optional[Dict]:
        """解析LLM响应中的Action"""
        # 查找 Action: 行
        lines = response.split('\n')
        action_line = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith('Action:'):
                action_line = line.strip()[7:].strip()
                # 尝试解析JSON
                try:
                    # 查找JSON块
                    json_start = response.find('{', response.find('Action:'))
                    if json_start != -1:
                        # 找到匹配的结束括号
                        depth = 0
                        json_end = json_start
                        for j, c in enumerate(response[json_start:]):
                            if c == '{':
                                depth += 1
                            elif c == '}':
                                depth -= 1
                                if depth == 0:
                                    json_end = json_start + j + 1
                                    break
                        
                        json_str = response[json_start:json_end]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
                
                # 简单解析
                if ':' in action_line:
                    tool_name = action_line.split(':')[0].strip()
                    return {'tool': tool_name, 'input': action_line}
        
        return None
    
    def execute_tool(self, action: Dict) -> ToolResult:
        """执行工具"""
        tool_name = action.get('tool', '')
        tool_input = action.get('input', {})
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_name}"
            )
        
        tool = self.tools[tool_name]
        
        # 根据工具类型调用
        if isinstance(tool_input, str):
            # 对于代码执行器
            if tool_name == 'python_executor':
                return tool.execute(tool_input)
            # 其他工具尝试解析为JSON
            try:
                tool_input = json.loads(tool_input)
            except:
                tool_input = {'query': tool_input}
        
        return tool.execute(tool_input)
    
    def run(self, task: str, context: Dict = None) -> Dict:
        """
        运行Agent
        
        Args:
            task: 任务描述
            context: 上下文信息
            
        Returns:
            执行结果
        """
        if context is None:
            context = {}
        
        # 构建提示
        prompt = self.get_prompt().format(
            task=task,
            context=json.dumps(context.get('context', {}), ensure_ascii=False),
            papers=json.dumps(context.get('papers', []), ensure_ascii=False)[:2000],
            draft=context.get('draft', '')[:1000]
        )
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"请处理以下任务：{task}")
        ]
        
        tool_outputs = []
        final_answer = None
        
        for iteration in range(self.max_iterations):
            # 调用LLM
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # 检查是否有Final Answer
            if 'Final Answer:' in response_text:
                final_idx = response_text.find('Final Answer:')
                final_answer = response_text[final_idx + 13:].strip()
                break
            
            # 解析Action
            action = self.parse_action(response_text)
            
            if action:
                # 执行工具
                result = self.execute_tool(action)
                
                tool_outputs.append({
                    'iteration': iteration + 1,
                    'action': action,
                    'result': {
                        'success': result.success,
                        'output': str(result.output)[:500] if result.output else None,
                        'error': result.error
                    }
                })
                
                # 添加观察结果到消息
                observation = f"Observation: {json.dumps(result.output, ensure_ascii=False)[:1000]}" if result.success else f"Observation: Error - {result.error}"
                messages.append(AIMessage(content=response_text))
                messages.append(HumanMessage(content=observation))
            else:
                # 没有Action，可能直接给出了答案
                final_answer = response_text
                break
        
        return {
            'agent_type': self.agent_type.value,
            'task': task,
            'iterations': len(tool_outputs),
            'tool_outputs': tool_outputs,
            'final_answer': final_answer,
            'success': final_answer is not None
        }


class ResearchAgentGraph:
    """
    科研助手多Agent图
    
    使用 LangGraph 构建多Agent协作流程
    """
    
    def __init__(self, config: Dict, retriever=None, memory=None, user_id: str = "default",
                 skills_dir=None):
        """
        初始化

        Args:
            config: 配置字典
            retriever: HybridRetriever 实例
            memory: MemoryStore 实例(可选,用于个性化记忆)
            user_id: 用户标识
            skills_dir: skill 目录(默认 <repo>/skills)
        """
        self.config = config
        self.retriever = retriever
        self.memory = memory
        self.user_id = user_id

        # skill 注册表与加载器（渐进披露：常驻只有 frontmatter）
        from ..skills import SkillLoader, SkillRegistry
        self.registry = SkillRegistry(skills_dir)
        self.loader = SkillLoader(self.registry)
        self.hops = 0  # 本次运行的路由跳数（度量 M2）
        
        # 初始化LLM
        llm_config = config.get('llm', {})
        api_keys_config = config.get('api_keys', {})
        api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY') or api_keys_config.get('openai_api_key')
        base_url = llm_config.get('base_url') or api_keys_config.get('openai_base_url')
        
        if api_key and HAS_LANGCHAIN:
            self.llm = ChatOpenAI(
                model=llm_config.get('model', 'deepseek-chat'),
                temperature=llm_config.get('temperature', 0.7),
                api_key=api_key,
                base_url=base_url
            )
        else:
            self.llm = None
            logger.warning("⚠️ LLM not initialized (missing API key or LangChain)")
        
        # 初始化工具
        self.tools = self._init_tools()
        
        # 初始化Agents
        self.agents = self._init_agents()
        
        # 构建图
        self.graph = self._build_graph() if HAS_LANGGRAPH else None
    
    def _init_tools(self) -> Dict[str, Any]:
        """初始化工具"""
        output_dir = self.config.get('output_dir', 'output/agent_outputs')
        
        tools = {
            'python_executor': PythonExecutorTool(output_dir=output_dir),
            'statistical_analysis': StatisticalAnalysisTool(),
            'data_visualization': DataVisualizationTool(output_dir=output_dir),
            'paper_retrieval': PaperRetrievalTool(retriever=self.retriever),
            'citation_manager': CitationTool(style=self.config.get('citation_style', 'ieee'))
        }
        
        return tools
    
    def _init_agents(self) -> Dict[str, ReActAgent]:
        """初始化Agents"""
        if not self.llm:
            return {}
        
        max_iterations = self.config.get('max_iterations', 5)
        
        agents = {
            'retrieval': ReActAgent(
                AgentType.RETRIEVAL,
                self.llm,
                {'paper_retrieval': self.tools['paper_retrieval']},
                max_iterations
            ),
            'experiment': ReActAgent(
                AgentType.EXPERIMENT,
                self.llm,
                {
                    'python_executor': self.tools['python_executor'],
                    'statistical_analysis': self.tools['statistical_analysis'],
                    'data_visualization': self.tools['data_visualization'],
                    'paper_retrieval': self.tools['paper_retrieval']
                },
                max_iterations
            ),
            'writing': ReActAgent(
                AgentType.WRITING,
                self.llm,
                {
                    'citation_manager': self.tools['citation_manager'],
                    'paper_retrieval': self.tools['paper_retrieval']
                },
                max_iterations
            )
        }
        
        return agents
    
    def _build_graph(self) -> Optional[StateGraph]:
        """构建LangGraph状态图"""
        if not HAS_LANGGRAPH:
            return None
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点（dispatch 取代原 supervisor：确定性 skill 选择，0 次 LLM 调用）
        workflow.add_node("dispatch", self._dispatch_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("experiment", self._experiment_node)
        workflow.add_node("writing", self._writing_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # 设置入口
        workflow.set_entry_point("dispatch")
        
        # dispatch 后按选中的 skill 进入对应节点（单跳）
        workflow.add_conditional_edges(
            "dispatch",
            self._route_task,
            {
                "retrieval": "retrieval",
                "experiment": "experiment",
                "writing": "writing",
                "end": "finalize"
            }
        )
        
        # Agent完成后的路由
        workflow.add_conditional_edges(
            "retrieval",
            self._check_next_agent,
            {
                "experiment": "experiment",
                "writing": "writing",
                "end": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "experiment",
            self._check_next_agent,
            {
                "writing": "writing",
                "end": "finalize"
            }
        )
        
        workflow.add_edge("writing", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _dispatch_node(self, state: AgentState) -> AgentState:
        """确定性分发节点（取代原 supervisor）。

        旧实现：调用 LLM 生成 JSON 执行计划 → 解析 → 路由（1 次 LLM 调用 + 易解析失败）。
        新实现：``SkillRegistry`` 按 skill frontmatter 的词元覆盖率打分选 skill。
        **0 次 LLM 调用**，且选择结果可解释、可复现。

        与旧实现的能力对齐：支持一条任务命中多个 skill（原 ``task_type == 'complex'``
        的串行链式能力）；未命中任何 skill 时回退到检索并在 trace 中标注。
        """
        task = state['task']
        metas = self.registry.select(task, top_k=3)
        state['skills'] = [m.name for m in metas]
        state['task_type'] = _SKILL_TO_TASK_TYPE.get(metas[0].name, 'retrieval') if metas else 'retrieval'
        state['skill_queue'] = list(state['skills'][1:])

        self.hops += 1  # 路由跳数：一次即完成（旧实现在这里就是一次 LLM 调用）
        state['route_hops'] = self.hops
        state['resident_chars'] = self.registry.resident_chars()
        state['llm_calls_for_routing'] = 0

        # 渐进披露：只为选中的 skill 加载正文，并注入为该 Agent 的 system prompt
        for m in metas:
            body = self.loader.load(m.name)
            if body is None:
                continue
            key = _SKILL_TO_AGENT_KEY.get(m.name)
            if key and key in self.agents:
                self.agents[key].skill_prompt = body.body
            state['skill_prompt_chars'] = (state.get('skill_prompt_chars', 0) + body.chars)

        logger.info("dispatch: task=%r -> skills=%s (0 LLM calls)", task[:40], state['skills'])
        return state
    
    def _route_task(self, state: AgentState) -> str:
        """（保留条件边契约）按 dispatch 选出的 skill 进入对应节点。"""
        task_type = state.get('task_type', 'retrieval')
        if task_type in ('retrieval', 'experiment', 'writing'):
            return task_type
        return 'end'
    
    def _check_next_agent(self, state: AgentState) -> str:
        """检查是否需要调用下一个Agent。

        新实现不再依赖 ``task_type == 'complex'`` 的硬编码串行顺序，
        而是消费 dispatch 阶段排好序的 ``skill_queue``（按相关度降序）。
        """
        iteration = state.get('iteration', 0)
        max_iter = state.get('max_iterations', 10)
        if iteration >= max_iter:
            return 'end'

        queue = state.get('skill_queue') or []
        while queue:
            nxt = queue.pop(0)
            if nxt in self.agents:
                state['task_type'] = _SKILL_TO_TASK_TYPE.get(nxt, 'retrieval')
                return nxt if nxt == 'retrieval' else state['task_type']

        return 'end'
    
    def _retrieval_node(self, state: AgentState) -> AgentState:
        """检索Agent节点"""
        state['current_agent'] = 'retrieval'
        
        if 'retrieval' in self.agents:
            result = self.agents['retrieval'].run(
                state['task'],
                {'context': state.get('tool_outputs', [])}
            )
            
            state['tool_outputs'].append(result)
            
            # 提取检索到的论文
            if result.get('tool_outputs'):
                for to in result['tool_outputs']:
                    if to.get('result', {}).get('output'):
                        try:
                            output = to['result']['output']
                            if isinstance(output, str):
                                output = json.loads(output)
                            if isinstance(output, dict) and 'results' in output:
                                state['retrieved_papers'].extend(output['results'])
                        except Exception as e:
                            logger.warning("Agent result parse failed: %s", e)
        
        state['iteration'] += 1
        return state
    
    def _experiment_node(self, state: AgentState) -> AgentState:
        """实验Agent节点"""
        state['current_agent'] = 'experiment'
        
        if 'experiment' in self.agents:
            result = self.agents['experiment'].run(
                state['task'],
                {
                    'context': state.get('tool_outputs', []),
                    'papers': state.get('retrieved_papers', [])
                }
            )
            
            state['tool_outputs'].append(result)
            
            # 保存实验方案
            if result.get('final_answer'):
                state['experiment_plan'] = {
                    'description': result['final_answer'],
                    'tool_outputs': result.get('tool_outputs', [])
                }
        
        state['iteration'] += 1
        return state
    
    def _writing_node(self, state: AgentState) -> AgentState:
        """写作Agent节点"""
        state['current_agent'] = 'writing'
        
        if 'writing' in self.agents:
            result = self.agents['writing'].run(
                state['task'],
                {
                    'context': state.get('tool_outputs', []),
                    'papers': state.get('retrieved_papers', []),
                    'draft': state.get('draft_content', '')
                }
            )
            
            state['tool_outputs'].append(result)
            
            # 保存写作内容
            if result.get('final_answer'):
                state['draft_content'] = result['final_answer']
        
        state['iteration'] += 1
        return state
    
    def _finalize_node(self, state: AgentState) -> AgentState:
        """最终节点：汇总结果"""
        # 汇总所有输出
        final_parts = []
        
        for output in state.get('tool_outputs', []):
            if output.get('final_answer'):
                final_parts.append(f"## {output.get('agent_type', 'Agent').title()} 结果\n\n{output['final_answer']}")
        
        if state.get('draft_content'):
            final_parts.append(f"## 最终内容\n\n{state['draft_content']}")
        
        state['final_output'] = '\n\n---\n\n'.join(final_parts) if final_parts else "任务处理完成，但没有生成输出。"
        
        return state
    
    def run(self, task: str, max_iterations: int = 10) -> Dict:
        """
        运行多Agent系统
        
        Args:
            task: 用户任务
            max_iterations: 最大迭代次数
            
        Returns:
            执行结果
        """
        # 重置度量计数器（M2 路由跳数 / M1 上下文加载量）
        self.hops = 0
        self.loader.reset_meter()

        # 个性化:注入用户科研偏好(软个性化)
        if self.memory:
            pref_text = format_preferences(self.memory.get_preferences(self.user_id))
            if pref_text:
                task = f"{pref_text}\n用户任务: {task}"

        # 初始状态
        initial_state: AgentState = {
            'messages': [HumanMessage(content=task)],
            'current_agent': '',
            'task': task,
            'task_type': '',
            'skills': [],
            'skill_queue': [],
            'route_hops': 0,
            'resident_chars': 0,
            'retrieved_papers': [],
            'experiment_plan': None,
            'draft_content': '',
            'tool_outputs': [],
            'iteration': 0,
            'max_iterations': max_iterations,
            'final_output': None,
            'error': None
        }

        if self.graph:
            # 使用LangGraph执行
            try:
                final_state = self.graph.invoke(initial_state)
                result = {
                    'success': True,
                    'task': task,
                    'task_type': final_state.get('task_type'),
                    'skills': final_state.get('skills', []),
                    'iterations': final_state.get('iteration'),
                    'retrieved_papers': final_state.get('retrieved_papers', []),
                    'experiment_plan': final_state.get('experiment_plan'),
                    'final_output': final_state.get('final_output'),
                    'tool_outputs': final_state.get('tool_outputs', [])
                }
                result['routing'] = {
                    'hops': final_state.get('route_hops', 0),
                    'llm_calls_for_routing': 0,
                    'resident_chars': final_state.get('resident_chars', 0),
                    'skill_prompt_chars': final_state.get('skill_prompt_chars', 0),
                }
            except Exception as e:
                result = {
                    'success': False,
                    'task': task,
                    'error': str(e)
                }
        else:
            # 降级：直接调用单个Agent
            result = self._fallback_run(task)

        result.setdefault('routing', {})
        result['routing'].update(self.loader.stats())
        result['routing']['hops'] = self.hops

        # 记录问答到长期记忆
        if self.memory:
            answer = result.get('final_output') or result.get('error', '')
            self.memory.add_qa(self.user_id, task, str(answer), success=result.get('success', False))

        return result
    
    def _fallback_run(self, task: str) -> Dict:
        """降级执行（不使用 LangGraph）。

        用 ``SkillRegistry`` 取代已删除的 ``_simple_task_classification``：
        同样 0 次 LLM 调用，但选择依据来自 skill 的 frontmatter 而非硬编码关键词表。
        """
        metas = self.registry.select(task, top_k=1)
        skill_name = metas[0].name if metas else None
        task_type = _SKILL_TO_TASK_TYPE.get(skill_name, 'retrieval')

        if task_type in self.agents:
            body = self.loader.load(skill_name) if skill_name else None
            if body is not None:
                self.agents[task_type].skill_prompt = body.body
            self.hops += 1
            result = self.agents[task_type].run(task, {})
            return {
                'success': result.get('success', False),
                'task': task,
                'task_type': task_type,
                'skills': [skill_name] if skill_name else [],
                'final_output': result.get('final_answer'),
                'tool_outputs': result.get('tool_outputs', [])
            }

        return {
            'success': False,
            'task': task,
            'error': 'No agents available'
        }
    
    def run_retrieval(self, query: str) -> Dict:
        """直接运行检索Agent"""
        if 'retrieval' not in self.agents:
            return {'success': False, 'error': 'Retrieval agent not available'}
        
        return self.agents['retrieval'].run(query, {})
    
    def run_experiment(self, task: str, papers: List[Dict] = None) -> Dict:
        """直接运行实验Agent"""
        if 'experiment' not in self.agents:
            return {'success': False, 'error': 'Experiment agent not available'}
        
        return self.agents['experiment'].run(task, {'papers': papers or []})
    
    def run_writing(self, task: str, papers: List[Dict] = None, draft: str = '') -> Dict:
        """直接运行写作Agent"""
        if 'writing' not in self.agents:
            return {'success': False, 'error': 'Writing agent not available'}
        
        return self.agents['writing'].run(task, {'papers': papers or [], 'draft': draft})


def create_research_agent(config: Dict, retriever=None) -> ResearchAgentGraph:
    """
    创建科研助手Agent
    
    Args:
        config: 配置字典
        retriever: HybridRetriever 实例
        
    Returns:
        ResearchAgentGraph 实例
    """
    return ResearchAgentGraph(config, retriever)


if __name__ == "__main__":
    # 测试代码
    config = {
        'llm': {
            'model': 'gpt-4-turbo-preview',
            'temperature': 0.7
        },
        'max_iterations': 5,
        'output_dir': 'output/agent_outputs'
    }
    
    agent_graph = ResearchAgentGraph(config)
    
    # 测试任务分类
    test_tasks = [
        "搜索关于图神经网络预测电池寿命的论文",
        "设计一个使用LSTM预测时间序列的实验",
        "帮我写一个关于深度学习的摘要",
        "分析这些数据并生成可视化图表"
    ]
    
    for task in test_tasks:
        task_type = agent_graph._simple_task_classification(task)
        print(f"Task: {task[:30]}... -> Type: {task_type}")
