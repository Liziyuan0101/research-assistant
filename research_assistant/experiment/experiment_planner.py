"""
Experiment Planner Module
自动规划实验方案的Agent
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from ..utils.llm import create_openai_client


class ExperimentPlanner:
    """实验规划器，基于研究问题和论文知识自动设计实验方案"""
    
    def __init__(self, config: dict, prompts: dict):
        self.config = config
        self.prompts = prompts
        self.agent_config = config.get('experiment_agent', {})
        
        # 初始化LLM
        llm_config = config.get('llm', {})
        
        self.client = create_openai_client(config)
        
        self.planning_model = self.agent_config.get('planning_model') or llm_config.get('model', 'deepseek-chat')
        self.max_iterations = self.agent_config.get('max_iterations', 10)
    
    def design_experiment(
        self, 
        research_question: str,
        papers_context: Optional[List[Dict]] = None,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        设计完整的实验方案
        
        Args:
            research_question: 研究问题
            papers_context: 相关论文上下文
            constraints: 约束条件（如计算资源、时间限制等）
            
        Returns:
            实验方案字典
        """
        print(f"🔬 Designing experiment for: {research_question}")
        
        # 构建论文上下文
        papers_text = self._format_papers_context(papers_context) if papers_context else "无相关论文参考"
        
        # 获取规划prompt
        prompt_template = self.prompts.get('experiment_design', {}).get('planning', '')
        
        prompt = prompt_template.format(
            research_question=research_question,
            papers_context=papers_text
        )
        
        # 添加约束条件
        if constraints:
            prompt += f"\n\n约束条件:\n{json.dumps(constraints, ensure_ascii=False, indent=2)}"
        
        if not self.client:
            return self._fallback_plan(research_question)
        
        try:
            response = self.client.chat.completions.create(
                model=self.planning_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一位经验丰富的科研实验设计专家，擅长规划严谨的实验方案。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # 解析实验方案
            experiment_plan = self._parse_experiment_plan(plan_text, research_question)
            
            print("✅ Experiment plan generated")
            return experiment_plan
            
        except Exception as e:
            print(f"❌ Error designing experiment: {e}")
            return self._fallback_plan(research_question)
    
    def suggest_parameters(
        self,
        model_type: str,
        dataset_size: int,
        task_type: str,
        paper_parameters: Optional[List[Dict]] = None
    ) -> Dict:
        """
        推荐实验参数
        
        Args:
            model_type: 模型类型
            dataset_size: 数据集大小
            task_type: 任务类型
            paper_parameters: 论文中的参数设置
            
        Returns:
            参数推荐
        """
        print(f"⚙️ Suggesting parameters for {model_type} on {task_type}")
        
        prompt_template = self.prompts.get('experiment_design', {}).get('parameter_suggestion', '')
        
        paper_params_text = ""
        if paper_parameters:
            paper_params_text = "\n".join([
                f"- 论文: {p.get('paper', 'Unknown')}\n  参数: {json.dumps(p.get('params', {}), indent=2)}"
                for p in paper_parameters
            ])
        
        prompt = prompt_template.format(
            model_type=model_type,
            dataset_size=dataset_size,
            task_type=task_type,
            paper_parameters=paper_params_text or "无参考"
        )
        
        if not self.client:
            return self._default_parameters(model_type, task_type)
        
        try:
            response = self.client.chat.completions.create(
                model=self.planning_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位机器学习专家，擅长超参数调优。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            params_text = response.choices[0].message.content.strip()
            
            # 尝试解析为结构化参数
            parameters = self._parse_parameters(params_text)
            
            return parameters
            
        except Exception as e:
            print(f"❌ Error suggesting parameters: {e}")
            return self._default_parameters(model_type, task_type)
    
    def generate_experiment_code(
        self,
        experiment_plan: Dict,
        framework: str = "pytorch"
    ) -> str:
        """
        根据实验方案生成代码
        
        Args:
            experiment_plan: 实验方案
            framework: 深度学习框架
            
        Returns:
            生成的代码
        """
        print(f"💻 Generating {framework} code for experiment")
        
        prompt_template = self.prompts.get('experiment_design', {}).get('code_generation', '')
        
        plan_text = json.dumps(experiment_plan, ensure_ascii=False, indent=2)
        prompt = prompt_template.format(experiment_plan=plan_text)
        
        if not self.client:
            return self._template_code(experiment_plan, framework)
        
        try:
            response = self.client.chat.completions.create(
                model=self.planning_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"你是一位专业的{framework}开发者，擅长编写高质量的机器学习代码。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低温度以获得更确定的代码
                max_tokens=8192
            )
            
            code = response.choices[0].message.content.strip()
            
            # 提取代码块
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            return code
            
        except Exception as e:
            print(f"❌ Error generating code: {e}")
            return self._template_code(experiment_plan, framework)
    
    def refine_plan(
        self,
        original_plan: Dict,
        feedback: str
    ) -> Dict:
        """
        根据反馈优化实验方案
        
        Args:
            original_plan: 原始方案
            feedback: 反馈意见
            
        Returns:
            优化后的方案
        """
        print("🔄 Refining experiment plan based on feedback")
        
        prompt = f"""
请根据以下反馈优化实验方案：

原始方案:
{json.dumps(original_plan, ensure_ascii=False, indent=2)}

反馈意见:
{feedback}

请提供优化后的完整实验方案。
"""
        
        if not self.client:
            return original_plan
        
        try:
            response = self.client.chat.completions.create(
                model=self.planning_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位科研实验设计专家。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            
            refined_text = response.choices[0].message.content.strip()
            refined_plan = self._parse_experiment_plan(
                refined_text, 
                original_plan.get('research_question', '')
            )
            
            return refined_plan
            
        except Exception as e:
            print(f"❌ Error refining plan: {e}")
            return original_plan
    
    def _format_papers_context(self, papers: List[Dict]) -> str:
        """格式化论文上下文"""
        context_parts = []
        
        for i, paper in enumerate(papers[:5], 1):  # 最多5篇
            context_parts.append(
                f"论文 {i}:\n"
                f"标题: {paper.get('title', '')}\n"
                f"作者: {', '.join(paper.get('authors', [])[:3])}\n"
                f"摘要: {paper.get('abstract', '')[:500]}...\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _parse_experiment_plan(self, plan_text: str, research_question: str) -> Dict:
        """解析实验方案文本为结构化格式"""
        return {
            'research_question': research_question,
            'plan_text': plan_text,
            'status': 'planned',
            'created_at': self._get_timestamp()
        }
    
    def _parse_parameters(self, params_text: str) -> Dict:
        """解析参数文本"""
        # 简单实现：返回文本描述
        return {
            'description': params_text,
            'parsed': False
        }
    
    def _fallback_plan(self, research_question: str) -> Dict:
        """备用实验方案"""
        return {
            'research_question': research_question,
            'plan_text': f"实验方案模板：\n1. 数据收集与预处理\n2. 模型选择与设计\n3. 训练与验证\n4. 结果分析",
            'status': 'template',
            'created_at': self._get_timestamp()
        }
    
    def _default_parameters(self, model_type: str, task_type: str) -> Dict:
        """默认参数推荐"""
        defaults = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'adam',
            'early_stopping_patience': 10
        }
        
        return {
            'parameters': defaults,
            'description': f"默认参数配置 for {model_type} on {task_type}",
            'source': 'default'
        }
    
    def _template_code(self, plan: Dict, framework: str) -> str:
        """模板代码"""
        return f"""
# Experiment Code Template
# Research Question: {plan.get('research_question', 'Unknown')}
# Framework: {framework}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: Implement data loading
def load_data():
    pass

# TODO: Implement model
def build_model():
    pass

# TODO: Implement training
def train_model(model, data):
    pass

# TODO: Implement evaluation
def evaluate_model(model, data):
    pass

if __name__ == "__main__":
    # Load data
    data = load_data()
    
    # Build model
    model = build_model()
    
    # Train
    train_model(model, data)
    
    # Evaluate
    results = evaluate_model(model, data)
    print(results)
"""
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_plan(self, plan: Dict, output_path: str):
        """保存实验方案"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Saved experiment plan to {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving plan: {e}")


if __name__ == "__main__":
    # 测试代码
    config = {
        'llm': {
            'model': 'gpt-4-turbo-preview'
        },
        'experiment_agent': {
            'planning_model': 'gpt-4-turbo-preview',
            'max_iterations': 10
        }
    }
    
    prompts_path = Path(__file__).parent.parent / 'config' / 'prompts.yaml'
    if prompts_path.exists():
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
    else:
        prompts = {}
    
    planner = ExperimentPlanner(config, prompts)
    
    # 测试实验设计
    plan = planner.design_experiment(
        research_question="如何使用图神经网络预测电池寿命？"
    )
    
    print(json.dumps(plan, ensure_ascii=False, indent=2))
