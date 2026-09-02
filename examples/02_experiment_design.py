"""
02_experiment_design.py - 实验方案设计模块示例

功能演示:
1. 基于论文上下文设计实验方案
2. 生成实验代码
3. 参数推荐

运行: python examples/02_experiment_design.py
前置条件: 需要设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY
"""

import os
import sys
import json
import warnings
from pathlib import Path

# 抑制第三方库输出
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载 .env 环境变量
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from research_assistant import ResearchAssistant


def demo_search_papers(assistant):
    """搜索相关论文作为实验设计的参考"""
    print("\n" + "="*60)
    print("📚 Step 1: 搜索相关论文")
    print("="*60)
    
    query = "lithium battery remaining useful life prediction deep learning"
    print(f"\n搜索查询: {query}")
    print("执行 API 搜索...")
    
    papers = assistant.search_papers(query, max_results=10, verbose=False)
    
    print(f"\n✅ 获取 {len(papers)} 篇参考论文:\n")
    for i, p in enumerate(papers[:5], 1):
        title = p.get('title', 'N/A')[:55]
        year = p.get('published', '')[:4]
        citations = p.get('citations', 0)
        print(f"  {i}. {title}...")
        print(f"     年份: {year} | 引用: {citations}")
    
    if len(papers) > 5:
        print(f"  ... 还有 {len(papers)-5} 篇论文")
    
    return papers


def demo_design_experiment(assistant, papers):
    """演示实验方案设计"""
    print("\n" + "="*60)
    print("🔬 Step 2: 实验方案设计")
    print("="*60)
    
    research_question = "如何使用Transformer模型预测锂电池剩余使用寿命(RUL)？"
    print(f"\n研究问题: {research_question}")
    
    print("\n🤖 使用 LLM 设计实验方案...")
    
    try:
        # 设计实验
        experiment_plan = assistant.design_experiment(
            research_question=research_question,
            papers=papers,
            constraints={
                'framework': 'pytorch',
                'gpu_memory': '4GB',
                'training_time': '< 2 hours'
            }
        )
        
        print("\n✅ 实验方案设计完成:\n")
        
        # 显示实验方案
        if isinstance(experiment_plan, dict):
            if 'objective' in experiment_plan:
                print(f"【实验目标】")
                print(f"  {experiment_plan['objective']}\n")
            
            if 'hypothesis' in experiment_plan:
                print(f"【研究假设】")
                print(f"  {experiment_plan['hypothesis']}\n")
            
            if 'dataset' in experiment_plan:
                print(f"【数据集】")
                if isinstance(experiment_plan['dataset'], list):
                    for ds in experiment_plan['dataset']:
                        print(f"  - {ds}")
                else:
                    print(f"  {experiment_plan['dataset']}\n")
            
            if 'model' in experiment_plan:
                print(f"\n【模型架构】")
                print(f"  {experiment_plan['model']}\n")
            
            if 'metrics' in experiment_plan:
                print(f"【评估指标】")
                if isinstance(experiment_plan['metrics'], list):
                    for m in experiment_plan['metrics']:
                        print(f"  - {m}")
                else:
                    print(f"  {experiment_plan['metrics']}")
            
            if 'parameters' in experiment_plan:
                print(f"\n【超参数】")
                params = experiment_plan['parameters']
                if isinstance(params, dict):
                    for k, v in params.items():
                        print(f"  - {k}: {v}")
        else:
            print(experiment_plan)
        
        return experiment_plan
        
    except Exception as e:
        print(f"\n❌ 实验设计失败: {e}")
        print("   请确保已设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
        return None


def demo_generate_code(assistant, experiment_plan):
    """演示实验代码生成"""
    print("\n" + "="*60)
    print("💻 Step 3: 生成实验代码")
    print("="*60)
    
    if experiment_plan is None:
        print("⚠️ 无实验方案，跳过代码生成")
        return
    
    print("\n🤖 使用 LLM 生成 PyTorch 实验代码...")
    
    try:
        code = assistant.generate_experiment_code(
            experiment_plan=experiment_plan,
            framework="pytorch"
        )
        
        print("\n✅ 代码生成完成:\n")
        
        # 显示代码片段
        lines = code.split('\n')
        if len(lines) > 50:
            print('\n'.join(lines[:50]))
            print(f"\n... (共 {len(lines)} 行，已省略)")
        else:
            print(code)
        
        # 保存代码
        output_dir = Path(__file__).parent.parent / 'output'
        output_dir.mkdir(exist_ok=True)
        code_path = output_dir / 'generated_experiment.py'
        
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"\n💾 代码已保存到: {code_path}")
        
        return code
        
    except Exception as e:
        print(f"\n❌ 代码生成失败: {e}")
        return None


def demo_save_plan(assistant, experiment_plan):
    """保存实验方案"""
    print("\n" + "="*60)
    print("💾 Step 4: 保存实验方案")
    print("="*60)
    
    if experiment_plan is None:
        print("⚠️ 无实验方案，跳过保存")
        return
    
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    plan_path = output_dir / 'experiment_plan.json'
    
    try:
        assistant.experiment_planner.save_plan(experiment_plan, str(plan_path))
        print(f"\n✅ 实验方案已保存到: {plan_path}")
    except Exception as e:
        # 直接保存JSON
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_plan, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 实验方案已保存到: {plan_path}")


def demo_manual_experiment():
    """演示手动创建实验方案（不需要LLM）"""
    print("\n" + "="*60)
    print("📝 备选: 手动创建实验方案")
    print("="*60)
    
    experiment_plan = {
        "title": "基于Transformer的锂电池RUL预测",
        "objective": "使用Transformer模型预测锂电池剩余使用寿命",
        "hypothesis": "Transformer的自注意力机制能够捕捉电池退化的长期依赖关系",
        "dataset": [
            "NASA Battery Dataset",
            "CALCE Battery Dataset"
        ],
        "model": {
            "architecture": "Transformer Encoder",
            "input_features": ["voltage", "current", "temperature", "capacity"],
            "output": "RUL (cycles)"
        },
        "parameters": {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        },
        "metrics": [
            "RMSE (Root Mean Square Error)",
            "MAE (Mean Absolute Error)",
            "R² Score"
        ],
        "baselines": [
            "LSTM",
            "GRU",
            "CNN-LSTM"
        ]
    }
    
    print("\n手动定义的实验方案:\n")
    print(json.dumps(experiment_plan, ensure_ascii=False, indent=2))
    
    return experiment_plan


def main():
    print("\n" + "="*60)
    print("🔬 实验方案设计模块示例")
    print("="*60)
    print("""
    完整流程:
    Step 1: 搜索相关论文 → Step 2: 设计实验方案 → Step 3: 生成代码 → Step 4: 保存方案
    
    ⚠️ 需要: DEEPSEEK_API_KEY 或 OPENAI_API_KEY
    """)
    
    # 初始化
    print("\n📦 初始化研究助手...")
    assistant = ResearchAssistant(verbose=False)
    
    # 1. 搜索论文
    papers = demo_search_papers(assistant)
    
    # 2. 设计实验
    print("\n是否使用LLM设计实验方案？[y/N]: ", end="")
    try:
        use_llm = input().strip().lower() == 'y'
    except:
        use_llm = False
    
    if use_llm:
        experiment_plan = demo_design_experiment(assistant, papers)
    else:
        experiment_plan = demo_manual_experiment()
    
    # 3. 生成代码 (可选)
    if experiment_plan and use_llm:
        print("\n是否生成实验代码？[y/N]: ", end="")
        try:
            if input().strip().lower() == 'y':
                demo_generate_code(assistant, experiment_plan)
        except:
            pass
    
    # 4. 保存方案
    demo_save_plan(assistant, experiment_plan)
    
    # 完成
    print("\n" + "="*60)
    print("🎉 实验方案设计完成！")
    print("="*60)


if __name__ == "__main__":
    main()
