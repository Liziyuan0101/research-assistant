"""
03_academic_writing.py - 学术写作模块示例

功能演示:
1. 论文解读
2. 摘要生成
3. 引言生成
4. 方法部分生成
5. 引用管理

运行: python examples/03_academic_writing.py
前置条件: 需要设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY
"""

import os
import sys
import warnings
from pathlib import Path

# 抑制第三方库输出
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from research_assistant import ResearchAssistant


def demo_search_papers(assistant):
    """搜索论文作为写作参考"""
    print("\n" + "="*60)
    print("📚 1. 搜索参考论文")
    print("="*60)
    
    query = "lithium battery state of health estimation"
    print(f"\n搜索查询: {query}")
    
    papers = assistant.search_papers(query, max_results=5, verbose=False)
    
    print(f"\n✅ 获取 {len(papers)} 篇参考论文:\n")
    for i, p in enumerate(papers, 1):
        title = p.get('title', 'N/A')[:55]
        authors = ', '.join(p.get('authors', [])[:2])
        if len(p.get('authors', [])) > 2:
            authors += ' et al.'
        print(f"  {i}. {title}...")
        print(f"     作者: {authors}")
    
    return papers


def demo_interpret_paper(assistant, papers):
    """演示论文解读"""
    print("\n" + "="*60)
    print("📖 2. 论文解读")
    print("="*60)
    
    if not papers:
        print("⚠️ 无论文可解读")
        return None
    
    paper = papers[0]
    print(f"\n解读论文: {paper.get('title', 'N/A')[:60]}...")
    print("\n正在解读... (需要LLM API)")
    
    try:
        result = assistant.interpret_paper(paper)
        
        print("\n✅ 解读完成:\n")
        
        if result.get('summary'):
            print("【摘要分析】")
            print(f"  {result['summary'][:300]}...")
        
        if result.get('key_findings'):
            print("\n【关键发现】")
            findings = result['key_findings']
            if isinstance(findings, list):
                for f in findings[:3]:
                    print(f"  • {f}")
            else:
                print(f"  {findings[:200]}...")
        
        if result.get('methodology'):
            print("\n【方法论】")
            print(f"  {result['methodology'][:200]}...")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 论文解读失败: {e}")
        print("   请确保已设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
        return None


def demo_generate_abstract(assistant):
    """演示摘要生成"""
    print("\n" + "="*60)
    print("✍️ 3. 摘要生成")
    print("="*60)
    
    print("\n输入论文信息:")
    
    # 论文信息
    title = "基于Transformer的锂电池剩余寿命预测方法研究"
    keywords = ["锂电池", "剩余寿命预测", "Transformer", "深度学习"]
    background = "锂电池广泛应用于电动汽车和储能系统，准确预测其剩余使用寿命(RUL)对于安全运行至关重要。"
    methods = "本文提出一种基于Transformer的RUL预测方法，利用自注意力机制捕捉电池退化的长期依赖关系。"
    results = "在NASA电池数据集上的实验表明，所提方法的RMSE为12.3个循环，优于传统LSTM方法15%。"
    conclusion = "Transformer模型能够有效预测锂电池RUL，为电池健康管理提供了新的技术路线。"
    
    print(f"  标题: {title}")
    print(f"  关键词: {', '.join(keywords)}")
    print(f"  背景: {background[:50]}...")
    
    print("\n正在生成摘要... (需要LLM API)")
    
    try:
        abstract = assistant.generate_abstract(
            title=title,
            keywords=keywords,
            background=background,
            methods=methods,
            results=results,
            conclusion=conclusion,
            language='zh'
        )
        
        print("\n✅ 生成的摘要:\n")
        print("-" * 50)
        print(abstract)
        print("-" * 50)
        
        return abstract
        
    except Exception as e:
        print(f"\n❌ 摘要生成失败: {e}")
        print("   请确保已设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
        return None


def demo_generate_introduction(assistant, papers):
    """演示引言生成"""
    print("\n" + "="*60)
    print("📝 4. 引言生成")
    print("="*60)
    
    print("\n正在生成引言... (需要LLM API)")
    
    # 准备相关工作
    related_work = ""
    if papers:
        for p in papers[:3]:
            related_work += f"- {p.get('title', '')}: {p.get('abstract', '')[:100]}...\n"
    
    try:
        introduction = assistant.academic_writer.generate_introduction(
            topic="锂电池剩余寿命预测",
            research_question="如何利用Transformer模型提高锂电池RUL预测精度？",
            related_work=related_work,
            contributions=[
                "提出基于Transformer的RUL预测框架",
                "设计适用于时序数据的位置编码方案",
                "在多个公开数据集上验证方法有效性"
            ]
        )
        
        print("\n✅ 生成的引言:\n")
        print("-" * 50)
        if len(introduction) > 500:
            print(introduction[:500] + "...")
        else:
            print(introduction)
        print("-" * 50)
        
        return introduction
        
    except Exception as e:
        print(f"\n❌ 引言生成失败: {e}")
        return None


def demo_generate_methodology(assistant):
    """演示方法部分生成"""
    print("\n" + "="*60)
    print("🔧 5. 方法部分生成")
    print("="*60)
    
    print("\n正在生成方法部分... (需要LLM API)")
    
    try:
        methodology = assistant.academic_writer.generate_methodology(
            method_name="Transformer-based RUL Prediction",
            technical_details="""
            1. 数据预处理：对电池充放电数据进行归一化和滑动窗口分割
            2. 特征提取：提取电压、电流、温度、容量等特征
            3. 模型架构：
               - Encoder: 3层Transformer Encoder
               - d_model: 64, n_heads: 4
               - 位置编码: 可学习的位置嵌入
            4. 训练策略：
               - 损失函数: MSE Loss
               - 优化器: Adam, lr=0.001
               - 早停: patience=10
            """
        )
        
        print("\n✅ 生成的方法部分:\n")
        print("-" * 50)
        if len(methodology) > 500:
            print(methodology[:500] + "...")
        else:
            print(methodology)
        print("-" * 50)
        
        return methodology
        
    except Exception as e:
        print(f"\n❌ 方法部分生成失败: {e}")
        return None


def demo_citation_manager(assistant, papers):
    """演示引用管理"""
    print("\n" + "="*60)
    print("📚 6. 引用管理")
    print("="*60)
    
    if not papers:
        print("⚠️ 无论文可添加引用")
        return
    
    print("\n添加引用...")
    
    # 添加引用
    for i, paper in enumerate(papers[:3], 1):
        try:
            assistant.citation_manager.add_citation(
                paper_id=f"paper_{i}",
                title=paper.get('title', ''),
                authors=paper.get('authors', []),
                year=paper.get('published', '')[:4] if paper.get('published') else '2024',
                venue=paper.get('venue', 'arXiv'),
                doi=paper.get('doi', '')
            )
            print(f"  ✓ 添加: {paper.get('title', '')[:40]}...")
        except Exception as e:
            print(f"  ✗ 添加失败: {e}")
    
    # 生成参考文献
    print("\n生成参考文献列表 (IEEE格式):\n")
    print("-" * 50)
    
    try:
        bibliography = assistant.citation_manager.generate_bibliography()
        print(bibliography)
    except Exception as e:
        print(f"生成失败: {e}")
    
    print("-" * 50)


def demo_save_document(assistant):
    """保存生成的文档"""
    print("\n" + "="*60)
    print("💾 7. 保存文档")
    print("="*60)
    
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # 示例文档内容
    document = """# 基于Transformer的锂电池剩余寿命预测方法研究

## 摘要

锂电池广泛应用于电动汽车和储能系统，准确预测其剩余使用寿命(RUL)对于安全运行至关重要。
本文提出一种基于Transformer的RUL预测方法，利用自注意力机制捕捉电池退化的长期依赖关系。
实验结果表明，所提方法在NASA电池数据集上取得了优异的预测性能。

## 1. 引言

随着电动汽车和可再生能源存储的快速发展，锂电池的健康管理变得越来越重要...

## 2. 方法

### 2.1 问题定义

给定电池历史运行数据，预测其剩余可用循环次数...

### 2.2 Transformer模型

本文采用Transformer Encoder架构...

## 3. 实验

### 3.1 数据集

使用NASA电池数据集进行验证...

### 3.2 结果

所提方法的RMSE为12.3个循环，优于基线方法...

## 4. 结论

本文提出的Transformer-based方法能够有效预测锂电池RUL...

## 参考文献

[1] ...
"""
    
    doc_path = output_dir / 'generated_paper.md'
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(document)
    
    print(f"\n✅ 示例文档已保存到: {doc_path}")


def main():
    print("\n" + "="*60)
    print("✍️ 学术写作模块示例")
    print("="*60)
    print("""
    本示例演示学术写作的完整流程:
    1. 搜索参考论文
    2. 论文解读
    3. 摘要生成
    4. 引言生成
    5. 方法部分生成
    6. 引用管理
    
    ⚠️ 注意: 步骤2-5需要LLM API (DEEPSEEK_API_KEY 或 OPENAI_API_KEY)
    """)
    
    # 初始化
    print("📦 初始化研究助手...")
    assistant = ResearchAssistant()
    
    # 1. 搜索论文
    papers = demo_search_papers(assistant)
    
    # 2. 论文解读 (可选)
    print("\n是否解读论文？(需要LLM API) [y/N]: ", end="")
    try:
        if input().strip().lower() == 'y':
            demo_interpret_paper(assistant, papers)
    except:
        pass
    
    # 3. 摘要生成 (可选)
    print("\n是否生成摘要？(需要LLM API) [y/N]: ", end="")
    try:
        if input().strip().lower() == 'y':
            demo_generate_abstract(assistant)
    except:
        pass
    
    # 4. 引言生成 (可选)
    print("\n是否生成引言？(需要LLM API) [y/N]: ", end="")
    try:
        if input().strip().lower() == 'y':
            demo_generate_introduction(assistant, papers)
    except:
        pass
    
    # 5. 方法部分生成 (可选)
    print("\n是否生成方法部分？(需要LLM API) [y/N]: ", end="")
    try:
        if input().strip().lower() == 'y':
            demo_generate_methodology(assistant)
    except:
        pass
    
    # 6. 引用管理
    demo_citation_manager(assistant, papers)
    
    # 7. 保存文档
    demo_save_document(assistant)
    
    # 完成
    print("\n" + "="*60)
    print("🎉 学术写作示例完成！")
    print("="*60)
    print("""
    核心功能:
    ✓ interpret_paper()        - 论文解读
    ✓ generate_abstract()      - 摘要生成
    ✓ generate_introduction()  - 引言生成
    ✓ generate_methodology()   - 方法部分生成
    ✓ citation_manager         - 引用管理
    
    输出文件:
    → output/generated_paper.md - 生成的论文
    
    完整工作流:
    → 使用 main.py --workflow 运行完整科研流程
    """)


if __name__ == "__main__":
    main()
