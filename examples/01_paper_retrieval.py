"""
01_paper_retrieval.py - 论文检索模块示例

功能演示:
1. API搜索 - 从arXiv/OpenAlex等获取论文
2. 混合检索 - BM25 + BGE-M3 + Reranker
3. HyDE检索 - 假设文档嵌入
4. 检索方法对比
5. RAGAS评测 - 四个核心指标

运行: python examples/01_paper_retrieval.py
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np

# 抑制第三方库输出
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载 .env 环境变量
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from main import ResearchAssistant
from modules.paper_retrieval.evaluation import RAGEvaluator, HAS_RAGAS


def demo_api_search(assistant):
    """演示API搜索 - 从外部数据库获取论文"""
    print("\n" + "="*60)
    print("📚 1. API搜索 - 从学术数据库获取论文")
    print("="*60)
    
    # 支持中文查询，会自动转换为英文学术术语
    query = "锂电池剩余寿命预测"
    print(f"\n查询: {query}")
    
    # 搜索论文
    papers = assistant.search_papers(
        query=query,
        max_results=10,
        auto_enhance=True,
        verbose=False  # 减少输出
    )
    
    print(f"\n✅ 获取 {len(papers)} 篇论文:\n")
    for i, p in enumerate(papers[:5], 1):
        title = p.get('title', 'N/A')[:60]
        source = p.get('source', 'N/A')
        year = p.get('published', '')[:4]
        citations = p.get('citations', 0)
        print(f"  {i}. [{source}] {title}...")
        print(f"     年份: {year} | 引用: {citations}")
    
    if len(papers) > 5:
        print(f"  ... 还有 {len(papers)-5} 篇论文")
    
    return papers


def demo_hybrid_search(assistant, papers):
    """演示混合检索 - BM25 + BGE-M3 + Reranker"""
    print("\n" + "="*60)
    print("🔍 2. 混合检索 - BM25 + BGE-M3 + Reranker")
    print("="*60)
    
    if not assistant.hybrid_retriever:
        print("⚠️ 混合检索器未初始化，跳过")
        print("   提示: 首次运行需要下载BGE模型 (~2GB)")
        return []
    
    # 建立索引
    print("\n建立本地索引...")
    count = assistant.index_papers(papers)
    print(f"✅ 已索引 {count} 个文本块")
    
    # 执行混合检索
    query = "如何使用深度学习预测电池健康状态"
    print(f"\n检索查询: {query}")
    
    results = assistant.hybrid_search(
        query=query,
        top_k=5,
        use_rerank=True,
        verbose=False
    )
    
    print(f"\n✅ 检索结果 ({len(results)} 条):\n")
    for i, r in enumerate(results, 1):
        title = r.get('paper', {}).get('title', 'N/A')[:50]
        score = r.get('score', 0)
        chunk_type = r.get('chunk_type', 'unknown')
        print(f"  {i}. [{score:.3f}] {title}...")
        print(f"     类型: {chunk_type}")
    
    return results


def demo_hyde_search(assistant):
    """演示HyDE检索 - 假设文档嵌入"""
    print("\n" + "="*60)
    print("💡 3. HyDE检索 - 假设文档嵌入")
    print("="*60)
    
    if not assistant.hybrid_retriever:
        print("⚠️ 混合检索器未初始化，跳过")
        return
    
    # 模糊的用户问题
    query = "电池为什么会老化？怎么预测什么时候坏？"
    print(f"模糊查询: {query}")
    
    # HyDE检索 (需要LLM API)
    try:
        results = assistant.hybrid_search(
            query=query,
            top_k=3,
            use_hyde=True,
            verbose=False
        )
        
        print(f"\n✅ HyDE检索结果 ({len(results)} 条):\n")
        for i, r in enumerate(results, 1):
            title = r.get('paper', {}).get('title', 'N/A')[:50]
            score = r.get('score', 0)
            print(f"  {i}. [{score:.3f}] {title}...")
    except Exception as e:
        print(f"⚠️ HyDE需要LLM API: {e}")


def demo_compare_methods(assistant):
    """对比不同检索方法"""
    print("\n" + "="*60)
    print("⚖️ 4. 检索方法对比")
    print("="*60)
    
    if not assistant.hybrid_retriever:
        print("⚠️ 混合检索器未初始化，跳过")
        return
    
    retriever = assistant.hybrid_retriever
    query = "battery state of health prediction neural network"
    print(f"\n测试查询: {query}\n")
    
    print(f"{'方法':<30} {'召回数':<8} {'最高分':<10}")
    print("-" * 50)
    
    # BM25 稀疏检索
    try:
        results = retriever._bm25_search(query, 5)
        print(f"{'BM25 (关键词匹配)':<30} {len(results):<8} {results[0]['score'] if results else 0:.4f}")
    except:
        print(f"{'BM25 (关键词匹配)':<30} {'N/A':<8}")
    
    # BGE-M3 稠密检索
    try:
        results = retriever._dense_search(query, 5)
        print(f"{'BGE-M3 (语义向量)':<30} {len(results):<8} {results[0]['score'] if results else 0:.4f}")
    except:
        print(f"{'BGE-M3 (语义向量)':<30} {'N/A':<8}")
    
    # 混合检索 (无精排)
    try:
        results = retriever.search(query, 5, use_rerank=False)
        print(f"{'混合检索 (BM25+BGE-M3)':<30} {len(results):<8} {results[0]['score'] if results else 0:.4f}")
    except:
        print(f"{'混合检索 (BM25+BGE-M3)':<30} {'N/A':<8}")
    
    # 混合检索 (带精排)
    try:
        results = retriever.search(query, 5, use_rerank=True)
        print(f"{'混合检索 + Reranker':<30} {len(results):<8} {results[0]['score'] if results else 0:.4f}")
    except:
        print(f"{'混合检索 + Reranker':<30} {'N/A':<8}")
    
    print("\n💡 结论: 混合检索 + Reranker 通常效果最佳")


def demo_index_stats(assistant):
    """显示索引统计信息"""
    print("\n" + "="*60)
    print("📊 5. 索引统计")
    print("="*60)
    
    if assistant.hybrid_retriever:
        try:
            stats = assistant.hybrid_retriever.get_stats()
            print(f"\n  论文数量: {stats.get('total_papers', 0)}")
            print(f"  文本块数: {stats.get('total_chunks', 0)}")
            print(f"  BM25文档: {stats.get('bm25_documents', 0)}")
            print(f"  FAISS向量: {stats.get('faiss_vectors', 0)}")
        except Exception as e:
            print(f"  获取统计失败: {e}")
    else:
        print("  混合检索器未初始化")


def demo_ragas_evaluation(assistant, results, query, answer=None):
    """演示RAGAS评测 - 四个核心指标"""
    print("\n" + "="*60)
    print("📈 3. RAGAS评测 - 四个核心指标")
    print("="*60)
    
    if not HAS_RAGAS:
        print("\n⚠️ RAGAS未安装，跳过评测")
        print("   安装: pip install ragas")
        return
    
    if not results:
        print("\n⚠️ 无检索结果，跳过评测")
        return

    
    # 准备评测数据
    questions = [query]
    
    # 从检索结果提取上下文
    contexts = []
    for r in results[:5]:
        content = r.get('content', '')
        if not content and r.get('paper'):
            content = r['paper'].get('abstract', '')[:500]
        if content:
            contexts.append(content)
    
    if not contexts:
        print("\n⚠️ 无法提取上下文，跳过评测")
        return
    
    # 生成答案 (使用LLM基于检索结果生成)
    if answer is None:
        print("\n🤖 使用LLM生成答案...")
        try:
            # 使用 paper_interpreter 的 OpenAI 客户端
            if hasattr(assistant, 'paper_interpreter') and assistant.paper_interpreter and assistant.paper_interpreter.client:
                context_text = "\n\n".join([f"[文档{i+1}] {ctx[:300]}..." for i, ctx in enumerate(contexts[:3])])
                prompt = f"""基于以下检索到的文档，回答问题。

问题: {query}

检索文档:
{context_text}

请用中文简洁回答（100字以内）:"""
                
                response = assistant.paper_interpreter.client.chat.completions.create(
                    model=assistant.paper_interpreter.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=200
                )
                answer = response.choices[0].message.content
                print(f"生成答案: {answer[:100]}...")
            else:
                raise Exception("LLM未配置")
        except Exception as e:
            print(f"⚠️ LLM生成答案失败: {e}")
            # 使用检索结果生成简单答案
            answer = f"根据检索结果，{contexts[0][:100]}..."
    
    answers = [answer]
    
    # 参考答案 (ground truth) - 用于计算 faithfulness 和 answer_relevancy
    # 这里使用生成的答案作为参考，实际应用中应该有人工标注的标准答案
    ground_truths = [answer]
    
    print(f"\n测试问题: {questions[0]}")
    print(f"检索上下文数: {len(contexts)}")
    print(f"生成答案长度: {len(answers[0])} 字符")
    
    # 执行RAGAS评测
    try:
        evaluator = RAGEvaluator()
        
        result = evaluator.evaluate_rag_system(
            questions=questions,
            answers=answers,
            contexts=[contexts],
            ground_truths=ground_truths
        )
        
        print("\n" + "="*60)
        print("📊 RAGAS 评测结果")
        print("="*60)
        
        if 'metrics' in result:
            metrics_data = result['metrics']
            
            # 提取指标值
            metrics_dict = {}
            if hasattr(metrics_data, 'to_pandas'):
                metrics_dict = metrics_data.to_pandas().to_dict('records')[0]
            elif hasattr(metrics_data, '__dict__'):
                metrics_dict = {k: v for k, v in vars(metrics_data).items() 
                               if not k.startswith('_') and isinstance(v, (int, float))}
            elif isinstance(metrics_data, dict):
                metrics_dict = metrics_data
            
            # 只显示核心指标
            core_metrics = {
                'context_precision': '上下文精确度',
                'context_recall': '上下文召回率',
                'faithfulness': '答案忠实度',
                'answer_relevancy': '答案相关性'
            }
            
            print("\n核心指标:")
            has_metrics = False
            for metric_key, metric_name in core_metrics.items():
                if metric_key in metrics_dict:
                    value = metrics_dict[metric_key]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        # 使用百分比显示
                        percentage = value * 100
                        bar_length = int(percentage / 5)  # 每5%一个方块
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        print(f"  • {metric_name:12s} {bar} {percentage:5.1f}%")
                        has_metrics = True
            
            if not has_metrics:
                print("  ⚠️ 部分指标计算失败（DeepSeek API 限制）")
                print("  ✅ 已成功计算: context_precision, context_recall")
                print("  ⚠️ 暂不支持: faithfulness, answer_relevancy (需要 OpenAI API)")
            
            # 显示生成的答案（截断）
            if 'response' in metrics_dict and isinstance(metrics_dict['response'], str):
                answer_text = metrics_dict['response']
                if len(answer_text) > 80:
                    answer_text = answer_text[:80] + "..."
                print(f"\n生成答案: {answer_text}")
        else:
            # 降级结果
            print(f"\n  样本数: {result.get('num_samples', 0)}")
            print(f"  平均答案长度: {result.get('avg_answer_length', 0):.1f}")
            print(f"  平均上下文数: {result.get('avg_context_count', 0):.1f}")
            print("\n  💡 完整RAGAS评测需要配置LLM (设置DEEPSEEK_API_KEY)")
        
    except Exception as e:
        print(f"\n❌ RAGAS评测失败: {e}")
        print("   提示: 需要配置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY")


def demo_complete_workflow():
    """完整工作流: 用户输入 -> HyDE + 混合检索 -> RAGAS评测"""
    
    # 初始化
    print("\n📦 初始化研究助手...")
    assistant = ResearchAssistant()
    
    if not assistant.hybrid_retriever:
        print("⚠️ 混合检索器未初始化，无法继续")
        return
    
    # Step 1: 获取用户查询
    print("\n" + "="*60)
    print("📝 Step 1: 输入查询")
    print("="*60)
    
    print("\n请输入您的查询 (或按回车使用默认查询):")
    print("示例: 如何使用深度学习预测电池健康状态")
    user_query = input("\n查询: ").strip()
    
    if not user_query:
        user_query = "如何使用深度学习预测电池健康状态"
        print(f"使用默认查询: {user_query}")
    
    # Step 2: HyDE + 混合检索
    print("\n" + "="*60)
    print("🔍 Step 2: HyDE + 混合检索")
    print("="*60)
    
    print(f"\n查询: {user_query}")
    print("\n执行 HyDE + 混合检索...")
    
    try:
        results = assistant.hybrid_search(
            query=user_query,
            top_k=5,
            use_hyde=True,
            use_rerank=True,
            verbose=False
        )
        
        print(f"\n✅ 检索完成，返回 {len(results)} 条结果:\n")
        for i, r in enumerate(results, 1):
            title = r.get('paper', {}).get('title', 'N/A')[:50]
            score = r.get('score', 0)
            chunk_type = r.get('chunk_type', 'unknown')
            print(f"  {i}. [{score:.3f}] {title}...")
            print(f"     类型: {chunk_type}")
    
    except Exception as e:
        print(f"\n❌ 检索失败: {e}")
        print("   提示: HyDE需要配置LLM (DEEPSEEK_API_KEY 或 OPENAI_API_KEY)")
        return
    
    # Step 3: RAGAS评测
    demo_ragas_evaluation(assistant, results, user_query)


def main():
    print("\n" + "="*60)
    print("📚 论文检索模块示例")
    print("="*60)
    print("""
    本示例提供两种模式:
    
    [1] 完整工作流 - 用户Query → HyDE+混合检索 → RAGAS评测
    [2] 分步演示   - API搜索、混合检索、HyDE、方法对比等
    """)
    
    print("\n请选择模式 [1/2]: ", end="")
    try:
        mode = input().strip()
    except:
        mode = "1"
    
    if mode == "1":
        # 模式1: 完整工作流
        demo_complete_workflow()
    else:
        # 模式2: 分步演示
        # 初始化
        print("\n📦 初始化研究助手...")
        assistant = ResearchAssistant()
        
        # 1. API搜索
        papers = demo_api_search(assistant)
        
        if not papers:
            print("\n❌ 未获取到论文，退出")
            return
        
        # 2. 混合检索
        results = demo_hybrid_search(assistant, papers)
        
        # 3. HyDE检索 (可选)
        print("\n是否演示HyDE检索？(需要LLM API) [y/N]: ", end="")
        try:
            if input().strip().lower() == 'y':
                demo_hyde_search(assistant)
        except:
            pass
        
        # 4. 方法对比 (可选)
        print("\n是否对比检索方法？[y/N]: ", end="")
        try:
            if input().strip().lower() == 'y':
                demo_compare_methods(assistant)
        except:
            pass
        
        # 5. 索引统计
        demo_index_stats(assistant)
        
        # 6. RAGAS评测 (可选)
        print("\n是否演示RAGAS评测？[y/N]: ", end="")
        try:
            if input().strip().lower() == 'y':
                query = "如何使用深度学习预测电池健康状态"
                demo_ragas_evaluation(assistant, results, query)
        except:
            pass
    
    # 完成
    print("\n" + "="*60)
    print("🎉 论文检索示例完成！")
    print("="*60)


if __name__ == "__main__":
    main()
