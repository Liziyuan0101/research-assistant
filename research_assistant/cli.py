"""
Research Assistant - Command Line Interface
科研论文智能助手命令行入口
"""

import argparse

from .assistant import ResearchAssistant
from .utils.logger import setup_logger


def main():
    """主函数"""
    # 配置日志输出,让库模块的状态/错误(INFO 及以上)通过 logging 可见
    setup_logger('research_assistant')
    parser = argparse.ArgumentParser(description='Research Paper Intelligent Assistant')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--query', type=str, help='Search query for papers')
    parser.add_argument('--workflow', action='store_true', help='Run complete workflow')
    parser.add_argument('--eval', action='store_true', help='Run retrieval evaluation')
    parser.add_argument('--health', action='store_true', help='Show module health status')

    args = parser.parse_args()

    # 初始化助手
    assistant = ResearchAssistant(config_path=args.config)

    if args.health:
        print("\n" + "=" * 50)
        print("模块健康状态")
        print("=" * 50)
        for module, status in assistant.health_report().items():
            print(f"  {module}: {status}")
        return

    if args.eval:
        summary = assistant.evaluate_retrieval()
        if summary:
            print("\n" + "=" * 50)
            print(f"📊 评测摘要 ({summary.get('total_queries', 0)} 个查询)")
            for metric, value in summary.get('metrics', {}).items():
                print(f"  {metric}: {value:.4f}")
        return

    if args.workflow and args.query:
        # 运行完整工作流
        assistant.complete_research_workflow(args.query)
    elif args.query:
        # 仅搜索论文
        papers = assistant.search_papers(args.query)
        print(f"\n✅ Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'][:3])}")
            print(f"   Published: {paper.get('published', 'N/A')}")
    else:
        print("Research Assistant initialized. Use --query to search papers or --workflow for complete workflow.")


if __name__ == "__main__":
    main()
