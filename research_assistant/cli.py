"""
Research Assistant - Command Line Interface
科研论文智能助手命令行入口
"""

import argparse

from .assistant import ResearchAssistant


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Research Paper Intelligent Assistant')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--query', type=str, help='Search query for papers')
    parser.add_argument('--workflow', action='store_true', help='Run complete workflow')

    args = parser.parse_args()

    # 初始化助手
    assistant = ResearchAssistant(config_path=args.config)

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
