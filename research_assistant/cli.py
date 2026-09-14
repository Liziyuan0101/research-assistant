"""
Research Assistant - Command Line Interface
科研论文智能助手命令行入口

示例:
    # 查看模块状态 / skill 列表
    research-assistant --health

    # 建索引后做个性化混合检索
    research-assistant --user-id lzy --lambda 0.25 \
        --pref "method=Bayesian deep learning for uncertainty quantification" \
        --pref "keyword=remaining useful life" \
        --search "remaining useful life prediction of lithium-ion batteries"

    # 让 Agent 自己挑 skill 干活（默认单 Agent + Skills，路由 0 次 LLM 调用）
    research-assistant --user-id lzy --agent "帮我设计一个消融实验，对比 PINN 和纯数据驱动模型"

    # 查看记忆画像（诊断个性化为何生效/不生效）
    research-assistant --user-id lzy --memory
"""

import argparse
import json
import sys

from .assistant import ResearchAssistant
from .utils.logger import setup_logger


def _parse_pref(spec: str):
    """``category=value``；``value`` 以 ``!`` 开头表示负面偏好。"""
    if '=' not in spec:
        raise argparse.ArgumentTypeError(f'偏好格式应为 category=value，收到: {spec}')
    category, value = spec.split('=', 1)
    category, value = category.strip(), value.strip()
    polarity = 1
    if value.startswith('!'):
        polarity, value = -1, value[1:].strip()
    return category, value, polarity


def _print_health(assistant):
    print('\n' + '=' * 56)
    print('模块健康状态')
    print('=' * 56)
    for module, status in assistant.health_report().items():
        print(f'  {module}: {status}')


def _print_skills(assistant):
    if assistant.agent is None:
        print('⚠️ 单 Agent 运行时未初始化')
        return
    reg = assistant.agent.registry
    loader = assistant.agent.loader
    print('\n' + '=' * 56)
    print(f'Skills（{len(reg.skills)} 个）— 渐进披露三级加载')
    print('=' * 56)
    for s in reg.skills:
        print(f'\n  ▸ {s.name}   (正文 {s.body_chars} 字符, {len(s.references)} 个 references)')
        print(f'      {s.description[:100]}')
    print(f"\n  L0 常驻索引 : {reg.resident_chars()} 字符")
    print(f"  L1+L2 已加载: {loader.loaded_chars} 字符")
    print('  路由: skill frontmatter 确定性命中，0 次 LLM 调用')


def _print_memory(assistant):
    print('\n' + '=' * 56)
    print(f'记忆画像 — user_id={assistant.user_id}')
    print('=' * 56)
    print(json.dumps(assistant.memory_report(), ensure_ascii=False, indent=2))
    prefs = assistant.get_preferences(include_negative=True)
    if any(prefs.values()):
        print('\n偏好明细:')
        for cat, values in prefs.items():
            if values:
                print(f'  {cat}: {values}')


def _print_results(results, limit=10):
    if not results:
        print('（无结果）')
        return
    for i, r in enumerate(results[:limit], 1):
        paper = r.get('paper') or {}
        title = (paper.get('title') or '?')[:66]
        print(f"  {i:>2}. [{r.get('score', 0):.4f}] {title}")
        if r.get('personalized'):
            print('      ↑ 个性化命中')


def main():
    """主函数"""
    setup_logger('research_assistant')
    parser = argparse.ArgumentParser(
        description='Research Paper Intelligent Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--user-id', default='default',
                        help='用户标识，用于个性化记忆（默认 default）')
    parser.add_argument('--backend', choices=['single', 'graph'], default='single',
                        help='运行时后端：single=单 Agent + Skills(默认)，graph=LangGraph')
    parser.add_argument('--lambda', dest='prior_lambda', type=float, default=None,
                        help='偏好回流检索先验强度 λ（0=关闭个性化）')
    parser.add_argument('--pref', action='append', default=[], metavar='CAT=VALUE',
                        help='写入一条偏好，可重复；值以 ! 开头表示负面偏好')
    parser.add_argument('--query', type=str, help='Search query for papers (external API)')
    parser.add_argument('--search', type=str, help='在本地索引上做混合检索（支持个性化）')
    parser.add_argument('--agent', type=str, help='交给 Agent 执行的任务（自动选 skill）')
    parser.add_argument('--chain', action='store_true',
                        help='--agent 时把命中的多个 skill 依次串起来')
    parser.add_argument('--workflow', action='store_true', help='Run complete workflow')
    parser.add_argument('--eval', action='store_true', help='Run retrieval evaluation')
    parser.add_argument('--health', action='store_true', help='Show module health status')
    parser.add_argument('--skills', action='store_true', help='List skills and context footprint')
    parser.add_argument('--memory', action='store_true', help='Show user memory profile')

    args = parser.parse_args()

    # 初始化助手
    assistant = ResearchAssistant(config_path=args.config, user_id=args.user_id,
                                  backend=args.backend,
                                  prior_lambda=args.prior_lambda)

    # 写入偏好（供个性化检索使用）
    for spec in args.pref:
        try:
            category, value, polarity = _parse_pref(spec)
        except argparse.ArgumentTypeError as e:
            print(f'⚠️ 跳过偏好: {e}')
            continue
        ok = assistant.add_preference(category, value, polarity=polarity)
        print(f"{'✅' if ok else '❌'} 偏好 {category} = {value}"
              f"{'（负面）' if polarity < 0 else ''}")

    if args.health:
        _print_health(assistant)
        return

    if args.skills:
        _print_skills(assistant)
        return

    if args.memory:
        _print_memory(assistant)
        return

    if args.eval:
        summary = assistant.evaluate_retrieval()
        if summary:
            print('\n' + '=' * 50)
            print(f"📊 评测摘要 ({summary.get('total_queries', 0)} 个查询)")
            for metric, value in summary.get('metrics', {}).items():
                print(f'  {metric}: {value:.4f}')
        return

    if args.agent:
        result = assistant.run_agent(args.agent, chain=args.chain)
        print('\n' + '=' * 56)
        print('Agent 执行结果')
        print('=' * 56)
        routing = result.get('routing', {})
        print(f"  命中 skills  : {routing.get('skills_matched')}")
        print(f"  执行 skills  : {routing.get('skills_executed')}")
        print(f"  路由跳数     : {routing.get('hops')}  (LLM 调用 {routing.get('llm_calls_for_routing')})")
        print(f"  常驻/已加载  : {routing.get('resident_index_chars')} / {routing.get('loaded_chars')} 字符")
        print(f"  画像         : {result.get('profile')}")
        print(f"  耗时         : {result.get('elapsed_ms')} ms")
        for step in result.get('steps', []):
            line = f"  ─ {step.get('skill')}: {step.get('status')}"
            if step.get('reason'):
                line += f"  ({step['reason']})"
            print(line)
            if step.get('backend'):
                print(f"      检索后端: {step['backend']}")
            out = step.get('output') or {}
            if out.get('papers'):
                print(f"      命中 {len(out['papers'])} 篇:")
                for i, p in enumerate(out['papers'][:5], 1):
                    print(f"        {i}. {(p.get('title') or '?')[:60]}")
            if out.get('plan'):
                keys = list(out['plan'])[:6]
                print(f"      实验方案字段: {keys}")
        return

    if args.search:
        results = assistant.hybrid_search(args.search, top_k=8, use_rerank=False)
        meta = getattr(assistant.hybrid_retriever, 'last_search_meta', {})
        backend = meta.get('backend', '?')
        print(f"\n🔍 混合检索（后端={backend}，个性化={meta.get('personalized')}）")
        print(f"   索引统计: {assistant.hybrid_retriever.get_stats() if assistant.hybrid_retriever else 'N/A'}")
        _print_results(results)
        return

    if args.workflow and args.query:
        assistant.complete_research_workflow(args.query)
    elif args.query:
        papers = assistant.search_papers(args.query)
        print(f'\n✅ Found {len(papers)} papers:')
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'][:3])}")
            print(f"   Published: {paper.get('published', 'N/A')}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
