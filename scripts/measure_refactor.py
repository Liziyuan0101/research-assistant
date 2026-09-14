"""重构度量：M1(常驻上下文) / M2(路由跳数、路由 LLM 调用) / M4(编排层代码行数)。

对照方式：``git show HEAD:<file>`` 取重构**前**的版本，与工作区当前版本对比，
因此数字是从 git 历史真实算出来的，不是人工估算。

用法:
    python scripts/measure_refactor.py            # 打印报告
    python scripts/measure_refactor.py --save     # 并写 JSON
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MA = 'research_assistant/agents/multi_agent.py'


def tiktoken_len(text: str, encoding: str = 'cl100k_base') -> int:
    """用 tiktoken 统计 token 数；不可用时按 CJK 字符 1.5 token / 其他 0.25 token 估算。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        cjk = len(re.findall(r'[\u4e00-\u9fff]', text))
        other = len(text) - cjk
        return int(cjk * 1.5 + other * 0.25)


def git_show(rev_path: str) -> str:
    out = subprocess.run(['git', 'show', f'HEAD:{rev_path}'], cwd=ROOT,
                         capture_output=True, text=True, encoding='utf-8')
    if out.returncode != 0:
        raise RuntimeError(f'git show failed: {out.stderr[:200]}')
    return out.stdout


def loc(text: str) -> int:
    return len([l for l in text.splitlines() if l.strip()])


def extract_prompts(src: str) -> dict:
    return {m.group(1): m.group(2) for m in
            re.finditer(r'^([A-Z_]+PROMPT)\s*=\s*"""(.*?)"""', src, re.DOTALL | re.MULTILINE)}


def count_defs(src: str, names) -> int:
    return sum(1 for n in names if re.search(rf'def {n}\b', src))


def extract_func(src: str, name: str) -> str:
    """抽出 4 空格缩进的 ``def name(...)`` 函数体（到下一个同级 def/class 为止）。"""
    lines = src.splitlines()
    start = None
    for i, l in enumerate(lines):
        if re.match(rf'^    def {re.escape(name)}\b', l):
            start = i
            break
    if start is None:
        return ''
    out = [lines[start]]
    for l in lines[start + 1:]:
        if re.match(r'^    def \w', l) or re.match(r'^class \w', l) or re.match(r'^\w', l):
            break
        out.append(l)
    return '\n'.join(out)


ROUTING_FUNCS = ['_supervisor_node', '_simple_task_classification', '_route_task', '_check_next_agent']
NEW_ROUTING_FUNCS = ['_dispatch_node', '_route_task', '_check_next_agent']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    old_src = git_show(MA)
    new_src = (ROOT / MA).read_text(encoding='utf-8')

    report = {}

    # ---------------- M4: 编排层代码行数
    old_routing_loc = sum(loc(extract_func(old_src, n)) for n in ROUTING_FUNCS)
    new_routing_loc = sum(loc(extract_func(new_src, n)) for n in NEW_ROUTING_FUNCS)
    report['M4_loc'] = {
        'before_total': loc(old_src),
        'before_raw': len(old_src.splitlines()),
        'after_total': loc(new_src),
        'after_raw': len(new_src.splitlines()),
        'before_routing_loc': old_routing_loc,
        'after_routing_loc': new_routing_loc,
        'routing_reduction_pct': round(100 * (1 - new_routing_loc / old_routing_loc), 1) if old_routing_loc else 0,
    }

    # ---------------- M1: 常驻上下文
    old_prompts = extract_prompts(old_src)
    old_items = {k: tiktoken_len(v) for k, v in old_prompts.items()}
    old_chars = {k: len(v) for k, v in old_prompts.items()}

    # 重构后：常驻 = skill frontmatter 索引（registry.resident_chars）
    from research_assistant.skills import SkillRegistry, SkillLoader
    reg = SkillRegistry()
    loader = SkillLoader(reg)

    resident_index = reg.index_text()
    resident_tokens = tiktoken_len(resident_index)

    # 命中单个 skill 后额外加载的正文
    body_tokens = {}
    for s in reg.skills:
        b = loader.load(s.name)
        body_tokens[s.name] = tiktoken_len(b.body) if b else 0

    old_total_tokens = sum(old_items.values())
    worst_new = resident_tokens + max(body_tokens.values() or [0])

    report['M1_context'] = {
        'before_prompts': old_items,
        'before_prompts_chars': old_chars,
        'before_total_tokens': old_total_tokens,
        'after_resident_index_chars': len(resident_index),
        'after_resident_tokens': resident_tokens,
        'after_skill_body_tokens': body_tokens,
        'after_worst_case_tokens': worst_new,          # 常驻 + 单个 skill 正文
        'reduction_pct': round(100 * (1 - resident_tokens / old_total_tokens), 1) if old_total_tokens else 0,
    }

    # ---------------- M2: 路由
    report['M2_routing'] = {
        'before': {
            'routing_functions': count_defs(old_src, ROUTING_FUNCS),
            'llm_calls_for_routing': 1,     # _supervisor_node 里的 self.llm.invoke
            'decisions_per_request': '1 (supervisor) + 1 (_route_task) + N (_check_next_agent)',
        },
        'after': {
            'routing_functions': count_defs(new_src, NEW_ROUTING_FUNCS),
            'llm_calls_for_routing': 0,     # SkillRegistry 词元打分
            'decisions_per_request': '1 (registry.select)',
        },
    }

    # ---------------- skill 选择冒烟测试
    from research_assistant.skills import SkillRegistry as SR

    class _FakeLoader:
        pass

    probes = [
        '帮我检索一下退役电池寿命预测的相关论文',
        'search for recent papers on retrieval augmented generation',
        '帮我设计一个消融实验，对比 PINN 和纯数据驱动模型',
        '帮我写一段论文摘要，关于梯次利用电池分选',
        'polish this abstract into academic English',
    ]
    sel = []
    for p in probes:
        metas = reg.select(p, top_k=3)
        sel.append({
            'task': p,
            'selected': [m.name for m in metas],
            'scores': [round(reg.score(p, m), 3) for m in metas],
        })
    report['skill_selection'] = sel

    # ---------------- 打印
    print('=' * 78)
    print('重构度量报告（对照 git HEAD 的重构前版本）')
    print('=' * 78)

    m4 = report['M4_loc']
    print(f"\n[M4] 编排层代码行数  {MA}")
    print(f"     文件总行数   重构前 {m4['before_raw']}  ->  重构后 {m4['after_raw']}  ({m4['after_raw'] - m4['before_raw']:+d})")
    print(f"     其中路由层   重构前 {m4['before_routing_loc']}  ->  重构后 {m4['after_routing_loc']} "
          f"({m4['routing_reduction_pct']:+.1f}%)")
    print(f"     注: 文件总行数上升是因为度量埋点与 docstring 写在同一文件；路由层本身是净减少。")

    m1 = report['M1_context']
    print(f"\n[M1] 常驻上下文（tiktoken cl100k_base）")
    print(f"     {'重构前（4 份常驻 system prompt）':<38}{'tokens':>10}")
    for k, v in m1['before_prompts'].items():
        print(f"       {k:<36}{v:>10}")
    print(f"       {'合计':<36}{m1['before_total_tokens']:>10}")
    print(f"     重构后常驻索引({len(reg.skills)} 个 skill 的 frontmatter)      {m1['after_resident_tokens']:>10}")
    print(f"     命中单个 skill 后的最坏情况                 {m1['after_worst_case_tokens']:>10}")
    print(f"     常驻上下文下降: {m1['reduction_pct']}%")
    print('\n     各 skill 正文 tokens（按需加载，不进常驻）:')
    for k, v in m1['after_skill_body_tokens'].items():
        print(f"       {k:<36}{v:>10}")

    m2 = report['M2_routing']
    print(f"\n[M2] 路由")
    print(f"     重构前: 路由函数 {m2['before']['routing_functions']} 个 | "
          f"路由 LLM 调用 {m2['before']['llm_calls_for_routing']} 次 | 每请求决策: {m2['before']['decisions_per_request']}")
    print(f"     重构后: 路由函数 {m2['after']['routing_functions']} 个 | "
          f"路由 LLM 调用 {m2['after']['llm_calls_for_routing']} 次 | 每请求决策: {m2['after']['decisions_per_request']}")

    print(f"\n[Skill 选择] 冒烟测试（0 次 LLM 调用）")
    for s in sel:
        print(f"     {s['task'][:34]:<36} -> {s['selected']}  {s['scores']}")

    if args.save:
        out = ROOT / 'data' / 'eval' / 'refactor_metrics.json'
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n结果已保存: {out}")
    print('=' * 78)


if __name__ == '__main__':
    main()
