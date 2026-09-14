"""偏好注入策略的契约测试：`boost` vs `tiebreak` 的语义差别。

用 stub 编码器（不加载真实模型），因此毫秒级完成。

语义约定：
* ``boost``     偏好**可以覆盖**相关性信号（这正是它在跨域实验里掉点的原因）
* ``tiebreak``  偏好**只能在近似并列的候选之间**决定次序，不能把明显更相关的文档挤下去
"""

import numpy as np
import pytest

from research_assistant.memory.personalize import PreferenceProfile, preference_prior


class _StubEncoder:
    """按文本查表返回 L2 归一化向量；不需要加载模型。"""

    available = True

    def __init__(self, table):
        self.table = table

    def encode(self, texts, **kwargs):
        return np.stack([self.table[t] for t in texts]).astype('float32')


def _vec(similarity_to_profile):
    """构造与 profile 向量余弦相似度 = similarity 的单位向量。"""
    s = float(similarity_to_profile)
    return np.array([s, np.sqrt(max(0.0, 1.0 - s * s))], dtype='float32')


@pytest.fixture()
def setup():
    """5 篇文档：doc0 基分远高；doc1/2/3 近似并列（先验递增）；doc4 基分最低。"""
    names = ['d0', 'd1', 'd2', 'd3', 'd4']
    priors = [0.10, 0.20, 0.50, 0.95, 0.05]
    enc = _StubEncoder({n: _vec(p) for n, p in zip(names, priors)})
    profile = PreferenceProfile(user_id='u')
    profile.pos_vector = np.array([1.0, 0.0], dtype='float32')
    profile.positive_texts = ['pref']
    profile.backend = 'semantic'
    base = np.array([0.9000, 0.5000, 0.4995, 0.4990, 0.1000], dtype='float32')
    return names, enc, profile, base


def test_boost_can_override_relevance(setup):
    """boost 在 λ 足够大时会把高先验但明显不相关的文档顶到第一 —— 记录该行为的代价。

    λ=0.3 时还不足以覆盖 0.4 的分差（doc0 仍第一）；λ=0.6 时被覆盖。
    这个"随 λ 增大而覆盖相关性"的转折正是它在跨域实验里掉点的原因。
    """
    names, enc, profile, base = setup
    mild = preference_prior(profile, names, enc, lam=0.3, base_scores=base)
    assert int(np.argmax(mild)) == 0, '小 λ 时不应覆盖相关性'

    strong = preference_prior(profile, names, enc, lam=0.6, base_scores=base)
    assert int(np.argmax(strong)) == 3, '大 λ 时 boost 会覆盖相关性（已知代价）'


def test_tiebreak_never_overrides_relevance(setup):
    """tiebreak 不能把明显更相关的 doc0 挤下去。"""
    names, enc, profile, base = setup
    scores = preference_prior(profile, names, enc, lam=0.3, base_scores=base, mode='tiebreak')
    assert int(np.argmax(scores)) == 0, 'tiebreak 不得覆盖 0.4 的分差'


def test_tiebreak_reorders_within_band(setup):
    """在近似并列的一组内部，先验高的应排前。"""
    names, enc, profile, base = setup
    scores = preference_prior(profile, names, enc, lam=0.3, base_scores=base, mode='tiebreak')
    order = list(np.argsort(-scores))
    # doc1(先验 .20) / doc2(.50) / doc3(.95) 三者按先验降序
    band = [i for i in order if i in (1, 2, 3)]
    assert band == [3, 2, 1], band


def test_tiebreak_is_closer_to_baseline_than_boost(setup):
    """tiebreak 对原排序的扰动应小于 boost。"""
    names, enc, profile, base = setup
    base_norm = (base - base.min()) / (base.max() - base.min())
    tb = preference_prior(profile, names, enc, lam=0.3, base_scores=base, mode='tiebreak')
    bo = preference_prior(profile, names, enc, lam=0.3, base_scores=base)
    assert np.abs(tb - base_norm).max() < np.abs(bo - base_norm).max()


def test_lambda_zero_returns_base_unchanged(setup):
    """λ=0 必须完全等价于不做个性化（可复现基线的前提）。"""
    names, enc, profile, base = setup
    out = preference_prior(profile, names, enc, lam=0.0, base_scores=base)
    base_norm = (base - base.min()) / (base.max() - base.min())
    assert np.allclose(out, base_norm)


def test_empty_profile_is_noop(setup):
    names, enc, _, base = setup
    empty = PreferenceProfile(user_id='u')
    out = preference_prior(empty, names, enc, lam=0.5, base_scores=base)
    base_norm = (base - base.min()) / (base.max() - base.min())
    assert np.allclose(out, base_norm)


def test_negative_preference_lowers_score():
    """负面偏好应把相似文档压低。"""
    names = ['good', 'bad']
    enc = _StubEncoder({'good': _vec(0.0), 'bad': _vec(1.0)})
    profile = PreferenceProfile(user_id='u')
    profile.neg_vector = np.array([1.0, 0.0], dtype='float32')
    profile.negative_texts = ['不偏好 X']
    profile.backend = 'semantic'
    base = np.array([0.5, 0.5], dtype='float32')
    out = preference_prior(profile, names, enc, lam=0.4, base_scores=base)
    assert out[1] < out[0], '与负面偏好相似的文档应被降分'


def test_encoder_unavailable_returns_base():
    """编码器不可用时必须原样返回 base，而不是返回全零假装有结果。"""
    class _Dead:
        available = False

        def encode(self, texts, **kwargs):
            return None

    profile = PreferenceProfile(user_id='u')
    profile.pos_vector = np.array([1.0, 0.0], dtype='float32')
    base = np.array([0.3, 0.7], dtype='float32')
    out = preference_prior(profile, ['a', 'b'], _Dead(), lam=0.5, base_scores=base)
    assert np.allclose(out, [0.0, 1.0])   # minmax(base)
