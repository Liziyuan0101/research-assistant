# 实验方案 schema 与验收标准

## 字段 schema

```json
{
  "objective":    "实验目标（一句话）",
  "hypotheses":   ["H1: ...", "H2: ..."],
  "data":         {"sources": [], "size": "", "preprocessing": []},
  "methods":      [{"name": "", "rationale": ""}],
  "setup":        {"params": {}, "hyperparams": {}, "hardware": ""},
  "metrics":      ["MAE", "RMSE", "..."],
  "baselines":    [{"name": "", "source": ""}],
  "expected":     {"result": "", "validation": ""},
  "risks":        [{"risk": "", "mitigation": ""}]
}
```

## 验收标准（自检清单）

- [ ] 每个假设都能被某个指标**证伪**（否则不是假设，是口号）
- [ ] 基线至少 2 个，且都注明来源（论文引用或 `paper-retrieval` 检索结果）
- [ ] 指标与假设一一对应，没有"多指标堆砌"
- [ ] 数据来源可获取（给出下载方式或说明为私有数据）
- [ ] **不含任何编造的结果数字** —— 预期结果只能写"预期趋势 + 验证方式"
- [ ] 无相关工作可参考时，已显式声明方案为通用模板
