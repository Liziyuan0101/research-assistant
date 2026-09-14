# 外部论文数据源

| 源 | 标识 | 鉴权 | 限流 | 字段完整度 | 失败处理 |
|---|---|---|---|---|---|
| arXiv | `arxiv` | 无 | 建议 ≤1 req/3s | 标题/摘要/作者/日期/id | 重试 2 次后跳过 |
| OpenAlex | `openalex` | 无（可选 mailto 提高配额） | 10 req/s | 含引用数、期刊 | 降级跳过 |
| Semantic Scholar | `semantic_scholar` | 可选 API key | 无 key 时 1 req/s | 含引用关系 | 慢，放最后 |
| CORE | `core` | **需 key** | 按 plan | 含全文链接 | 无 key ⇒ 直接禁用 |
| Dimensions | `dimensions` | **需 key** | 按 plan | 含资助信息 | 无 key ⇒ 直接禁用 |

## 约定

- 无 key 的源必须能工作；有 key 的源在缺 key 时**跳过并在 `trace` 中记录**，不要抛错。
- 检索结果统一归一化为 `{'id', 'title', 'authors', 'abstract', 'published', 'source'}`。
  注意 `paper_id`（带版本，如 `2502.07070v1`）与 `id`（不带版本，如 `2502.07070`）
  **是两个字段**，去重按 `id`，评测标注按 `id`。
- 入库前按 `id` 去重；`MetadataStore.remove_duplicate_chunks()` 可清理历史重复。
