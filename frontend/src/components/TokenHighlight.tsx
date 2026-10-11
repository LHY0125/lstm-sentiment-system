import type { Contrib } from "../types";

interface Props {
  /** 完整分词序列，用于渲染整句骨架。长度**不受** CONTRIB_MAX_TOKENS 限制。 */
  tokens: string[];
  /** 逐词贡献，只覆盖前 CONTRIB_MAX_TOKENS 个词位，须按 token 查表、不能当词序列用。 */
  contributions: Contrib[];
}

/** 逐词贡献高亮的**占位实现**——真正的着色在 Task 12 补上。
 *
 *  本文件此刻存在的唯一理由，是让 `SingleAnalyzer` 能编译通过、并**先把接口钉死**。
 *  Task 12 会换掉整个 body，但沿用这两个输入：
 *  `contributions` 被截断在 100 个词位，`tokens` 才是完整句；Task 12 必须用 `tokens`
 *  渲染整句、用 `contributions` 只做 delta 查表——否则长句会被截到前 100 个词，
 *  未分析的后半句会被静默渲染成「贡献为 0」，那是有意禁止的信息销毁。
 *  两个输入恒同生命周期（同为 null 或同为非 null），调用方不得只给其一。
 */
export function TokenHighlight({ tokens, contributions }: Props) {
  return (
    <div className="text-sm text-slate-400">
      逐词贡献：{contributions.length} 个词位（整句共 {tokens.length} 个词位）
    </div>
  );
}
