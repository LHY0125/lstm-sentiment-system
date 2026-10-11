import type { Contrib } from "../types";

interface Props {
  contributions: Contrib[];
}

/** 逐词贡献高亮的**占位实现**——真正的着色在 Task 12 补上。
 *
 *  本文件此刻存在的唯一理由是让 `SingleAnalyzer` 能编译通过：Task 11 已把
 *  `/api/explain` 的 `contributions` 拿到手，但「怎么把 delta 映射成颜色」是
 *  Task 12 的题目。这里先如实把「收到几个词」显示出来，既不假装完成了着色，
 *  也保留了 Task 12 要沿用的 props 签名（`contributions: Contrib[]`）。
 */
export function TokenHighlight({ contributions }: Props) {
  return <div className="text-sm text-slate-400">逐词贡献：{contributions.length} 个词</div>;
}
