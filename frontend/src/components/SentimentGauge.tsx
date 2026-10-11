import type { SentimentLabel } from "../types";

interface Props {
  probPos: number;
  probNeg: number;
  label: SentimentLabel;
}

/** 正负面概率条（纯展示，不自己算标签）。**
 *
 *  **用一条水平双色条、而不是两个独立进度条**：两个进度条在视觉上读作两个互不相干的
 *  量，观众会下意识去比它们的长度；而一条被切开的条读作「同一个整体的两个部分」——
 *  这正是二分类概率的语义（两者互补，和为 1）。
 *
 *  标签由调用方传入而非此组件内推导：判定规则全应用只有 `types.labelOf` 一处，
 *  在这里再写一遍 `probPos >= 0.5` 就会与阈值滑块脱节——滑块动了、这条却不跟着动。
 */
export function SentimentGauge({ probPos, probNeg, label }: Props) {
  const pct = Math.round(probPos * 100);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline gap-3">
        <span className="text-sm text-slate-500">判定结果</span>
        <span
          className={`rounded-full px-3 py-0.5 text-sm font-medium ${
            label === "积极" ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"
          }`}
        >
          {label}
        </span>
      </div>

      <div className="flex h-6 overflow-hidden rounded-full bg-slate-100">
        <div
          className="flex items-center justify-start bg-emerald-500 pl-2 text-xs text-white transition-all"
          style={{ width: `${pct}%` }}
        >
          {pct >= 15 && `积极 ${pct}%`}
        </div>
        <div
          className="flex flex-1 items-center justify-end bg-rose-500 pr-2 text-xs text-white transition-all"
          style={{ width: `${100 - pct}%` }}
        >
          {100 - pct >= 15 && `消极 ${100 - pct}%`}
        </div>
      </div>

      <div className="flex justify-between font-mono text-xs text-slate-500">
        <span>积极 {probPos.toFixed(4)}</span>
        <span>消极 {probNeg.toFixed(4)}</span>
      </div>
    </div>
  );
}
