import Papa from "papaparse";
import { type BatchItem, labelOf } from "../types";

interface Props {
  items: BatchItem[];
  threshold: number;
}

/** 批量结果的表格 + CSV 导出。
 *
 *  **标签一律用 `labelOf(prob_pos, threshold)` 现算**，不清后端要 label。批量响应本来
 *  就只有概率（没有 `label`、没有 `tokens`），这不是缺字段而是设计：阈值是用户手里的活旋钮，
 *  它一变，**表格与导出必须立刻跟着变，且不发请求**。
 */
export function ResultTable({ items, threshold }: Props) {
  /** 导出**屏幕上看到的**结果。
   *
   *  标签按当前阈值现算，不是某个概率快照——否则用户拖完滑块再导出，拿到的是与屏幕对不上的文件。
   *
   *  前缀 `\uFEFF`（BOM）不可省：Excel 见不到 BOM 就按系统代码页（简中 Windows 是 GBK）解码，
   *  中文整片乱码。导出中文却没有 BOM，等于导出一个废文件。 */
  function exportCsv() {
    const rows = items.map((it) => ({
      文本: it.text,
      标签: labelOf(it.prob_pos, threshold),
      积极概率: it.prob_pos.toFixed(4),
      消极概率: it.prob_neg.toFixed(4),
    }));
    const csv = Papa.unparse(rows);
    const blob = new Blob([`\uFEFF${csv}`], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sentiment_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  // 统计也走 labelOf：屏幕上的分类口径与表内标签、导出的 CSV 三者只由这一个函数定义
  const posCount = items.filter((it) => labelOf(it.prob_pos, threshold) === "积极").length;

  // ⚠️ 不要用数组下标当 key：Biome 的 suspicious/noArrayIndexKey 会判 error（全局约束）。
  // 先把 items 摊成带**显式身份字段**的渲染单元，再在 JSX 里 key={row.id}。
  // 批量行的稳定身份就是它在输入里的次序（该序列在单次结果内定序、行不持有任何 state），
  // 故 id 由 idx 构造；关键是 **key 处不出现下标表达式本身**。
  const rows = items.map((it, idx) => ({ id: `row-${idx}`, index: idx, ...it }));

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-slate-500">
          共 {items.length} 条，积极 {posCount} 条 / 消极 {items.length - posCount} 条
        </p>
        <button
          type="button"
          onClick={exportCsv}
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50"
        >
          导出 CSV
        </button>
      </div>

      {/* 1000 行也只在**同一条**滚动容器里滚：表头 sticky 才不会跟着跑掉。
          行内只做 toFixed 与一次数值比较，没有任何按行重算的昂贵操作，故不需要虚拟列表。 */}
      <div className="max-h-96 overflow-auto rounded-lg border border-slate-200">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-slate-50">
            <tr className="text-left text-xs text-slate-500">
              <th className="px-3 py-2 font-normal">#</th>
              <th className="px-3 py-2 font-normal">文本</th>
              <th className="px-3 py-2 font-normal">标签</th>
              <th className="px-3 py-2 text-right font-normal">积极概率</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const label = labelOf(row.prob_pos, threshold);
              return (
                <tr key={row.id} className="border-t border-slate-100">
                  <td className="px-3 py-2 text-slate-400">{row.index + 1}</td>
                  {/* 长评论用 truncate 截断，title 里留全文：1000 行时不能让单条把整表撑出横向滚动。
                      直接渲染成文本节点（不用 innerHTML）——评论原文是外部数据，不参与任何解析。 */}
                  <td className="max-w-md truncate px-3 py-2" title={row.text}>
                    {row.text}
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs ${
                        label === "积极"
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-rose-100 text-rose-800"
                      }`}
                    >
                      {label}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-slate-600">
                    {row.prob_pos.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
