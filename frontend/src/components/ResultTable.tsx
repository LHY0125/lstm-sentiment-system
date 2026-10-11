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
   *  中文整片乱码。导出中文却没有 BOM，等于导出一个废文件。
   *
   *  **不做 CSV 公式注入（formula injection）防护——这是有意的决定，不是遗漏。**
   *  以 `=` `+` `-` `@` 开头的单元格会被 Excel 当公式执行（`=HYPERLINK(...)`、DDE 之类），
   *  而这里的文本列原样来自外部数据。业界通用的缓解法是给这类单元格前缀一个单引号，
   *  但**Excel 读 CSV 时并不会吃掉那个单引号**——它只是「键盘输入」的标记，
   *  对解析文件不生效，于是它会作为一个真实字符写进导出值：既污染了用户在屏幕上看到的
   *  同一份数据，也让重新导入后的值发生变化。那直接违反「导出所见即所得」，
   *  缓解手段比症状更糟，所以这里**保持字节忠实**。
   *
   *  被接受的威胁模型：本机单用户、导出的是用户自己的输入。真要防，正确的位置是在导出前
   *  显式提示、或另加一列标注来源，而不是偷偷改值。 */
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
