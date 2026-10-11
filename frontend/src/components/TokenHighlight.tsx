import type { Contrib } from "../types";

interface Props {
  /** 完整分词结果（与 /api/predict 的 tokens 相同）——**渲染骨架**。 */
  tokens: string[];
  /** 逐词贡献，按原句词序，只覆盖前 CONTRIB_MAX_TOKENS 个位置——**delta 查表**。 */
  contributions: Contrib[];
}

/** 整句高亮的一个渲染单元。 */
interface Cell {
  /** 词位身份：同一个词在一句里可能出现多次，只有词位是唯一的。 */
  id: string;
  token: string;
  /** 该词位的贡献；`undefined` = **未分析**（不是「测得为零」）。 */
  contribution: Contrib | undefined;
  /** 该词位的着色强度 |delta| / max|delta| ∈ [0,1]；未分析词位恒为 0 且渲染时不读取。 */
  intensity: number;
  /** 悬停提示：颜色之外的第二条通道，色盲用户也能读出方向与数值。 */
  detail: string;
}

/** 当前结果里最大的 |delta|，作为归一化分母。
 *
 *  下界 1e-9 是空数组与「全部 delta 为 0」的兜底，避免除以零得到 NaN。
 *
 *  **不建 `token → 强度` 的表**：一句话里同一个词可以出现多次，而每次出现的 |delta|
 *  是各自独立测出来的。按 token 建表会让后一次的强度盖掉前一次，于是两个**测得不同**的
 *  词位画成同一个深度——与图例承诺的「颜色越深影响越大」直接矛盾，也属于把不同的测量
 *  结果渲染成同一种样子。故强度一律按**词位**算。 */
function maxAbsDelta(contributions: Contrib[]): number {
  return Math.max(...contributions.map((c) => Math.abs(c.delta)), 1e-9);
}

/** 依据 delta 取背景色。
 *
 *  delta 定义为「删词后的积极概率 − 原始积极概率」：
 *    delta < 0  → 该词推动「积极」，绿色
 *    delta > 0  → 该词推动「消极」，红色
 *
 *  用发散色阶（绿—灰—红）而非单一色阶，因为贡献是有方向的量，
 *  单色阶会把「强烈积极」和「强烈消极」画成同一个样子。 */
function background(c: Contrib, intensity: number): string {
  const alpha = 0.15 + intensity * 0.55; // 保留一个可见的下限，弱贡献也能被看见
  return c.delta < 0
    ? `color-mix(in oklch, var(--contrib-pos) ${alpha * 100}%, transparent)`
    : `color-mix(in oklch, var(--contrib-neg) ${alpha * 100}%, transparent)`;
}

/** 逐词贡献高亮：整句着色 + 影响最大的词排名表。
 *
 *  **两个输入缺一不可**：`tokens` 是完整渲染骨架（长可达 MAX_LEN），`contributions`
 *  只覆盖前 `CONTRIB_MAX_TOKENS`（100）个位置。只按 `contributions` 渲染会把长句截断到
 *  前 100 词，并把「未分析」静默画成「测得零贡献」——那是销毁信息。 */
export function TokenHighlight({ tokens, contributions }: Props) {
  // 归一化分母取当前结果内最大的 |delta|：绝对值随模型/输入浮动，只有相对深浅才稳定可比。
  const maxAbs = maxAbsDelta(contributions);

  // contributions 是 tokens 的前缀（服务端保序），故按下标一一对应。
  // 超出 contributions 长度的尾部是「**未分析**」——不是「测得贡献为零」。
  const covered = contributions.length;

  // 先把整句摊成渲染单元，再进 JSX：两个状态（已分析 / 未分析）的判定与 key 的构造
  // 都不是一行能写完的表达式，放进 JSX 会把它读成模板，而不是逻辑。
  // key 用「词 + 词位」而不是纯 token：一句话里同一个词可以重复出现，纯 token 会撞 key。
  // 该序列在给定结果下是定序的、这些 span 也不持有任何 state，故位置身份是安全的。
  const cells: Cell[] = tokens.map((token, i) => {
    const contribution = i < covered ? contributions[i] : undefined;
    return {
      id: `${token}#${i}`,
      token,
      contribution,
      intensity: contribution ? Math.abs(contribution.delta) / maxAbs : 0,
      detail: contribution
        ? `删去该词后积极概率变化 ${contribution.delta >= 0 ? "+" : ""}${contribution.delta.toFixed(4)}`
        : "未分析：逐词贡献只覆盖前若干词",
    };
  });

  // 按 |delta| 取前 10 —— 排序是**呈现**所需，服务端已改为保序返回，故在这里排。
  // 必须先复制：contributions 按原句词序来，原地排序会毁掉上面整句渲染所依赖的顺序。
  const top = [...contributions]
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, 10)
    .map((c, rank) => ({ id: `${c.token}#${rank}`, token: c.token, delta: c.delta }));

  return (
    <div className="flex flex-col gap-3">
      <div>
        <h3 className="mb-2 text-sm font-medium text-slate-700">逐词贡献（遮挡法）</h3>
        <p className="flex flex-wrap gap-1 text-base leading-relaxed">
          {cells.map((cell) =>
            cell.contribution ? (
              <span
                key={cell.id}
                style={{ backgroundColor: background(cell.contribution, cell.intensity) }}
                className="rounded px-1 py-0.5"
                title={cell.detail}
              >
                {cell.token}
              </span>
            ) : (
              // 未分析段：必须与「测过且确实无影响」视觉可分，否则用户无法区分
              // 「没测」和「测了是零」——那等于销毁信息。用**虚线空心**而不是中性填充色，
              // 是因为「测得接近零」本来就是浅色填充，只靠颜色深浅分不开。
              <span
                key={cell.id}
                className="rounded border border-dashed border-slate-300 px-1 py-0.5 text-slate-400"
                title={cell.detail}
              >
                {cell.token}
              </span>
            ),
          )}
        </p>
        <p className="mt-2 text-xs text-slate-400">
          绿色 = 推高积极判定；红色 = 推高消极判定；颜色越深影响越大。悬停可看具体数值。
          {covered < tokens.length && (
            <>
              {" "}
              虚线框的词
              <strong className="font-semibold text-slate-500">未分析</strong>
              （逐词贡献只覆盖前 {covered} 词），与「测得无影响」不是一回事。
            </>
          )}
        </p>
        {/* 防御性提示：contributions 比 tokens 还长属于响应自相矛盾（后端契约不会产生），
            但静默丢掉多出来的贡献又是一次信息销毁，所以让它可见。 */}
        {covered > tokens.length && (
          <p className="mt-1 text-xs text-amber-700">
            响应不一致：逐词贡献 {covered} 项多于分词结果 {tokens.length} 项，整句已按分词结果渲染。
          </p>
        )}
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-slate-700">影响最大的词</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs text-slate-400">
              <th className="py-1 font-normal">词</th>
              <th className="py-1 text-right font-normal">影响</th>
              {/* 方向列写「积极/消极」而不是只放一个色块：颜色不能是唯一的语义通道 */}
              <th className="py-1 text-right font-normal">方向</th>
            </tr>
          </thead>
          <tbody>
            {top.map((row) => (
              <tr key={row.id} className="border-b border-slate-100">
                <td className="py-1">{row.token}</td>
                <td className="py-1 text-right font-mono">{Math.abs(row.delta).toFixed(4)}</td>
                <td className="py-1 text-right">
                  <span className={row.delta < 0 ? "text-emerald-700" : "text-rose-700"}>
                    {row.delta < 0 ? "积极" : "消极"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
