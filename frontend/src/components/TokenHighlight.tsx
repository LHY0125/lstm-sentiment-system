import type { Contrib } from "../types";

interface Props {
  /** 完整分词结果（与 /api/predict 的 tokens 相同）——**渲染骨架**。 */
  tokens: string[];
  /** 逐词贡献，按原句词序，只覆盖前 CONTRIB_MAX_TOKENS 个位置——**delta 查表**。 */
  contributions: Contrib[];
}

/** 一个词位的**方向**。注意它有三个值而不是两个：
 *
 *  `delta === 0` 是**真的会出现**的第三种状态，不是 `delta > 0` 的边界情况。
 *  后端按「把该位置改写成 UNK_IDX」做遮挡；若该词本来就在词表外，改写前后序列完全一样，
 *  概率不变，`after - base` 恰为 `0.0`——**这次测量是空操作，什么都没测到**，
 *  既不是「模型认为它中性」，也不是「把判定推向消极」。
 *  实测：`位置太方便了…干干净净…` 一句里 `干干净净` 不在词表内，
 *  它是该句唯一 delta 恰为 0.0 的词位。
 *
 *  **已知局限**：`/api/explain` 只回 `{token, delta}`，不回该词是否在词表内，
 *  因此界面分不出「零变化因为空操作」与「零变化因为模型真的不敏感」，
 *  只能把这一档统一措辞成「本次遮挡没有产生可分辨的变化」（见文件末的图例）。
 *  要真正分开需要后端多回一个字段，属契约改动，已记入项目遗留项。 */
type Polarity = "pos" | "neg" | "neutral";

/** 方向的中文名。表格方向列与读屏文案共用，避免两处各写一套。 */
const POLARITY_LABEL: Record<Polarity, string> = {
  pos: "积极",
  neg: "消极",
  neutral: "无方向",
};

/** delta → 方向。**全组件唯一一处判定**：着色、表格文字、悬停/读屏文案都从这里取，
 *  免得三处各写一遍 `delta < 0` 再慢慢漂移。 */
function polarityOf(delta: number): Polarity {
  if (delta === 0) return "neutral";
  return delta < 0 ? "pos" : "neg";
}

/** 一个词位渲染出来的样子。用可辨识联合而不是「一堆可选字段」：
 *
 *  未分析的词位**没有**方向、强度、数值可言，若把它们做成可选字段就得填哨兵值，
 *  而任何哨兵值（比如 `"neutral"`）都可能被误渲染成「测得无方向」——那又变回冒充。
 *  做成联合之后，类型系统直接不允许未分析词位被当成已分析来画。 */
type Cell =
  | {
      kind: "analysed";
      /** 词位身份：同一个词在一句里可能出现多次，只有词位是唯一的。 */
      id: string;
      token: string;
      contribution: Contrib;
      polarity: Polarity;
      /** 着色强度 |delta| / max|delta| ∈ [0,1]。 */
      intensity: number;
      /** 方向 + 数值的说明，同时供 `title`（鼠标）与 `aria-label`（读屏）使用。 */
      description: string;
    }
  | {
      /** **未分析**词位——不是「测得为零」。 */
      kind: "unanalysed";
      id: string;
      token: string;
    };

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

/** 方向 + 数值的一句话说明。悬停与读屏共用同一个串，两者不可能对不上。
 *
 *  零变化那一档**不能说「测得无方向」**：`delta` 恰为 0 时这次遮挡可能根本没生效
 *  （词不在词表内），说「测得」就是把一次空操作讲成一次测量结论。
 *  故措辞落在「本次遮挡未产生可分辨的变化」上，并直接点出空操作这一可能。 */
function describe(polarity: Polarity, delta: number): string {
  const value = `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`;
  const cause = `删去该词后积极概率变化 ${value}`;
  return polarity === "neutral"
    ? `本次遮挡未产生可分辨的变化（${cause}）：若该词不在词表内，遮挡是空操作，不代表模型认为它中性`
    : `${POLARITY_LABEL[polarity]}贡献：${cause}`;
}

/** 依据方向取**填充色**。
 *
 *  delta 定义为「删词后的积极概率 − 原始积极概率」：
 *    delta < 0  → 该词推动「积极」，绿色
 *    delta > 0  → 该词推动「消极」，红色
 *    delta = 0  → **无方向**，中性灰（不是「偏消极」，见 Polarity 的说明）
 *
 *  用发散色阶（绿—灰—红）而非单一色阶，因为贡献是有方向的量，
 *  单色阶会把「强烈积极」和「强烈消极」画成同一个样子。 */
function background(polarity: Polarity, intensity: number): string {
  if (polarity === "neutral") {
    // 零变化这一档不乘强度：强度编码的是「影响有多大」，而这里的影响恒为 0，
    // 乘完 0.15 的下限还是几乎看不见。变量本身就是浅灰，直接用。
    // 用**填充色**而不是虚线空心，与「未分析」那一档区分开：那一档是「词位压根没进分析范围」
    // （超出 CONTRIB_MAX_TOKENS），这一档是「在分析范围内，但这次遮挡没产生可分辨的变化」。
    // 两者都读不出方向，差别在**有没有为它跑过前向**；而这一档内部还混着「空操作」那一类，
    // 界面分不开，见文件顶部 Polarity 的已知局限与 README 的「已知局限」小节。
    return "var(--contrib-neutral)";
  }
  const alpha = 0.15 + intensity * 0.55; // 保留一个可见的下限，弱贡献也能被看见
  return `color-mix(in oklch, var(--contrib-${polarity}) ${alpha * 100}%, transparent)`;
}

/** 表格里方向文字的**文字色**。
 *
 *  不能直接拿 `--contrib-*` 当文字色：这三个变量是按**填充**调的，亮度 0.66~0.92，
 *  压在白底上只有 ~2:1 对比度，比原来的 `text-*-700` 还难读。所以对同一个变量往黑里
 *  压一档——改 `index.css` 就能同时挪动整句与表格（只有一个色源），可读性又不丢。 */
function directionTextColor(polarity: Polarity): string {
  return `color-mix(in oklch, var(--contrib-${polarity}) 60%, black)`;
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
    if (!contribution) {
      return { kind: "unanalysed", id: `${token}#${i}`, token };
    }
    const polarity = polarityOf(contribution.delta);
    return {
      kind: "analysed",
      id: `${token}#${i}`,
      token,
      contribution,
      polarity,
      intensity: Math.abs(contribution.delta) / maxAbs,
      description: describe(polarity, contribution.delta),
    };
  });

  // 单一异常状态：**长度**或**身份**任一不对，都说明「contributions 逐位对齐 tokens 前缀」
  // 这个前提破了。此时显示的词来自 tokens、颜色与数值来自 contributions，两者已经错配——
  // 画出来是假数据，而且**不会抛错**。合并成一条提示，只留一个可见状态。
  const extraContributions = Math.max(covered - tokens.length, 0);
  const misplaced = cells.filter(
    (c) => c.kind === "analysed" && c.contribution.token !== c.token,
  ).length;
  const misalignedReason = [
    extraContributions > 0 ? `多出 ${extraContributions} 项贡献` : "",
    misplaced > 0 ? `${misplaced} 个词位的词对不上` : "",
  ]
    .filter(Boolean)
    .join("、");

  // 按 |delta| 取前 10 —— 排序是**呈现**所需，服务端已改为保序返回，故在这里排。
  // 必须先复制：contributions 按原句词序来，原地排序会毁掉上面整句渲染所依赖的顺序。
  const top = [...contributions]
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, 10)
    .map((c, rank) => ({
      id: `${c.token}#${rank}`,
      token: c.token,
      delta: c.delta,
      polarity: polarityOf(c.delta),
    }));

  return (
    <div className="flex flex-col gap-3">
      <div>
        <h3 className="mb-2 text-sm font-medium text-slate-700">逐词贡献（遮挡法）</h3>
        <p className="flex flex-wrap gap-1 text-base leading-relaxed">
          {cells.map((cell) =>
            cell.kind === "analysed" ? (
              <span
                key={cell.id}
                style={{ backgroundColor: background(cell.polarity, cell.intensity) }}
                className="rounded px-1 py-0.5"
                title={cell.description}
              >
                {cell.token}
                {/* 颜色不能是唯一的语义通道，而 `title` 只有鼠标够得着。这里用的是**真实文本**
                    而不是 `aria-label`：`aria-label` 在无角色的 <span>（ARIA 的 generic）上
                    被规范禁止命名，Biome 的 a11y/useAriaPropsSupportedByRole 会直接报错；
                    而文本内容在所有读屏组合下都保证被朗读，也不需要给一个「词」硬套
                    role="img" 那种错误语义。对鼠标用户它不可见（sr-only 裁成 1px）。 */}
                <span className="sr-only">{cell.description}</span>
              </span>
            ) : (
              // 未分析段：必须与「测过且确实无影响」视觉可分，否则用户无法区分
              // 「没测」和「测了是零」——那等于销毁信息。用**虚线空心**而不是中性填充色，
              // 是因为「测得无方向」本来就是中性填充，只靠颜色（尤其灰色）分不开。
              <span
                key={cell.id}
                className="rounded border border-dashed border-slate-300 px-1 py-0.5 text-slate-400"
                title="未分析：逐词贡献只覆盖前若干词"
              >
                {cell.token}
              </span>
            ),
          )}
        </p>
        <p className="mt-2 text-xs text-slate-400">
          绿色 = 推高积极判定；红色 = 推高消极判定；颜色越深影响越大。灰色 = 本次遮挡没有产生
          可分辨的变化（删去该词后概率不变）——这不等于「模型认为它中性」：若该词不在词表内，
          遮挡本身是空操作（改写前后序列完全相同，delta 字面为 0），这次测量什么都没测到。
          悬停可看具体数值。
          {covered < tokens.length && (
            <>
              {" "}
              虚线框的词
              <strong className="font-semibold text-slate-500">未分析</strong>
              （逐词贡献只覆盖前 {covered} 词），与「测得无影响」不是一回事。
            </>
          )}
        </p>
        {/* 防御性提示：长度或身份对不上都属于响应自相矛盾（后端契约不会产生），
            但静默按错位的对子渲染又是一次信息销毁，所以让它可见。 */}
        {misalignedReason !== "" && (
          <p className="mt-1 text-xs text-amber-700">
            响应异常：逐词贡献与分词结果没有逐位对齐（{misalignedReason}
            ）。显示的词来自分词结果、颜色与数值来自逐词贡献，两者已经错配，请勿据此下结论。
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
              {/* 方向列写「积极/消极/无方向」而不是只放一个色块：颜色不能是唯一的语义通道 */}
              <th className="py-1 text-right font-normal">方向</th>
            </tr>
          </thead>
          <tbody>
            {top.map((row) => (
              <tr key={row.id} className="border-b border-slate-100">
                <td className="py-1">{row.token}</td>
                <td className="py-1 text-right font-mono">{Math.abs(row.delta).toFixed(4)}</td>
                <td className="py-1 text-right">
                  <span style={{ color: directionTextColor(row.polarity) }}>
                    {POLARITY_LABEL[row.polarity]}
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
