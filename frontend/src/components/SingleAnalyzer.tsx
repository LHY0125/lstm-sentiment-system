import { useState } from "react";
import { fetchExplain, fetchPredict } from "../api/client";
import { type Contrib, labelOf } from "../types";
import { SentimentGauge } from "./SentimentGauge";
import { TokenHighlight } from "./TokenHighlight";

interface Props {
  modelId: string;
  maxLen: number;
  useExplain: boolean;
  threshold: number;
}

/** 三个演示例句，覆盖正/负/中性三种结局。
 *
 *  存在理由是「演示不依赖现场手打中文」：讲的人点一下就能出一张概率条。
 *  标签写在按钮上（而不是「示例 1/2/3」）是因为演示时需要的正是「哪个是正面例子」，
 *  编号只告诉观众顺序、不告诉内容。
 */
const EXAMPLES: { label: string; text: string }[] = [
  {
    label: "正面",
    text: "位置太方便了，出地铁站步行三分钟就到，前台还主动帮我升级了江景房，房间收拾得干干净净，早餐种类也多",
  },
  {
    label: "负面",
    text: "空调外机整晚嗡嗡响根本睡不着，跟前台反映两次都没人管，床单上还有头发，这个价位实在说不过去",
  },
  {
    label: "中性",
    text: "房间不大但挺干净，就是隔音一般，楼下在装修早上八点就开始吵，位置和价格都还说得过去，凑合住一晚没问题",
  },
];

/** 单条评论分析面板：输入 → 调后端 → 概率条（可选逐词贡献）。
 *
 *  **阈值 / MAX_LEN / 逐词贡献都不是本地 state**，而是 Task 10 的 `InferenceSettings`
 *  通过父组件传下来的 props。原因见 `InferenceSettings` 顶部注释：阈值是纯前端参数，
 *  它的「当前值」属于设置面板；在这里再存一份就会有两个真相。
 *
 *  本地 state 只管本组件自己的事：输入文本、最近一次结果、请求中标志、错误信息。
 */
export function SingleAnalyzer({ modelId, maxLen, useExplain, threshold }: Props) {
  const [text, setText] = useState("");
  const [probPos, setProbPos] = useState<number | null>(null);
  // tokens 与 contributions 恒同生命周期（同为 null 或同为非 null）：前者是完整句，
  // 后者只覆盖前 CONTRIB_MAX_TOKENS 个词位。Task 12 用 tokens 渲染整句、用 contributions
  // 查表取 delta——少了 tokens，长句会被截断并把它未被分析的尾部当作「贡献为 0」渲染。
  const [tokens, setTokens] = useState<string[] | null>(null);
  const [contributions, setContributions] = useState<Contrib[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function analyze() {
    const input = text.trim();
    if (!input) return;
    setBusy(true);
    setError(null);
    try {
      if (useExplain) {
        // **explain 的响应已是 predict 的超集**（含 label / prob_pos / prob_neg / tokens），
        // 所以这里只调这一个接口。若先调 predict 再补一次 explain，会对同一条文本多跑一遍
        // 前向——而 explain 本身就是 101 次前向，这一遍纯属浪费。两条路径是**一个分支**，
        // 不是两次顺序调用。
        const r = await fetchExplain(input, modelId, maxLen);
        setProbPos(r.prob_pos);
        setTokens(r.tokens);
        setContributions(r.contributions);
      } else {
        const r = await fetchPredict(input, modelId, maxLen);
        setProbPos(r.prob_pos);
        // 关掉逐词贡献后，上一次 explain 留下的高亮与词序列必须一起清掉，
        // 否则会显示与当前结果无关的旧数据
        setTokens(null);
        setContributions(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      // 出错时连旧结果一起清空：一条新鲜的错误信息旁边挂着上一次的概率条，比什么都不显示更糟
      setProbPos(null);
      setTokens(null);
      setContributions(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        placeholder="输入一条中文评论，例如：这家酒店位置很好，但隔音太差……"
        className="input resize-y"
      />

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void analyze()}
          // 空输入与请求进行中都不许再发请求：前者后端没有可分析的文本，后者会产生竞态
          disabled={busy || !text.trim()}
          className="btn btn-primary"
        >
          {/* explain 是 101 次前向、单次预测只要 1 次，等待时长差一个量级；
              文案分开写，让用户知道「慢」是预期而不是卡住了 */}
          {busy ? (useExplain ? "逐词分析中…" : "分析中…") : "分析"}
        </button>

        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            onClick={() => setText(ex.text)}
            className="btn-ghost text-xs"
          >
            {ex.label}示例
          </button>
        ))}
      </div>

      {error && <p className="alert alert-error">{error}</p>}

      {probPos !== null && (
        <>
          <SentimentGauge
            probPos={probPos}
            probNeg={1 - probPos}
            // 标签在这里用 labelOf 现算，不取后端返回的 label 字段：后端那个是在固定
            // 0.5 阈值下算好的，而本界面的阈值是活的——用后端的值会让滑块「拖了没反应」
            label={labelOf(probPos, threshold)}
          />
          {/* 高亮用 `useExplain` 在**渲染期派生**门控，而不是监听 props 变化去重置 state：
              切换复选框不重跑 analyze()，旧高亮会一直挂到下次点「分析」。React 不鼓励
              「用 effect 把 prop 变化写回 state」，派生门控是这个场景的正解。
              「关→开」时高亮会重新出现，但那时展示的 probPos 正是该高亮的同一次分析结果，
              二者一致，故不是陈旧数据。同时要求 tokens 非空——它与 contributions 同生命周期，
              多这一项既让 TS 收窄，也让不变量万一被破坏时「什么都不渲染」而非静默渲染空句。 */}
          {useExplain && tokens && contributions && (
            <TokenHighlight tokens={tokens} contributions={contributions} />
          )}
        </>
      )}
    </div>
  );
}
