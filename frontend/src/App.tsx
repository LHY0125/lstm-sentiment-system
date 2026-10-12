import { useState } from "react";
import { BatchAnalyzer } from "./components/BatchAnalyzer";
import { InferenceSettings } from "./components/InferenceSettings";
import { ModelSelector } from "./components/ModelSelector";
import { SingleAnalyzer } from "./components/SingleAnalyzer";
import { useModels } from "./hooks/useModels";
import { DEFAULT_THRESHOLD } from "./types";

/** MAX_LEN 输入框的初值。与后端 `senti/config.py` 的 `MAX_LEN` 保持一致——
 *  这里只是「用户没动过时显示什么」，真正生效的是随每次请求发出去的那个数。
 *  写成字面量 150 会在两处各有一个真相，改一处就静默分叉。 */
const DEFAULT_MAX_LEN = 150;

type Tab = "single" | "batch";

/** 主界面：把模型选择、推理设置与两个分析面板组装起来。
 *
 *  **全部状态就下面这五份**（当前页签、用户选过的模型 id、阈值、MAX_LEN、逐词贡献开关），
 *  没有路由、没有状态管理库——组件之间是「父传子」的单向数据流，
 *  唯一需要跨两处共享的是 modelId，还没到值得引入别的东西的规模。
 *
 *  **模型选择与设置面板都渲染在侧栏、且只渲染一次**：Task 10 的组件用的是字面量 id
 *  （`model-select` / `inference-threshold` …），同一份文档里出现两份实例就会有重复 id，
 *  于是 `<label htmlFor>` 全部指向第一个实例，第二个面板的滑块点标签不再生效。
 *  单条分析与批量分析是互斥的两个页签，同一时刻只有一个在场，故这里天然只有一份。 */
export default function App() {
  const { models, loading, error } = useModels();
  const [pickedId, setPickedId] = useState("");
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [maxLen, setMaxLen] = useState(DEFAULT_MAX_LEN);
  const [useExplain, setUseExplain] = useState(true);
  const [tab, setTab] = useState<Tab>("single");

  // 「实际生效的模型」在**渲染期派生**，而不是用 effect 把列表变化写回 state。
  // 派生一次解决两件事：列表到达时自动选中（useModels 已按 val_acc 降序返回，
  // 故 models[0] 就是验证集最优的那个，而非测试集最大值），
  // 以及刷新后旧 id 失效时的回退。
  // 回退这一条是承重的：<select> 完全受控于 value，value 指向一个不存在的 option 时
  // 浏览器把选择框画成**空白**，用户会以为「没有模型可选」而不是「选择被重置了」。
  // 放在渲染期而不是 effect 里，是因为 effect 要等下一次提交才生效——
  // 中间那一帧 modelId 仍是空的，会把空 id 传给下面两个面板。
  const modelId = models.some((m) => m.id === pickedId) ? pickedId : (models[0]?.id ?? "");

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-6 py-4">
          <h1 className="text-lg font-semibold text-slate-900">网络评论情感分类系统</h1>
          <p className="text-xs text-slate-500">基于 LSTM 的中文评论情感二分类 · 遮挡法逐词贡献</p>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-6 py-6">
        {loading && <p className="text-sm text-slate-500">正在加载模型列表…</p>}
        {error && (
          <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">
            无法连接后端：{error}
            <br />
            请先启动后端：<code>cd backend &amp;&amp; ..\.venv\Scripts\python.exe run_api.py</code>
          </p>
        )}

        {!loading && !error && models.length === 0 && (
          <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">
            后端尚无已训练模型。请先运行 <code>python -m senti.train</code> 完成训练。
          </p>
        )}

        {/* 两个面板都要求 modelId 非空才渲染：派生出来的 id 在列表非空时必然非空，
            但仍显式判一次——把「下拉框是空白」和「面板拿到空 id」隔开，
            免得将来改坏了选择逻辑时，出错现场从「点分析报 400」挪到别处。 */}
        {models.length > 0 && modelId !== "" && (
          <div className="grid gap-6 md:grid-cols-[1fr_320px]">
            <div className="flex flex-col gap-4">
              <div className="flex gap-1 border-b border-slate-200">
                {(["single", "batch"] as Tab[]).map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setTab(t)}
                    className={`-mb-px border-b-2 px-4 py-2 text-sm ${
                      tab === t
                        ? "border-slate-800 font-medium text-slate-900"
                        : "border-transparent text-slate-500 hover:text-slate-800"
                    }`}
                  >
                    {t === "single" ? "单条分析" : "批量分析"}
                  </button>
                ))}
              </div>

              {/* 两个面板是互斥渲染而不是 display:none 隐藏：隐藏的那一个仍持有上一次的
                  结果与在飞请求，切回来时会拿旧结果唬人。卸载顺带把它的局部状态一起丢掉，
                  这是这里需要的行为。 */}
              {tab === "single" ? (
                <div className="flex flex-col gap-3">
                  {/* 「示例」按钮上的三个标签是**句子的大意**，不是标准答案。
                      这句话必须留在界面里：模型只有 6 组权重，同一句在不同模型下会给出
                      不同的概率，个别短句甚至可能翻面。
                      不写这句，演示时第一个被问到的就是「你这系统是不是坏了」——
                      而它没坏，是单句判定的正常波动，不该被当成缺陷去掩盖。
                      刻意不点名具体模型**也不写具体数值**：那会把一句关于方法论的说明
                      写成版本相关的八卦，换一次 checkpoint 就过期（上一版这里抄的
                      0.4234 就随着合并语料重训而失效了）。 */}
                  <p className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500">
                    「正面 / 负面 /
                    中性示例」按钮只是演示用的句子，不是正确答案。单句判定随所选模型的
                    权重变化，同一句在不同模型下可能给出相反的标签——那是模型之间的差异，不是界面出错。
                  </p>
                  <SingleAnalyzer
                    modelId={modelId}
                    maxLen={maxLen}
                    useExplain={useExplain}
                    threshold={threshold}
                  />
                </div>
              ) : (
                <BatchAnalyzer modelId={modelId} maxLen={maxLen} threshold={threshold} />
              )}
            </div>

            <aside className="flex flex-col gap-4">
              <div className="rounded-lg border border-slate-200 bg-white p-4">
                <ModelSelector models={models} selectedId={modelId} onSelect={setPickedId} />
              </div>
              <div className="rounded-lg bg-white">
                <InferenceSettings
                  threshold={threshold}
                  onThresholdChange={setThreshold}
                  maxLen={maxLen}
                  onMaxLenChange={setMaxLen}
                  useExplain={useExplain}
                  onUseExplainChange={setUseExplain}
                />
              </div>
            </aside>
          </div>
        )}
      </main>
    </div>
  );
}
