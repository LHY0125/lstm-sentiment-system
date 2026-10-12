import { DEFAULT_THRESHOLD, MAX_THRESHOLD, MIN_THRESHOLD } from "../types";

interface Props {
  threshold: number;
  onThresholdChange: (v: number) => void;
  maxLen: number;
  onMaxLenChange: (v: number) => void;
  useExplain: boolean;
  onUseExplainChange: (v: boolean) => void;
}

/** 推理期参数面板。
 *
 *  **阈值是纯前端参数**：后端只返回概率，「积极概率 ≥ 阈值」这个判断在浏览器里做
 *  （规则定义在 `types.labelOf`，全应用仅此一处）。所以拖动滑块**不能**发请求——
 *  这正是阈值放在这一侧的全部理由。将来的读者若想在这里补一句「改完刷新一下」，
 *  那是把免费的即时反馈换成一次往返延迟。
 *
 *  MAX_LEN 与逐词贡献恰好相反：它们改变送进模型的东西（分词结果 / 计算方式），
 *  改完**必须**重新请求。两种旋钮长得像、代价完全不同，故文案里各自写明。
 */
export function InferenceSettings({
  threshold,
  onThresholdChange,
  maxLen,
  onMaxLenChange,
  useExplain,
  onUseExplainChange,
}: Props) {
  return (
    <div className="card flex flex-col gap-4 p-4">
      <div className="flex flex-col gap-2">
        <div className="flex items-baseline justify-between">
          <label htmlFor="inference-threshold" className="field-label">
            判定阈值
          </label>
          <span className="num text-sm text-foreground-subtle">{threshold.toFixed(2)}</span>
        </div>
        <input
          id="inference-threshold"
          type="range"
          min={MIN_THRESHOLD}
          max={MAX_THRESHOLD}
          step={0.01}
          value={threshold}
          onChange={(e) => onThresholdChange(Number(e.target.value))}
          aria-describedby="inference-threshold-hint"
          className="w-full accent-primary"
        />
        <p id="inference-threshold-hint" className="hint">
          积极概率 ≥ 阈值即判为积极。阈值只影响标签，不改变概率，拖动即时生效、不重新请求。
          调高阈值会减少「积极」判定（更保守），调低则相反。
        </p>
        <button
          type="button"
          onClick={() => onThresholdChange(DEFAULT_THRESHOLD)}
          className="btn-ghost self-start text-xs"
        >
          恢复默认 {DEFAULT_THRESHOLD.toFixed(2)}
        </button>
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="inference-max-len" className="field-label">
          最大词数（MAX_LEN）
        </label>
        <input
          id="inference-max-len"
          type="number"
          min={5}
          max={1000}
          value={maxLen}
          onChange={(e) => {
            // 空串、"1e" 这类中间态下 valueAsNumber 是 NaN。此时不往上抛：
            // 抛出去会变成 NaN，或（若改用 Number("")）变成 0，两者都是后端读不懂的
            // 截断长度。合法数字才交给父组件——范围裁剪是父组件的职责，这里只管不抛垃圾。
            const next = e.target.valueAsNumber;
            if (!Number.isNaN(next)) onMaxLenChange(next);
          }}
          aria-describedby="inference-max-len-hint"
          className="input"
        />
        <p id="inference-max-len-hint" className="hint">
          超出部分截断。与阈值不同，这一项改变的是送进模型的分词结果，改完需重新请求才生效。
          本界面限定 5–1000；超出后端可接受范围的值不报错，会被静默换成默认的 150 词。
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <input
            id="inference-use-explain"
            type="checkbox"
            checked={useExplain}
            onChange={(e) => onUseExplainChange(e.target.checked)}
            aria-describedby="inference-use-explain-hint"
            className="accent-primary"
          />
          <label htmlFor="inference-use-explain" className="text-sm text-foreground">
            显示逐词贡献（遮挡法，明显更慢）
          </label>
        </div>
        <p id="inference-use-explain-hint" className="hint">
          勾选后请求 /api/explain：逐个词位遮住重跑一遍前向（最多 100 次，另加 1 次基线），
          而单次预测只要 1 次；概率与标签不变，多出来的是每个词的遮挡敏感度。
        </p>
      </div>
    </div>
  );
}
