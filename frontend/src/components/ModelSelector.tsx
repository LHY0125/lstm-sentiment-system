import type { ModelInfo } from "../types";

interface Props {
  models: ModelInfo[];
  selectedId: string;
  onSelect: (id: string) => void;
}

/** 模型下拉。把 meta.json 里的指标直接摊在选项上——选模型时看得见代价。
 *
 *  **排序与默认选中来自 val_acc，这里展示的却是 test_acc**，两者不一致是有意的：
 *  `useModels` 按验证准确率降序返回，本组件把测试准确率显示给人看。本项目的规则是
 *  「用验证集做选择、用测试集做报告」——若按测试准确率排序，默认选中的就是 6 个
 *  test_acc 里的极大值，那个数字已被选择过程本身挑过一遍，是乐观偏置。
 *  两个数字都摊开，人才能自己判断差距（6 个模型的 test_acc 只差约 1.3 个百分点，
 *  不把数字摆出来就无从比较）。
 *
 *  `train_time_sec` 是**本机墙钟测量值**，不是模型的固有属性：换机器、换批次大小
 *  都会变。故标注为「本机训练耗时」，避免被当成可复现的模型指标引用。
 */
export function ModelSelector({ models, selectedId, onSelect }: Props) {
  const selected = models.find((m) => m.id === selectedId);

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="model-select" className="text-sm font-medium text-slate-700">
        模型
      </label>
      <select
        id="model-select"
        value={selectedId}
        onChange={(e) => onSelect(e.target.value)}
        className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm
                   focus:border-slate-500 focus:outline-none"
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.layers} 层 · 隐藏 {m.hidden_dim} · 准确率 {(m.test_acc * 100).toFixed(2)}%
          </option>
        ))}
      </select>

      {selected && (
        <dl className="grid grid-cols-3 gap-2 rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
          <div>
            <dt className="text-slate-500">测试准确率</dt>
            <dd className="font-mono text-slate-900">{(selected.test_acc * 100).toFixed(2)}%</dd>
          </div>
          <div>
            <dt className="text-slate-500">参数量</dt>
            {/* params 是 2.5M–3.4M 的裸整数，分组显示；不分组就是一长串数字，读不出量级 */}
            <dd className="font-mono text-slate-900">{selected.params.toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-slate-500">本机训练耗时</dt>
            <dd className="font-mono text-slate-900">{selected.train_time_sec.toFixed(0)} s</dd>
          </div>
        </dl>
      )}
    </div>
  );
}
