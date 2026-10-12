import type { ModelInfo } from "../types";

interface Props {
  models: ModelInfo[];
  selectedId: string;
  onSelect: (id: string) => void;
}

/** 模型下拉。把 meta.json 里的指标摊开——选模型时看得见代价。
 *
 *  **排序与默认选中来自 val_acc，展示的却是「平衡准确率」**，不一致是有意的：
 *  `useModels` 按验证准确率降序返回（规则是「用验证集做选择、用测试集做报告」），
 *  而展示用平衡准确率而**不是**普通的测试准确率——合并语料下测试集 74.3% 是积极，
 *  一个「无脑判积极」的模型白拿 74.3% 准确率，那个数字单独看会把人骗过去。
 *  平衡准确率（两类召回率的均值）不受类别比例影响，是偏斜数据上可比的量。
 *
 *  详情里把**三个数并排**给出：准确率、平衡准确率、多数类基线。
 *  三个一起看才能回答真正该问的问题——「这个模型比什么都不做强多少」。
 *
 *  展示哪一个都不能驱动排序：`val_acc` 决定顺序，因为它是唯一没被测试集
 *  挑过一遍的量（按 test_acc 排序会让默认项变成六个测试分数的极大值，
 *  那个值带选择性偏差）。
 *
 *  `train_time_sec` 是**本机墙钟测量值**，不是模型的固有属性：换机器、换批次大小
 *  都会变。故标注为「本机训练耗时」，避免被当成可复现的模型指标引用。
 */
export function ModelSelector({ models, selectedId, onSelect }: Props) {
  const selected = models.find((m) => m.id === selectedId);

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="model-select" className="field-label">
        模型
      </label>
      <select
        id="model-select"
        value={selectedId}
        onChange={(e) => onSelect(e.target.value)}
        className="input"
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.layers} 层 · 隐藏 {m.hidden_dim} · 平衡准确率{" "}
            {(m.test_balanced_acc * 100).toFixed(2)}%
          </option>
        ))}
      </select>

      {selected && (
        <dl className="grid grid-cols-3 gap-2 rounded-lg bg-surface-muted p-3 text-xs text-foreground-muted">
          <div>
            <dt className="text-foreground-subtle">测试准确率</dt>
            <dd className="num text-foreground">{(selected.test_acc * 100).toFixed(2)}%</dd>
          </div>
          <div>
            <dt className="text-foreground-subtle">平衡准确率</dt>
            <dd className="num text-foreground">
              {(selected.test_balanced_acc * 100).toFixed(2)}%
            </dd>
          </div>
          <div>
            <dt className="text-foreground-subtle">多数类基线</dt>
            {/* 测试集里多数类的占比：准确率的「无脑参照线」。没有它就不知道
                92% 到底是好还是仅仅「数据集偏斜」 */}
            <dd className="num text-foreground">
              {(selected.majority_baseline * 100).toFixed(2)}%
            </dd>
          </div>
          <div>
            <dt className="text-foreground-subtle">参数量</dt>
            {/* params 是 2.6M–3.5M 的裸整数，分组显示；不分组就是一长串数字，读不出量级 */}
            <dd className="num text-foreground">{selected.params.toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-foreground-subtle">本机训练耗时</dt>
            <dd className="num text-foreground">{selected.train_time_sec.toFixed(0)} s</dd>
          </div>
        </dl>
      )}
    </div>
  );
}
