/** 与后端 API 契约一一对应的类型定义。改后端契约必须同步改这里。 */

export interface ModelInfo {
  id: string;
  layers: number;
  hidden_dim: number;
  embedding_dim: number;
  test_acc: number;
  val_acc: number;
  params: number;
  train_time_sec: number;
}

export interface PredictResponse {
  label: number;
  prob_pos: number;
  prob_neg: number;
  tokens: string[];
}

export interface Contrib {
  /** 词 */
  token: string;
  /** 删词后的积极概率 − 原始积极概率。
   *  负值 = 该词起积极作用；正值 = 起消极作用。 */
  delta: number;
}

export interface ExplainResponse {
  label: number;
  prob_pos: number;
  prob_neg: number;
  /** 完整分词结果，与 predict 返回的 tokens 相同——渲染骨架用这个。
   *  contributions 只覆盖前 CONTRIB_MAX_TOKENS 个位置，不能当词序列用。 */
  tokens: string[];
  contributions: Contrib[];
}

export interface BatchItem {
  text: string;
  prob_pos: number;
  prob_neg: number;
}

/** 判定阈值：把「积极概率」映射成标签的临界值。
 *
 *  阈值是纯前端参数——后端只返回概率，拖动滑块无需重新请求。 */
export const DEFAULT_THRESHOLD = 0.5;
export const MIN_THRESHOLD = 0.05;
export const MAX_THRESHOLD = 0.95;

export type SentimentLabel = "积极" | "消极";

/** 由概率与阈值导出标签。全应用只有这一处定义标签判定规则。 */
export function labelOf(probPos: number, threshold: number = DEFAULT_THRESHOLD): SentimentLabel {
  return probPos >= threshold ? "积极" : "消极";
}

/** 单条批量输入的最大条数，与后端 MAX_BATCH_SIZE 保持一致。
 *
 *  跨边界常量：后端 `senti/api.py` 独立持有一份 MAX_BATCH_SIZE 并在超限时回 400。
 *  只改一边的后果是前端放行、后端拒绝——要改必须两边一起改。 */
export const MAX_BATCH_SIZE = 1000;

/** 单条文本的字符数上限（**码点**数，不是 UTF-16 码元数），与后端 `senti/config.py` 的
 *  `MAX_TEXT_CHARS` 一致。
 *
 *  跨边界常量，与上面的 MAX_BATCH_SIZE 同一套路：后端 `senti/api.py` 会对**任一**超长文本回
 *  400，前端这份只用于**提前拦**，两边必须一起改。
 *
 *  计数口径也必须对齐：Python 的 `len()` 数码点，JS 的 `String.length` 数 UTF-16 码元——
 *  一个 emoji 分别是 1 和 2。比对时要用码点数（见 `BatchAnalyzer` 的 `charCount`）。 */
export const MAX_TEXT_CHARS = 2000;
