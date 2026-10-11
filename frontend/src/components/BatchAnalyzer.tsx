import Papa from "papaparse";
import { useState } from "react";
import { fetchBatch } from "../api/client";
import { type BatchItem, MAX_BATCH_SIZE } from "../types";
import { ResultTable } from "./ResultTable";

interface Props {
  modelId: string;
  maxLen: number;
  threshold: number;
}

/** 单条文本的字符数上限，与后端 `senti/config.py` 的 `MAX_TEXT_CHARS` 一致。
 *
 *  跨边界常量：后端 `senti/api.py` 会对**任一**超长文本回 400（`types.ts` 的
 *  `MAX_BATCH_SIZE` 同理）。这里这份只用于**提前拦**，两边必须一起改——只改一边的后果是
 *  前端放行、后端拒绝。
 *
 *  放在本文件而不是 `types.ts`：目前只有批量面板需要它（单条输入框本来就粘不下 2000 字）。 */
const MAX_TEXT_CHARS = 2000;

/** 认定为「文本列」的表头名。比较前统一转小写，故 ASCII 名只列小写形式。 */
const TEXT_COLUMN_NAMES = ["text", "文本", "评论", "content"];

/** U+FFFD（替换字符）。UTF-8 解码遇到非法字节时产出它，是「源文件不是 UTF-8」的唯一可靠信号。
 *  本产品是中文评论分析，而简中 Windows 上的 Excel 默认把 CSV 存成 GBK——踩中的概率不低。
 *  用码位构造而不是写转义字面量：源码里不留任何不可见字符。 */
const REPLACEMENT_CHAR = String.fromCodePoint(0xfffd);

/** 从 CSV 抽出文本列的结果。
 *
 *  **失败时给一句可直接展示的原因，而不是空数组**：空数组会被上层当成「文件里没有文本」，
 *  于是要么静默发一次空分析、要么让用户对着一张空表猜——两种都比直接说明差。 */
type CsvParse =
  | {
      ok: true;
      texts: string[];
      /** 「用的是哪一列」的可展示说明。不报出来，用户无法判断进对没进对。 */
      column: string;
      /** 解码后出现了替换字符，源文件很可能不是 UTF-8。 */
      suspectEncoding: boolean;
    }
  | { ok: false; reason: string };

/** 从 CSV 文本里抽出待分析文本列：认 `text` / `文本` / `评论` / `content` 表头，
 *  没有识别到表头时取第一列。
 *
 *  **用 `header: false` 解析完再自己认表头**，而不是 `header: true`：papaparse 的表头模式会
 *  把文件第一行**吃掉**当字段名——无表头的文件因此会丢掉第一条评论，而且字段名恰好就是那条
 *  评论本身，于是「解析成功」却少了一条数据。自己看第一行，有表头与无表头两种情况都对。
 *
 *  导出这个函数只为一件事：本模块没有测试运行器，验收靠外部脚本喂固定样本
 *  （见 task-13-report.md）。纯函数比走 UI 好构造，也免得为测试新增依赖。
 */
export function extractTexts(csvText: string): CsvParse {
  // 先去掉 UTF-8 BOM：Excel「另存为 CSV UTF-8」会在文件开头写入 U+FEFF，
  // 不清掉的话表头首格变成 "\uFEFFtext"，认不出列名而被当成无表头文件。
  const clean = csvText.replace(/^\uFEFF/, "");
  const suspectEncoding = clean.includes(REPLACEMENT_CHAR);

  const parsed = Papa.parse<string[]>(clean, { header: false, skipEmptyLines: true });
  // 逐格 trim，再丢掉「整行都空白」的行：导出工具常在末尾留一个空行或只剩逗号的行。
  // `?? ""` 不是多余的：CSV 的行是**参差**的，短行在该下标处拿到的是 undefined，
  // 而类型声明只说得出 string（papaparse 的类型不编码行的长度）。
  const rows = parsed.data
    .map((r) => r.map((c) => (c ?? "").trim()))
    .filter((r) => r.some((c) => c !== ""));

  if (rows.length === 0) {
    return { ok: false, reason: "文件里没有任何可读内容（可能只有空行）" };
  }

  const header = rows[0];
  const headerCol = header.findIndex((c) => TEXT_COLUMN_NAMES.includes(c.toLowerCase()));
  const hasHeader = headerCol >= 0;
  const col = hasHeader ? headerCol : 0;
  const body = hasHeader ? rows.slice(1) : rows;

  // 空单元格直接丢弃：一条空白评论没有可分析的东西，留着只会让「共 N 条」虚高
  const texts = body.map((r) => r[col] ?? "").filter((c) => c !== "");

  if (texts.length === 0) {
    return {
      ok: false,
      reason: hasHeader ? `表头「${header[col]}」那一列整列为空` : "第一列整列为空",
    };
  }

  return {
    ok: true,
    texts,
    column: hasHeader
      ? `第 ${col + 1} 列「${header[col]}」`
      : "第 1 列（没认出表头，按无表头文件取第一列）",
    suspectEncoding,
  };
}

/** 请求发出**之前**在本地说清越界在哪，以及越了多少；两条上限都没踩到就返回 null。
 *
 *  后端两条上限都会回 400，而往返一次只为换一句报错，既花用户的时间也占服务端资源。
 *  不把按钮置灰来「防」超限：置灰之后用户面对一个点不动的按钮，反而不知道哪里越界了。
 *
 *  放在模块级（而不是组件里）是照 TokenHighlight 的做法——不依赖任何 state 的纯函数就该是
 *  模块级函数。导出同样只为外部验收脚本喂固定样本（见 task-13-report.md）。 */
export function limitViolation(list: string[]): string | null {
  // 先判条数再判长度：一次 5000 条的超大粘贴，条数才是真正要用户先解决的那件事
  if (list.length > MAX_BATCH_SIZE) {
    return `一次最多分析 ${MAX_BATCH_SIZE} 条，当前 ${list.length} 条（超出 ${
      list.length - MAX_BATCH_SIZE
    } 条）。请删减后重试。`;
  }
  const over = list.filter((t) => t.length > MAX_TEXT_CHARS);
  if (over.length > 0) {
    const at = list.findIndex((t) => t.length > MAX_TEXT_CHARS) + 1;
    return `第 ${at} 条有 ${over[0].length} 字，超过单条上限 ${MAX_TEXT_CHARS} 字${
      over.length > 1 ? `（共 ${over.length} 条超长）` : ""
    }。请拆分或缩短后重试。`;
  }
  return null;
}

/** 批量分析面板：每行一条文本，或上传 CSV。
 *
 *  **两种输入不是两条代码路径**：上传 CSV 的结果被写回 textarea，再由渲染期派生的
 *  `texts` 走同一个 `analyze()`。于是「CSV 进来的内容」与「手打的内容」完全同权——
 *  用户可以先上传再改两行，也可以把 CSV 内容直接粘进来，分析路径只有一条。
 */
export function BatchAnalyzer({ modelId, maxLen, threshold }: Props) {
  const [raw, setRaw] = useState("");
  const [items, setItems] = useState<BatchItem[] | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 每次渲染都从 textarea 现推，不另存一份「待分析文本」。存了就会有两个真相
  // （手打的与 CSV 来的），早晚对不上——这正是 SingleAnalyzer 里不复制 props 的同一条理由。
  const texts = raw
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);

  /** 唯一的失败出口。**错误与结果不能并存**：一条新鲜的报错旁边挂着上一次的表格，
   *  用户会以为那就是这次的结果。同理，上一次的 CSV 列说明也一起撤掉——它描述的是
   *  另一个文件，留着只会误导。 */
  function fail(message: string) {
    setError(message);
    setItems(null);
    setNotice(null);
  }

  async function analyze() {
    if (!texts.length) return;
    const violation = limitViolation(texts);
    if (violation) {
      fail(violation);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setItems(await fetchBatch(texts, modelId, maxLen));
    } catch (e) {
      // 错误信息直接来自后端（client.ts 取的是响应体里的 error 字段），原样转述不加工
      fail(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onFile(file: File) {
    let content: string;
    try {
      content = await file.text();
    } catch (e) {
      fail(`读取「${file.name}」失败：${e instanceof Error ? e.message : String(e)}`);
      return;
    }

    const parsed = extractTexts(content);
    if (!parsed.ok) {
      fail(
        `未能从「${file.name}」读出文本列：${parsed.reason}。` +
          "需要含 text / 文本 / 评论 / content 表头的列，或无表头的单列文本。",
      );
      return;
    }

    // 一条评论渲染成 textarea 的一行。**已知折衷**：CSV 单元格内部若有换行（带引号的多行字段），
    // 会被这里的 join 与上面的 split 拆成两条。中文评论极少跨行，而为了它再引入一份
    // 「待分析文本」state 就会让输入与结果各有一个真相——代价大于收益，故如实说明而非静默处理。
    setRaw(parsed.texts.join("\n"));
    setError(null);
    // 输入被整体换掉了：上一轮的结果已经不属于屏幕上的这段输入，留着只会让表格与
    // 紧挨着的「已读取 N 条」互相矛盾
    setItems(null);
    setNotice(
      `已从「${file.name}」读取 ${parsed.texts.length} 条，取${parsed.column}。` +
        (parsed.suspectEncoding
          ? " 文件编码似乎不是 UTF-8（中文可能已乱码），建议在 Excel 里另存为「CSV UTF-8」后重传。"
          : ""),
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <label htmlFor="batch-input" className="text-sm font-medium text-slate-700">
          每行一条评论
        </label>
        <textarea
          id="batch-input"
          value={raw}
          onChange={(e) => {
            setRaw(e.target.value);
            // 上一次 CSV 的列说明里带着条数，改一个字它就可能不再成立
            setNotice(null);
          }}
          rows={8}
          placeholder={"这家酒店很好，服务周到\n隔音差，晚上被吵醒\n性价比一般，凑合"}
          className="w-full resize-y rounded-lg border border-slate-300 p-3 text-sm
                     focus:border-slate-500 focus:outline-none"
        />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void analyze()}
          // 空输入与请求进行中都不许再发：前者没有可分析的文本，后者会产生竞态
          disabled={busy || !texts.length}
          className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-white
                     hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "分析中…" : `分析（${texts.length} 条）`}
        </button>

        <label className="cursor-pointer text-sm text-slate-500 underline hover:text-slate-800">
          或上传 CSV
          <input
            type="file"
            accept=".csv,.txt"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              // 立刻清空 input 的值：浏览器只在值真的变化时才派发 change，
              // 不清的话「改完 CSV 再传同一个文件名」会静默无反应。
              // 先取出 File 再清空是安全的——File 是独立对象，不随 input 的值消失。
              e.target.value = "";
              if (f) void onFile(f);
            }}
          />
        </label>
      </div>

      {error && <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
      {notice && (
        <p className="rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-600">{notice}</p>
      )}

      {/* 表格把活的 threshold 透传下去：拖动阈值滑块，表里的标签与导出内容一起跟着变，不发请求 */}
      {items && <ResultTable items={items} threshold={threshold} />}
    </div>
  );
}
