# 基于 LSTM 的网络评论情感分类系统 — 设计文档

> ⚠️ 环境注（2026-11 追加）：文中 `Python 3.13.13` 路径已失效——2026-11-08 全量迁移至 **Python 3.14.10**。本文保留为历史记录。

- **课程**：神经网络与深度学习（河南理工大学计算机学院）
- **选题**：选题(3) 基于 LSTM 的网络评论情感分类系统设计与实现
- **文档日期**：2026-10-11
- **文档状态**：已评审 → 已实现（本文档是设计阶段的产物；实现后的事实——语料、版本、依赖——以 README 为准）

---

## 1. 项目概述

### 1.1 选题依据

任务书要求基于长短期记忆网络（LSTM）实现网络评论的二分类情感识别，需完成
文本数据预处理、分词、词向量表示、LSTM 建模，并设计文本输入界面。

选择本题目的依据：

1. **与已完成的课程实验四同构**。`作业/神经网络与深度学习/实验四_情感分析/`
   已实现 `Embedding + LSTM + Linear` 的英文 IMDb 情感二分类，其数据预处理、
   模型结构、训练循环、评估与绘图逻辑可直接迁移。
2. **实验四的原始设计本就面向中文**。实验四指导书原方案为"字符级特征 + 固定长度填充"，
   项目 README 明确记录该方案"针对中文数据集（中文单字即词，字符级合理）"，
   当时因未获得中文评论语料才改用英文 IMDb。本课题即把该设计"归位"。
3. **精度门槛可达**。ChnSentiCorp 为二分类公开语料，词级 LSTM 的典型测试准确率
   在 88%~92% 区间，高于任务书要求的 85%。
4. **已知坑点已有解决方案**。实验四已定位并修复 `output[-1]` 取到 `<pad>` 填充导致
   模型不收敛的问题（改用 `lengths + gather` 取最后一个真实词，测试准确率
   68.62% → 80.10%），该修正直接继承。

### 1.2 目标与验收标准

| 编号 | 验收项 | 判据 |
|------|--------|------|
| A1 | 数据预处理完整 | 完成语料解析、jieba 分词、词表构建、索引化与填充，产出 `lengths` |
| A2 | 模型达标 | 测试集二分类准确率 **≥85%** |
| A3 | 人机交互 | 前后端分离的 Web 界面，支持评论输入、情感预测、结果展示 |
| A4 | 可解释性 | 以遮挡法给出逐词贡献，可排序、可高亮 |
| A5 | 批量分析 | 支持多行文本与 CSV 上传的批量预测，结果可导出 |
| A6 | 参数可调 | 可切换模型 checkpoint；阈值、`MAX_LEN` 可调 |
| A7 | 模块化 | 后端按数据/模型/训练/推理/接口五层划分，模块边界清晰 |
| A8 | 可复现 | 固定随机种子，README 记录实测结果 |

### 1.3 明确不做的事

- **不写课程设计报告**（本轮范围，报告后置）
- **不做完整训练/验证/测试三分**：沿用数据集自带的 train/test 划分，
  验证集通过从训练集切分获得
- **不引入预训练语言模型**（BERT 等）。任务书指定 LSTM，且引入预训练模型
  会掩盖对 LSTM 本身的理解
- **不做实时训练**（见第 8 节方案选择）
- **不做 `.ipynb` 双份**。兄弟项目曾因 `.py` / `.ipynb` 两版分叉而需要双向同步，
  本项目只维护 `.py`

---

## 2. 技术选型

| 层 | 选型 | 理由 |
|----|------|------|
| 深度学习框架 | PyTorch 2.12.0+cu132 | 与课程实验一致；本机已有本地 wheel |
| 中文分词 | jieba 0.42.1 | 中文分词事实标准；全局 `uv_env` 已有，本项目 `.venv` 单独安装 |
| 后端框架 | Flask + flask-cors | 轻量；仅需 REST 接口，不需 FastAPI 的异步与自动文档 |
| 数据读写 | 标准库 csv + pyarrow | 语料为 CSV/TSV/parquet；csv 流式读、pyarrow 读原生划分，不引重依赖 |
| 绘图 | matplotlib | 沿课程实验惯例（含 `common/plot_font.py` 中文字体配置） |
| 前端 | React 19 + Vite + TypeScript + Tailwind CSS v4 | 与用户既有项目技术栈一致；`pnpm create vite --template react-ts` 当前产出 React 19 |
| 前端工具链 | pnpm + Biome | 同上 |
| 测试 | pytest | 仅覆盖纯函数与接口契约（见第 10 节） |

### 2.1 分词粒度决策

**采用词级（jieba）**，不做字级对照。

理由：词级为中文情感分析的主流做法；字级分支虽成本低，但用户已明确选择"只做词级"，
遵循 YAGNI 不为未确定的对比需求增加分支。

---

## 3. 系统架构

### 3.1 核心原则：训练与推理的物理隔离

训练期只**产出产物**，推理期只**消费产物**，两侧不互相 import。

```
训练期（离线，分钟级）                     推理期（在线，毫秒级）
────────────────────                      ─────────────────────
数据集/ChnSentiCorp/                        前端 React (Vite dev server)
        │                                          │ JSON over /api
   [数据层] data.py                                ▼
   jieba 分词 + 建词表                    [接口层] api.py (Flask)
        ▼                                          ▼
   [模型层] model.py                       [推理层] infer.py
   Embedding + LSTM + Linear               载入 *.pt + vocab.json
        ▼                                          ▼
   [训练层] train.py                        概率 → 阈值 → 标签
   网格跑 6 组                                     │
        ▼                                          ▼
   产物 ──────────────────────────────────→ 返回 JSON
   *.pt + vocab.json + meta.json
```

这条边界的意义：**"模型可切换"只是换一个 `.pt` 文件，不触发任何训练**。

### 3.2 目录结构

```
基于 LSTM 的网络评论情感分类系统设计与实现/
├── docs/
│   ├── （课程材料：任务书与报告模板）              # 仅本地保留，不入库
│   └── superpowers/specs/                    # 本设计文档
├── 数据集/
│   └── ChnSentiCorp/                         # 语料（不入库）
├── backend/
│   ├── senti/
│   │   ├── __init__.py
│   │   ├── config.py                         # 路径与超参数常量
│   │   ├── data.py                           # 数据层
│   │   ├── model.py                          # 模型层
│   │   ├── train.py                          # 训练层（CLI 入口）
│   │   ├── infer.py                          # 推理层
│   │   └── api.py                            # 接口层（Flask）
│   ├── common/plot_font.py                   # 中文字体配置（复制自兄弟项目）
│   ├── tests/                                # pytest
│   ├── run_api.py                            # 后端启动入口
│   └── artifacts/                            # 训练产物（不入库）
│       ├── lstm_l1_h64.pt, lstm_l1_h128.pt, lstm_l1_h256.pt
│       ├── lstm_l2_h64.pt, lstm_l2_h128.pt, lstm_l2_h256.pt
│       ├── vocab.json
│       └── meta.json
├── frontend/
│   ├── src/                                  # 见第 6 节
│   ├── vite.config.ts                        # 含 /api 代理
│   ├── biome.json
│   └── package.json
├── .venv/                                    # 项目虚拟环境（不入库）
├── .gitignore
├── README.md                                 # 权威文档（含实测结果表格）
└── CLAUDE.md                                 # 给 AI 的项目说明
```

六层模块划分刻意对应任务书"模块化程序设计"的要求，报告第 3 章的"各模块功能说明"
可直接照搬该结构。

---

## 4. 后端模块设计

### 4.1 `config.py` — 常量中心

集中全部路径与超参数，训练循环内不得出现字面量。

```python
# 路径
PROJECT_ROOT, DATA_DIR, ARTIFACT_DIR, OUTPUT_DIR

# 数据
MAX_LEN = 150            # 词级截断长度
MAX_VOCAB = 20000
MIN_FREQ = 2
PAD_IDX, UNK_IDX = 0, 1  # 0 保留给 <pad>，1 给 <unk>，实词从 2 开始

# 模型
EMBEDDING_DIM = 128      # 固定
DROPOUT = 0.3

# 训练
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
VAL_RATIO = 0.1          # 从训练集切分验证集，用于选最优轮次

# 推理
CONTRIB_MAX_TOKENS = 100 # 逐词贡献分析的参与词数上限（超出部分不分析）

# 网格：6 组 checkpoint
GRID = [(layers, hidden) for layers in (1, 2) for hidden in (64, 128, 256)]
```

### 4.2 `data.py` — 数据层

| 函数 | 职责 |
|------|------|
| `download()` | 从 hf-mirror 下载语料到 `数据集/ChnSentiCorp/`，已存在则跳过 |
| `load_split()` | 解析语料 → `(texts, labels)`，标签 1=积极 / 0=消极 |
| `tokenize(text)` | jieba 分词 + 去除空白与标点 |
| `build_vocab(texts)` | **仅用训练集**统计词频，返回 `word2idx` |
| `encode(texts, word2idx)` | → `(ids: int64[max_len], lengths: int64)` |
| `save_vocab()/load_vocab()` | `vocab.json` 读写 |
| `build_dataloaders()` | 封装 `TensorDataset(ids, lengths, labels)` |

`lengths` 记录每条评论的**真实词数（不含填充）**，是模型取最后真实词的必要输入，
不可省略。

**词表只从训练集统计**，避免测试集信息泄漏。

### 4.3 `model.py` — 模型层

```python
class SentimentLSTM(nn.Module):
    """Embedding + LSTM + Linear。

    三个组件：
        Embedding  词 → 连续向量（padding_idx=0，填充位不参与梯度）
        LSTM       序列 → 语义表示
        Linear     语义表示 → 2 维（积极/消极 logits）
    """
    def forward(self, x, lengths):
        # x: (seq_len, batch)；lengths: (batch,) 每条真实词数
        # 按真实长度 gather 取最后一个真实词的输出，
        # 而非固定取 output[-1]（后者在大量短评论上会读到 <pad>）
```

**必须继承实验四的修正**：`output[-1]` 在 IMDb 上取到填充导致前 5 轮几乎不学习；
本模型一律用 `lengths` 通过 `gather` 取真实末词。

### 4.4 `train.py` — 训练层

| 函数 | 职责 |
|------|------|
| `train_one(layers, hidden)` | 训练单个配置，返回**指标字典**（`{model_id, layers, hidden_dim, embedding_dim, val_acc, test_acc, params, train_time_sec, vocab_size}`），供 `meta.json` 直接汇总。**每轮在验证集上评估，取验证准确率最高的那轮权重** |
| `evaluate(loader)` | 指定数据集上的准确率（`@torch.no_grad()`） |
| `visualize()` | 损失曲线 + 准确率曲线 → `输出/` |
| `run_grid()` | 遍历 `GRID` 训练 6 组，逐一保存 `*.pt` |
| `write_meta()` | 汇总各组指标写 `meta.json` |
| `main()` | CLI 入口；支持 `--config` 只训单组、`--overfit-test` 只跑过拟合验证 |

`meta.json` 结构（`GET /api/models` 直接返回其内容）：

```json
{
  "dataset": "ChnSentiCorp",
  "train_size": 0, "test_size": 0,
  "vocab_size": 0,
  "max_len": 150,
  "models": [
    {"id": "lstm_l2_h128", "layers": 2, "hidden_dim": 128, "embedding_dim": 128,
     "test_acc": 0.0, "params": 0, "train_time_sec": 0.0}
  ]
}
```

> 上表中的 `0` 为结构占位，实际数值在训练完成后写入。README 中记录最终实测值。

### 4.5 `infer.py` — 推理层

```python
class Predictor:
    """持有「模型 + 词表」配对，提供推理能力。

    一次只载入一个 checkpoint，切换模型时重新载入（含词表一致性断言）。
    """
    def __init__(self, model_id): ...
    def predict(self, text) -> dict: ...          # 单条
    def batch_predict(self, texts) -> list[dict]: # 批量
    def explain(self, text) -> dict: ...          # 逐词贡献
```

**逐词贡献采用遮挡法（occlusion）**：对每个词位，将其替换为 `<unk>` 后重新前向，
记录积极概率的变化量 Δ。

- Δ > 0：删掉该词后积极概率**上升** → 该词起**消极**作用
- Δ < 0：删掉该词后积极概率**下降** → 该词起**积极**作用

返回 `contributions`，**按原句词序排列**（与 `predict` 返回的 `tokens` 同序）。复杂度 O(N) 次前向，评论词数在 150 以内，
毫秒级可接受。

> **为什么不按 |Δ| 排序返回**：前端需要同时呈现「原句高亮」与「Top-N 排序表」。
> 排序是客户端零成本可做的，而**从乱序还原原句序是不可能的**——jieba 在服务端，
> 客户端无法重新分词。因此服务端必须返回保序数据，排序交给消费方。
> `TokenHighlight` 组件自行对 `contributions` 排序得到 Top-N 表。

**不使用 attention 权重作为可解释性依据**——注意力反映信息路由而非重要性，
二者不等价。遮挡法问的是反事实问题"缺了这个词结果会变吗"，对任意模型成立。

### 4.6 `api.py` — 接口层（薄层）

只做参数校验、调用 `Predictor`、序列化响应。不含业务逻辑。

---

## 5. 接口契约

| 方法 | 路径 | 请求 | 响应 |
|------|------|------|------|
| GET | `/api/health` | — | `{status: "ok"}` |
| GET | `/api/models` | — | `[{id, layers, hidden_dim, embedding_dim, test_acc, params, train_time_sec}]` |
| POST | `/api/predict` | `{text, model_id, max_len}` | `{label, prob_pos, prob_neg, tokens}` |
| POST | `/api/explain` | `{text, model_id, max_len}` | `{label, prob_pos, prob_neg, tokens, contributions: [{token, delta}]}` |
| POST | `/api/batch` | `{texts: [], model_id, max_len}` | `[{text, prob_pos, prob_neg}]` |

约定：

- 所有响应概率为浮点数，`prob_pos + prob_neg == 1`（误差 < 1e-6）
- **`/api/explain` 与 `/api/predict` 对同一 `(text, model_id, max_len)` 必须给出相同的
  `label` / `prob_pos` / `prob_neg` / `tokens`。** `explain` 只在 `predict` 的结果之上
  **追加**逐词贡献，绝不改变被预测的序列——`CONTRIB_MAX_TOKENS` 只限制「分析多少个位置」，
  不截断被预测的输入。两个端点对同一输入给出不同答案是静默矛盾，不可接受。
- `contributions` **按原句词序**返回，且只覆盖前 `CONTRIB_MAX_TOKENS` 个位置；
  `tokens` 则是完整的（与 `predict` 相同）。前端用 `tokens` 做渲染骨架、
  `contributions` 作 delta 查表。
- **未覆盖的位置必须在视觉上独立区分，不得渲染为中性色。** 尾部词是「**未分析**」
  （`contributions` 里没有对应条目），不是「测得贡献为零」。用 `token in delta_map` 判断，
  给未分析段一个单独的视觉状态并提示「仅分析前 N 词」。
  把两者混同等于**销毁信息**——用户无法分辨「测过且确实无影响」与「根本没测」。
- **后端不接收阈值参数**。阈值是"概率 → 标签"的纯前端映射（见第 7 节）
- 错误响应：`{error: "<中文说明>"}` + 恰当的 HTTP 状态码
  （400 参数错误 / 404 模型不存在 / 500 内部错误）
- 单条文本长度上限 2000 字符，超限返回 400

---

## 6. 前端设计

### 6.1 组件结构

```
frontend/src/
├── main.tsx
├── App.tsx                    # 布局 + Tab（单条分析 / 批量分析）
├── types.ts                   # 与后端响应对齐的类型定义
├── api/client.ts              # 类型化 fetch 封装（唯一网络出口）
├── hooks/useModels.ts         # 拉取模型列表与元数据
└── components/
    ├── ModelSelector.tsx      # 模型下拉：显示准确率 / 参数量 / 层数×维度
    ├── InferenceSettings.tsx  # 阈值滑块 / MAX_LEN / 逐词贡献开关
    ├── SentimentGauge.tsx     # 正负面概率条
    ├── TokenHighlight.tsx     # 逐词贡献高亮 + Top-N 排序列表
    ├── SingleAnalyzer.tsx     # 单条输入面板
    ├── BatchAnalyzer.tsx      # 批量：粘贴多行 / 拖入 CSV
    └── ResultTable.tsx        # 批量结果表格 + 导出 CSV
```

**不引入状态管理库**。全应用状态仅三项（模型列表、推理设置、当前结果），
`useState` + 自定义 hook 足够。

### 6.2 交互流程

**单条分析**：输入评论文本 → 点「分析」→ 显示情感标签、正负面概率条、分词结果。

- 「逐词贡献」关闭时：调用 `/api/predict`
- 「逐词贡献」开启时：**只调 `/api/explain`**。该接口的响应已包含标签与概率，
  再调一次 `/api/predict` 会重复一次前向计算，是多余的往返

拖入 CSV → 点「批量分析」→
表格展示 `文本 / 标签 / 积极概率 / 消极概率` → 「导出 CSV」。

**阈值滑块**：范围 `0.05 ~ 0.95`，默认 `0.5`。**纯前端重算标签，不发请求**——
后端返回概率，标签由 `prob_pos >= threshold` 在前端即时导出。
拖动滑块即可观察 precision / recall 的此消彼长。

### 6.3 CORS 处理

`vite.config.ts` 配置 `server.proxy` 将 `/api` 转发到 `http://127.0.0.1:5000`。
前端一律使用相对路径，开发期同源、无需 CORS。

演示期可执行 `pnpm build`，由 Flask 托管 `frontend/dist/`，实现
**单端口、单进程**，降低现场故障概率。

### 6.4 逐词贡献配色

按 Δ 做**发散色阶**：Δ 为负（积极贡献）→ 蓝系，Δ 为正（消极贡献）→ 橙系，
|Δ| 极小 → 中性灰。色阶需满足正负对称与色盲可辨。实现时依据 data-viz
配色规范确定具体色值，不凭直觉取色。

---

## 7. 参数分层设计

五类参数按"影响解释还是影响计算"分拨：

| 参数 | 生效位置 | 是否发请求 | 原因 |
|------|---------|-----------|------|
| 判定阈值 | 纯前端 | 否 | 标签是概率的无状态纯映射 |
| 逐词贡献开关 | 后端 | 是（首次开启） | 需 N 次额外前向 |
| 模型选择 | 后端 | 是 | 切换 checkpoint 与词表 |
| `MAX_LEN` | 后端 | 是 | 影响截断，即影响 token 序列 |
| 批量文本 | 后端 | 是 | 推理本身 |

### 7.1 "模型参数可调"的实现方式（已决策）

**采用多组预训练 checkpoint，不做后端按需训练。**

前端提交超参 → 后端训练 → 轮询进度：形式上最"完整"，但需引入任务队列与状态管理，
训练期间 GPU 占满、服务阻塞，演示现场不可控。

改为离线网格训练 6 组（`layers ∈ {1,2}` × `hidden ∈ {64,128,256}`），
产物落盘 + `meta.json` 记录指标。前端下拉切换，响应为毫秒级。
训练总时长约为单次的 6 倍（按实验四约 1 分钟/次估算，合计 6~10 分钟），一次性完成。

---

## 8. 数据来源

| 项 | 内容 |
|----|------|
| 数据集 | ChnSentiCorp（中文情感分析语料，含酒店、电脑/电子、图书三域） |
| 主下载源 | `hf-mirror.com`（已验证可达） |
| 备选源 | `raw.githubusercontent.com`（已验证可达）、`gitee.com`（已验证可达） |
| 不可用 | `huggingface.co` 直连（已验证不可达） |
| 落盘位置 | `数据集/ChnSentiCorp/`（不入库） |
| 划分 | 沿用语料自带 train/test；验证集从**训练集**按 `VAL_RATIO=0.1` 切分 |
| 集合用途 | **训练集**更新参数；**验证集**选最优轮次（每轮评估、取验证最优的权重）；**测试集**只在全部训练结束后报告最终指标，不参与任何选择 |

测试集不参与选轮次或调参，避免测试集信息泄漏——这是报告第 5 章会被追问的点。
| 标签 | 1 = 积极，0 = 消极 |

**数据集获取是整个方案唯一的未知数**，实现阶段第 0 步先单独验证可获取性，
再动其他模块。若三个源均失败，降级方案见第 11 节。

---

## 9. 环境与依赖

### 9.1 后端

项目级 `.venv`（Python 3.13.13，uv 创建），不依赖全局解释器。

```
torch 2.12.0+cu132         ← 本地 wheel 安装（1.80 GB，已验证存在）
jieba
flask, flask-cors
matplotlib
pytest
```

**不安装 torchvision。** 本项目的数据是 CSV（标准库 csv 读取）与 jieba 分词，
模型只用 `nn.Embedding / nn.LSTM / nn.Linear`，不涉及任何图像变换或
`torchvision.datasets`，因此 torchvision 无用途。

这不只是省一个依赖——兄弟项目 README 记录的坑是"torchvision 与 torch 版本必须严格配对
（2.12.0 ↔ 0.27.0），且腾讯云镜像不含 CUDA 版 torchvision，需加
`--index-url https://download.pytorch.org/whl/cu132` 与
`--index-strategy unsafe-best-match`，否则报
`RuntimeError: operator torchvision::nms does not exist`"。
去掉 torchvision 后，该坑整条消失，环境可由单个本地 wheel 离线装成。

### 9.2 前端

Node.js + pnpm。依赖：`react`, `react-dom`, `vite`, `typescript`,
`tailwindcss`, `@biomejs/biome`, `papaparse`（CSV 解析）。

---

## 10. 验证策略

| 层 | 方法 | 通过判据 |
|----|------|---------|
| 数据层 | 打印条数、类别分布、长度分位数、`<unk>` 占比 | 条数与语料一致、两类均衡、`<unk>` 占比合理 |
| **模型层** | **小样本过拟合测试**：取 20~50 条不加正则训 30 轮 | 训练准确率能冲到 100% |
| 整体 | 6 组 checkpoint 各跑测试集 | 最优组准确率 **≥85%** |
| 接口层 | pytest 打 API（Flask test client，无需起服务） | 结构正确、概率和为 1、批量长度一致、错误码正确 |
| 可解释性 | 构造含明显强情感词的句子，断言删词后概率变化方向 | 方向正确 |

**小样本过拟合测试是其中最有价值的一条**：把"模型不学习"与"数据管线有错"两个症状分开。
管线正确时几十条样本必然能过拟合；若不能，说明词表、编码或取末词逻辑有 bug。
该技巧在兄弟项目实验三中已用于验证管线（"过拟合测试 20 轮达 88%"）。

### 10.1 测试范围（已决策：中间路线）

对**纯函数**与接口契约写 pytest：`encode`、`build_vocab`、`explain`、
API 响应结构。**不追训练循环的覆盖率**——训练是数值实验，断言意义有限。

---

## 11. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| ChnSentiCorp 三个源均不可达 | 阻塞 | 降级：① 换 `weibo_senti_100k` 等公开中文情感语料；② 用实验四已有的 IMDb 数据改做英文（能跑通但与选题不符，仅作保底） |
| 测试准确率 < 85% | 不达标 | ① 增大 `MAX_LEN`；② 增加 EPOCHS、调 lr；③ 引入预训练词向量（word2vec，`gensim`）；④ 改双向 LSTM / 加注意力池化 |
| 阈值放前端导致两侧标签定义漂移 | 结果不一致 | 标签判定逻辑单点定义在前端 `types.ts`，并由接口测试断言后端不参与判定 |
| 词表与 checkpoint 不配对 | 静默准确率下降 | `vocab.json` 与 `*.pt` 同目录同生命周期；载入时断言 `vocab_size` 一致 |
| 逐词贡献在长文本上变慢 | 交互卡顿 | 以 `CONTRIB_MAX_TOKENS = 100` 截断参与分析的词数，前端提示"仅分析前 100 词" |
| 前端构建产物与后端托管路径不一致 | 演示失败 | 交付前用 `pnpm build` + Flask 托管路径走通一遍 |
| 中文字体缺失导致图表乱码 | 报告图表不可用 | 复用 `common/plot_font.py` 自动探测系统字体 |

---

## 12. 未决事项

> **（实现阶段更新）** 下表是设计阶段留下的待决项，实现后均已落定：U1 已建库并推送、
> U2 已核实（并进一步合并为五语料）、U3 定为 150、U4 超出本轮代码范围。
> 原表保留，以记录当时的判断依据。

| 编号 | 事项 | 处理时机 |
|------|------|---------|
| U1 | 项目 `.git` 初始化与远端仓库 | **待用户决定**（兄弟项目均为 git 仓库，本项目尚未初始化；不擅自执行） |
| U2 | 语料实际规模与划分（原版 7000 条 vs 镜像版 9600+1200） | 数据获取后核实并写入 `meta.json` |
| U3 | `MAX_LEN` 默认值的最终取值 | 数据长度分布统计后确定，当前暂定 150 |
| U4 | 报告撰写（第 1~6 章） | 本轮范围外，代码完成后再议 |

---

## 附：关键设计决策记录

| 决策 | 选择 | 备选与否决理由 |
|------|------|---------------|
| 交付物范围 | 只做代码 | 报告后置，避免两线并行 |
| 分词粒度 | 词级（jieba） | 字级对照成本低但属 YAGNI |
| 界面形态 | 前后端分离，React 前端 | tkinter 零依赖但不符用户偏好 |
| 前端技术栈 | React + Vite + Tailwind | 与用户既有项目一致 |
| 功能范围 | 可解释 + 批量 + 参数可调 | 最完整档 |
| 参数可调实现 | 多组预训练 checkpoint | 后端按需训练会阻塞服务、演示不可控 |
| 测试策略 | 纯函数 + 接口契约 | 直采 80% 覆盖率对数值实验无意义；兄弟项目惯例为无测试框架 |
| 环境 | 项目级 `.venv` | 复用兄弟项目环境会破坏项目独立性 |
| 代码形态 | 仅 `.py`，不做 `.ipynb` 双份 | 兄弟项目两版分叉需双向同步，成本高 |
| 可解释性方法 | 遮挡法 | attention 权重反映路由而非重要性 |
