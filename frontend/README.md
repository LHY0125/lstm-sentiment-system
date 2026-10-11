# 前端 · 网络评论情感分类系统

本目录是《神经网络与深度学习》课程设计「基于 LSTM 的网络评论情感分类系统」的前端：
单条分析（含遮挡法逐词贡献）、批量分析与推理参数调节三个界面。

技术栈：React 19 + TypeScript + Vite + Tailwind CSS v4，包管理用 pnpm。

## 它需要后端

所有数据都来自 Flask 后端，默认地址 **`http://127.0.0.1:5000`**。前端不内嵌任何模型或样例数据，
后端没起来时页面会显示「无法连接后端」而不是空转。

## 运行

```powershell
pnpm install
pnpm dev        # 开发服务器，http://127.0.0.1:5173
```

开发模式下 `/api` 由 Vite 代理转发到 `127.0.0.1:5000`（见 `vite.config.ts`），
浏览器看到的是同源请求，因此开发期不需要 CORS。

**演示时不必开两个进程**：在仓库根目录构建一次后，由 Flask 托管 `dist/`，只需一个进程：

```powershell
pnpm build                              # 产出 dist/
cd ..\backend; ..\.venv\Scripts\python.exe run_api.py
# 然后访问 http://127.0.0.1:5000/
```

> 不要用 `pnpm preview` 做联调检查：`vite preview` 只托管 `dist/`，**不带** `server.proxy`
> （要用得另配 `preview.proxy`），所以那里的 `/api` 请求必然失败——那是预览配置的正常现象，
> 不是接口坏了。要单进程就用上面这条 Flask 托管的路子。

## 命令

| 命令 | 作用 |
|------|------|
| `pnpm dev` | 开发服务器（5173，含 `/api` 代理） |
| `pnpm build` | `tsc -b` + `vite build`，产出 `dist/`（Flask 托管的产物） |
| `pnpm typecheck` | `tsc -b --noEmit` 类型检查 |
| `pnpm lint` | Biome 检查（格式 + 规则），**必须零错误** |
| `pnpm lint:fix` | Biome 自动修复格式与可自动修的规则 |
| `pnpm preview` | 仅预览 `dist/`（无代理，别用它验接口） |

本项目的 linter 与格式化器都是 **Biome**（配置在 `biome.json`）。
`package.json` 里的 `oxlint` 与 `.oxlintrc.json` 是 Vite 脚手架自带的残留，**本项目不使用**；
`pnpm lint` 跑的只有 Biome。清理它们需要同步改 `pnpm-lock.yaml`，故暂留原样。

## 目录结构

```
src/
├── api/client.ts          # 唯一发网络请求的模块，5 个后端端点的薄封装
├── components/
│   ├── ModelSelector.tsx      # 模型下拉 + 该模型的指标
│   ├── InferenceSettings.tsx  # 阈值 / MAX_LEN / 逐词贡献开关
│   ├── SingleAnalyzer.tsx     # 单条分析面板
│   ├── SentimentGauge.tsx     # 概率条（纯展示）
│   ├── TokenHighlight.tsx     # 整句着色 + 影响最大的词表
│   ├── BatchAnalyzer.tsx      # 批量面板（粘贴或上传 CSV）
│   └── ResultTable.tsx        # 批量结果表 + CSV 导出
├── hooks/useModels.ts     # 拉模型列表并按验证准确率排序
├── types.ts               # 与后端契约一一对应的类型 + 全部跨边界常量
├── App.tsx                # 组装：页签、模型选择、设置与两个面板
└── index.css              # Tailwind 入口 + 逐词贡献的三个色阶变量
```

## 改动前必读的几条约定

1. **阈值是纯前端参数。** 后端只返回概率，`/api/*` 一律不接收阈值；
   `积极概率 ≥ 阈值 → 积极` 这条规则全应用只在 `types.ts` 的 `labelOf` 定义一处。
   在小部件里再写一遍判定，就会出现「滑块动了、某个地方不跟着动」。同理，
   拖动阈值**不允许发请求**——那是把免费的即时反馈换成一次往返延迟。
2. **`MAX_LEN` 与逐词贡献开关改完必须重新请求**：它们改变的是送进模型的东西（分词截断、计算方式），
   与阈值不是一类旋钮，尽管三个控件长得很像。
3. **跨边界常量有两份，必须一起改**：`types.ts` 里的 `MAX_BATCH_SIZE` / `MAX_TEXT_CHARS`
   与后端 `senti/api.py` / `senti/config.py` 里各自的同名常量。只改一边的后果是前端放行、后端拒绝
   （或反过来拦掉合法输入）。
4. **列表的 `key` 不能是数组下标**：Biome 的 `suspicious/noArrayIndexKey` 判 error。
   做法是先把数组摊成带显式身份字段的渲染单元，再 `key={unit.id}`；本仓库不开 `biome-ignore` 豁免。
5. **不新建 `tailwind.config.js`**：Tailwind v4 是 CSS 优先配置，主题写在 `index.css`。
6. **不要在 JSX 里内联十六进制颜色**：色阶只有 `index.css` 的 `--contrib-*` 三个变量一个来源。
7. **modelId 必须始终是一个已加载的模型**：模型下拉是完全受控的，`value` 指向不存在的
   `<option>` 时浏览器会把它画成空白。列表刷新后旧 id 失效的情况由 `App.tsx` 在渲染期派生兜住。
