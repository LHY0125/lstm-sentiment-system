# CLAUDE.md

本目录是《神经网络与深度学习》课程设计——「基于 LSTM 的网络评论情感分类系统」。
前后端分离：Python 后端（训练 + 推理）+ React 前端。

## 项目结构

- `backend/senti/` —— 五层划分：`data`（数据层）/ `model`（模型层）/ `train`（训练层）/
  `infer`（推理层）/ `api`（接口层）；`config.py` 是全局路径与超参数常量
- `backend/artifacts/` —— 训练产物（checkpoint + 词表 + 元数据），**不入库**
- `backend/tests/` —— pytest；`backend/run_api.py` —— 后端启动入口
- `frontend/` —— React 19 + Vite + TypeScript + Tailwind v4（pnpm + Biome）
- `docs/superpowers/` —— 设计文档与实现计划
- `.superpowers/sdd/` —— 逐任务的 brief / report / 审查 diff（过程记录）
- `.venv/` —— 项目专用虚拟环境，**不要用全局解释器替代**

## 常用命令

所有后端命令在 `backend/` 下执行：

```powershell
..\.venv\Scripts\python.exe -m senti.train --overfit-test   # 数据管线体检
..\.venv\Scripts\python.exe -m senti.train                  # 训练全部 6 组
..\.venv\Scripts\python.exe -m senti.train --config 2 128   # 只训一组（层数 隐藏维度）
..\.venv\Scripts\python.exe run_api.py                      # 启动后端（含前端静态托管）
..\.venv\Scripts\python.exe -m pytest -q                    # 全部测试
```

前端在 `frontend/` 下执行：`pnpm dev` / `pnpm build` / `pnpm typecheck` / `pnpm lint`。

## 关键约定（改动前必读）

1. **测试集只用于最终报告指标**。选最优轮次、给模型列表排序、挑前端默认项，
   一律依据**验证**准确率。不要写 `if test_acc > best: best = test_acc` —— 那是测试集泄漏，
   而「按测试集挑出的最大值」还会带上 winner's curse。
2. **词表与 checkpoint 必须同源**。`vocab.json` 与 `*.pt` 同生命周期，
   `Predictor.__init__` 有 `vocab_size` 一致性断言。改词表必须重训。
3. **阈值是纯前端参数**。后端只返回概率，`/api/*` 不接收阈值。
   标签判定规则只在 `frontend/src/types.ts` 的 `labelOf` 定义一处，
   且拖动阈值**不允许发请求**。
4. **`MAX_LEN` 改了不需要重训**：它只影响推理期的截断，但它会改变输入表示，
   故由后端参数接收、改完必须重新请求，不走前端本地计算。
5. **超参数只在 `backend/senti/config.py` 定义**，训练/推理代码内不得出现字面量。
   跨语言的那几个常量（`MAX_BATCH_SIZE`、`MAX_TEXT_CHARS`）在前后端各有一份，
   **必须一起改**，见 `frontend/src/types.ts` 的注释。
6. **不装 torchvision**：本项目无图像任务，装了反而引入版本配对坑。
7. **前端列表的 `key` 不能用数组下标**：Biome 的 `suspicious/noArrayIndexKey` 判 error
   → `pnpm lint` 退出码 1。先摊成带显式身份字段的渲染单元再 `key={unit.id}`，
   不要为此开 `biome-ignore`。
8. **`/api/` 下的路径不得被前端静态托管的兜底路由吞掉**：路由匹配先于错误处理器，
   放行会让 `/api/nope` 变成 200 + `index.html`、`GET /api/predict` 丢掉 405，
   且**只在 `frontend/dist/` 存在时**（也就是演示时）发作。

## 验证方式

- 后端：`pytest`（纯函数 + 接口契约，含 5 条路由的成功与失败路径、错误信封）
- 数据管线：`--overfit-test`，必须能过拟合
- 前端：`pnpm typecheck` + `pnpm lint` + `pnpm build`，再用浏览器人工走通验收表
  （单条/批量/阈值不发请求/CSV 导出与上传/后端掉线不白屏）
