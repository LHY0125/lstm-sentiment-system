# 贡献指南

感谢你有兴趣参与。这个项目是一个课程设计作品，代码规模不大但要求较严——
下面几条是**动手前请先看的**。

## 环境

后端需要 Python 3.13，依赖见 [`backend/requirements.txt`](backend/requirements.txt)：

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe -r backend\requirements.txt
```

前端需要 Node 20+ 与 pnpm：

```powershell
cd frontend; pnpm install
```

**语料不入库**（`数据集/` 已在 `.gitignore` 中）。首次运行需要自己下载：

```powershell
cd backend
..\.venv\Scripts\python.exe -m senti.train --overfit-test          # 拉主语料并体检
..\.venv\Scripts\python.exe -m senti.corpora                       # 拉四个扩充语料（约 180 MB）
```

## 提交改动前

```powershell
# 后端
cd backend; ..\.venv\Scripts\python.exe -m pytest -q

# 前端（三条都要过）
cd frontend; pnpm typecheck; pnpm lint; pnpm build
```

**测试必须全绿**。前端没有测试运行器，`typecheck` + `lint` + `build` 是自动化能覆盖的全部；
涉及界面的改动请自行在浏览器里走一遍（单条分析 / 批量分析 / 阈值 / CSV 导出与上传 / 后端掉线）。

## 代码约定

**改动前请先读 [`CLAUDE.md`](CLAUDE.md) 的「关键约定」一节**——那里列了 8 条容易踩的坑，
其中最要紧的是：

1. **测试集只用于最终报告**：任何"选择"（选轮次、排模型顺序、挑默认项）一律依据**验证**集。
2. **词表与 checkpoint 必须同源**：改词表就要重训，否则推理时静默错位。
3. **超参数只在 `backend/senti/config.py`**：训练与推理代码里不出现字面量。
4. **跨语言常量有两份**（前后端各一），必须一起改。

其余风格约定：Python 跟随 PEP 8、注释用中文并说明**为什么**而不是复述代码；
前端由 Biome 统一格式与规则（不要手工与它对格式）。

## 提交信息

用 [约定式提交](https://www.conventionalcommits.org/zh-hans/)：

```
<type>: <描述>

可选正文：说清「为什么」，而不是「改了什么」——后者 diff 已经写了。
```

`type` 取 `feat` / `fix` / `refactor` / `docs` / `test` / `chore` / `perf` / `ci` 之一。

**一个提交只做一件事。** 若一次改动里既有重构又有行为变化，拆成两个提交——
审查者需要能分别判断它们。

## 拉取请求

1. 从 `main` 切出分支，命名如 `feat/xxx` 或 `fix/xxx`。
2. 在描述里写清：**动机**（为什么要改）、**做法**（怎么改的）、**验证**（跑了什么命令、结果如何）。
3. 如果改动影响了 README 里的实测数字或接口契约，**请一并更新文档**——
   文档与代码不一致时，读者不会知道该信哪个。

### 关于指标

若你的改动会影响模型准确率，请在描述里同时给出**验证准确率**与**测试准确率**，
并说明你的"选择"依据的是哪一个。合并语料下还要给出**平衡准确率**与**多数类基线**——
类别偏斜时，准确率单独看会骗人。

## 报告问题

- 功能缺陷或建议：开 Issue。
- **安全相关问题：请不要开公开 Issue**，见 [`SECURITY.md`](SECURITY.md)。

参与本项目即表示你同意遵守[行为准则](CODE_OF_CONDUCT.md)。
