## 这个 PR 做了什么

<!-- 一句话说清。 -->

## 动机

<!-- 为什么要改？解决了什么问题？（diff 已经说明"改了什么"，这里说"为什么"。） -->

## 做法

<!-- 关键设计取舍。若某个做法看起来绕，解释为什么不能更直接。 -->

## 验证

<!-- 跑了什么命令、结果如何。原样贴输出比"测试通过"有说服力。 -->

```
# 后端
cd backend; ..\.venv\Scripts\python.exe -m pytest -q

# 前端
cd frontend; pnpm typecheck; pnpm lint; pnpm build
```

## 检查清单

- [ ] 测试全绿，且新行为有对应测试覆盖
- [ ] 若改了接口契约，README 与 `frontend/src/types.ts` 已同步
- [ ] 若改了超参数，只改在 `backend/senti/config.py`
- [ ] 若涉及"选择"（选轮次、排序、默认项），依据的是**验证**集而非测试集
- [ ] 若影响模型准确率，PR 描述里同时给出了验证准确率与测试准确率
- [ ] 没有把 `数据集/`、`backend/artifacts/`、`dist/`、`node_modules/` 提交进来

## 关联 Issue

<!-- 如 closes #123 -->
