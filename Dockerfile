# 基于 LSTM 的中文评论情感分类系统 —— 单容器镜像
#
# 单容器是**顺着项目已有的设计**来的：Flask 在 frontend/dist 存在时会顺带托管前端，
# 于是前端构建产物一进镜像，就不需要第二个进程、也不需要反向代理。
#
# 三条「本机假设」在容器里都会**静默失效**（不报错，只是行为变成另一个东西），
# Dockerfile 里的每一处非显然写法都对应其中一条：
#
#   1. 目录布局必须复刻成 /app/backend/senti/config.py。
#      config.PROJECT_ROOT = Path(__file__).resolve().parents[2]，布局差一层就算不出
#      /app/frontend/dist，兜底路由**不注册**：/api 一切正常、首页 404，
#      现场看起来像「前端没构建」。
#   2. run_api.py 默认绑 127.0.0.1 → 容器里必须 HOST=0.0.0.0（见该文件注释）。
#   3. checkpoint（67 MB）**不入库**，故不打进镜像，由卷挂载到 /app/backend/artifacts。
#      没挂时容器照起、健康检查照过、/api/models 返回空数组——那是「没挂模型」，
#      不是「镜像坏了」。两者必须能分辨，所以健康检查**刻意不查模型**。
#
# 构建：
#   docker build -t senti-lstm:latest .
# 运行：
#   docker run --rm -p 8050:5000 -v /宿主/backend/artifacts:/app/backend/artifacts:ro senti-lstm:latest

# ── 阶段 1：构建前端（需要 Node 与 devDependencies） ────────────────────────
FROM node:24-slim AS web
WORKDIR /web
# 先只拷依赖清单再安装：改前端源码不会让这一层缓存失效
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN npm i -g pnpm@11.7.0 && pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

# ── 阶段 2：运行时 ─────────────────────────────────────────────────────────
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=5000

WORKDIR /app

# 依赖单独成层：之后改后端源码不会触发 torch 重装（那是本镜像最贵的一层）
COPY backend/requirements-runtime.txt /app/backend/requirements-runtime.txt
RUN pip install --no-cache-dir -r /app/backend/requirements-runtime.txt

COPY backend/ /app/backend/
COPY --from=web /web/dist /app/frontend/dist

# 非 root 运行。artifacts 目录先建好并交给 app：
# 宿主目录若不存在，Docker 会以 root 建挂载点，容器内就以非 root 读不到。
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/backend/artifacts \
    && chown -R app:app /app

USER app

EXPOSE 5000

# 只断言「进程活着、接口答得上话」。**刻意不查模型**——见文件顶部第 3 条。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://127.0.0.1:5000/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "backend/run_api.py"]
