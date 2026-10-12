# -*- coding: utf-8 -*-
"""后端启动入口。

运行：..\\.venv\\Scripts\\python.exe run_api.py

debug 必须保持 False：调试模式的重载器会再起一个子进程，
每个进程都会各自载入六份 checkpoint，显存/内存瞬间翻倍。

**监听地址可用环境变量覆盖，默认值与本机原有行为完全一致**：
    HOST  默认 127.0.0.1
    PORT  默认 5000

容器里必须设 HOST=0.0.0.0：绑在回环上的进程**只有容器自己的回环能访问**，
端口映射形同虚设。这条不会报任何错——进程正常起、日志正常打、容器内
`curl 127.0.0.1:5000` 也通，只有从外面连不上，是最难排查的一类。
"""

import os

from senti.api import create_app

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    create_app().run(host=host, port=port, debug=False)
