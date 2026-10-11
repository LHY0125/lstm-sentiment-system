# -*- coding: utf-8 -*-
"""后端启动入口。

运行：..\\.venv\\Scripts\\python.exe run_api.py

debug 必须保持 False：调试模式的重载器会再起一个子进程，
每个进程都会各自载入六份 checkpoint，显存/内存瞬间翻倍。
"""

from senti.api import create_app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
