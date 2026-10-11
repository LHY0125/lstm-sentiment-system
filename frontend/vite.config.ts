import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // 绑定 IPv4 回环：Vite 默认 host="localhost"，本机 localhost 先解析到 ::1，
    // 于是只监听 [::1]:5173，而文档与浏览器里写的是 127.0.0.1 —— 那种地址会连不上。
    host: "127.0.0.1",
    // 开发期把 /api 代理到 Flask，前端一律用相对路径。
    // 这样开发期同源、完全不需要 CORS 配置。
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
});
