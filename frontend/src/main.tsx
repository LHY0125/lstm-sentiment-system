import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";

// 显式取到挂载点：取不到就立刻抛出可读错误，而不是用 `!` 断言把 null 推迟成
// 运行期的 "Cannot read properties of null"。前提是 index.html 里有 <div id="root">。
const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("未找到 #root 挂载点：请检查 index.html");
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
