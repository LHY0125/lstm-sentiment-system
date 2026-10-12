import { useCallback, useEffect, useState } from "react";

/** 主题取值。**只有两个**——不提供「跟随系统」第三态：那会让切换钮变成三态循环，
 *  且用户回到「跟随系统」后无法判断当前到底跟的是哪一套。跟随系统只作为**首次进入**
 *  的默认值，用户一旦手动切过，就以他的选择为准。 */
export type Theme = "light" | "dark";

/** localStorage 键。与 `index.html` 里那段防白闪的内联脚本**必须一致**。 */
export const THEME_STORAGE_KEY = "senti-theme";

/** 首次渲染时读当前主题。
 *
 *  优先读 `<html>` 上已经存在的类——那段内联脚本在首帧之前就把它打上了，
 *  以它为准能保证「脚本算出来的」与「React 算出来的」永远一致；
 *  万一脚本没跑（被 CSP 拦、或将来有人删了它），再退回同一套计算规则。
 *  localStorage 在隐私模式下可能直接抛异常，故整段包在 try 里。 */
function initialTheme(): Theme {
  if (typeof document !== "undefined" && document.documentElement.classList.contains("dark")) {
    return "dark";
  }
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

/** 主题状态：写 <html class="dark">、并持久化到 localStorage。
 *
 *  **不在这里读 DOM 决定渲染什么**：组件拿到的 `theme` 就是唯一真相，
 *  类名是它的**副作用**。反过来（读 DOM 再 setState）会在 StrictMode 的双调用下
 *  产生一帧不一致。
 */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // 隐私模式 / 配额满：持久化失败不影响本次会话的主题，静默跳过
    }
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  }, []);

  return { theme, toggle };
}
