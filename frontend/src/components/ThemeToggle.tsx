import type { Theme } from "../hooks/useTheme";

interface Props {
  theme: Theme;
  onToggle: () => void;
}

/** 明暗切换钮。
 *
 *  **文案说的是「点了会发生什么」，不是「现在是什么」**：
 *  一个写着「深色」的按钮，用户无法判断它是「当前已是深色」还是「点它变深色」——
 *  这类按钮在真实产品里是高频误读点。写成动作（「切换到深色」）就没有第二种读法。
 *
 *  `aria-pressed` 同时给出**状态**，供读屏用户听到「按下/未按下」，
 *  与可见文案里的动作描述互补而不重复。
 */
export function ThemeToggle({ theme, onToggle }: Props) {
  const isDark = theme === "dark";
  const label = isDark ? "切换到浅色主题" : "切换到深色主题";

  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={isDark}
      aria-label={label}
      title={label}
      className="btn btn-secondary shrink-0 gap-1.5 text-xs"
    >
      {/* 图标只是装饰（文案已说清），故对读屏隐藏，避免念出「月亮」这种无意义的字符名 */}
      <span aria-hidden="true">{isDark ? "☀" : "☾"}</span>
      <span aria-hidden="true">{isDark ? "浅色" : "深色"}</span>
    </button>
  );
}
