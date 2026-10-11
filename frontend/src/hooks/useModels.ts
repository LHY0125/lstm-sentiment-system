import { useCallback, useEffect, useState } from "react";
import { fetchModels } from "../api/client";
import type { ModelInfo } from "../types";

/** 拉取可用模型列表，并按**验证**准确率降序排列。
 *
 *  用验证集而非测试集排序：排序即「选择」，而测试集只负责报告。
 *  若按测试准确率排序，默认选中的就是 6 个测试准确率的极大值，会被高估。 */
export function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await fetchModels();
      setModels([...list].sort((a, b) => b.val_acc - a.val_acc));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { models, loading, error, reload };
}
