# -*- coding: utf-8 -*-
"""config.py 常量与辅助函数的测试。"""

from pathlib import Path

from senti import config


def test_grid_has_six_combinations():
    """网格必须是 2 种层数 × 3 种隐藏维度 = 6 组。"""
    assert len(config.GRID) == 6
    assert (1, 64) in config.GRID
    assert (2, 256) in config.GRID


def test_model_id_format():
    assert config.model_id(2, 128) == "lstm_l2_h128"


def test_paths_are_absolute_and_under_project_root():
    """路径必须绝对，且都能追到项目根，避免依赖当前工作目录。"""
    for p in (config.DATA_DIR, config.ARTIFACT_DIR, config.OUTPUT_DIR):
        assert isinstance(p, Path)
        assert p.is_absolute()
        assert config.PROJECT_ROOT in p.parents


def test_special_token_indices():
    """0 号必须留给 <pad>，1 号留给 <unk>，实词从 2 开始。"""
    assert config.PAD_IDX == 0
    assert config.UNK_IDX == 1
    # 关系和数值一起锁：只钉死 2 的话，将来若增删特殊符号，
    # 这里不会跟着报错，词表与 encode 会静默错位。
    assert config.FIRST_WORD_IDX == config.UNK_IDX + 1 == 2
    assert config.MAX_LEN > 0
    assert config.CONTRIB_MAX_TOKENS > 0
