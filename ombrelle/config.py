"""意匠パラメータの保存と読み込み。

絵は目で決めるものなので、値は実行中に触って決める。ただし決めた値が
「HUD を目で読んで書き写す」でしか外に出せないと、写し間違いと再現不能が起きる。

  * `p` キーで現在の値を config.json に保存する
  * 起動時に config.json があれば**引数の既定値として**読み込む
    (コマンドラインで明示した引数の方が優先される)
  * `s` キーのスクリーンショットには同名の .json を並べて書く
    → どのスクショがどの設定だったかが後から必ず分かる
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

# 保存対象。ここに無いものは実行ごとの状態であって意匠ではない
KEYS = ("haze", "chroma", "brush", "split", "inject", "flow_gain", "cam_lod", "paint_mix", "energy_floor")


def load(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in data.items() if k in KEYS}


def save(state, energy_floor: float, path: Path = CONFIG_PATH) -> Path:
    data = {
        "haze": round(float(state.haze), 3),
        "chroma": round(float(state.chroma), 3),
        "brush": round(float(state.brush), 3),
        "split": round(float(state.split), 3),
        "inject": round(float(state.inject), 3),
        "flow_gain": round(float(state.flow_gain), 3),
        "cam_lod": round(float(state.cam_lod), 3),
        "paint_mix": round(float(state.paint_mix), 3),
        "energy_floor": round(float(energy_floor), 4),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def snapshot(state, energy_floor: float, extra: dict | None = None) -> dict:
    d = {
        "haze": round(float(state.haze), 3),
        "chroma": round(float(state.chroma), 3),
        "brush": round(float(state.brush), 3),
        "split": round(float(state.split), 3),
        "inject": round(float(state.inject), 3),
        "flow_gain": round(float(state.flow_gain), 3),
        "cam_lod": round(float(state.cam_lod), 3),
        "paint_mix": round(float(state.paint_mix), 3),
        "energy_floor": round(float(energy_floor), 4),
        "view": int(state.view),
        "depth": state.depth_kind,
    }
    if extra:
        d.update(extra)
    return d
