"""ombrelle-live のエントリポイント。

カメラ → (フロー / 深度) → 筆触レンダリング のメインループ。

推論は描画より遅い、という前提で組んでいる:
  * フローは軽いのでメインループ内で同期実行する(筆の向きを駆動するので遅延が効く)
  * 深度は重いので別スレッド。**最新結果が無ければ前回の結果を使い続ける**。
    描画を待たせない = フレームを落としても絵は 60fps で動く。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import glfw
import numpy as np

from . import config as cfg
from .color import WhiteBalance
from .flow import FlowField, gust_env
from .gl.renderer import Renderer
from .metrics import Meter, build_hud, hud_lines
from .source import open_source


class State:
    def __init__(self, args) -> None:
        self.view = float(args.view)
        self.paint_mix = args.paint_mix
        self.flow_gain = args.flow_gain
        self.cam_lod = args.cam_lod
        self.haze = args.haze
        self.chroma = args.chroma
        self.brush = args.brush
        self.split = args.split
        self.hud = not args.no_hud
        self.quit = False
        self.shot = False
        self.save = False
        self.depth_kind = args.depth  # "teacher" | "student" | "off"


def make_key_callback(state: State):
    def cb(window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            state.quit = True
        elif glfw.KEY_0 <= key <= glfw.KEY_3:
            state.view = float(key - glfw.KEY_0)
        elif key == glfw.KEY_S:
            state.shot = True
        elif key == glfw.KEY_P:
            state.save = True
        elif key == glfw.KEY_H:
            state.hud = not state.hud
        elif key == glfw.KEY_LEFT_BRACKET:
            state.flow_gain = max(0.0, state.flow_gain - 0.5)
        elif key == glfw.KEY_RIGHT_BRACKET:
            state.flow_gain = min(40.0, state.flow_gain + 0.5)
        elif key == glfw.KEY_COMMA:
            state.cam_lod = max(0.0, state.cam_lod - 0.25)
        elif key == glfw.KEY_PERIOD:
            state.cam_lod = min(8.0, state.cam_lod + 0.25)
        elif key == glfw.KEY_MINUS:
            state.paint_mix = max(0.0, state.paint_mix - 0.1)
        elif key == glfw.KEY_EQUAL:
            state.paint_mix = min(1.0, state.paint_mix + 0.1)
        elif key == glfw.KEY_K:
            state.haze = max(0.0, state.haze - 0.05)
        elif key == glfw.KEY_L:
            state.haze = min(1.5, state.haze + 0.05)
        elif key == glfw.KEY_N:
            state.chroma = max(0.5, state.chroma - 0.05)
        elif key == glfw.KEY_M:
            state.chroma = min(2.5, state.chroma + 0.05)
        elif key == glfw.KEY_V:
            state.brush = max(0.2, state.brush - 0.1)
        elif key == glfw.KEY_B:
            state.brush = min(6.0, state.brush + 0.1)
        elif key == glfw.KEY_T:
            state.split = max(0.0, state.split - 0.05)
        elif key == glfw.KEY_Y:
            state.split = min(1.5, state.split + 0.05)
        elif key == glfw.KEY_D:
            order = ["teacher", "student", "off"]
            state.depth_kind = order[(order.index(state.depth_kind) + 1) % 3]
    return cb


def sample_depth(depth: np.ndarray | None, xy: tuple[float, float]) -> float:
    """深度マップを (x, y=上向き) で1点サンプルする。"""
    if depth is None:
        return 0.6
    h, w = depth.shape[:2]
    x = int(np.clip(xy[0], 0.0, 1.0) * (w - 1))
    y = int(np.clip(1.0 - xy[1], 0.0, 1.0) * (h - 1))
    return float(depth[y, x])


def main() -> None:
    ap = argparse.ArgumentParser(description="現実を印象派に変換するリアルタイム処理系")
    ap.add_argument(
        "--source", default="cam:0",
        help="cam:0 | 動画/画像のパス | synthetic",
    )
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--render-width", type=int, default=1280)
    ap.add_argument("--render-height", type=int, default=720)
    ap.add_argument("--view", type=int, default=0, help="0=筆触 1=生カメラ 2=深度 3=フロー")
    ap.add_argument("--depth", choices=["teacher", "student", "off"], default="off")
    ap.add_argument("--student-ckpt", default="checkpoints/student.pt")
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--flow-gain", type=float, default=1.5)
    ap.add_argument("--cam-lod", type=float, default=2.0)
    ap.add_argument("--paint-mix", type=float, default=1.0,
                    help="0=グレーディングのみ 1=完全に絵")
    ap.add_argument("--brush", type=float, default=2.4,
                    help="筆の大きさ。1.0 が原典の値。カメラの近接被写体では大きめが要る")
    ap.add_argument("--split", type=float, default=0.50,
                    help="色彩分割の強さ(暖色側/寒色側への振り分け)")
    ap.add_argument("--haze", type=float, default=0.35)
    ap.add_argument("--chroma", type=float, default=1.35)
    ap.add_argument("--energy-floor", type=float, default=0.0,
                    help="風の最低値。動いていなくても絵を動かしたいとき / 検証用")
    ap.add_argument("--frames", type=int, default=0, help=">0 なら N フレームで終了(検証用)")
    ap.add_argument("--shot", default="", help="終了直前にこのパスへ保存(検証用)")
    # config.json があればそれを既定値にする。明示した引数の方が優先される
    saved = cfg.load()
    if saved:
        ap.set_defaults(**saved)
    args = ap.parse_args()
    if saved:
        print(f"config.json を読み込みました: {saved}")

    state = State(args)
    meter = Meter()

    cam = open_source(args.source, args.cam_width, args.cam_height, mirror=not args.no_mirror)
    first = cam.wait_first()
    print(f"source {args.source}: {first.shape[1]}x{first.shape[0]}")

    renderer = Renderer(
        win_w=args.render_width, win_h=args.render_height,
        render_w=args.render_width, render_h=args.render_height,
    )
    key_cb = make_key_callback(state)
    glfw.set_key_callback(renderer.window, key_cb)

    flowf = None if args.no_flow else FlowField()
    wb = WhiteBalance()

    depther = None
    if args.depth != "off":
        from .depth import DepthWorker
        depther = DepthWorker(kind=args.depth, ckpt=args.student_ckpt).start()

    latest_depth: np.ndarray | None = None
    last_seq = -1
    adv = 0.0
    hud_at = 0.0
    t0 = time.perf_counter()
    n = 0

    try:
        while not renderer.should_close() and not state.quit:
            renderer.poll()
            dt = meter.tick()
            t = time.perf_counter() - t0

            frame, stamp, seq = cam.latest()
            if frame is not None and seq != last_seq:
                last_seq = seq
                renderer.update_camera(frame)
                wb.update(frame)
                if flowf is not None:
                    s = time.perf_counter()
                    flowf.update(frame, stamp)
                    meter.add_stage("flow", time.perf_counter() - s)
                    renderer.update_flow(flowf.field)
                if depther is not None:
                    depther.submit(frame)

            if depther is not None:
                d = depther.latest()
                if d is not None:
                    latest_depth = d
                    renderer.update_depth(d)
                    meter.add_stage("depth", depther.last_infer_s)

            energy = flowf.energy if flowf is not None else 0.0
            energy = max(energy, args.energy_floor)
            seed = flowf.centroid if flowf is not None else (0.5, 0.5)
            wind = flowf.wind if flowf is not None else (-1.0, 0.12)
            # 人が動いた分だけ風が進む
            adv += gust_env(t) * (0.30 + 2.5 * energy) * dt

            if state.hud and t - hud_at > 0.25:
                hud_at = t
                src = state.depth_kind if depther is not None else "off"
                renderer.update_hud(
                    build_hud(renderer.render_w, renderer.render_h,
                              hud_lines(meter, state, energy, src))
                )
            elif not state.hud:
                renderer.update_hud(None)

            renderer.draw({
                "uTime": t,
                "uAdv": adv,
                "uView": state.view,
                "uFlowGain": state.flow_gain,
                "uCamLod": state.cam_lod,
                "uSeed": (float(seed[0]), float(seed[1])),
                "uSeedDepth": sample_depth(latest_depth, seed),
                "uWind": (float(wind[0]), float(wind[1])),
                "uEnergy": float(energy),
                "uPaintMix": state.paint_mix,
                "uHaze": state.haze,
                "uChroma": state.chroma,
                "uBrush": state.brush,
                "uSplit": state.split,
                "uWhite": tuple(float(x) for x in wb.gain),
            })
            meter.add_latency(time.perf_counter() - stamp)

            if state.shot:
                state.shot = False
                stem = f"{datetime.now():%Y%m%d-%H%M%S}"
                p = renderer.screenshot(Path("shots") / f"{stem}.png")
                # どのスクショがどの設定だったかを必ず残す
                side = p.with_suffix(".json")
                side.write_text(
                    json.dumps(cfg.snapshot(state, args.energy_floor, {
                        "source": args.source,
                        "fps": round(meter.fps, 1),
                        "e2e_ms": round(meter.latency_ms, 1),
                        "energy": round(float(energy), 5),
                    }), indent=2) + "\n", encoding="utf-8")
                # 生のカメラフレームも残す。これがあれば Claude 側で
                # --source shots/xxx_raw.png として同じ画で意匠を詰められる
                if frame is not None:
                    raw = p.with_name(p.stem + "_raw.png")
                    cv2.imwrite(str(raw), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"saved {p} + {side.name} + {raw.name}")
                else:
                    print(f"saved {p} + {side.name}")

            if state.save:
                state.save = False
                print(f"saved {cfg.save(state, args.energy_floor)}  ← 次回起動時に自動で読まれます")

            n += 1
            if args.frames and n >= args.frames:
                break

        if args.shot:
            print(f"saved {renderer.screenshot(args.shot)}")
        el = time.perf_counter() - t0
        # 単体のレイテンシではなく「描画と GPU を共有した状態で深度の場が
        # 毎秒何回更新されたか」が体験に効く量。捨てた入力フレーム数も一緒に出す。
        dhz = (depther.done / el) if depther is not None else 0.0
        drop = depther.dropped if depther is not None else 0
        print(
            f"frames={n} elapsed={el:.1f}s fps={meter.fps:.1f} e2e={meter.latency_ms:.1f}ms "
            f"flow={meter.ms('flow'):.1f}ms depth_infer={meter.ms('depth'):.1f}ms "
            f"depth_updates={dhz:.1f}Hz dropped={drop}"
        )
    finally:
        cam.stop()
        if depther is not None:
            depther.stop()
        renderer.close()


if __name__ == "__main__":
    main()
