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
from .palette import PaletteMeter, Stabilizer
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
        self.inject = args.inject
        self.memory = args.memory
        self.idle_wind = args.idle_wind
        self.compose = 1.0 if args.compose else 0.0
        self.stand = args.stand
        # 計測は既定 OFF。人物マットに 8ms 掛かる分だけ深度の更新率が落ちるので、
        # 見るときだけ点ける (a キー)。意匠ではないので config.json には保存しない
        self.palette = bool(args.palette)
        # 色の操作をどの平面で行うか。既定は今までの luma 対立色平面
        self.oklab = 1.0 if args.oklab else 0.0
        # 実測した R_C を目標帯へ寄せる閉ループ。測っていないと動かせないので
        # 点けたら計測も点く
        self.stabilize = bool(args.stabilize)
        if self.stabilize:
            self.palette = True
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
        elif key == glfw.KEY_F:
            state.inject = max(0.0, state.inject - 0.02)
        elif key == glfw.KEY_G:
            state.inject = min(0.6, state.inject + 0.02)
        elif key == glfw.KEY_I:
            state.memory = max(0.0, state.memory - 0.05)
        elif key == glfw.KEY_O:
            state.memory = min(1.0, state.memory + 0.05)
        elif key == glfw.KEY_W:
            state.idle_wind = max(0.0, state.idle_wind - 0.1)
        elif key == glfw.KEY_E:
            state.idle_wind = min(1.0, state.idle_wind + 0.1)
        elif key == glfw.KEY_C:
            state.compose = 0.0 if state.compose > 0.5 else 1.0
        elif key == glfw.KEY_R:
            state.stand = max(0.2, state.stand - 0.05)
        elif key == glfw.KEY_U:
            state.stand = min(1.0, state.stand + 0.05)
        elif key == glfw.KEY_A:
            state.palette = not state.palette
        elif key == glfw.KEY_J:
            state.oklab = 0.0 if state.oklab > 0.5 else 1.0
        elif key == glfw.KEY_X:
            state.stabilize = not state.stabilize
            if state.stabilize:
                state.palette = True      # 測っていないと制御できない
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
    ap.add_argument("--brush", type=float, default=1.6,
                    help="筆の大きさ。1.0 が原典の値。カメラの近接被写体では大きめが要る")
    ap.add_argument("--split", type=float, default=0.60,
                    help="色彩分割の強さ。筆の大きさに連動するので、"
                         "筆を変えても見た目の揺らめき量は保たれる")
    ap.add_argument("--inject", type=float, default=0.28,
                    help="中性面への色の注入量。白い壁を暖色と寒色で描くための量")
    ap.add_argument("--memory", type=float, default=0.90,
                    help="記憶色(肌)の保護。肌の近くの色相だけ振れ幅を抑える。0で保護なし")
    ap.add_argument("--idle-wind", type=float, default=0.0,
                    help="静止時の微風。0=誰も動かなければ筆も止まる 1=従来 (実行中は w e)")
    ap.add_argument("--compose", action="store_true",
                    help="人物をモネ風の風景の中へ合成する (実行中は c キー)")
    ap.add_argument("--palette", action="store_true",
                    help="人物と背景の dL / R_C / dh を測って HUD に出す (実行中は a キー)")
    ap.add_argument("--oklab", action="store_true",
                    help="色の注入/圧縮/分割/天井を Oklab で行う (実行中は j キー)")
    ap.add_argument("--stabilize", action="store_true",
                    help="実測した R_C を目標帯へ寄せる (実行中は x キー、計測も自動で点く)")
    ap.add_argument("--stand", type=float, default=0.75,
                    help="合成時に人物が立つ奥行き 0=遠 1=手前")
    ap.add_argument("--haze", type=float, default=0.35)
    ap.add_argument("--chroma", type=float, default=1.30)
    ap.add_argument("--energy-floor", type=float, default=0.0,
                    help="風の最低値。動いていなくても絵を動かしたいとき / 検証用")
    ap.add_argument("--fps", type=float, default=60.0,
                    help="描画の上限 fps。0 で無制限")
    ap.add_argument("--frames", type=int, default=0, help=">0 なら N フレームで終了(検証用)")
    ap.add_argument("--shot", default="", help="終了直前にこのパスへ保存(検証用)")
    ap.add_argument("--wait-ready", action="store_true",
                    help="深度と計測が揃ってから --frames を数え始める(比較画像の生成用)")
    ap.add_argument("--ready-timeout", type=float, default=30.0,
                    help="揃うのを待つ上限(秒)。超えたら警告して先へ進む")
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

    # 深度とマットは 1 本のスレッドで回す。2 本にすると GL のメインスレッドと
    # 合わせて 3 経路が同じ GPU を奪い合い、間欠的にハングする (ombrelle/vision.py 参照)
    vision = None
    if args.depth != "off" or args.compose or args.palette:
        from .vision import VisionWorker
        vision = VisionWorker(depth_kind=args.depth, ckpt=args.student_ckpt,
                              want_matte=args.compose or args.palette).start()
    # 入力(カメラ)と出力(描いた絵)を別々に測る。処理系が彩度比をどう動かしたかは
    # 片方だけ見ても分からない。出力側は FBO の読み戻しで GPU を止めるので低レート
    pal_in = PaletteMeter()
    pal_out = PaletteMeter(ema=0.6)
    pal_out_at = 0.0
    PAL_OUT_HZ = 2.0
    stab = Stabilizer()

    def shot_meta() -> dict:
        """スクショの脇に置く記録。設定値と実測値を必ず 1 枚と対にする。

        s キーでも --shot でも同じものを書く。tools/sweep.py はこれを読んで
        比較画像に数値を添える
        """
        extra = {
            "source": args.source,
            "fps": round(meter.fps, 1),
            "e2e_ms": round(meter.latency_ms, 1),
            "energy": round(float(energy), 5),
        }
        if pal_in.stats is not None:
            extra["palette_in"] = pal_in.stats.as_dict()
        if pal_out.stats is not None:
            extra["palette_out"] = pal_out.stats.as_dict()
        if state.stabilize:
            extra["stab"] = {"subj_chroma": round(stab.subj_chroma, 3),
                             "split_scale": round(stab.split_scale, 3)}
        return cfg.snapshot(state, args.energy_floor, extra)

    latest_depth: np.ndarray | None = None
    latest_matte: np.ndarray | None = None
    last_seq = -1
    energy = 0.0
    adv = 0.0
    hud_at = 0.0
    t0 = time.perf_counter()
    n = 0
    ready = False
    warned = False

    try:
        while not renderer.should_close() and not state.quit:
            renderer.poll()
            dt = meter.tick()
            t = time.perf_counter() - t0

            # 絵に渡す時刻は、検証実行 (--frames) では実時間ではなくフレーム番号から作る。
            # 呼吸 (gustEnv) と粒子感の乱数が uTime に乗っているので、実時間のままだと
            # 同じ設定でも回すたびに絵が変わり、1 軸だけ振った比較が成立しない
            step = 1.0 / max(args.fps if args.fps > 0.0 else 60.0, 1.0)
            ts, dts = (n * step, step) if args.frames else (t, dt)

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
                if vision is not None:
                    vision.submit(frame)

            if vision is not None:
                # c キーで合成を始めたら、その場でマットも作らせる。
                # 計測 (a キー) もマットを要るので、どちらかが立っていれば作る
                vision.want_matte = state.compose > 0.5 or state.palette
                d = vision.latest_depth()
                if d is not None:
                    latest_depth = d
                    renderer.update_depth(d)
                    meter.add_stage("depth", vision.depth_s)
                mt = vision.latest_matte()
                if mt is not None:
                    latest_matte = mt
                    renderer.update_matte(mt)
                    meter.add_stage("seg", vision.matte_s)
                    # マットが更新されたフレームだけ測る。毎フレーム Oklab に
                    # 変換する必要はない (統計は EMA で均されている)。
                    # マットは 1〜2 フレーム古いが、領域の代表値には影響しない
                    if state.palette and frame is not None:
                        s = time.perf_counter()
                        pal_in.update(frame, mt)
                        meter.add_stage("palette", time.perf_counter() - s)
            if not state.palette:
                pal_in.reset()
                pal_out.reset()
            if not state.stabilize and stab.subj_chroma != 1.0:
                stab.update(None)          # 切ったときも同じ変化率で 1.0 へ帰す

            energy = flowf.energy if flowf is not None else 0.0
            energy = max(energy, args.energy_floor)
            seed = flowf.centroid if flowf is not None else (0.5, 0.5)
            wind = flowf.wind if flowf is not None else (-1.0, 0.12)
            # 人が動いた分だけ風が進む
            adv += gust_env(ts) * (0.30 + 2.5 * energy) * dts

            if state.hud and t - hud_at > 0.25:
                hud_at = t
                src = state.depth_kind if vision is not None else "off"
                renderer.update_hud(
                    build_hud(renderer.render_w, renderer.render_h,
                              hud_lines(meter, state, energy, src,
                                        pal_in.stats, pal_out.stats, stab))
                )
            elif not state.hud:
                renderer.update_hud(None)

            renderer.draw({
                "uTime": ts,
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
                "uInject": state.inject,
                "uMemory": state.memory,
                "uIdleWind": state.idle_wind,
                "uOklab": state.oklab,
                "uSubjChroma": stab.subj_chroma,
                "uSplitScale": stab.split_scale,
                "uCompose": state.compose,
                "uStand": state.stand,
                "uWhite": tuple(float(x) for x in wb.gain),
            })
            meter.add_latency(time.perf_counter() - stamp)

            # 描いた結果を測る。読み戻しは GPU を止めるので 2Hz に絞る。
            # 絵は前回の描画結果、マットは数フレーム前のもので、厳密には
            # 同一時刻ではない。領域の代表値としては許容範囲
            if (state.palette and latest_matte is not None
                    and t - pal_out_at > 1.0 / PAL_OUT_HZ):
                pal_out_at = t
                s = time.perf_counter()
                pal_out.update(renderer.read_scene(), latest_matte)
                meter.add_stage("palette_out", time.perf_counter() - s)
                # 制御は測った直後にだけ動かす。描画のたびに動かすと、
                # 同じ観測に対して何度も補正を掛けることになる
                if state.stabilize:
                    stab.update(pal_out.stats)

            ready = ((vision is None or latest_depth is not None or args.depth == "off")
                     and (not state.palette or pal_out.stats is not None))

            if state.shot:
                state.shot = False
                stem = f"{datetime.now():%Y%m%d-%H%M%S}"
                p = renderer.screenshot(Path("shots") / f"{stem}.png")
                # どのスクショがどの設定だったかを必ず残す
                side = p.with_suffix(".json")
                side.write_text(json.dumps(shot_meta(), indent=2) + "\n", encoding="utf-8")
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

            # 検証実行では「準備できてから」数える。深度もマットも非同期なので、
            # 冷えた状態で数え始めると、比較のつもりの 2 枚が別の状態の絵になる
            # (実際 sweep の 1 枚目と 2 枚目だけ計測が空だった)。
            # n は絵に渡す時刻でもあるので、揃えると絵の位相も揃う
            if args.frames and args.wait_ready and not ready:
                if t < args.ready_timeout:
                    continue
                if not warned:
                    warned = True
                    print(f"準備が {args.ready_timeout:.0f}s で整いませんでした。"
                          f"depth={latest_depth is not None} palette={pal_out.stats is not None}")
            n += 1
            if args.frames and n >= args.frames:
                break

            # 描画の上限。速いこと自体は目的ではないので、GPU と電力を推論に残す。
            #
            # 注: 当初これを「深度が更新されなくなる」問題の対策として入れたが、
            # **その仮説は測って否定された** (40fps でも無制限の 120fps でも同様に起きた)。
            # 真因はワーカースレッドが 2 本あったことで、vision.py に統合して解決した。
            # この上限自体は無害で有用なので残してある。
            if args.fps > 0.0:
                spare = (1.0 / args.fps) - (time.perf_counter() - meter._last)
                if spare > 0.0005:
                    time.sleep(spare)

        if args.shot:
            p = renderer.screenshot(args.shot)
            side = p.with_suffix(".json")
            side.write_text(json.dumps(shot_meta(), indent=2) + "\n", encoding="utf-8")
            print(f"saved {p} + {side.name}")
        el = time.perf_counter() - t0
        # 単体のレイテンシではなく「描画と GPU を共有した状態で深度の場が
        # 毎秒何回更新されたか」が体験に効く量。捨てた入力フレーム数も一緒に出す。
        dhz = (vision.done / el) if vision is not None else 0.0
        drop = vision.dropped if vision is not None else 0
        print(
            f"frames={n} elapsed={el:.1f}s fps={meter.fps:.1f} e2e={meter.latency_ms:.1f}ms "
            f"flow={meter.ms('flow'):.1f}ms depth_infer={meter.ms('depth'):.1f}ms "
            f"depth_updates={dhz:.1f}Hz dropped={drop}"
        )
        # 検証実行 (--frames/--shot) でも数値が標準出力に残るようにする。
        # HUD を目で読んで書き写す経路しか無いと、写し間違いと再現不能が起きる
        if state.palette:
            for label, m in (("in ", pal_in), ("out", pal_out)):
                v = json.dumps(m.stats.as_dict()) if m.stats else "なし(人物が見えていません)"
                print(f"palette {label} {v}")
            print(f"palette cost in={meter.ms('palette'):.2f}ms out={meter.ms('palette_out'):.2f}ms")
            if state.stabilize:
                print(f"stabilize  人物の彩度 x{stab.subj_chroma:.3f}  分割 x{stab.split_scale:.3f}")
    finally:
        cam.stop()
        if vision is not None:
            vision.stop()
        renderer.close()


if __name__ == "__main__":
    main()
