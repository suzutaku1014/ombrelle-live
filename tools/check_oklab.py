"""brush.frag の Oklab 関数を切り出して実行し、ombrelle/oklab.py と数値を照合する。

なぜ要るか:

色空間の実装が 2 つある。測定 (Python) と描画 (GLSL) で、これが食い違うと
「測った数値」と「描いた絵」が別の空間の話になり、A/B が意味を失う。
係数を目で見比べても分からない種類のズレが実際に出た:

  伝達関数の**色域外の扱い**が違っていた (Python は符号を保存せず、GLSL は保存)。
  sRGB の [0,1] の内側では完全に一致するので、通常の絵では気づかない。
  ところがシェーダの色はグレーディングで 1.0 を超え、色彩分割で負にもなる。
  そこだけで最大 0.65 ずれていた。

だから範囲外の入力を必ず含めて照合する。

    uv run python tools/check_oklab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import moderngl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ombrelle.oklab import oklab_to_srgb, srgb_to_oklab  # noqa: E402

TOL = 1e-4       # float32 の往復として妥当な範囲


def _extract_functions() -> str:
    """uniform を参照しない部分だけを brush.frag から取り出す。

    変換関数 (cbrt_s 〜 oklabToRgb) と、その後ろの定数 + gamutMap。
    間に挟まる toPlane/fromPlane と、cref() 以降は uOklab を見るので外す。
    """
    src = Path("ombrelle/gl/shaders/brush.frag").read_text(encoding="utf-8")
    conv = src[src.index("float cbrt_s"):src.index("// RGB ⇄ 平面")]
    gamut = src[src.index("const float OK_C"):src.index("float cref()")]
    return conv + gamut


def main() -> int:
    ctx = moderngl.create_standalone_context()
    prog = ctx.program(
        vertex_shader=(
            "#version 330\nvoid main(){ gl_Position = vec4("
            "(gl_VertexID==1)?3.0:-1.0, (gl_VertexID==2)?3.0:-1.0, 0.0, 1.0); }"
        ),
        fragment_shader=(
            "#version 330\nuniform sampler2D uSrc;\nuniform float uInv;\nout vec4 o;\n"
            + _extract_functions()
            + "void main(){ vec3 c = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0).rgb;"
              "  o = vec4(uInv > 1.5 ? gamutMap(c)"
              "         : uInv > 0.5 ? oklabToRgb(c) : rgbToOklab(c), 1.0); }"
        ),
    )

    def run(data: np.ndarray, mode: float) -> np.ndarray:
        """mode 0=rgbToOklab / 1=oklabToRgb / 2=gamutMap"""
        h = len(data)
        tex = ctx.texture((1, h), 3, data=np.ascontiguousarray(data.reshape(h, 1, 3)), dtype="f4")
        tex.use(0)
        prog["uSrc"] = 0
        prog["uInv"] = mode
        out = ctx.texture((1, h), 4, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[out])
        fbo.use()
        ctx.viewport = (0, 0, 1, h)
        ctx.vertex_array(prog, []).render(moderngl.TRIANGLES, vertices=3)
        return np.frombuffer(fbo.read(components=4, dtype="f4"),
                             dtype=np.float32).reshape(h, 4)[:, :3]

    rng = np.random.default_rng(1)
    n = 512
    rgb = np.concatenate([
        rng.random((n, 3)),                     # 色域内
        rng.uniform(-0.2, 1.6, (n, 3)),         # 色域外 (ここで一度ズレた)
    ]).astype(np.float32)

    checks = [
        ("rgbToOklab", np.abs(run(rgb, 0.0) - srgb_to_oklab(rgb)).max()),
    ]
    lab = srgb_to_oklab(rgb).astype(np.float32)
    checks.append(("oklabToRgb", np.abs(run(lab, 1.0) - oklab_to_srgb(lab)).max()))
    checks.append(("GLSL 往復", np.abs(run(lab, 1.0) - rgb).max()))

    # ---- 色域マッピング ----
    # 「L と h を保って C だけ落とす」が守られているか。成分ごとの clamp なら
    # ここで色相がずれるので、この 3 つが同時に通れば naive clip ではないと言える
    out = run(rgb, 2.0)
    lab_i, lab_o = srgb_to_oklab(rgb), srgb_to_oklab(out)
    floor_l = float(_extract_functions().split("FLOOR_L = ")[1].split(";")[0])
    checks.append(("色域内に収まる", max(0.0, float(np.max(np.abs(out - np.clip(out, 0.0, 1.0)))))))
    checks.append(("明度 L の保存", float(np.abs(
        lab_o[:, 0] - np.clip(lab_i[:, 0], floor_l, 1.0)).max())))
    # 色相は彩度がある画素だけ見る。無彩色に色相は無く、色域外が極端で
    # C をほぼ 0 まで落とした画素は出力側の色相が数値ノイズになる
    Ci = np.hypot(lab_i[:, 1], lab_i[:, 2])
    Co = np.hypot(lab_o[:, 1], lab_o[:, 2])
    sel = (Ci > 0.02) & (Co > 0.02)
    hi_, ho_ = np.arctan2(lab_i[sel, 2], lab_i[sel, 1]), np.arctan2(lab_o[sel, 2], lab_o[sel, 1])
    dh = np.abs(hi_ - ho_) % (2 * np.pi)
    checks.append(("色相 h の保存", float(np.minimum(dh, 2 * np.pi - dh).max())))

    ok = True
    for name, err in checks:
        mark = "OK " if err < TOL else "NG "
        ok &= err < TOL
        print(f"{mark}{name:<14} 最大誤差 {err:.2e}  (許容 {TOL:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
