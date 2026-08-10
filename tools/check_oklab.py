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
    """uniform を参照しない色空間の関数群だけを brush.frag から取り出す。"""
    src = Path("ombrelle/gl/shaders/brush.frag").read_text(encoding="utf-8")
    a = src.index("float cbrt_s")
    b = src.index("// RGB ⇄ 平面")     # ここから先は uOklab を見るので切り離す
    return src[a:b]


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
              "  o = vec4(uInv > 0.5 ? oklabToRgb(c) : rgbToOklab(c), 1.0); }"
        ),
    )

    def run(data: np.ndarray, inverse: bool) -> np.ndarray:
        h = len(data)
        tex = ctx.texture((1, h), 3, data=np.ascontiguousarray(data.reshape(h, 1, 3)), dtype="f4")
        tex.use(0)
        prog["uSrc"] = 0
        prog["uInv"] = 1.0 if inverse else 0.0
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
        ("rgbToOklab", np.abs(run(rgb, False) - srgb_to_oklab(rgb)).max()),
    ]
    lab = srgb_to_oklab(rgb).astype(np.float32)
    checks.append(("oklabToRgb", np.abs(run(lab, True) - oklab_to_srgb(lab)).max()))
    checks.append(("GLSL 往復", np.abs(run(lab, True) - rgb).max()))

    ok = True
    for name, err in checks:
        mark = "OK " if err < TOL else "NG "
        ok &= err < TOL
        print(f"{mark}{name:<12} 最大誤差 {err:.2e}  (許容 {TOL:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
