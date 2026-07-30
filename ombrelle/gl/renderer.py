"""moderngl による描画。

設計の要点:
  * 筆触パスは 1 フラグメントあたり 5x5 の楕円探索 + 24 枚の花びらを回すので重い。
    Retina のフレームバッファ(2560x1440 等)で直接描くと帯域も演算も無駄になるため、
    **固定の内部解像度**の FBO に描いてからウィンドウへ引き伸ばす。
    筆のサイズは正規化座標で決まるので、内部解像度を変えても筆の大きさは変わらない
    (変わるのは鮮明さだけ)。スクリーンショットの再現性も得られる。
  * カメラテクスチャは mipmap を張る。一筆は楕円の中心 1 点しかサンプルしないので、
    フル解像度から拾うと写真の高周波がノイズとして出る。先に面積分ぼかすのが正しい。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import glfw
import moderngl
import numpy as np

SHADER_DIR = Path(__file__).parent / "shaders"


def _load(name: str) -> str:
    return (SHADER_DIR / name).read_text(encoding="utf-8")


class Renderer:
    def __init__(
        self,
        win_w: int = 1280,
        win_h: int = 720,
        render_w: int = 1280,
        render_h: int = 720,
        title: str = "ombrelle-live",
    ) -> None:
        if not glfw.init():
            raise RuntimeError("glfw の初期化に失敗しました")

        # macOS の OpenGL は 4.1 打ち止め。3.3 core + forward compat が最も素直に通る
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        self.window = glfw.create_window(win_w, win_h, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("ウィンドウを作成できませんでした")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync

        self.ctx = moderngl.create_context()
        self.render_w, self.render_h = render_w, render_h

        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")  # 全画面を覆う1枚の三角形
        self._vbo = self.ctx.buffer(quad.tobytes())

        self.brush = self.ctx.program(
            vertex_shader=_load("fullscreen.vert"),
            fragment_shader=_load("brush.frag"),
        )
        self.present = self.ctx.program(
            vertex_shader=_load("fullscreen.vert"),
            fragment_shader=_load("present.frag"),
        )
        self._vao_brush = self.ctx.vertex_array(self.brush, [(self._vbo, "2f", "in_pos")])
        self._vao_present = self.ctx.vertex_array(self.present, [(self._vbo, "2f", "in_pos")])

        self.scene_tex = self.ctx.texture((render_w, render_h), 3, dtype="f1")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.scene_tex])

        self.cam_tex: moderngl.Texture | None = None
        self.depth_tex: moderngl.Texture | None = None
        self.flow_tex: moderngl.Texture | None = None
        self.hud_tex: moderngl.Texture | None = None
        self._hud_on = False

    # ------------------------------------------------------------ テクスチャ更新
    def update_camera(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        if self.cam_tex is None or self.cam_tex.size != (w, h):
            if self.cam_tex is not None:
                self.cam_tex.release()
            self.cam_tex = self.ctx.texture((w, h), 3, dtype="f1")
            self.cam_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.cam_tex.repeat_x = False
            self.cam_tex.repeat_y = False
        self.cam_tex.write(np.ascontiguousarray(rgb))
        self.cam_tex.build_mipmaps()

    def update_depth(self, depth: np.ndarray) -> None:
        """depth: float32/float16, 0=遠 1=近 に正規化済み"""
        h, w = depth.shape[:2]
        if self.depth_tex is None or self.depth_tex.size != (w, h):
            if self.depth_tex is not None:
                self.depth_tex.release()
            self.depth_tex = self.ctx.texture((w, h), 1, dtype="f2")
            self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.depth_tex.repeat_x = False
            self.depth_tex.repeat_y = False
        self.depth_tex.write(np.ascontiguousarray(depth.astype("f2")))

    def update_flow(self, flow: np.ndarray) -> None:
        """flow: (h, w, 2) float, 画面幅を1とした1フレームあたりの移動量"""
        h, w = flow.shape[:2]
        if self.flow_tex is None or self.flow_tex.size != (w, h):
            if self.flow_tex is not None:
                self.flow_tex.release()
            self.flow_tex = self.ctx.texture((w, h), 2, dtype="f2")
            self.flow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.flow_tex.repeat_x = False
            self.flow_tex.repeat_y = False
        self.flow_tex.write(np.ascontiguousarray(flow.astype("f2")))

    def update_hud(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            self._hud_on = False
            return
        h, w = rgba.shape[:2]
        if self.hud_tex is None or self.hud_tex.size != (w, h):
            if self.hud_tex is not None:
                self.hud_tex.release()
            self.hud_tex = self.ctx.texture((w, h), 4, dtype="f1")
            self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hud_tex.write(np.ascontiguousarray(rgba))
        self._hud_on = True

    # ------------------------------------------------------------ 描画
    def draw(self, uniforms: dict[str, object]) -> None:
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.render_w, self.render_h)

        u = dict(uniforms)
        u["uRes"] = (float(self.render_w), float(self.render_h))
        u["uHasDepth"] = 1.0 if self.depth_tex is not None else 0.0
        u["uHasFlow"] = 1.0 if self.flow_tex is not None else 0.0

        if self.cam_tex is not None:
            self.cam_tex.use(0)
            self.brush["uCam"] = 0
        if self.depth_tex is not None:
            self.depth_tex.use(1)
            self.brush["uDepth"] = 1
        if self.flow_tex is not None:
            self.flow_tex.use(2)
            self.brush["uFlow"] = 2

        for k, v in u.items():
            if k in self.brush:
                self.brush[k] = v
        self._vao_brush.render(moderngl.TRIANGLES, vertices=3)

        # ウィンドウへ提示
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, fb_w, fb_h)
        self.scene_tex.use(0)
        self.present["uScene"] = 0
        self.present["uRes"] = (float(fb_w), float(fb_h))
        self.present["uHudOn"] = 1.0 if self._hud_on else 0.0
        if self.hud_tex is not None:
            self.hud_tex.use(1)
            self.present["uHud"] = 1
        self._vao_present.render(moderngl.TRIANGLES, vertices=3)

        glfw.swap_buffers(self.window)

    def screenshot(self, path: str | Path) -> Path:
        """内部解像度の FBO を読み出して保存する(ウィンドウ倍率に依存しない)"""
        raw = self.fbo.read(components=3, alignment=1)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.render_h, self.render_w, 3)
        img = np.flipud(img)  # GL は y 上向き
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return path

    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.window))

    def poll(self) -> None:
        glfw.poll_events()

    def close(self) -> None:
        glfw.terminate()
