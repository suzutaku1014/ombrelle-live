#version 330 core
// 内部解像度で描いた絵をウィンドウへ引き伸ばす。HUD を上に重ねる。
out vec4 fragColor;
uniform vec2 uRes;        // ウィンドウ(フレームバッファ)解像度
uniform sampler2D uScene; // 内部解像度の描画結果
uniform sampler2D uHud;   // RGBA の文字オーバーレイ
uniform float uHudOn;

void main() {
    vec2 q = gl_FragCoord.xy / uRes;
    vec3 col = texture(uScene, q).rgb;
    if (uHudOn > 0.5) {
        // HUD テクスチャは OpenCV 由来なので y を反転して読む
        vec4 h = texture(uHud, vec2(q.x, 1.0 - q.y));
        col = mix(col, h.rgb, h.a);
    }
    fragColor = vec4(col, 1.0);
}
