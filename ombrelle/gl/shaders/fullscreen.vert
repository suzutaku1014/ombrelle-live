#version 330 core
// 全画面を覆う1枚の四角形。頂点は -1..1 の2次元だけで足りる。
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
