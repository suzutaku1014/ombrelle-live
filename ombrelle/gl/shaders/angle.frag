#version 330 core
// 筆の向きの場を先に焼いておくパス。
//
// 格子を固定したことで、各画素は 7x7 のセルそれぞれについて「そのセル中心の向き」を
// 知る必要がある。向きの計算は構造テンソル + fbm で重いので、画素あたり 49 回
// 評価すると 130fps → 21fps まで落ちた (実測)。
//
// 向きの場は本質的に低周波なので、低解像度に一度描いてテクスチャから読めばよい。
//
// 出力は角度そのものではなく **2 倍角ベクトル (cos2θ, sin2θ)**。
// 筆の向きは「方向」ではなく「軸」で、角度を線形補間すると ±π で破綻するため。
// 2 倍角ベクトルなら補間もフィルタリングも素直に効く。
//
// 注意: このパスを切り出したとき (描画の 2 パス化) に、元の実装が持っていた
// **一貫性ゲート (coh) と structW の閾値**を落としていた。そのせいで平らな面の
// ノイズを輪郭と取り違え、筆が回り続けていた。両方を戻してある。
// 向きの実装をここ以外に置かないこと (二重に持つと必ず片方が置き去りになる)。

out vec4 fragColor;

uniform vec2  uRes;        // このパスの解像度(低い)
uniform float uTime;
uniform float uAdv;
uniform float uBrush;
uniform sampler2D uDepth;
uniform sampler2D uFlow;
uniform float uHasDepth;
uniform float uHasFlow;
uniform float uFlowGain;
uniform vec2  uWind;
uniform float uEnergy;
uniform float uIdleWind;   // 静止時の微風。0=誰も動かなければ筆も止まる / 1=従来

float hash21(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  f = f*f*(3.0-2.0*f);
  return mix(mix(hash21(i), hash21(i+vec2(1,0)), f.x),
             mix(hash21(i+vec2(0,1)), hash21(i+vec2(1,1)), f.x), f.y);
}
float fbm2(vec2 p){
  float a = 0.5, s = 0.0;
  for(int i=0;i<4;i++){ s += a*vnoise(p); p = p*2.02 + vec2(13.7, 7.1); a *= 0.5; }
  return s;
}
float gustEnv(float t){
  float br = 0.62 + 0.38*sin(6.28318*0.22*t + 0.8*sin(6.28318*0.031*t));
  float ph = mod(t, 45.0);
  float pf = (ph > 43.0) ? 1.0 - smoothstep(43.0, 44.2, ph) : smoothstep(0.0, 0.9, ph);
  return max(br*pf, 0.05);
}
vec2 img(vec2 q){ return vec2(q.x, 1.0 - q.y); }
float depthAt(vec2 q){
  if (uHasDepth < 0.5) return clamp(1.0 - q.y, 0.0, 1.0);
  return texture(uDepth, img(clamp(q, 0.0, 1.0))).r;
}
vec2 flowAt(vec2 q){
  if (uHasFlow < 0.5) return vec2(0.0);
  vec2 f = texture(uFlow, img(clamp(q, 0.0, 1.0))).rg;
  return vec2(f.x, -f.y) * uFlowGain;
}
vec2 windDirBase(vec2 q){
  vec2 arc = normalize(vec2(-1.0, 0.12 + 0.70*max(q.x - 0.25, 0.0)));
  return normalize(mix(arc, normalize(uWind + vec2(1e-4)), 0.45));
}
vec2 windF(vec2 q, float t){
  float turb = 0.7 + 0.3*fbm2(q*6.0 - vec2(uAdv*1.3, uAdv*0.16));
  vec2 f = flowAt(q) * turb;
  // brush.frag の同名関数と必ず同じ式にすること (向きの場と絵で別々に動くと破綻する)
  float g = gustEnv(t) * 0.06 * uIdleWind * (1.0 - smoothstep(0.010, 0.10, uEnergy));
  return f + windDirBase(q) * g * turb;
}

// 2 倍角ベクトルどうしの補間 = 軸としての平均
vec2 axisOf(float a){ return vec2(cos(2.0*a), sin(2.0*a)); }

void main(){
  float asp = uRes.x / uRes.y;
  vec2  q = gl_FragCoord.xy / uRes;
  vec2  p = vec2((q.x - 0.5)*asp, q.y - 0.5);
  float bs = max(uBrush, 0.05);

  vec2 f = windF(q, uTime);
  float flowA = atan(f.y, f.x);

  // 深度の勾配は筆の大きさに合わせた幅で取る(細かい凹凸に反応させない)。
  // さらに 3x3 を構造テンソル(2倍角ベクトルの和)で平均する。勾配ベクトルを
  // そのまま平均すると符号が反転して打ち消し合うため、軸として平均するのが正しい。
  float e = clamp(2.0*bs/uRes.y, 1.0/uRes.y, 0.05);
  vec2 acc = vec2(0.0);
  float wsum = 0.0;
  for (int j = -1; j <= 1; j++)
  for (int i = -1; i <= 1; i++){
    vec2 o = vec2(float(i), float(j)) * e;
    vec2 gd = vec2(depthAt(q + o + vec2(e,0)) - depthAt(q + o - vec2(e,0)),
                   depthAt(q + o + vec2(0,e)) - depthAt(q + o - vec2(0,e)));
    float m = length(gd);
    if (m > 1e-6){ acc += axisOf(atan(gd.x, -gd.y)) * m; wsum += m; }
  }
  float gm = wsum / 9.0;
  float contourA = (dot(acc, acc) > 1e-12) ? 0.5*atan(acc.y, acc.x) : 1.5708;

  // 向きが揃っている(テンソルの長さが重みの和に近い)ほど輪郭に従わせる。
  //
  // **ノイズ除去の本体はここ。** 本物の輪郭は近傍で勾配の向きが揃うが、
  // ノイズ由来の勾配は近傍でばらばらなので打ち消し合って coh が下がる。
  // 大きさだけを見る閾値ではノイズと弱い輪郭を区別できない。
  float coh = (wsum > 1e-6) ? length(acc)/wsum : 0.0;

  // 平らな壁や机では深度の勾配がほぼ 0 になる。そこで向きを定数に倒すと
  // 筆が一斉に揃い、大きな筆では帯状に融合して「塗り」になる。
  // 根拠が無い場所は低周波で散らす。ただし全周(2π)振ると渦模様になるので、
  // ゆるい斜めの筋目を基本にして振れ幅は ±0.8 rad に留める。
  float hatchA = 0.9 + (fbm2(p*1.1/bs + 11.0) - 0.5)*1.6;
  // 大きさの閾値は元のまま。ここを上げるとノイズも落ちるが、顔のような
  // ゆるい陰影の勾配まで切り捨てて、筆が形をなぞらなくなる (試して却下した)。
  // ノイズと弱い輪郭を分けるのは大きさではなく coh の仕事。
  float structW = smoothstep(0.0015, 0.0150, gm) * smoothstep(0.35, 0.85, coh);
  vec2 base = mix(axisOf(hatchA), axisOf(contourA), structW);

  // 動いている所は動きに沿った筆、止まっている所は形をなぞる筆
  float w = smoothstep(0.10, 0.75, length(f));
  vec2 v = mix(base, axisOf(flowA), w);

  // 筆ごとのゆらぎ(低周波)
  float jit = (fbm2(p*1.6/bs) - 0.5) * 0.55;
  float a = 0.5*atan(v.y, v.x) + jit;

  fragColor = vec4(cos(2.0*a), sin(2.0*a), 0.0, 1.0);
}
