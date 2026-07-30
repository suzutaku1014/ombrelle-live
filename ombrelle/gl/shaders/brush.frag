#version 330 core
// ombrelle-live — reference/ombrelle_v11_3.frag のカメラ翻案
//
// 原典との対応:
//   renderScene()  … 手続き生成の丘/空  →  scenePainterly() = カメラ + 印象派グレーディング
//   depth (草の帯) … 稜線からの距離     →  単眼深度推定の出力 (uDepth)
//   windF()        … 解析的な風の場     →  人の動きのオプティカルフロー (uFlow)
//   SEAT           … 空席(不在の一点)   →  動きのエネルギー重心 (uSeed)
//
// 原典の検証済みの法則はそのまま維持する:
//   「絵の層のパラメータ(筆サイズ/色相/彩度)はdepthの関数にする。ただし目が理屈に勝つ。」
//
// 座標系の約束:
//   q      … 0..1、**y は上向き**(GL 流儀)
//   テクスチャ … OpenCV 由来なので y は下向き → img() で反転して読む
//   フローの y … 画像座標では下が正 → flowAt() で符号を反転して GL 流儀に揃える

out vec4 fragColor;

uniform vec2  uRes;        // 内部描画解像度
uniform float uTime;
uniform float uAdv;        // ∫ (呼吸 × 動きのエネルギー) dt — CPU側で積算
uniform float uView;       // 0=筆触 1=生カメラ 2=深度 3=フローの場

uniform sampler2D uCam;    // RGB8 + mipmap
uniform sampler2D uDepth;  // R16F  0=遠 1=近 に正規化済み
uniform sampler2D uFlow;   // RG16F 画面幅を1とした「1フレームあたりの移動量」

uniform float uHasDepth;   // 0=未接続(ダミー深度を使う) 1=実測
uniform float uHasFlow;
uniform float uFlowGain;   // フローを O(1) に持ち上げる利得
uniform float uCamLod;     // カメラの mip レベル: 一筆が代表する面積分だけ先にぼかす
uniform vec2  uSeed;       // 動きの重心 (0..1, y上向き)
uniform float uSeedDepth;  // 重心位置の深度
uniform vec2  uWind;       // 画面全体の代表的な流れの向き (正規化前)
uniform float uEnergy;     // フローのエネルギー 0..1 くらい
uniform float uPaintMix;   // 0=グレーディングのみ 1=完全に絵

#define PETALS 24

const vec3 HAZE_COOL = vec3(0.86, 0.87, 0.94);
const vec3 PINK_HAZE = vec3(1.03, 0.86, 0.88);
const vec3 SHADOW_V  = vec3(0.30, 0.32, 0.48);   // 影は黒ではなく青紫
const vec3 LIGHT_W   = vec3(1.00, 0.95, 0.84);   // 光は暖色へ
const vec3 LW        = vec3(0.299, 0.587, 0.114);

// ---------------------------------------------------------------- 原典から無改変
float hash21(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float hash11(float p){ p = fract(p*0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
// 注: 元の名前 noise2 は desktop GLSL の予約済み組み込み関数と衝突するため改名
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
vec3 hueRotate(vec3 c, float a){
  const vec3 k = vec3(0.5773503);
  float ca = cos(a), sa = sin(a);
  return c*ca + cross(k, c)*sa + k*dot(k, c)*(1.0-ca);
}
// 呼吸 0.22Hz + 45秒ごとに息を呑む。絵の生命感の根拠なので消さない
float gustEnv(float t){
  float br = 0.62 + 0.38*sin(6.28318*0.22*t + 0.8*sin(6.28318*0.031*t));
  float ph = mod(t, 45.0);
  float pf;
  if(ph > 43.0) pf = 1.0 - smoothstep(43.0, 44.2, ph);
  else          pf = smoothstep(0.0, 0.9, ph);
  return max(br*pf, 0.05);
}

// ---------------------------------------------------------------- 入力の読み取り
vec2 img(vec2 q){ return vec2(q.x, 1.0 - q.y); }   // GL(y上) → 画像(y下)

vec3 camAt(vec2 q, float lod){
  return textureLod(uCam, img(clamp(q, 0.0, 1.0)), lod).rgb;
}
// 0=遠 1=近。未接続時は「下が手前」というダミーで筆触だけ先に検証できるようにする
float depthAt(vec2 q){
  if (uHasDepth < 0.5) return clamp(1.0 - q.y, 0.0, 1.0);
  return texture(uDepth, img(clamp(q, 0.0, 1.0))).r;
}
vec2 flowAt(vec2 q){
  if (uHasFlow < 0.5) return vec2(0.0);
  vec2 f = texture(uFlow, img(clamp(q, 0.0, 1.0))).rg;
  return vec2(f.x, -f.y) * uFlowGain;   // 画像は y 下向き → GL の y 上向きへ
}

// ---------------------------------------------------------------- 風の場
// 原典 windF() の置き換え。人の動き(実測フロー)が主、原典の呼吸が従。
//
// 原典の「全体で上向きの弧を描く風」は、動きが無いときの下地として残す。
// 実測フローの向きへ弧を寄せるので、人が動くと下地ごとその向きへ傾く。
vec2 windDirBase(vec2 q){
  vec2 arc = normalize(vec2(-1.0, 0.12 + 0.70*max(q.x - 0.25, 0.0)));
  return normalize(mix(arc, normalize(uWind + vec2(1e-4)), 0.45));
}
vec2 windF(vec2 q, float t){
  float turb = 0.7 + 0.3*fbm2(q*6.0 - vec2(uAdv*1.3, uAdv*0.16));
  vec2 f = flowAt(q) * turb;
  // 動きが無いときも絵が完全に死なないよう、呼吸だけの微風を残す(主役ではない)
  float g = gustEnv(t) * 0.06 * (1.0 - smoothstep(0.010, 0.10, uEnergy));
  return f + windDirBase(q) * g * turb;
}

// ---------------------------------------------------------------- 印象派グレーディング
// 写真に筆触を乗せただけでは「フィルタ」に見える。印象派に見せる第一原理は
// **黒を使わないこと**。暗部は青紫へ、明部は暖色へ振り分けて色で明暗を作る。
vec3 grade(vec3 c){
  float l = dot(c, LW);
  float sh = 1.0 - smoothstep(0.00, 0.38, l);
  c = mix(c, mix(c, SHADOW_V, 0.60), sh);
  float hi = smoothstep(0.60, 1.00, l);
  c = mix(c, mix(c, LIGHT_W, 0.45), hi);
  return c;
}

// 空気遠近法。原典と同形。遠いものは冷たい白へ溶け、稜線の一線だけピンクに転ぶ
vec3 aerial(vec3 c, float d, vec2 q){
  float asp = uRes.x / uRes.y;
  vec2 rd = normalize(vec2((q.x - 0.5)*asp, q.y - 0.24) + vec2(1e-4));
  float sunAmt = pow(max(dot(rd, normalize(vec2(0.88, 0.30))), 0.0), 7.0);
  vec3 hazeC = mix(HAZE_COOL, vec3(0.97, 0.97, 0.96), clamp(sunAmt*0.8, 0.0, 1.0));
  float distP = mix(2.0, 0.15, pow(clamp(d, 0.0, 1.0), 0.7));
  c = mix(c, hazeC, 1.0 - exp(-distP*0.32));
  c = mix(c, PINK_HAZE, 0.12*pow(1.0 - clamp(d, 0.0, 1.0), 4.0));
  return c;
}

// ---------------------------------------------------------------- 花びら
// 原典は空席(SEAT)から生まれ風下へ散った。ここでは「動きの重心」から生まれる。
// 生まれた瞬間は手前、時間とともに奥へ沈む → 人物の背後へ回り込む(オクルージョン)。
vec3 petals(vec3 col, vec2 q, float sceneD, float t){
  float asp = uRes.x / uRes.y;
  float born = smoothstep(0.008, 0.080, uEnergy);
  if (born < 0.002) return col;
  vec2 w = normalize(uWind + vec2(1e-4, 1e-4));
  for (int i = 0; i < PETALS; i++){
    float fi = float(i);
    float h1 = hash11(fi*7.31), h2 = hash11(fi*3.17), h3 = hash11(fi*9.53);
    float life = 5.0 + 4.0*h3;
    float lt = mod(t - h1*97.0, life);
    float pr = lt / life;

    vec2 pp = uSeed + vec2((h2 - 0.5)*0.17, (h3 - 0.5)*0.17);
    pp += w * (0.50*(0.7 + 0.6*h1)) * pr;      // 風下へ流れる
    pp += vec2(0.0, 0.16*pr*pr - 0.02*pr);     // 舞い上がる
    pp.x += 0.040*sin(lt*1.1 + fi);
    pp.y += 0.020*sin(lt*1.7 + fi*2.0);

    float rot = lt*(0.6 + 0.5*h3) + fi;
    float ca = cos(rot), sa = sin(rot);
    vec2 d2 = vec2((q.x - pp.x)*asp, q.y - pp.y);
    vec2 dr = vec2(ca*d2.x - sa*d2.y, sa*d2.x + ca*d2.y);
    float tumble = 0.40 + 0.60*abs(sin(lt*2.1 + fi*1.3));
    float sx = 0.011 + 0.008*h3, sy = sx*0.47*tumble;
    float g2 = exp(-0.5*(dr.x*dr.x/(sx*sx) + dr.y*dr.y/(sy*sy)));
    if (g2 < 0.002) continue;

    // 深度: 生まれた場所の深度から、奥へ沈んでいく
    float pd = mix(uSeedDepth, uSeedDepth*0.30, smoothstep(0.10, 0.80, pr));
    // 手前にある物体(=sceneD が大きい)より奥なら隠れる
    float vis = (uHasDepth < 0.5) ? 1.0 : smoothstep(pd + 0.06, pd - 0.06, sceneD);

    float warmth = exp(-dot(pp - uSeed, pp - uSeed)*30.0);
    vec3 pCol = mix(vec3(0.98, 0.72, 0.78), vec3(1.05, 0.88, 0.62), warmth);
    col = mix(col, pCol, clamp(g2, 0.0, 1.0)*0.85*sin(3.14159*pr)*born*vis);
  }
  return col;
}

// ---------------------------------------------------------------- 物理の層
// 原典の renderScene() に相当。一筆が代表する「現実の一点の色」。
vec3 scenePainterly(vec2 q, float t, float lod){
  float d = depthAt(q);
  vec3 c = camAt(q, lod);
  c = grade(c);
  c = aerial(c, d, q);
  // 動きが凝った場所は暖かく光る(原典の「空席の光溜まり」の意味を転換)
  float asp = uRes.x / uRes.y;
  vec2 dS = vec2((q.x - uSeed.x)*asp, q.y - uSeed.y);
  float pool = exp(-dot(dS, dS)/(2.0*0.085*0.085)) * smoothstep(0.01, 0.12, uEnergy);
  c = mix(c, vec3(1.00, 0.94, 0.84), pool*0.26);
  c = petals(c, q, d, t);
  return c;
}

// ---------------------------------------------------------------- 筆の向き
// 向きは「方向」ではなく「軸」なので、角度を直接 mix すると ±π で破綻する。
// 2倍角ベクトルで補間する(構造テンソルと同じ考え方)。
float mixOrient(float a, float b, float w){
  vec2 va = vec2(cos(2.0*a), sin(2.0*a));
  vec2 vb = vec2(cos(2.0*b), sin(2.0*b));
  vec2 v  = mix(va, vb, w);
  if (dot(v, v) < 1e-8) return a;
  return 0.5*atan(v.y, v.x);
}

// 動いている所は動きに沿った筆、止まっている所は等深線に沿った筆(形をなぞる)
float strokeAngle(vec2 q, vec2 p, float t){
  vec2 f = windF(q, t);
  float flowA = atan(f.y, f.x);

  float e = 2.0/uRes.y;
  vec2 gd = vec2(depthAt(q + vec2(e,0)) - depthAt(q - vec2(e,0)),
                 depthAt(q + vec2(0,e)) - depthAt(q - vec2(0,e)));
  // 等深線 = 勾配に直交する向き
  float contourA = (dot(gd, gd) > 1e-9) ? atan(gd.x, -gd.y) : 1.5708;

  float w = smoothstep(0.10, 0.75, length(f));
  return mixOrient(contourA, flowA, w) + (fbm2(p*1.6) - 0.5)*0.55;
}

void main(){
  float asp = uRes.x / uRes.y;
  vec2  q = gl_FragCoord.xy / uRes;
  vec2  p = vec2((q.x - 0.5)*asp, q.y - 0.5);
  float t = uTime;
  vec3  col;

  if (uView > 2.5) {
    // --- フローの場を矢印で見る(原典の矢印描画を流用) ---
    col = mix(camAt(q, 3.0), vec3(0.93, 0.94, 0.96), 0.62);
    vec2 pa = vec2(q.x*asp, q.y);
    vec2 c  = (floor(pa/0.055) + 0.5)*0.055;
    vec2 w  = windF(vec2(c.x/asp, c.y), t);
    float m = length(w);
    vec2 u2 = w/max(m, 1e-4);
    float len = 0.021*clamp(m/0.8, 0.22, 1.0);   // 向きが読めるよう最小長を確保
    vec2 d = pa - c;
    float tt = clamp(dot(d, u2)/max(len, 1e-5), -1.0, 1.0);
    float dd = length(d - u2*tt*len);
    float line = 1.0 - smoothstep(0.0011, 0.0028, dd);
    vec3 ink = mix(vec3(0.16, 0.25, 0.38), vec3(0.78, 0.30, 0.12), smoothstep(0.2, 1.0, tt));
    col = mix(col, ink, line*0.92);
    // 動きの重心
    vec2 dS = vec2((q.x - uSeed.x)*asp, q.y - uSeed.y);
    float ring = 1.0 - smoothstep(0.0012, 0.0032, abs(length(dS) - 0.075));
    col = mix(col, vec3(0.72, 0.42, 0.18), ring*0.7);

  } else if (uView > 1.5) {
    // --- 深度マップ(近いほど明るい) ---
    float d = depthAt(q);
    col = mix(vec3(0.07, 0.10, 0.22), vec3(1.00, 0.93, 0.70), d);
    col = mix(col, vec3(0.85, 0.30, 0.25), smoothstep(0.80, 1.0, d)*0.35);

  } else if (uView > 0.5) {
    // --- 生カメラ(比較用の素の写真) ---
    col = camAt(q, 0.0);

  } else {
    // ---- 楕円筆触 + 等輝度色彩分割(絵側の層) ----
    float d0 = depthAt(q);
    float ang = strokeAngle(q, p, t);
    vec2 sdir = vec2(cos(ang), sin(ang));
    vec2 perp = vec2(-sdir.y, sdir.x);
    vec2 s = vec2(dot(p, sdir), dot(p, perp));

    // タッチの遠近法: 手前ほど大きい筆
    float szG = mix(0.50, 1.30, clamp(d0, 0.0, 1.0));

    vec2 pitch = vec2(0.019, 0.0062);
    vec2 base = floor(s/pitch);
    float best = -1.0;
    vec2 bestC = (base + 0.5)*pitch;
    vec2 bestId = base;
    float fuzz = (vnoise(p*52.0) - 0.5)*0.55;
    // 5x5 近傍から1枚の楕円を勝たせる = 一つの楕円は一色 = 一筆
    for (int j = -2; j <= 2; j++)
    for (int i = -2; i <= 2; i++){
      vec2 cid = base + vec2(float(i), float(j));
      vec2 rnd = vec2(hash21(cid + 1.7), hash21(cid + 7.3));
      vec2 ctr = (cid + 0.5 + (rnd - 0.5)*0.9)*pitch;
      float sz = (0.65 + 1.00*hash21(cid + 8.8))*szG;
      float aa = 0.0155*sz;
      float bb = 0.0046*(0.7 + 0.8*hash21(cid + 4.4))*min(sz, 1.6);
      float rr2 = (hash21(cid + 2.2) - 0.5)*0.55;
      float cr2 = cos(rr2), sr2 = sin(rr2);
      vec2 dd = s - ctr;
      vec2 dr = vec2(cr2*dd.x - sr2*dd.y, sr2*dd.x + cr2*dd.y);
      float q2 = dr.x*dr.x/(aa*aa) + dr.y*dr.y/(bb*bb);
      q2 *= 1.0 + fuzz;
      float pr2 = hash21(cid + 5.1);
      if (q2 < 1.0 && pr2 > best){ best = pr2; bestC = ctr; bestId = cid; }
    }
    vec2 pc = sdir*bestC.x + perp*bestC.y;
    vec2 qc = vec2(pc.x/asp + 0.5, pc.y + 0.5);

    // 一筆が覆う面積の分だけ先にぼかしてから1点サンプルする(写真の高周波を落とす)
    float lod = uCamLod + log2(max(szG, 0.25));
    col = scenePainterly(qc, t, lod);

    float d = depthAt(qc);
    float l0 = dot(col, LW);
    // 色彩分割: 三族に量子化(青緑/緑/黄)。非対称で、赤方向には決して届かない
    float h1r = hash21(bestId + 3.1);
    float h2r = hash21(bestId + 17.9);
    float fam = floor(h1r*3.0) - 1.0;
    float famG = ((fam < 0.0) ? -0.58 : fam*0.16) + (h2r - 0.5)*0.12;
    famG *= 0.55 + 0.45*d;              // 霞んだ遠景は回転させない(土色の発生源)
    col = hueRotate(col, famG);

    float satG = 1.18 + 0.28*d;         // 彩度ブーストも手前だけ
    col = mix(vec3(dot(col, LW)), col, satG + 0.30*(hash21(bestId + 13.0) - 0.5));
    col *= l0 / max(dot(col, LW), 1e-3);   // 輝度は保存
    col *= 0.985 + 0.030*hash21(bestId + 9.7);
    col = clamp(col, 0.0, 1.6);

    // 絵の強さを 0..1 で混ぜられるようにしておく(比較用スライダ)
    if (uPaintMix < 0.999){
      vec3 raw = grade(camAt(q, 0.0));
      col = mix(raw, col, uPaintMix);
    }
  }

  // 縁はごくわずかに沈む(包みの暗がり)
  if (uView < 0.5 || uView > 2.5) col *= 1.0 - 0.10*smoothstep(0.60, 1.25, length(p));

  // 粒子感・黒禁止
  col += (hash21(gl_FragCoord.xy*0.7 + mod(t, 10.0)) - 0.5)*0.012;
  col = max(col, vec3(0.08, 0.09, 0.11));

  fragColor = vec4(col, 1.0);
}
