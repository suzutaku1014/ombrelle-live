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
uniform sampler2D uAngle;  // RG16F 筆の向きの場を 2 倍角ベクトルで焼いたもの

uniform float uHasDepth;   // 0=未接続(ダミー深度を使う) 1=実測
uniform float uHasFlow;
uniform float uFlowGain;   // フローを O(1) に持ち上げる利得
uniform float uCamLod;     // カメラの mip レベル: 一筆が代表する面積分だけ先にぼかす
uniform vec2  uSeed;       // 動きの重心 (0..1, y上向き)
uniform float uSeedDepth;  // 重心位置の深度
uniform vec2  uWind;       // 画面全体の代表的な流れの向き (正規化前)
uniform float uEnergy;     // フローのエネルギー 0..1 くらい
uniform float uPaintMix;   // 0=グレーディングのみ 1=完全に絵
uniform float uHaze;       // 空気遠近の強さ
uniform float uChroma;     // 全体の彩度ブースト(写真はモネより地味なので持ち上げる)
uniform float uBrush;      // 筆の大きさ。これが「絵に見えるか」を最も強く決める
uniform float uSplit;      // 色彩分割の強さ(隣り合う筆を暖色側/寒色側へ振り分ける量)
uniform vec3  uWhite;      // 照明の色かぶり(グレーワールド)。これで割ってから絵の色を決める
uniform float uInject;     // 中性面への色の注入量。分割は平均を保存するので彩度は増えない
uniform float uMemory;     // 記憶色(肌)の保護。肌の近くだけ色相の振れ幅を抑える
uniform sampler2D uMatte;  // R16F 人物 1 / 背景 0
uniform float uHasMatte;
uniform float uCompose;    // 0=現実を絵にする  1=モネ風の風景の中へ人物を合成する
uniform float uStand;      // 合成時に人物が立つ奥行き

#define PETALS 24

const vec3 HAZE_COOL = vec3(0.86, 0.87, 0.94);
const vec3 PINK_HAZE = vec3(1.03, 0.86, 0.88);
// 影/光の色は「混ぜる色」ではなく「掛ける係数」として持つ。
// 混色だと明度まで動いて絵が褪せる (下の grade() のコメント参照)
const vec3 SHADOW_T  = vec3(0.72, 0.82, 1.18);   // 影は青紫へ倒す
const vec3 LIGHT_T   = vec3(1.10, 1.02, 0.86);   // 光は暖色へ倒す
const vec3 LW        = vec3(0.299, 0.587, 0.114);
const vec3 HAZE_WARM = vec3(1.00, 0.93, 0.82);
const vec3 SKY_ZEN   = vec3(0.46, 0.64, 0.94);   // 青は青く
const vec3 SKY_HOR   = vec3(0.99, 0.95, 0.90);   // 地平は白へ
const vec3 GRASS_SUN = vec3(0.55, 0.66, 0.26);
const vec3 GRASS_MID = vec3(0.34, 0.50, 0.28);
const vec3 GRASS_SHD = vec3(0.18, 0.31, 0.32);
const vec3 CLOUD_LIT = vec3(1.00, 0.98, 0.95);
const vec3 CLOUD_SHD = vec3(0.72, 0.76, 0.90);
const vec3 WARM      = vec3(1.00, 0.82, 0.58);   // 光の側
const vec3 COOL      = vec3(0.62, 0.75, 1.00);   // 影の側

// ---------------------------------------------------------------- 原典から無改変
float hash21(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
// 4 個の乱数を 1 回で返す (Dave Hoskins 系)。セルあたり 6 回の hash21 が
// 2 回の hash42 になる。7x7 = 49 セルを回すのでここが効く
vec4 hash42(vec2 p){
  vec4 p4 = fract(vec4(p.xyxy) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
  p4 += dot(p4, p4.wzxy + 33.33);
  return fract((p4.xxyz + p4.yzzw) * p4.zywx);
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
  vec3 c = textureLod(uCam, img(clamp(q, 0.0, 1.0)), lod).rgb;
  // 絵の色は「影は寒色・光は暖色」という相対的な決め方をしている。
  // カメラのホワイトバランスが電球色に転んでいると、そこへ暖色を重ねることになり
  // 白まで黄色くなる。照明の色かぶりを先に外してから絵の色を決める。
  return c / max(uWhite, vec3(1e-3));
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

  // 印象派の原則は「黒い絵具を使わない」であって「暗い値を使わない」ではない。
  // 影を暗い青紫と**混色**すると明度まで一緒に上がり、絵全体が褪せる
  // (実写で紙も机も中間調に潰れた)。色を掛けてから輝度を元に戻すことで、
  // **色相だけを振って明度は動かさない**。
  float sh = 1.0 - smoothstep(0.02, 0.42, l);
  float hi = smoothstep(0.55, 1.00, l);
  vec3 t = mix(vec3(1.0), SHADOW_T, sh*0.60);
  t = mix(t, LIGHT_T, hi*0.38);
  c *= t;
  c *= l / max(dot(c, LW), 1e-3);

  // 値の構造をむしろ強める。カメラの素材は絵より眠い
  float lc = clamp(l, 0.0, 1.0);
  float lS = mix(lc, lc*lc*(3.0 - 2.0*lc), 0.45);   // ゆるい S 字
  c *= lS / max(l, 1e-3);

  // 彩度は筆触パスの satG で一箇所だけ決める。ここで掛けると二重になる
  return c;
}

// 空気遠近法。原典と同形。遠いものは冷たい白へ溶け、稜線の一線だけピンクに転ぶ
vec3 aerial(vec3 c, float d, vec2 q){
  float asp = uRes.x / uRes.y;
  vec2 rd = normalize(vec2((q.x - 0.5)*asp, q.y - 0.24) + vec2(1e-4));
  float sunAmt = pow(max(dot(rd, normalize(vec2(0.88, 0.30))), 0.0), 7.0);
  vec3 hazeC = mix(HAZE_COOL, vec3(0.97, 0.97, 0.96), clamp(sunAmt*0.8, 0.0, 1.0));
  // 原典は distP = mix(2.0, 0.15, ...) だったが、それは「丘の草」の帯に対する値で、
  // 空には霞を掛けていなかった。カメラでは depth=0 が空そのものなので、
  // 同じ係数を全画面に掛けると空が灰紫に洗われる。遠側を圧縮して uHaze で調整する。
  float dd = clamp(d, 0.0, 1.0);
  float distP = mix(1.6, 0.12, pow(dd, 0.7));
  c = mix(c, hazeC, uHaze * (1.0 - exp(-distP*0.32)));
  c = mix(c, PINK_HAZE, uHaze * 0.10*pow(1.0 - dd, 4.0));
  return c;
}

// ---------------------------------------------------------------- 色彩分割
// 原典のコメントにある設計意図:
//   「色彩分割: 草は三族に量子化(青緑/緑/黄——**緑を成分に分解**)」
//
// 本物の色彩分割は「色を成分に分解して隣り合う筆に振り分け、目の中で混ぜる」こと。
// 要点は **平均すると元の色に戻る**ことで、そのために各筆は元より彩度が高くなる。
// これが「絵具を混ぜるより明るく見える」理由 (混色は彩度を落とすが、並置は落とさない)。
//
// 対立色平面 (輝度を抜いた 2 次元) で色度ベクトルを ±Δ 回すと、2 つの平均は
// cos(Δ) 倍に縮む。だから各筆を 1/cos(Δ) 倍に伸ばしておけば平均が厳密に元へ戻る。
// 色相を固定角で回す原典の方式と違い、これは元の色相が何であっても成立する。
vec3 divide(vec3 c, float ang, float sat){
  float Y = dot(c, LW);
  vec2 v = vec2(c.r - Y, c.b - Y);                 // 対立色平面の色度ベクトル
  float ca = cos(ang), sa = sin(ang);
  v = vec2(ca*v.x - sa*v.y, sa*v.x + ca*v.y) * sat;
  float R = Y + v.x;
  float B = Y + v.y;
  float G = (Y - LW.r*R - LW.b*B) / LW.g;          // 輝度は定義から保存される
  return vec3(R, G, B);
}

// パレットの圧縮。
//
// 絵具には最高彩度がある。画家は自然の彩度をそのまま写さず、**パレットの範囲へ
// 圧縮する**。低い彩度は持ち上げ、高い彩度は抑える。
//
// これをやらないと、白い壁を背景にした肌のように「一箇所だけ彩度が高い」被写体で
// 肌だけが浮く (実写で発生。入力の顔/壁の彩度比は 4.1 倍あった)。
// さらに、色彩分割の平均保存のための彩度補正 (1/shrink) は**平均**しか保証せず、
// 個々の筆の彩度に上限が無い。元が高彩度の肌に掛かるとネオンになる。
// 上限はここで一度だけ与える。
vec3 compand(vec3 c, float level, float d){
  float Y = dot(c, LW);
  vec2 v = vec2(c.r - Y, c.b - Y);
  float C = length(v);
  if (C < 1e-5) return c;
  const float CREF = 0.18;                       // パレットの基準彩度
  const float G    = 0.60;                       // <1 で圧縮 (低彩度を持ち上げ高彩度を抑える)
  float Cn = CREF * pow(C/CREF, G) * level * (0.92 + 0.20*d);
  Cn = min(Cn, CREF*2.2);                        // 絵具の上限
  v *= Cn / C;
  float R = Y + v.x, B = Y + v.y;
  return vec3(R, (Y - LW.r*R - LW.b*B)/LW.g, B);
}

// 絵具の上限。色彩分割の彩度補正はあくまで**平均**を保証する式なので、
// 個々の筆が上限を超えうる。分割の後にもう一度だけ天井を当てる。
vec3 ceilChroma(vec3 c, float cmax){
  float Y = dot(c, LW);
  vec2 v = vec2(c.r - Y, c.b - Y);
  float C = length(v);
  if (C <= cmax || C < 1e-5) return c;
  v *= cmax / C;
  float R = Y + v.x, B = Y + v.y;
  return vec3(R, (Y - LW.r*R - LW.b*B)/LW.g, B);
}

// 中性面への色の注入。
//
// divide() は色度ベクトルを回す操作なので、**中性色は回しても中性のまま**。
// 白い壁や白飛びした天井では分割が空振りする (実写で画面の大半がこれだった)。
// 画家は白い壁を「暖色と寒色」で描く。分割の前に、輝度に応じた色度を与えておく。
//   明るい面 → 暖色側、暗い面 → 寒色側
// 注入量は元の彩度が低いほど大きく、中間調で最大にする
// (実際の絵具も最高彩度は中間調にあり、白飛びと黒潰れでは彩度が落ちる)。
vec3 inject(vec3 c, float amt, float jit){
  float Y = dot(c, LW);
  vec2 v = vec2(c.r - Y, c.b - Y);
  float bias = clamp((Y - 0.45)*2.0, -1.0, 1.0);        // 明るいほど暖色へ
  float mid  = clamp(4.0*Y*(1.0 - Y), 0.0, 1.0);        // 中間調で最大
  // 元が有彩色でも温度差は残す。0 にすると顔や木で明暗の色分けが消える
  float lack = 0.25 + 0.75*(1.0 - smoothstep(0.02, 0.16, length(v)));
  // 注入の軸を筆ごとに散らす。単一の軸だと中性面が同じ 2 色の繰り返しになる
  float a0 = -0.66 + jit*0.45;                          // 暖色方向のまわり ±0.45rad
  vec2 axis = vec2(cos(a0), sin(a0));
  v += axis * bias * amt * mid * lack;
  float R = Y + v.x, B = Y + v.y;
  float G = (Y - LW.r*R - LW.b*B) / LW.g;
  return vec3(R, G, B);
}


// 前方宣言 (茎は風景より後ろで定義されるが、風景から呼ぶ)
vec4 stalks(vec2 q, float t, out float sd);

// ---------------------------------------------------------------- モネ風の風景
// reference/ombrelle_v11_3.frag の renderScene() から**風景だけ**を移植した。
// 花びらと光溜まりは移植していない。こちらには既にフロー(人の動き)で駆動する版があり、
// 二重に持つ意味がないため。
//
// 草の揺れには原典の解析的な風ではなく、**このプログラムの windF()**(= 人の動き)を使う。
// これで「人が動くと、絵の中の草と雲が動く」が成立する。
float crest(float x){
  return 0.24 + 0.10*exp(-pow((x - 0.58)/0.30, 2.0))
       + 0.008*sin(x*7.0 + 1.3) + 0.005*sin(x*17.0 + 4.0);
}
float farRidge(float x){
  return 0.252 + 0.014*sin(x*2.6 + 2.0) + 0.007*sin(x*7.3 + 1.0);
}
float cloudField(vec2 q){
  vec2 pp = vec2(q.x*2.1, q.y*3.4) + normalize(vec2(-1.0, 0.35))*uAdv*0.075;
  float f = (fbm2(pp*1.6) + 0.35*fbm2(pp*4.1 + 7.0)) / 1.35;
  vec2 d1 = vec2((q.x - 0.32)*0.60, (q.y - 0.76)*1.1);
  vec2 d2 = vec2((q.x - 0.78)*0.75, (q.y - 0.52)*1.3);
  float mass = 1.15*exp(-dot(d1,d1)*1.6) + 0.9*exp(-dot(d2,d2)*2.0);
  float veil = 0.30*smoothstep(0.30, 0.90, q.y);
  return smoothstep(0.40, 0.72, f) * (mass + veil);
}

// 風景の深度。0=遠 1=近 という本プログラムの約束に合わせる
// (原典の草の depth は「稜線=0 手前=1」で、既に同じ向きだった)
float sceneDepth(vec2 q){
  float cr = crest(q.x);
  if (q.y > cr) return 0.0;                       // 空は最遠
  return clamp((cr - q.y)/max(cr, 1e-3), 0.0, 1.0);
}

vec3 ombrelleScene(vec2 q, float t){
  float asp = uRes.x / uRes.y;
  vec2 rd = normalize(vec2((q.x - 0.5)*asp, q.y - 0.24) + vec2(1e-4));
  float sunAmt = pow(max(dot(rd, normalize(vec2(0.88, 0.30))), 0.0), 7.0);
  vec3 haze = mix(HAZE_COOL, HAZE_WARM, clamp(0.22 + sunAmt, 0.0, 1.0));
  haze = mix(haze, PINK_HAZE, 0.25);

  vec3 col = mix(SKY_HOR, SKY_ZEN, smoothstep(0.22, 1.05, q.y));
  col += vec3(0.20, 0.12, 0.03) * sunAmt * 0.5;

  float cd  = cloudField(q);
  float cd2 = cloudField(q + vec2(0.020, 0.012));
  float lit = clamp(0.5 + (cd - cd2)*4.5, 0.0, 1.0);
  col = mix(col, mix(mix(CLOUD_SHD, CLOUD_LIT, lit), haze, 0.15), clamp(cd*1.4, 0.0, 0.97));

  float rr = farRidge(q.x);
  vec3 ridge = mix(mix(GRASS_MID, vec3(0.52, 0.60, 0.62), 0.5), haze, 0.72);
  col = mix(col, ridge, smoothstep(rr + 0.008, rr - 0.012, q.y)*0.9);

  float cr = crest(q.x);
  vec2 wCr = windF(vec2(q.x, cr), t);            // ← 人の動きが草を揺らす
  float rise = max(q.y - cr, 0.0);
  float comb = vnoise(vec2(q.x*asp*150.0 - wCr.x*rise*900.0, 3.7));
  comb = max(comb, vnoise(vec2((q.x*asp*150.0 - wCr.x*rise*900.0)*0.47 + 11.0, 8.1)));
  float bladeH = 0.008 + 0.050*comb*comb;
  float edge = smoothstep(cr + bladeH + 0.003, cr + bladeH - 0.005, q.y);
  if (edge > 0.001){
    float dep = sceneDepth(q);
    float sway = wCr.x * mix(0.40, 0.03, dep);
    float scale = mix(190.0, 60.0, dep);
    vec2 sp = vec2(q.x*asp + sway, q.y);
    float tex = fbm2(vec2(sp.x*scale*0.85, sp.y*scale*0.10))*0.65
              + vnoise(vec2(sp.x*scale*1.5, sp.y*scale*0.12))*0.50;
    float litG = clamp(0.20 + 0.42*(1.0 - dep) + 0.38*tex + 0.22*smoothstep(0.05, 0.95, q.x), 0.0, 1.0);
    vec3 g = mix(GRASS_SHD, GRASS_MID, smoothstep(0.0, 0.45, litG));
    g = mix(g, GRASS_SUN, smoothstep(0.45, 0.92, litG));
    g = mix(g, GRASS_SHD, smoothstep(0.35, 1.0, q.x)*smoothstep(0.30, 0.02, q.y)*0.60);
    col = mix(col, g, edge);
  }
  // 中景の茎
  float sd;
  vec4 st = stalks(q, t, sd);
  col = mix(col, st.rgb, st.a);
  return col;
}

// 茎を含めた風景の深度。合成の深度テストはこちらを見る
float sceneDepthFull(vec2 q, float t){
  float sd;
  vec4 st = stalks(q, t, sd);
  return max(sceneDepth(q), sd);
}

// 人物を絵の光へ合わせ直す。
// 人物は部屋の光(電球色や蛍光灯)で撮られていて、絵の中は屋外の日向。
// 光源が違うまま貼ると「切り抜いて置いた」感じが抜けない。
// 明部は日向の色、暗部は空からの回り込みの色へ倒す。輝度は保存する。
vec3 relight(vec3 c){
  float l = dot(c, LW);
  const vec3 SUN = vec3(1.08, 1.00, 0.86);
  const vec3 SKY = vec3(0.78, 0.88, 1.14);
  c *= mix(SKY, SUN, smoothstep(0.25, 0.75, l));
  return c * (l / max(dot(c, LW), 1e-3));
}

float matteAt(vec2 q){
  if (uHasMatte < 0.5 || uCompose < 0.5) return 1.0;   // 合成しないなら全部が「人物」= 現実
  return clamp(texture(uMatte, img(clamp(q, 0.0, 1.0))).r, 0.0, 1.0);
}



// ---------------------------------------------------------------- 中景の茎
// 「手前に草の帯がある」だけでは、まだ層が 2 枚しかない。
// **人物より手前のものと奥のものが同時にある**と、初めて空間として読める。
//
// 茎ごとに固定の深度を持たせる。合成側は深度テストなので、
// 人物 (uStand) より深い茎は自動で手前に、浅い茎は自動で奥になる。
// 人が左右に歩けば、茎の間を抜けたり回り込んだりする。
// x位置 / 深度 / 高さ
vec3 stalkDef(int i){
  if (i == 0) return vec3(0.10, 0.97, 0.78);   // 手前・高い
  if (i == 1) return vec3(0.31, 0.58, 0.42);   // 奥
  if (i == 2) return vec3(0.62, 0.90, 0.68);   // 手前・高い
  if (i == 3) return vec3(0.79, 0.50, 0.36);   // 奥
  if (i == 4) return vec3(0.94, 0.94, 0.60);   // 手前
  return              vec3(0.47, 0.66, 0.30);  // 中
}

// 戻り値 rgb=色 a=被覆率、out で最も手前の茎の深度
vec4 stalks(vec2 q, float t, out float sd){
  float asp = uRes.x / uRes.y;
  vec2 w = windF(q, t);
  vec4 acc = vec4(0.0);
  sd = 0.0;
  for (int i = 0; i < 6; i++){
    vec3 st = stalkDef(i);
    float dep = st.y;
    // 手前の茎ほど太く、風で大きく傾ぐ (根元は動かない)
    float rise = clamp(q.y / max(st.z, 1e-3), 0.0, 1.0);
    float lean = w.x * 0.055 * dep * rise * rise;
    float dx = (q.x - st.x - lean) * asp;
    float wdt = (0.010 + 0.020*dep) * (1.0 - 0.50*rise);
    float body = smoothstep(wdt, wdt*0.3, abs(dx)) * smoothstep(st.z, st.z*0.80, q.y);
    // 穂 (花): 深度に応じた大きさ。ここが「物」として読める要
    vec2 hd = vec2(dx, (q.y - st.z*0.94));
    float a2 = atan(hd.y, hd.x);
    float rr = (0.030 + 0.028*dep)*(0.74 + 0.26*cos(5.0*a2 + float(i)*1.7));
    float head = smoothstep(rr, rr*0.55, length(vec2(hd.x, hd.y*1.35)));
    float cov = clamp(body + head, 0.0, 1.0);
    if (cov > 0.002){
      float h = hash11(float(i)*5.7);
      vec3 c = mix(GRASS_SHD, GRASS_SUN, 0.30 + 0.55*h);
      // 花は暖色。奥の花ほど霞に沈むので、後段の空気遠近と併せて距離が読める
      vec3 fl = mix(vec3(1.02, 0.84, 0.86), vec3(1.02, 0.93, 0.62), h);
      c = mix(c, fl, head*0.92);
      c = mix(c, vec3(0.98, 0.78, 0.32), smoothstep(rr*0.42, 0.0, length(vec2(hd.x, hd.y*1.35)))*0.8);
      // 遠い茎は空気に溶かす
      c = mix(c, HAZE_COOL, (1.0 - dep)*0.45);
      acc = mix(acc, vec4(c, 1.0), cov);
      sd = max(sd, dep*step(0.5, cov));
    }
  }
  return acc;
}

// ---------------------------------------------------------------- 前景
// 「背景の前に立っている」と「空間の中にいる」の差は、**手前に何かがあるか**。
// これまで描いていたものは全部人物より奥だったので、書き割りに見えていた。
//
// 画面の下から生える草の穂と、いくつかの花を**人物より手前の深度**に置く。
// あとは合成側の深度テストが自動で人物を隠す。個別のマスクは要らない。
//
// 戻り値: rgb = 色、a = 被覆率。深度は fgDepth() が返す
vec4 foreground(vec2 q, float t){
  float asp = uRes.x / uRes.y;
  vec2 w = windF(q, t);
  vec4 acc = vec4(0.0);

  // 手前の草の穂: 下端から生え、風で傾ぐ。根元は動かず穂先が振れる
  float x = q.x*asp;
  for (int i = 0; i < 3; i++){
    float sc = 26.0 + 11.0*float(i);
    float ph = float(i)*7.31;
    float lean = w.x * 0.10;
    float u = x*sc + ph + lean*sc*q.y*6.0;
    float h = 0.14 + 0.16*vnoise(vec2(floor(u), ph));      // 穂の高さ
    float bl = fract(u);
    float wdt = 0.16 + 0.12*vnoise(vec2(floor(u)+3.0, ph));
    float body = smoothstep(wdt, wdt*0.35, abs(bl - 0.5))  // 縦の帯
               * smoothstep(h, h*0.55, q.y);               // 上へ細る
    if (body > 0.001){
      float lit = 0.35 + 0.5*vnoise(vec2(floor(u)*1.7, ph + 2.0));
      vec3 c = mix(GRASS_SHD, GRASS_SUN, lit);
      acc = mix(acc, vec4(c, 1.0), body*0.95);
    }
  }

  // 花: 少数を離して置く。人物の前を横切る位置に来ると空間が一気に読める
  for (int i = 0; i < 7; i++){
    float fi = float(i);
    float h1 = hash11(fi*4.11), h2 = hash11(fi*9.77);
    vec2 fp = vec2(h1, 0.045 + 0.19*h2);
    fp += w * 0.012 * (0.5 + h2);                          // 風でわずかに揺れる
    vec2 d = vec2((q.x - fp.x)*asp, q.y - fp.y);
    float r = length(d);
    float pet = 0.019 + 0.011*h2;
    // 五弁の花: 半径を角度で変調する
    float a = atan(d.y, d.x);
    float rr = pet*(0.72 + 0.28*cos(5.0*a + fi));
    float m = smoothstep(rr, rr*0.55, r);
    if (m > 0.001){
      vec3 c = mix(vec3(1.02, 0.86, 0.86), vec3(1.02, 0.94, 0.66), h1);
      c = mix(c, vec3(0.98, 0.80, 0.35), smoothstep(pet*0.45, 0.0, r));  // 芯
      acc = mix(acc, vec4(c, 1.0), m);
    }
  }
  return acc;
}

// 前景は人物より手前に置く。uStand より確実に大きくする
float fgDepth(){ return clamp(uStand + 0.18, 0.0, 1.0); }

// 接地の影。人物の**真上**にマットがある地面を暗くする。
// 足元に影が無いと、どれだけ前後関係を作っても figure が浮いたままになる。
float contactShadow(vec2 q){
  float s = 0.0;
  for (int i = 1; i <= 6; i++){
    float dy = float(i) * 0.013;
    s = max(s, matteAt(q + vec2(0.0, dy)) * (1.0 - float(i)/7.0));
  }
  return s;
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

    // 深度: 発生点の深度を中心に散らす。半分は人物より手前、半分は奥に置くことで
    // シルエットの境界で「回り込み」が読める。時間とともにわずかに奥へ沈む。
    float pd = clamp(uSeedDepth + (h1 - 0.5)*0.50 - 0.10*pr, 0.0, 1.0);
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
// 合成後の深度。人物と風景を**1 本の深度場**にまとめる。
// こうしないと、筆サイズ・空気遠近・花びらのオクルージョンが人物と丘で別々に
// 振る舞い、境界が見える。
float depthComposite(vec2 q){
  if (uCompose < 0.5) return depthAt(q);
  float m  = matteAt(q);
  float pd = clamp(uStand + (depthAt(q) - 0.5)*0.25, 0.0, 1.0);
  float sd = sceneDepthFull(q, uTime);
  float vis = m * smoothstep(pd + 0.03, pd - 0.03, sd);
  return mix(sd, pd, vis);
}

vec3 scenePainterly(vec2 q, float t, float lod){
  float d = depthComposite(q);
  vec3 c;
  if (uCompose > 0.5){
    // 担当を分ける。
    //   写真 → 絵の言語へ翻訳する処理 (relight / grade / aerial) は**人物だけ**。
    //     風景は既に絵として描かれていて、空気遠近も焼き込んである。
    //     そこへもう一度霞を掛けると空が白茶ける (実写で発生)。
    //   絵具の載せ方 (compand / divide / 楕円の探索) は**合成後に一様**。
    //     ここを分けると継ぎ目で筆の挙動が変わって必ず見える。
    // **合成は筆触パスの前**なので、同じ一筆が境界をまたぎ、マットの粗さは筆に隠れる。
    float m  = matteAt(q);
    float pd = clamp(uStand + (depthAt(q) - 0.5)*0.25, 0.0, 1.0);

    vec3 land = ombrelleScene(q, t);
    // 接地の影: 人物の真下の地面を沈める
    land *= 1.0 - 0.45*contactShadow(q)*smoothstep(0.0, 0.25, sceneDepth(q));

    vec3 person = aerial(grade(relight(camAt(q, lod))), d, q);

    // **マットではなく深度で決める**。風景の方が手前なら風景が勝つ。
    // これで画面下の近い草が人物の足元を隠し、「草の中に立っている」になる。
    float vis = m * smoothstep(pd + 0.03, pd - 0.03, sceneDepthFull(q, t));
    c = mix(land, person, vis);

    // 前景 (手前の草の穂と花) は最後に、人物より手前として乗せる
    vec4 fg = foreground(q, t);
    if (fg.a > 0.001 && fgDepth() > pd - 0.02) c = mix(c, fg.rgb, fg.a);
  } else {
    c = grade(camAt(q, lod));
    c = aerial(c, d, q);
  }
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
//
// **重要**: 筆の格子は画素ごとに向きで回転する。向きの場が「筆1本の大きさ」より
// 速く変化すると、同じ一筆に属するはずの画素が別々の格子を見て格子が破れる。
// 大きい筆にしたら向きの場も同じ比率で滑らかにしなければならない。
// 以下、空間周波数と深度の差分幅をすべて筆の大きさ bs で割っている。
float strokeAngle(vec2 q, vec2 p, float t, float bs){
  vec2 f = windF(q, t);
  float flowA = atan(f.y, f.x);

  // 深度の勾配も筆の大きさに合わせた幅で取る(細かい凹凸に反応させない)
  float e = clamp(2.0*bs/uRes.y, 1.0/uRes.y, 0.05);

  // 1点の勾配をそのまま使うと、深度が不連続な輪郭で向きが数画素の間に大きく振れ、
  // 一筆の中で格子が破れて縁が毛羽立つ。**構造テンソル**で筆の footprint の中を
  // 平均する。向きは軸なので、平均は 2 倍角ベクトルの和で取るのが正しい
  // (勾配ベクトルをそのまま平均すると符号が反転して打ち消し合う)。
  vec2 tensor = vec2(0.0);
  float wsum = 0.0;
  for (int i = 0; i < 5; i++){
    vec2 o = (i == 0) ? vec2(0.0)
           : vec2(cos(1.2566*float(i)), sin(1.2566*float(i))) * e * 1.6;
    vec2 g = vec2(depthAt(q + o + vec2(e,0)) - depthAt(q + o - vec2(e,0)),
                  depthAt(q + o + vec2(0,e)) - depthAt(q + o - vec2(0,e)));
    float m = length(g);
    if (m > 1e-6){
      float a = atan(g.x, -g.y);                  // 等深線 = 勾配に直交する向き
      tensor += vec2(cos(2.0*a), sin(2.0*a)) * m; // 大きい勾配ほど強く効く
      wsum += m;
    }
  }
  float contourA = (dot(tensor, tensor) > 1e-12) ? 0.5*atan(tensor.y, tensor.x) : 1.5708;
  // 向きが揃っている(=テンソルの長さが重みの和に近い)ほど輪郭に従わせる。
  // 輪郭が入り組んで向きが打ち消し合う場所では従わない
  float coh = (wsum > 1e-6) ? length(tensor)/wsum : 0.0;
  vec2 gd = vec2(wsum/5.0, 0.0);   // 以降の structW 用に平均勾配の大きさだけ渡す

  // 平らな壁や机では勾配がほぼ 0 で、向きを決める根拠が無い。定数に倒すと
  // 筆が一斉に揃って帯状に融合するので、ゆるい斜めの筋目を基本にして
  // 低周波で振る。全周(2π)振ると渦模様になるので振れ幅は ±0.8 rad に留める。
  float hatchA = 0.9 + (fbm2(p*1.1/bs + 11.0) - 0.5)*1.6;
  float structW = smoothstep(0.004, 0.045, length(gd)/max(bs, 0.2)) * smoothstep(0.35, 0.85, coh);
  float baseA = mixOrient(hatchA, contourA, structW);

  float w = smoothstep(0.10, 0.75, length(f));
  return mixOrient(baseA, flowA, w) + (fbm2(p*1.6/bs) - 0.5)*0.45;
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
    float d0 = depthComposite(q);
    float bs = max(uBrush, 0.05);
    // タッチの遠近法: 手前ほど大きい筆
    float szG = mix(0.50, 1.30, clamp(d0, 0.0, 1.0));

    // ---- 格子は回さない ----
    // 原典は「筆の向きで座標系ごと回してから格子を切る」方式だった。原典の向きの場は
    // 解析的で滑らかなので、これで成立していた。
    //
    // 実測の深度から作った向きの場は**輪郭で急回転する**(そこでは急回転が正しい)。
    // 座標系ごと回すと、同じ一筆に属するはずの隣り合う画素が**別々の格子**を見て、
    // 筆が細い糸屑に砕ける (髪と顔の境界に櫛状の筋が出た)。
    // 向きの場を平滑化しても消えない。輪郭で速く変わること自体は正しいからである。
    //
    // 格子は画面に固定し、**各セルの楕円だけをセル中心の向きで回す**。
    // こうすればすべての画素が同じセル集合に同意するので、糸屑は原理的に起きない。
    // 筆は依然として向きの場に沿う (向きはセルごとに引く)。
    // 代わりに格子は等方にする必要がある (回転した楕円が縦横どちらにも伸びるため)。
    // 格子は等方。ピッチと楕円の比が探索範囲を決めるので、5x5 に収まる比にする
    // (7x7 は 49 セルで、向きをテクスチャ化しても 34fps までしか戻らなかった)
    float pitch = 0.0175 * bs;
    vec2 base = floor(p/pitch);
    float reach2 = (2.35*pitch)*(2.35*pitch);   // セルが届きうる最大距離の二乗
    float best = -1.0;
    vec2 bestC = (base + 0.5)*pitch;
    vec2 bestId = base;
    // どの楕円にも入らない画素は、格子の素のセル中心に落ちる。これが格子に揃った
    // 不連続 = 櫛状のギザギザした縁を作る。最も近い楕円を控えに取っておく
    float nearQ = 1e9;
    vec2 nearC = bestC, nearId = base;
    float fuzz = (vnoise(p*52.0/bs) - 0.5)*0.55;
    // 5x5 近傍から1枚の楕円を勝たせる = 一つの楕円は一色 = 一筆
    for (int j = -2; j <= 2; j++)
    for (int i = -2; i <= 2; i++){
      vec2 cid = base + vec2(float(i), float(j));
      // 揺らぎ前のセル中心で早期に足切りする(ハッシュを引く前なので実質ただ)
      vec2 d0v = p - (cid + 0.5)*pitch;
      if (dot(d0v, d0v) > reach2) continue;

      vec4 r0 = hash42(cid + 1.7);          // rnd.xy / sz / bb
      vec2 ctr = (cid + 0.5 + (r0.xy - 0.5)*0.9)*pitch;
      float sz = (0.65 + 1.00*r0.z)*szG;
      float aa = 0.0155*sz*bs;
      float bb = 0.0046*(0.7 + 0.8*r0.w)*min(sz, 1.6)*bs;

      // 向きは**セル中心**で引く。画素ごとではないので一筆の中で必ず一定になる。
      // 場は angle.frag に焼いてあるので 1 回のテクスチャ参照で済む
      // (その場で計算すると画素あたり 25 回になり 130fps -> 21fps に落ちた)
      vec2 cq = vec2(ctr.x/asp + 0.5, ctr.y + 0.5);
      vec2 av = texture(uAngle, img(clamp(cq, 0.0, 1.0))).rg;
      vec4 r1 = hash42(cid + 5.1);          // 向きの揺らぎ / 優先度 / 明度
      // 場は 2 倍角ベクトル (cos2a, sin2a) で焼いてある。
      // atan で角度に戻してから cos/sin を取ると、セルあたり atan+cos+sin の 3 回。
      // 25 セル分では効く (実測 36fps)。**半角公式**で直接 cos a, sin a を得る:
      //   cos a = sqrt((1+cos2a)/2),  sin a = sign(sin2a)·sqrt((1-cos2a)/2)
      // 揺らぎは 2 倍角のまま回してから半角に落とす。
      float jj = (r1.x - 0.5)*1.10;         // = 2 * (r1.x-0.5)*0.55
      float cj = cos(jj), sj = sin(jj);
      vec2 avr = normalize(vec2(av.x*cj - av.y*sj, av.x*sj + av.y*cj) + vec2(1e-6, 0.0));
      float cr2 =  sqrt(max(0.5*(1.0 + avr.x), 0.0));
      float sr2 = -sign(avr.y)*sqrt(max(0.5*(1.0 - avr.x), 0.0));   // 符号は -a 側
      vec2 dd = p - ctr;
      vec2 dr = vec2(cr2*dd.x - sr2*dd.y, sr2*dd.x + cr2*dd.y);
      float q2 = (dr.x*dr.x/(aa*aa) + dr.y*dr.y/(bb*bb)) * (1.0 + fuzz);
      if (q2 < 1.0 && r1.y > best){ best = r1.y; bestC = ctr; bestId = cid; }
      if (q2 < nearQ){ nearQ = q2; nearC = ctr; nearId = cid; }
    }
    if (best < 0.0){ bestC = nearC; bestId = nearId; }   // 隙間は最寄りの筆で埋める
    vec2 qc = vec2(bestC.x/asp + 0.5, bestC.y + 0.5);

    // 一筆が覆う面積の分だけ先にぼかしてから1点サンプルする(写真の高周波を落とす)
    float lod = uCamLod + log2(max(szG, 0.25));
    col = scenePainterly(qc, t, lod);

    float d = depthComposite(qc);
    float l0 = dot(col, LW);
    float h1r = hash21(bestId + 3.1);
    float h2r = hash21(bestId + 17.9);

    // ---- 色彩分割 ----
    // 役割を 2 段に分ける:
    //   inject() … 「光は暖色・影は寒色」の法則。平均を動かしてよい(意図的な脚色)
    //   divide() … 色彩分割そのもの。**平均は厳密に保存する**
    // 以前は divide 側で輝度による暖寒の偏りも付けていたが、それをやると平均が
    // 保存されず「色彩分割」の定義から外れる。担当を分けた。
    if (uInject > 1e-3) col = inject(col, uInject, h2r*2.0 - 1.0);

    // ---- 順序が効く ----
    // 「分割してから圧縮」だと、高彩度の肌の上で色相を大きく振ってから抑えることになり、
    // 緑と桃色の斑が残る (実写で発生)。**パレットを先に決めて、その中で分割する**。
    //   inject  … 中性面に色を入れる
    //   compand … パレットの水準と圧縮を決める (ここで画面全体の彩度がほぼ揃う)
    //   divide  … その水準のまま色相だけ散らす
    //   ceil    … 分割の彩度補正が天井を超えた分だけ戻す
    col = compand(col, uChroma * (0.92 + 0.24*(h2r - 0.5)), d);

    float dlt = uSplit * (0.30 + 0.70*d) * 1.90 / max(bs, 0.4);
    dlt = min(dlt, 1.90);

    // ---- 記憶色 (肌) の保護 ----
    // 色相の振れ幅を全色相に一律で掛けていたが、**肌だけは気持ち悪くなる**。
    // 人は肌の色ズレに極端に敏感で (放送機器に肌色補正が入っているのもそのため)、
    // 草や壁で心地よい振れ幅でも、肌に緑が乗ると生理的に拒否される。
    //
    // 対立色平面での角度を測ると、肌は -0.77rad、木 -0.87rad に対し、
    // 草 -2.11rad / 空 +2.04rad と十分離れている。肌の周りだけ狙って抑えられる。
    if (uMemory > 0.001){
      float Yc = dot(col, LW);
      vec2 vc = vec2(col.r - Yc, col.b - Yc);
      if (dot(vc, vc) > 1e-8){
        const float SKIN_ANG = -0.78;
        float dd = abs(atan(vc.y, vc.x) - SKIN_ANG);
        dd = min(dd, 6.28318 - dd);                       // 円周上の距離
        dlt *= mix(1.0 - 0.72*uMemory, 1.0, smoothstep(0.35, 0.78, dd));
      }
    }                              // 遠景は分割を弱める(霞に彩度を掛けると濁る)
    if (dlt > 1e-3){
      // 分離角は筆ごとに**連続分布**から引く。
      // ±Δ の二値にすると局所色ごとに色が 2 つしか現れず、
      // 「緑と赤が同じ色ばかり」に見える (実写で指摘された)。
      //
      // 分布の形も効く。実写での指摘は 2 つあり、**逆方向を向いていた**:
      //   「同じ色すぎる」 → 均等に散ってほしい (一様分布が有利)
      //   「肌色に見える」 → 緑や紫まで振り切ってほしい (裾の重い分布が有利)
      // 一様だと大半が中途半端にずれた同系色になり、全体は肌色のまま。
      // 裾を重くすると大半が局所色の近くに集まり、かえって 2 色に偏る。
      //
      // 両立させるため**混合分布**にする:
      //   確率 1-R … θ ~ U(-Δ, Δ)      基調。均等に散らす
      //   確率 R   … θ ~ U(-Δf, Δf)    飛び道具。肌から緑や紫へ振り切る
      // 平均の縮みは一様分布の sinc の混合で、閉じた式で書ける:
      //   (1-R)·sin(Δ)/Δ + R·sin(Δf)/Δf
      // その逆数を掛ければ平均が厳密に元へ戻る。
      // 飛び道具の割合。顔のように「見慣れた色」の面では、
      // 緑の筆が 26% もあると面全体がオリーブに見える (実写で発生)。
      // 少数の差し色として効く水準まで下げる。
      const float R = 0.14;
      // Δf を補色(π)近くまで振ると、元が高彩度の肌で虹色になる。
      // 橙から緑や紫へ届く程度 (最大 ~85°) に留める
      float dltF = min(dlt*1.8, 1.25);
      bool far = h2r < R;
      float dd2 = far ? dltF : dlt;
      float th  = (h1r*2.0 - 1.0) * dd2;
      float shr = (1.0 - R)*sin(dlt)/max(dlt, 1e-3)
                +       R *sin(dltF)/max(dltF, 1e-3);
      float sat = 1.0 / max(shr, 0.25);
      col = divide(col, th, sat);
      // 補色の差し色: 暗部にごく少数だけ混ぜる(影に反対色を置く印象派の常套)
      if (h2r < 0.05*uSplit && l0 < 0.42) col = mix(col, divide(col, 3.14159, 0.60), 0.45);
    }

    col = ceilChroma(col, 0.18*2.2);

    col *= l0 / max(dot(col, LW), 1e-3);   // 輝度は保存(原典の思想はここで維持)
    // 実際の絵では隣り合う筆は色相より**明度**が違う。原典の ±1.5% では
    // 平らな面がのっぺりしたまま残るので、分割の強さに連動させて振る
    col *= 1.0 + uSplit*0.20*(hash21(bestId + 9.7) - 0.5);
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
