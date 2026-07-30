// ombrelle v11.3 — モネ《日傘をさす女(右向き)》1886 翻案 / 統合参考実装
// 実行可能版: reference.html (ブラウザで開くだけ)
//
// uniforms:
//   uRes   : 解像度 (Shadertoyなら iResolution.xy)
//   uTime  : 経過秒 (iTime)
//   uAdv   : ∫gust(t)dt — 風速の積分。CPU側で advT += gustEnv(t)*dt と積算して渡す。
//            Shadertoy移植時の近似: uAdv ≈ 0.58*iTime (呼吸の平均値。停止の演出は消える)
//   uView  : 0=筆触(完成) 1=物理のみ 2=風の場デバッグ
//
// 構成: renderScene()=物理の層(丘/空/雲/空席/花びら) → main()の筆触パス=絵の層
// 検証済みの法則: 絵の層のパラメータ(筆サイズ/色相/彩度)はdepthの関数にする。ただし目が理屈に勝つ。

precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform float uAdv;    // ∫gust dt — 風の積算(CPUで積分)
uniform float uView;   // 0=筆触 1=物理のみ 2=風の場

const vec2 SEAT = vec2(0.60, 0.42);
const vec3 HAZE_COOL = vec3(0.86, 0.87, 0.94);
const vec3 HAZE_WARM = vec3(1.00, 0.93, 0.82);
const vec3 PINK_HAZE = vec3(1.03, 0.86, 0.88);
const vec3 SKY_ZEN   = vec3(0.46, 0.64, 0.94);   // 青は青く
const vec3 SKY_HOR   = vec3(0.99, 0.95, 0.90);   // 地平は白へ
const vec3 GRASS_SUN = vec3(0.55, 0.66, 0.26);   // 明るい緑は鮮やかに
const vec3 GRASS_MID = vec3(0.34, 0.50, 0.28);
const vec3 GRASS_SHD = vec3(0.18, 0.31, 0.32);
const vec3 CLOUD_LIT = vec3(1.00, 0.98, 0.95);
const vec3 CLOUD_SHD = vec3(0.72, 0.76, 0.90);
const vec3 ACCENT    = vec3(0.85, 0.25, 0.15);

float hash21(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float hash11(float p){ p = fract(p*0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float noise2(vec2 p){
  vec2 i = floor(p), f = fract(p);
  f = f*f*(3.0-2.0*f);
  return mix(mix(hash21(i), hash21(i+vec2(1,0)), f.x),
             mix(hash21(i+vec2(0,1)), hash21(i+vec2(1,1)), f.x), f.y);
}
float fbm2(vec2 p){
  float a = 0.5, s = 0.0;
  for(int i=0;i<4;i++){ s += a*noise2(p); p = p*2.02 + vec2(13.7, 7.1); a *= 0.5; }
  return s;
}
vec3 hueRotate(vec3 c, float a){
  const vec3 k = vec3(0.5773503);
  float ca = cos(a), sa = sin(a);
  return c*ca + cross(k, c)*sa + k*dot(k, c)*(1.0-ca);
}

// ---- 呼吸 0.22Hz + 45秒ごとに息を呑む(強さ。移流はuAdv) ----
float gustEnv(float t){
  float br = 0.62 + 0.38*sin(6.28318*0.22*t + 0.8*sin(6.28318*0.031*t));
  float ph = mod(t, 45.0);
  float pf;
  if(ph > 43.0) pf = 1.0 - smoothstep(43.0, 44.2, ph);
  else          pf = smoothstep(0.0, 0.9, ph);
  return max(br*pf, 0.05);
}

// ---- 風の指向性: 全体で上向きの弧を描く(右下から入り、左上へ抜ける) ----
vec2 windDir(vec2 q){
  return normalize(vec2(-1.0, 0.12 + 0.70*max(q.x - 0.25, 0.0)));
}

// ---- 単一の風の場 + 航跡(見えない重心移動) ----
vec2 windF(vec2 q, float t, float asp){
  float g = gustEnv(t);
  float turb = 0.7 + 0.3*fbm2(q*6.0 - vec2(uAdv*1.3, uAdv*0.16));
  vec2 w = windDir(q) * g * turb;
  float lt = mod(t, 90.0);
  float u = clamp((lt - 20.0)/42.0, 0.0, 1.0);
  float e = u*u*(3.0 - 2.0*u);
  vec2 wk = mix(vec2(-0.15, 0.28), SEAT - vec2(0.0, 0.05), e);
  float on = step(0.001, u) * (1.0 - smoothstep(0.90, 1.0, u));
  vec2 d = vec2((q.x - wk.x)*asp, (q.y - wk.y)*1.6);
  w += vec2(0.55, 0.30) * exp(-dot(d,d)*70.0) * on;
  return w;
}

float crest(float x){
  return 0.24 + 0.10*exp(-pow((x - 0.58)/0.30, 2.0))
       + 0.008*sin(x*7.0 + 1.3) + 0.005*sin(x*17.0 + 4.0);
}
float farRidge(float x){
  return 0.252 + 0.014*sin(x*2.6 + 2.0) + 0.007*sin(x*7.3 + 1.0);
}

float cloudField(vec2 q){
  vec2 p = vec2(q.x*2.1, q.y*3.4) + normalize(vec2(-1.0, 0.35))*uAdv*0.075;  // 雲も弧の風で
  float f = fbm2(p*1.6) + 0.35*fbm2(p*4.1 + 7.0);
  f /= 1.35;
  // 雲塊は風の弧に沿って二つ——大胆に、主役級に
  vec2 d1 = vec2((q.x - 0.32)*0.60, (q.y - 0.76)*1.1);
  vec2 d2 = vec2((q.x - 0.78)*0.75, (q.y - 0.52)*1.3);
  float mass = 1.15*exp(-dot(d1,d1)*1.6) + 0.9*exp(-dot(d2,d2)*2.0);
  float veil = 0.30*smoothstep(0.30, 0.90, q.y);
  return smoothstep(0.40, 0.72, f) * (mass + veil);
}

// ---- 物理の層: 丘と空と不在(v0ベース) ----
vec3 renderScene(vec2 q, float t, float asp){
  // 空気の方向成分: 太陽方向(右)の霞だけ暖かい。円盤は描かない
  vec2 rd = normalize(vec2((q.x - 0.5)*asp, q.y - 0.24) + vec2(1e-4));
  float sunAmt = pow(max(dot(rd, normalize(vec2(0.88, 0.30))), 0.0), 7.0);
  vec3 haze = mix(HAZE_COOL, HAZE_WARM, clamp(0.22 + sunAmt*1.0, 0.0, 1.0));
  haze = mix(haze, PINK_HAZE, 0.25);                       // 霞は少しピンクに転ぶ(感情設計値)

  // 空(縦構図: 画面の7割)
  vec3 col = mix(SKY_HOR, SKY_ZEN, smoothstep(0.22, 1.05, q.y));
  col += vec3(0.20, 0.12, 0.03) * sunAmt * 0.5;

  // 雲: 同じ風で移流
  float cd  = cloudField(q);
  float cd2 = cloudField(q + vec2(0.020, 0.012));
  float lit = clamp(0.5 + (cd - cd2)*4.5, 0.0, 1.0);
  vec3 cCol = mix(mix(CLOUD_SHD, CLOUD_LIT, lit), haze, 0.15);
  col = mix(col, cCol, clamp(cd*1.4, 0.0, 0.97));

  // 遠い尾根(ほぼ空気に溶けている)
  float rr = farRidge(q.x);
  vec3 ridge = mix(mix(GRASS_MID, vec3(0.52, 0.60, 0.62), 0.5), haze, 0.72);
  col = mix(col, ridge, smoothstep(rr + 0.008, rr - 0.012, q.y)*0.9);

  // 丘の草: 稜線は縦のブレード、上に行くほど風下へ流される
  float cr = crest(q.x);
  vec2 wCr = windF(vec2(q.x, cr), t, asp);
  float rise = max(q.y - cr, 0.0);
  float leanX = q.x*asp*150.0 - wCr.x*rise*900.0;
  float comb = noise2(vec2(leanX, 3.7));
  comb = max(comb, noise2(vec2(leanX*0.47 + 11.0, 8.1)));
  float bladeH = 0.008 + 0.050*comb*comb;
  float edge = smoothstep(cr + bladeH + 0.003, cr + bladeH - 0.005, q.y);
  if(edge > 0.001){
    float depth = clamp((cr - q.y)/max(cr, 1e-3), 0.0, 1.0);   // 0=稜線(遠) 1=手前
    // 波は稜線に沿った一本(wCr): 列ごとに根元から一体で傾ぎ、穂先が最大・根元は静止
    float sway = wCr.x * mix(0.40, 0.03, depth);
    float scale = mix(190.0, 60.0, depth);
    vec2 sp = vec2(q.x*asp + sway, q.y);
    float strokes = fbm2(vec2(sp.x*scale*0.85, sp.y*scale*0.10));  // 強い縦異方性
    float blades  = noise2(vec2(sp.x*scale*1.5, sp.y*scale*0.12));
    float tex = strokes*0.65 + blades*0.50;
    float litG = clamp(0.20 + 0.42*(1.0 - depth) + 0.38*tex
               + 0.22*smoothstep(0.05, 0.95, q.x), 0.0, 1.0);
    vec3 g = mix(GRASS_SHD, GRASS_MID, smoothstep(0.0, 0.45, litG));
    g = mix(g, GRASS_SUN, smoothstep(0.45, 0.92, litG));
    float sh = smoothstep(0.35, 1.0, q.x) * smoothstep(0.30, 0.02, q.y);
    g = mix(g, GRASS_SHD, sh*0.60);                 // 右下は暗く=光源差し替え

    // 空気遠近法: 草の霞は冷たい白(暖ピンクを混ぜると緑が泥になる)
    vec3 hazeG = mix(HAZE_COOL, vec3(0.97, 0.97, 0.96), clamp(sunAmt*0.8, 0.0, 1.0));
    float distP = mix(2.0, 0.15, pow(depth, 0.7));
    g = mix(g, hazeG, 1.0 - exp(-distP*0.32));
    g = mix(g, PINK_HAZE, 0.12*pow(1.0 - depth, 4.0));   // ピンクは稜線の一線だけ
    col = mix(col, g, edge);
  }

  // 不在: 空席の光溜まり——空気が一点だけ濃く暖かく凝る
  vec2 dS = vec2((q.x - SEAT.x)*asp, q.y - SEAT.y);
  float pool = exp(-dot(dS,dS)/(2.0*0.075*0.075));
  col = mix(col, vec3(1.00, 0.94, 0.84), pool*0.28);
  col += vec3(0.055, 0.042, 0.016) * exp(-dot(dS,dS)/(2.0*0.035*0.035));

  // 花びら: 空席から生まれ、風下へ散る
  float born = smoothstep(8.0, 16.0, mod(t, 90.0));
  for(int i=0;i<26;i++){
    float fi = float(i);
    float h1 = hash11(fi*7.31);
    float h2 = hash11(fi*3.17);
    float h3 = hash11(fi*9.53);
    float lt2 = mod(t - h1*90.0, 26.0);
    float pr = lt2/26.0;
    vec2 pp = SEAT + vec2((h2 - 0.5)*0.14, 0.02 + h3*0.14);
    // 風の弧に乗って、風下へ流れながら上へ舞い上がる
    pp += vec2(-0.48*(0.7 + 0.6*h1)*pr, 0.34*pr*pr - 0.04*pr);
    pp.x += 0.040*sin(lt2*1.1 + fi);
    pp.y += 0.020*sin(lt2*1.7 + fi*2.0);
    float rot = lt2*(0.6 + 0.5*h3) + fi;
    float ca = cos(rot), sa = sin(rot);
    vec2 d2 = vec2((q.x - pp.x)*asp, q.y - pp.y);
    vec2 dr = vec2(ca*d2.x - sa*d2.y, sa*d2.x + ca*d2.y);
    float tumble = 0.40 + 0.60*abs(sin(lt2*2.1 + fi*1.3));
    float sx = 0.011 + 0.008*h3, sy = sx*0.47*tumble;
    float g2 = exp(-0.5*(dr.x*dr.x/(sx*sx) + dr.y*dr.y/(sy*sy)));
    float warmth = exp(-dot(pp - SEAT, pp - SEAT)*30.0);     // 光溜まりの傍でだけ金に透ける
    vec3 pCol = mix(vec3(0.98, 0.72, 0.78), vec3(1.05, 0.88, 0.62), warmth);
    col = mix(col, pCol, clamp(g2, 0.0, 1.0)*0.85*sin(3.14159*pr)*born);
  }
  return col;
}

void main(){
  float asp = uRes.x / uRes.y;
  vec2 q = gl_FragCoord.xy / uRes;
  vec2 p = vec2((q.x - 0.5)*asp, q.y - 0.5);
  float t = uTime;
  vec3 col;

  if(uView > 1.5){
    // 風の場の矢印(V1判定)
    col = mix(renderScene(q, t, asp), vec3(0.93, 0.94, 0.96), 0.62);
    vec2 pa = vec2(q.x*asp, q.y);
    vec2 c = (floor(pa/0.055) + 0.5)*0.055;
    vec2 w = windF(vec2(c.x/asp, c.y), t, asp);
    float m = length(w);
    vec2 u2 = w/max(m, 1e-4);
    float len = 0.021*clamp(m/1.2, 0.06, 1.0);
    vec2 d = pa - c;
    float tt = clamp(dot(d, u2)/max(len, 1e-5), -1.0, 1.0);
    float dd = length(d - u2*tt*len);
    float line = 1.0 - smoothstep(0.0011, 0.0028, dd);
    vec3 ink = mix(vec3(0.16, 0.25, 0.38), vec3(0.78, 0.30, 0.12), smoothstep(0.2, 1.0, tt));
    col = mix(col, ink, line*0.92);
    vec2 dS = vec2((q.x - SEAT.x)*asp, q.y - SEAT.y);
    float ring = 1.0 - smoothstep(0.0012, 0.0032, abs(length(dS) - 0.075));
    col = mix(col, vec3(0.72, 0.42, 0.18), ring*0.7);
  } else if(uView > 0.5){
    // 物理のみ(写真側の層)
    col = renderScene(q, t, asp);
  } else {
    // ---- 楕円筆触 + 等輝度色彩分割(絵側の層) ----
    // 筆の向きは中身に従う: 空は「弧を描く風」に沿い、草は縦(風に傾ぐブレード)
    float crA = crest(q.x);
    float mixSky = smoothstep(crA - 0.03, crA + 0.05, q.y);
    float angG = 1.5708 + 0.45*windF(q, t, asp).x;
    vec2 wdir = windDir(q);
    float angS = atan(wdir.y, wdir.x);
    float ang = mix(angG, angS, mixSky) + (fbm2(p*1.6) - 0.5)*0.55;
    vec2 sdir = vec2(cos(ang), sin(ang));
    vec2 perp = vec2(-sdir.y, sdir.x);
    vec2 s = vec2(dot(p, sdir), dot(p, perp));
    // タッチの遠近法: 草は手前ほど大きく。空は上まで細かいまま(粗い天頂は雑に見えるだけ)
    float cr = crest(q.x);
    float band;
    float szG;
    if(q.y > cr){
      band = clamp((q.y - cr)*1.6, 0.0, 1.0);
      szG = mix(0.60, 0.82, clamp((q.y - cr)*1.2, 0.0, 1.0));
    } else {
      band = clamp((cr - q.y)/max(cr,1e-3), 0.0, 1.0);
      szG = mix(0.50, 1.30, band);
    }
    vec2 pitch = vec2(0.019, 0.0062);
    vec2 su = s/pitch;
    vec2 base = floor(su);
    float best = -1.0;
    vec2 bestC = (base + 0.5)*pitch;
    vec2 bestId = base;
    float fuzz = (noise2(p*52.0) - 0.5)*0.55;
    for(int j=-2;j<=2;j++)
    for(int i=-2;i<=2;i++){
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
      if(q2 < 1.0 && pr2 > best){ best = pr2; bestC = ctr; bestId = cid; }
    }
    vec2 pc = sdir*bestC.x + perp*bestC.y;
    vec2 qc = vec2(pc.x/asp + 0.5, pc.y + 0.5);
    col = renderScene(qc, t, asp);                  // 一つの楕円は一色=一筆
    const vec3 LW = vec3(0.299, 0.587, 0.114);
    float l0 = dot(col, LW);
    // 色彩分割: 草は三族に量子化(青緑/緑/黄——緑を成分に分解)、空は連続ジッタ±20°
    float h1r = hash21(bestId + 3.1);
    float h2r = hash21(bestId + 17.9);
    // 三族は非対称: 青側は深く、黄側はささやき——赤方向には決して届かない
    float fam = floor(h1r*3.0) - 1.0;
    float famG = ((fam < 0.0) ? -0.58 : fam*0.16) + (h2r - 0.5)*0.12;
    famG *= 0.55 + 0.45*band;                    // 霞んだ稜線際は回転させない(土色の発生源)
    float jitS = (h1r - 0.5)*0.70;
    col = hueRotate(col, mix(famG, jitS, mixSky));
    float satG = 1.18 + 0.28*band;               // 彩度ブーストも手前だけ(霞に彩度を掛けると濁る)
    col = mix(vec3(dot(col, LW)), col, mix(satG, 1.10, mixSky) + 0.30*(hash21(bestId + 13.0) - 0.5));
    col *= l0 / max(dot(col, LW), 1e-3);                       // 輝度は保存
    col *= 0.985 + 0.030*hash21(bestId + 9.7);
    col = clamp(col, 0.0, 1.6);
  }

  // 縁はごくわずかに沈む(包みの暗がり)
  if(uView < 1.5) col *= 1.0 - 0.10*smoothstep(0.60, 1.25, length(p));

  // 粒子感・黒禁止
  col += (hash21(gl_FragCoord.xy*0.7 + mod(t, 10.0)) - 0.5)*0.012;
  col = max(col, vec3(0.08, 0.09, 0.11));

  gl_FragColor = vec4(col, 1.0);
}
