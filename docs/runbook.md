# 手順書 — 実写での仕上げ

初日で合成シーンによる検証は終わっている。ここからは**実写**に移す作業。
各ステップに「何を見るか」と「Claude に報告すること」を書いた。報告があれば私が続きを実装する。

所要時間の目安: STEP 0〜2 で 30 分、STEP 3〜6 で 2 時間半（うち学習の待ち 1 時間）。

---

## STEP 0 — ディスクを空ける（必須・最初に）

**残り 2.1 GB / 460 GB（100% 使用）。** このままでは STEP 4 のデータ収集が途中で死ぬ。
必要な空き容量は **10 GB 以上**を推奨（データ自体は約 600 MB だが、macOS 自体が
数 GB の作業領域を要求する）。

安全に消せそうな候補（**確認してから消してください。私は勝手に消しません**）:

```bash
du -sh ~/Downloads/* 2>/dev/null | sort -rh | head -20
```

上位はこうなっています:

| サイズ | 中身 | 判断 |
|---|---|---|
| 3.3 G | `DavinciResolve17-IntrotoEditPT1.mov` | チュートリアル動画。再取得可能 |
| 3.2 G | `DaVinci-Resolve-17-Edit-...Part1.zip` | 同上（上の .mov と重複の可能性） |
| 2.7 G | `Microsoft_365_..._Installer.pkg` | インストール済みなら不要 |
| 758 M | `AcroRdrSCADC...dmg` | 同上 |
| 696 M | `TouchDesigner...dmg` | 同上 |
| 592 M | `Visual Studio Code.app` | `/Applications` にあるなら重複 |
| 463 M | `Docker.dmg` / 454 M `Codex.dmg` / 439 M `UnityHubSetup.dmg` / 318 M `Claude (2).dmg` | インストーラ。再取得可能 |

上位 3 つだけで 9 GB 空きます。

もっと大きいのは `~/Library/Caches` の **29 GB** ですが、中身はアプリ次第なので
機械的に消すのは勧めません。何が入っているか見るなら:

```bash
du -sh ~/Library/Caches/* 2>/dev/null | sort -rh | head -15
```

**確認:**
```bash
df -h /Users/suzukitakumi | tail -1
```

**報告すること:** 空けた後の空き容量。10 GB 未満なら収集する枚数を減らす設計に変えます。

---

## STEP 1 — カメラ権限

**システム設定 → プライバシーとセキュリティ → カメラ → Claude をオン**

一覧に `Claude` が無い場合は、いちど下のコマンドを実行すると要求が飛びます（その後に一覧へ出る）。

```bash
cd /Users/suzukitakumi/magic-effect && uv run python -c "import cv2; c=cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION); print('opened:', c.isOpened()); c.release()"
```

`opened: True` が出れば完了。

> **注意**: macOS はカメラ権限の付与を**アプリの再起動後**に反映することがあります。
> `opened: False` のままなら Claude.app を再起動してください。その場合この会話は切れるので、
> 新しいセッションで「ombrelle-live の手順書 STEP 2 から」と言ってもらえれば続けられます
> （`docs/runbook.md` と `docs/devlog.md` を読めば私が状況を復元できます）。

**報告すること:** `opened:` の結果。

---

## STEP 2 — 実写で初見、意匠を追い込む

```bash
uv run python -m ombrelle.app --source cam:0 --depth teacher
```

**ビューをこの順で見てください。**（診断の順序です。いきなり `0` を見て「変」と思っても
どの段が原因か切り分けられない）

| キー | 見るもの | 正常なら |
|---|---|---|
| `1` | 生カメラ | 左右反転している（自撮り像）。明るさが極端でない |
| `2` | 深度マップ | **自分がいちばん明るい**（近い）。壁や奥が暗い。じっとしていてチラつかない |
| `3` | フローの場 | 手を振ると**その場所の矢印だけ**が伸びる。オレンジの環が動いた場所に付く |
| `0` | 筆触（完成） | 手前ほど筆が大きい。奥ほど細かく霞む |

**ここで意匠を触ります。** HUD に現在値が出ます。

| キー | 効果 | 迷ったら |
|---|---|---|
| `k` / `l` | 霞を弱く / 強く | 空や壁が灰紫に洗われるなら `k` で下げる |
| `n` / `m` | 彩度を下げる / 上げる | 「写真にフィルタをかけただけ」に見えるなら `m` で上げる |
| `[` / `]` | 風の利得 | 手を振っても筆が傾がないなら `]` で上げる |
| `,` / `.` | 筆のぼかし | ザラついてノイズっぽいなら `.` で上げる（一筆が覆う面積分ぼかす） |
| `-` / `=` | 絵の強さ | `-` を連打すると写真寄りに戻る。効果の確認に便利 |
| `s` | スクショ → `shots/` に保存 | 気に入った状態を残す |

**報告すること:**
- 気に入った状態の **haze / chroma / flowGain / lod の 4 数値**（HUD の 3〜4 行目）→ 既定値に反映します
- `shots/` に保存したスクショ
- 「ここが変」と思った点。特に **view `2` の深度がチラつくか**（これが STEP 3 の核心）

---

## STEP 3 — EMA 検証用の固定クリップを撮る

初日に残した宿題①のための実験。深度の正規化（毎フレーム min-max vs 非対称 EMA）を
**同一の映像**で比べる必要があります。ライブカメラでは 4 条件が別々のフレームを見てしまい、
比較になりません。

**QuickTime Player → ファイル → 新規ムービー収録** で **20〜30 秒**、以下を1本に入れて撮ってください。

1. 誰もいない状態を 3 秒（← 深度レンジの基準）
2. 画面の外から歩いて入る
3. カメラに**ぐっと近づく**（顔が画面の半分くらいまで） ← ここでレンジが跳ねる
4. 元の位置まで下がる
5. 画面の外へ出る
6. 誰もいない状態を 3 秒

**3 と 5 が実験の本体です。** 深度レンジが急に広がる／急に縮む瞬間で、
EMA の非対称追従（広がるのは速く、縮むのは遅く）が効くかどうかが決まります。

保存先: `data/clip_range.mov`

```bash
mkdir -p data && ls -lh data/clip_range.mov
uv run python -m ombrelle.app --source data/clip_range.mov --depth teacher --view 2
```

**報告すること:** クリップが置けたこと。以降の測定は私が回します。

---

## STEP 4 — 学習用データを収集する

ここが結果を決めます。**枚数より多様性**です。

初日の実測で、60fps で撮ると **8 割が近重複**でした（`collect.py` が自動で捨てます）。
つまり「長く回す」より「条件を変える」方が効きます。

### 条件を変えて 4〜5 セッション、別ディレクトリに保存

```bash
# 1. 昼の自然光・部屋の中を歩き回る（カメラを持って動くと背景が最も変わる → 効果大）
uv run python -m train.collect --source cam:0 --out data/day-walk    --seconds 240

# 2. 夜／室内灯だけ
uv run python -m train.collect --source cam:0 --out data/night-lamp  --seconds 240

# 3. 逆光（窓を背にして立つ）— 破綻しやすい条件を意図的に入れる
uv run python -m train.collect --source cam:0 --out data/backlit     --seconds 180

# 4. 距離を変える（0.5m / 1.5m / 3m を行き来する）
uv run python -m train.collect --source cam:0 --out data/distance    --seconds 180

# 5. （人がいれば）2人以上で映る
uv run python -m train.collect --source cam:0 --out data/multi       --seconds 120
```

各セッション中にやること: 歩く、腕を振る、しゃがむ、椅子に座る、物を持つ、
カメラの前を横切る。**止まっている時間を作らない**（止まると近重複で全部捨てられます）。

### 進行の確認

実行中に `kept / seen (skip N)` が出ます。目安は **4 分で 1500〜3000 枚**。
終了時の `meta.json` を見て:

```bash
cat data/day-walk/meta.json
```

- `count` が 300 未満 → 動きが足りないか閾値が高い。`--novelty 2.0` に下げて撮り直す
- `skipped_near_duplicate / seen` が 0.9 超 → ほぼ止まっていた。もっと動く

**合計 5000〜8000 枚**が目標。ディスク使用量は 1000 枚あたり約 100 MB。

**報告すること:** 各 `meta.json` の `count` と `skipped_near_duplicate`。

---

## STEP 5 — 再学習

全セッションをまとめて学習させます（`--data` に並べる。ディレクトリごとに
train/val が連続ブロックで切られるので、条件をまたいだ漏れは起きません）。

```bash
uv run python -m train.distill \
  --data data/day-walk data/night-lamp data/backlit data/distance data/multi \
  --epochs 20 --batch 16 --workers 4 \
  --out checkpoints/student.pt --run runs/real
```

**所要 約 1 時間**（実測 26 枚/秒 → 5000 枚で 200 秒/epoch × 20）。
裏で回して構いません。合成データでの実測では 20 epoch 前後で val MAE が飽和しました。

見るところ: `val mae` が下がり続けているか。`acc<0.05` が上がるか。
途中で val mae が上がり始めたら過学習なので epoch を減らして再実行（best は自動保存）。

**報告すること:** 最終行の `best val mae` と、`runs/real/log.jsonl`。

---

## STEP 6 — 評価と A/B

```bash
# 精度・レイテンシ・時間的一貫性を測って docs/bench.md を生成
uv run python -m train.eval \
  --data data/day-walk --ckpt checkpoints/student.pt \
  --source data/clip_range.mov --frames 300 \
  --out docs/bench.md --figure docs/depth_compare.png

# アプリに載せた状態の A/B（実行中に d キーでも切り替えられます）
uv run python -m ombrelle.app --source cam:0 --depth teacher --frames 900 --no-hud
uv run python -m ombrelle.app --source cam:0 --depth student --frames 900 --no-hud
```

`--source data/clip_range.mov` を指定するのが要点。STEP 3 のクリップで測ることで
teacher / student × 毎フレーム min-max / EMA の 4 条件が**同じフレーム列**を見ます。

**報告すること:** `docs/bench.md`。ここで宿題①（EMA が実写で効くか）に決着がつきます。
効いていなければ「効かなかった」と devlog に書きます。それも結果です。

---

## STEP 7 — デモ素材

`ffmpeg` は入っています（`/opt/homebrew/bin/ffmpeg`）。

```bash
# QuickTime で画面収録 → mp4 を GIF に（README 用、10秒・幅720）
ffmpeg -i ~/Desktop/screen.mov -t 10 -vf "fps=15,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" docs/images/demo.gif
```

撮るべきカット（各 5〜8 秒）:
1. **止まっている → 腕を大きく振る → 止まる** ＝ 筆触が傾いで、また凪ぐ（この作品の核心）
2. **`0` → `2` → `3` → `0` とビューを切り替える** ＝ 中身が見える
3. **`d` キーで teacher ↔ student を切り替え、HUD の fps が変わる** ＝ 蒸留の効果
4. **人物の背後に花びらが回り込む**瞬間（オクルージョン）

**報告すること:** 撮れた素材。README の画像を実写に差し替えます。

---

## 詰まったとき

| 症状 | 対処 |
|---|---|
| `opened: False` のまま | Claude.app を再起動（STEP 1 の注意書き） |
| ウィンドウが真っ黒 | 別アプリがカメラを占有している。Zoom / Photo Booth を閉じる |
| view `2` の深度が激しくチラつく | まさに測りたい現象。`s` でスクショを撮って報告 |
| fps が 30 を割る | `--render-width 960 --render-height 540` で内部解像度を下げる |
| 絵が「写真にフィルタ」に見える | `m` で彩度、`.` で筆のぼかしを上げる。それでも駄目なら報告（`grade()` の設計を見直します） |
| `collect` が `採用フレームが 0 枚` | `--novelty 1.5` に下げる |
| 学習が MPS で落ちる | `--batch 8` に下げる |

---

## 一気にやるなら（コピペ用）

```bash
cd /Users/suzukitakumi/magic-effect

# STEP 1
uv run python -c "import cv2; c=cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION); print('opened:', c.isOpened()); c.release()"

# STEP 2
uv run python -m ombrelle.app --source cam:0 --depth teacher

# STEP 4
uv run python -m train.collect --source cam:0 --out data/day-walk   --seconds 240
uv run python -m train.collect --source cam:0 --out data/night-lamp --seconds 240
uv run python -m train.collect --source cam:0 --out data/backlit    --seconds 180
uv run python -m train.collect --source cam:0 --out data/distance   --seconds 180

# STEP 5
uv run python -m train.distill --data data/day-walk data/night-lamp data/backlit data/distance \
  --epochs 20 --out checkpoints/student.pt --run runs/real

# STEP 6
uv run python -m train.eval --data data/day-walk --ckpt checkpoints/student.pt \
  --source data/clip_range.mov --frames 300 --out docs/bench.md --figure docs/depth_compare.png
```
