# 手順書 — 実写での仕上げ

初日で合成シーンによる検証は終わっている。ここからは**実写**に移す作業。
各ステップに「何を見るか」と「Claude に報告すること」を書いた。報告があれば私が続きを実装する。

所要時間の目安: STEP 0〜2 で 30 分、STEP 3〜6 で 2 時間半（うち学習の待ち 1 時間）。

---

## STEP 0 — ディスクを空ける ✅ 完了

2.1 GB → **29 GB 空き**。データ収集に十分。

---

## 再開するとき（Claude.app 再起動後）

新しいセッションで、これを貼ってください:

> ombrelle-live の続き。docs/runbook.md と docs/devlog.md を読んで STEP 1 から。
> カメラ権限は付与済みなので、まず `--source cam:0 --depth teacher` を実写で確認して。

Claude 側でやること (再開時のチェックリスト):

1. **カメラは Claude から使えない (下記「決着済み」参照)。実写の実行はユーザーに依頼する**
2. ユーザーが `s` / `p` で残した `shots/*.json` と `config.json` を読む
3. 数値を既定値に反映し、スクショを見て次の調整を提案する
4. 動画ファイル (`--source data/*.mov`) なら Claude 側で回せる。測定はそちらで行う
5. STEP 3 以降 (クリップ撮影 → データ収集 → 再学習 → 評価) へ

### 決着済み: Claude のプロセスからカメラは使えない (2026-07-30 検証)

Claude.app にカメラ権限を付与し、Claude.app を再起動しても解決しなかった。
AVFoundation に直接要求させた結果:

```
before: 0 notDetermined
callback: False          ← ダイアログを出さずに即座に拒否
after:  0 notDetermined  ← 記録すら作られない
devices: ['MacBook Pro Camera', '鈴木拓海のiPhone Camera']   ← 列挙はできる
```

TCC が要求元アプリを特定できないプロセスの挙動。**拒否されているのではなく
許可を求める資格が無い**ので、設定に該当項目が現れず、切り替えでも再起動でも直らない。
このやり取りを再度試さないこと。実写の実行は必ずユーザーの `Terminal.app` から。

初日時点の状態: M0〜M6 完了、合成シーンで全段検証済み、コミット済み。
未解決の宿題は devlog の「未着手 / 宿題」を参照。

---

## STEP 1 — 自分のターミナルから起動する

カメラは**あなたが動かす**方針なので、Claude 側の権限は不要。
`Terminal.app`（または iTerm）から実行すると、そのアプリに対して権限ダイアログが正しく出ます。

```bash
cd /Users/suzukitakumi/magic-effect
uv run python -m ombrelle.app --source cam:0 --depth teacher
```

初回にカメラ許可を聞かれるので許可してください。ウィンドウが開けば完了。

> 私（Claude）はカメラ映像を直接見られません。**`s` キーのスクリーンショットと
> `p` キーの設定ファイルが、私への唯一の報告経路**になります。次のステップで使います。

---

## STEP 2 — 実写で初見、意匠を追い込む

**ビューをこの順で見てください。**（診断の順序です。いきなり `0` を見て「変」と思っても
どの段が原因か切り分けられません）

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

### 決まったら 2 つのキーを押すだけ

| キー | 何が起きるか |
|---|---|
| **`s`** | `shots/` に PNG と **同名の .json**（そのときの全設定 + fps + energy）を保存 |
| **`p`** | `config.json` に現在の意匠パラメータを保存。**次回起動時に自動で読まれます** |

これで HUD の数字を目で読んで書き写す必要がありません。
気に入った状態を見つけたら `s` → `p`。気に入らない候補も `s` だけ押しておくと比較できます。

**報告すること: 「保存した」の一言だけ。** 私が `config.json` と `shots/*.json` を読んで、
既定値に反映し、スクショを見て次の調整を提案します。

> 迷ったら「良い / 悪い」を 3〜4 枚ずつ `s` で残してください。
> 私は数値と絵を突き合わせられるので、言葉で説明するより速く原因に辿り着けます。

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
| カメラ許可のダイアログが出ない | Terminal.app から起動しているか確認。システム設定 → プライバシーとセキュリティ → カメラ に `ターミナル` があるか見る |
| ウィンドウが真っ黒 | 別アプリがカメラを占有している。Zoom / Photo Booth を閉じる |
| もっと良い画で撮りたい | iPhone が Continuity Camera として見えている。`--source cam:1` で切り替わる (内蔵より高画質・構図も作りやすい) |
| view `2` の深度が激しくチラつく | まさに測りたい現象。`s` でスクショを撮って報告 |
| fps が 30 を割る | `--render-width 960 --render-height 540` で内部解像度を下げる |
| 絵が「写真にフィルタ」に見える | `m` で彩度、`.` で筆のぼかしを上げる。それでも駄目なら報告（`grade()` の設計を見直します） |
| `collect` が `採用フレームが 0 枚` | `--novelty 1.5` に下げる |
| 学習が MPS で落ちる | `--batch 8` に下げる |

---

## 一気にやるなら（コピペ用）

```bash
cd /Users/suzukitakumi/magic-effect

# STEP 1-2  (意匠が決まったら s でスクショ、p で config.json に保存)
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
