# SAM3 Anime Mask (Forge Neo)

**English** | [日本語](#日本語)

Anime-oriented inpaint mask helper for **Stable Diffusion WebUI Forge Neo** (Gradio 4.x).

> **v1 only converts anime presets into SAM3 text prompts and cuts masks.** There is no automatic full-object inventory.

- Target GPU (tested): NVIDIA RTX 3090 24GB  
- UI: `SAM3 Anime Mask` accordion on txt2img / img2img (default export: **img2img → Inpaint upload**)

## Features

1. Auto-reads img2img init image when extension Image is empty
2. Auto-loads SAM3 on Generate if not already loaded
3. Check anime presets (face / hair / clothes / …) or add free-text concepts
4. Generate per-concept masks, preview, dilate/erode, invert, combine
5. After generate, toggle which concept masks are used (no re-inference)
6. Export image + mask to **img2img → Inpaint upload**
7. Unload / Free VRAM when done

Free text is always applied when non-empty (the `extra` checkbox is optional).
Comma-separated phrases become separate masks (`cat ears, ribbon` → 2 masks).

## Install

```text
Clone this repo into <webui>/extensions/sd-forge-neo-sam3-anime
Put an official SAM3 checkpoint into <webui>/models/SAM3/
Restart WebUI
Open the accordion → Load SAM3
```

`install.py` tries to install the official package (failure does **not** break WebUI):

```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

Official requirements (upstream): Python 3.12+, PyTorch 2.7+, CUDA 12.6+.

This extension does **not** install `segment-anything` (name collision).

## Checkpoint (gated — access request required)

SAM3 weights on Hugging Face are **gated**. You must request access and log in before download.

1. Request access: [facebook/sam3](https://huggingface.co/facebook/sam3) (and/or [facebook/sam3.1](https://huggingface.co/facebook/sam3.1))
2. Authenticate: `hf auth login` (or use a HF token)
3. Place the official checkpoint here:
   - `<webui>/models/SAM3/sam3.pt` **(recommended)**
   - or set env `SAM3_CHECKPOINT` to a file path

**Use the official `sam3.pt`.** Unofficial `.safetensors` exports often use a different key layout and may load with missing weights / poor masks.

Links:

- https://github.com/facebookresearch/sam3
- https://huggingface.co/facebook/sam3

## Usage

1. Put an image on img2img (or drop it into the extension Image box — if empty, img2img init image is used automatically)
2. Check presets (default: `face`). Optional free text: `cat ears, ribbon, sword` (comma-split into separate masks; `extra` checkbox not required)
3. **Generate masks** — SAM3 is auto-loaded if not already loaded
4. Toggle **使用するマスク** to include/exclude individual concept masks (updates preview/export without re-running SAM3)
5. Adjust `dilate / erode`, `invert mask` as needed
6. **Export to inpaint** — always targets **img2img → Inpaint upload**
   - On **img2img**: Export is primary
   - On **txt2img**: Download mask PNG is primary; Export switches to img2img Inpaint upload
7. Optional: **Unload SAM3** / **Free VRAM**

If no checkpoint is found, the UI shows the expected path (`models/SAM3/sam3.pt`) and HF gated-access notes.

### Presets → SAM3 prompts

| id | label | prompt |
|---|---|---|
| face | 顔 | anime face |
| hair | 髪 | anime hair |
| eyes | 目 | anime eyes |
| mouth | 口 | anime mouth |
| body | 体 | anime body |
| clothes | 服 | anime clothes |
| hands | 手 | hands |
| fingers | 指 | fingers |
| legs | 脚 | legs |
| feet | 足 | feet |
| sky | 空 | sky |
| background | 背景 | background |
| extra | free | your English short phrase |

Japanese labels are UI-only. SAM3 receives **English short noun phrases**.

## Settings

- **Unload SAM3 after each generate** (default OFF) — free VRAM after each generate; next generate reloads.

## Known limitations (v1)

- No automatic inventory of all objects in the image
- No GroundingDINO / YOLO / WD14 integration
- No video / SAM2 tracker
- No ControlNet direct send
- No batch folder processing
- `background` may not fully exclude foreground characters
- Text prompts only (no point/box prompts yet)

## Logs

Console prefix: `[SAM3 Anime]`

## Contributing

**Pull requests are welcome.** Please keep changes scoped, avoid monkey-patching WebUI internals, and note any new SAM3 API assumptions in the PR description.

## License

This extension: [MIT](./LICENSE).

SAM 3 itself is by Meta and uses the **SAM License** — see [facebookresearch/sam3](https://github.com/facebookresearch/sam3). This repo does not redistribute model weights.

---

# 日本語

**Stable Diffusion WebUI Forge Neo**（Gradio 4.x）向けのアニメ用インペイントマスク拡張です。

> **v1 はアニメ用プリセットを SAM3 のテキストプロンプトに変換してマスクを切るだけです。** 物体の自動認識一覧はありません。

- 検証 GPU: NVIDIA RTX 3090 24GB  
- UI: txt2img / img2img の Accordion「SAM3 Anime Mask」（Export 先は **img2img → Inpaint upload**）

## できること

1. 拡張内の Image が空なら img2img の画像を自動参照
2. 未ロード時の Generate で SAM3 を自動ロード
3. アニメプリセット（顔・髪・服など）にチェック、または自由入力
4. コンセプトごとのマスク生成、プレビュー、膨張/収縮、反転、合成
5. 生成後に「使用するマスク」で個別ON/OFF（再推論なし）
6. 画像 + マスクを **img2img → Inpaint upload** へ Export
7. 終わったら Unload / Free VRAM

自由入力は空でなければ常に使われます（`extra` チェック不要）。
カンマ区切りの文は別マスクになります（`cat ears, ribbon` → 2枚）。

## インストール

```text
このリポジトリを <webui>/extensions/sd-forge-neo-sam3-anime に clone
公式チェックポイントを <webui>/models/SAM3/ に置く
WebUI 再起動
Accordion から Load SAM3
```

`install.py` が公式パッケージのインストールを試みます（失敗しても WebUI は起動します）:

```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

公式要件: Python 3.12+ / PyTorch 2.7+ / CUDA 12.6+  
`segment-anything` v1 は入れません（名前衝突防止）。

## チェックポイント（**Gated — アクセス申請が必要**）

Hugging Face 上の SAM3 重みは **Gated** です。申請して承認されないとダウンロードできません。

1. アクセス申請: [facebook/sam3](https://huggingface.co/facebook/sam3)（必要なら [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)）
2. 認証: `hf auth login`（または HF トークン）
3. 公式チェックポイントの配置先:
   - `<webui>/models/SAM3/sam3.pt` **（推奨）**
   - または環境変数 `SAM3_CHECKPOINT` にファイルパス

**公式の `sam3.pt` を使ってください。** 非公式の `.safetensors` はキー構成が違い、重みが欠けたりマスク精度が落ちたりします。

リンク:

- https://github.com/facebookresearch/sam3
- https://huggingface.co/facebook/sam3

## 使い方

1. img2img に画像を置く（拡張内の Image が空なら img2img の画像を自動参照）
2. プリセットにチェック（既定: `face`）。自由入力例: `cat ears, ribbon, sword`（カンマで分割・`extra` チェック不要）
3. **Generate masks**（未ロード時は自動ロード）
4. **使用するマスク** で個別ON/OFF（プレビュー/Export へ即時反映、再推論なし）
5. `dilate / erode`、`invert mask` を調整
6. **Export to inpaint** — 送信先は常に **img2img → Inpaint upload**
   - **img2img** タブ: Export が主ボタン
   - **txt2img** タブ: マスク PNG Download が主。Export を押すと img2img Inpaint upload へ自動切替
7. 任意で **Unload SAM3** / **Free VRAM**

チェックポイントが無いときは、配置先パス（`models/SAM3/sam3.pt`）と HF gated 申請の案内を UI に表示します。

### プリセット → SAM3 プロンプト

| id | ラベル | プロンプト |
|---|---|---|
| face | 顔 | anime face |
| hair | 髪 | anime hair |
| eyes | 目 | anime eyes |
| mouth | 口 | anime mouth |
| body | 体 | anime body |
| clothes | 服 | anime clothes |
| hands | 手 | hands |
| fingers | 指 | fingers |
| legs | 脚 | legs |
| feet | 足 | feet |
| sky | 空 | sky |
| background | 背景 | background |
| extra | 自由入力 | 英語の短い名詞句 |

日本語は UI ラベル用です。SAM3 には **英語の短い名詞句** を渡します。

## Settings

- **Unload SAM3 after each generate**（既定 OFF）— Generate ごとに VRAM を解放。次回 Generate で再ロード。

## 既知の制限（v1）

- 画像内の全物体を自動で一覧化しない
- GroundingDINO / YOLO / WD14 連携なし
- 動画 / SAM2 トラッカーなし
- ControlNet へ直接送らない
- バッチフォルダ処理なし
- `background` は前景人物を除外しきれないことがある
- テキストプロンプトのみ（点・ボックス指定は未対応）

## ログ

コンソールプレフィックス: `[SAM3 Anime]`

## コントリビュート

**プルリクエスト歓迎です。** 変更はスコープを絞り、WebUI 内部へのモンキーパッチは避け、SAM3 API の新しい仮定があれば PR 説明に書いてください。

## ライセンス

本拡張: [MIT](./LICENSE)

SAM 3 本体は Meta の **SAM License** です — [facebookresearch/sam3](https://github.com/facebookresearch/sam3) を参照。モデル重みは同梱していません。
