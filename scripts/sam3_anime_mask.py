"""scripts.Script entry for SAM3 Anime Mask (Forge Neo).

Heavy imports stay inside functions so a missing sam3/torch never kills the UI.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
from modules import script_callbacks, scripts, shared
from PIL import Image

# Allow `sam3_anime` package import from extension root
_EXT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))

from sam3_anime import constants  # noqa: E402


def _log(msg: str) -> None:
    print(f"{constants.LOG_PREFIX} {msg}", flush=True)


# Captured img2img components (created before script ui()).
_INPAINT_BASE: Any = None
_INPAINT_MASK: Any = None
# Forge Canvas stores images in LogicalImage textboxes (base64 data URLs).
# Creation order in ui.py: img2img, sketch, inpaint, inpaint_sketch.
_CANVAS_BACKGROUNDS: list[Any] = []


def _node_source_blob(node) -> str:
    """Collect id / label / class text used to classify image sources."""
    parts: list[str] = []
    for attr in ("elem_id", "label", "elem_classes"):
        value = getattr(node, attr, None)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value)
    return " ".join(parts)


def _parent_id_chain(component, limit: int = 50) -> list[str]:
    ids: list[str] = []
    try:
        node = getattr(component, "parent", None)
        depth = 0
        while node is not None and depth < limit:
            pid = getattr(node, "elem_id", None)
            if isinstance(pid, str):
                ids.append(pid)
            # Labels catch accordion titles like "ControlNet" with no elem_id.
            label = getattr(node, "label", None)
            if isinstance(label, str) and label:
                ids.append(label)
            node = getattr(node, "parent", None)
            depth += 1
    except Exception:
        pass
    return ids


def _inside_img2img(component) -> bool:
    """True only for legitimate img2img init canvases.

    Rejects ControlNet / third-party unit images and txt2img canvases.
    Unknown block trees are rejected too — grabbing a random image is worse
    than asking the user to set the extension 参照画像 or img2img init.
    """
    from sam3_anime import ui_state

    eid = getattr(component, "elem_id", None) or ""
    parents = _parent_id_chain(component)
    return ui_state.accepts_as_img2img_init(
        eid,
        parents,
        class_blob=_node_source_blob(component),
    )


def _capture_inpaint_components(component, **kwargs) -> None:
    global _INPAINT_BASE, _INPAINT_MASK
    eid = getattr(component, "elem_id", None) or kwargs.get("elem_id") or ""
    classes = list(getattr(component, "elem_classes", None) or kwargs.get("elem_classes") or [])
    if eid == "img_inpaint_base":
        _INPAINT_BASE = component
        _log("captured img_inpaint_base")
    elif eid == "img_inpaint_mask":
        _INPAINT_MASK = component
        _log("captured img_inpaint_mask")
    elif "logical_image_background" in classes:
        if _inside_img2img(component):
            _CANVAS_BACKGROUNDS.append(component)
            _log(f"captured canvas background #{len(_CANVAS_BACKGROUNDS)} ({eid})")
        else:
            _log(f"skipped canvas background outside img2img ({eid})")


script_callbacks.on_after_component(_capture_inpaint_components)


def _on_ui_settings() -> None:
    section = ("sam3_anime", "SAM3 Anime Mask")
    shared.opts.add_option(
        "sam3_anime_unload_after_generate",
        shared.OptionInfo(
            False,
            "Unload SAM3 after each generate",
            section=section,
        ).info("Generate 完了後に VRAM を解放する。次回 Generate で再ロードします。"),
    )


script_callbacks.on_ui_settings(_on_ui_settings)


def _unload_after_generate_enabled() -> bool:
    try:
        return bool(shared.opts.data.get("sam3_anime_unload_after_generate", False))
    except Exception:
        return False


class Sam3AnimeMaskScript(scripts.Script):
    # High unique value keeps this accordion near the end of the script list
    # (rendered after WebUI built-in accordions on txt2img/img2img).
    sorting_priority = 260209400

    def title(self) -> str:
        return constants.ACCORDION_LABEL

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        tab = "img2img" if is_img2img else "txt2img"
        with gr.Accordion(
            constants.ACCORDION_LABEL,
            open=False,
            elem_id=f"sam3_anime_acc_{tab}",
        ):
            gr.Markdown(
                "画像が空なら img2img を参照。未ロードでも Generate で自動ロード。"
                "自由入力はカンマ分割（`extra` 不要）。プレビュー画像はクリックで拡大。"
                "オーバーレイはコンセプト別の色分け表示。個別マスクをクリックで単体表示。"
            )

            # --- model ---
            ckpts = _safe_list_checkpoints()
            default_ckpt = _preferred_ckpt(ckpts)
            with gr.Row():
                ckpt = gr.Dropdown(
                    label="SAM3 checkpoint",
                    choices=ckpts,
                    value=default_ckpt,
                    interactive=True,
                    elem_id=f"sam3_anime_ckpt_{tab}",
                    scale=3,
                )
                refresh_ckpt_btn = gr.Button("Refresh", elem_id=f"sam3_anime_refckpt_{tab}", scale=1)
            ckpt_note = gr.Markdown(
                _ckpt_warning_text(default_ckpt),
                elem_id=f"sam3_anime_ckpt_note_{tab}",
            )
            with gr.Row():
                load_btn = gr.Button("Load", elem_id=f"sam3_anime_load_{tab}")
                unload_btn = gr.Button("Unload", elem_id=f"sam3_anime_unload_{tab}")
                free_btn = gr.Button("Free VRAM", elem_id=f"sam3_anime_free_{tab}")
                cancel_btn = gr.Button("Cancel", elem_id=f"sam3_anime_cancel_{tab}")
            status = gr.Textbox(
                label="Status",
                value="SAM3: not loaded",
                interactive=False,
                lines=1,
                max_lines=3,
                elem_id=f"sam3_anime_status_{tab}",
            )

            # --- input + presets side by side ---
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="参照画像（クリックで拡大）",
                        type="pil",
                        image_mode="RGB",
                        height=220,
                        elem_id=f"sam3_anime_input_{tab}",
                    )
                    load_ref_btn = gr.Button(
                        "img2img から取得",
                        elem_id=f"sam3_anime_loadref_{tab}",
                        size="sm",
                    )
                with gr.Column(scale=1):
                    preset_ids = [p[0] for p in constants.PRESETS]
                    preset_labels = [f"{p[1]} ({p[0]})" if p[0] != "extra" else p[1] for p in constants.PRESETS]
                    presets = gr.CheckboxGroup(
                        choices=list(zip(preset_labels, preset_ids, strict=True)),
                        value=list(constants.DEFAULT_SELECTED),
                        label="アニメプリセット",
                        elem_id=f"sam3_anime_presets_{tab}",
                    )
                    free_text = gr.Textbox(
                        label="自由入力（カンマ区切り・extra 不要）",
                        placeholder="例: cat ears, ribbon, sword",
                        elem_id=f"sam3_anime_freetext_{tab}",
                    )

            # --- generate + params ---
            with gr.Row():
                generate_btn = gr.Button("Generate masks", variant="primary", elem_id=f"sam3_anime_gen_{tab}", scale=1)
                threshold = gr.Slider(
                    label="score threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    info=(
                        "検出スコアがこの値以上の領域のみマスク化（低い=多く拾う/高い=厳選）。"
                        "出力にスコアがない場合は効果なし。マスクの二値化は固定（0.5）です。"
                    ),
                    scale=2,
                )
                max_side = gr.Slider(label="max side", minimum=512, maximum=1536, value=1024, step=64, scale=2)
            with gr.Row():
                dilate = gr.Slider(
                    label="dilate / erode (px)",
                    minimum=-64,
                    maximum=64,
                    value=0,
                    step=1,
                    info="プラスで膨張 / マイナスで収縮。ドラッグ中も更新",
                    scale=3,
                )
                feather = gr.Slider(
                    label="feather (px)",
                    minimum=0,
                    maximum=32,
                    value=0,
                    step=1,
                    info=(
                        "マスクの輪郭をぼかして塗り跡を軽減。"
                        " inpaint はマスク値を補間するため、境界がなじみます(0=オフ)。"
                    ),
                    scale=2,
                )
                invert = gr.Checkbox(label="invert", value=False, scale=1)

            # --- outputs (compact) ---
            gallery = gr.Gallery(
                label="個別マスク",
                columns=5,
                height=180,
                object_fit="contain",
                elem_id=f"sam3_anime_gallery_{tab}",
            )
            active_masks = gr.CheckboxGroup(
                label="使用するマスク（生成後に個別ON/OFF）",
                choices=[],
                value=[],
                elem_id=f"sam3_anime_active_{tab}",
            )
            with gr.Row(elem_id=f"sam3_anime_quick_{tab}"):
                quick_all_btn = gr.Button("全ON", size="sm", scale=1)
                quick_none_btn = gr.Button("全OFF", size="sm", scale=1)
                quick_face_btn = gr.Button("顔まわり", size="sm", scale=1)
                quick_person_btn = gr.Button("人物", size="sm", scale=1)
                quick_invert_btn = gr.Button("反転", size="sm", scale=1)
            legend_md = gr.Markdown("", elem_id=f"sam3_anime_legend_{tab}")
            with gr.Row():
                combined_img = gr.Image(
                    label="合成マスク（クリックで拡大）",
                    type="pil",
                    height=260,
                    elem_id=f"sam3_anime_combined_{tab}",
                    scale=1,
                )
                with gr.Column(scale=1):
                    overlay_img = gr.Image(
                        label="オーバーレイ（色分け・クリックで拡大）",
                        type="pil",
                        height=260,
                        elem_id=f"sam3_anime_overlay_{tab}",
                    )
                    show_color_btn = gr.Button(
                        "色分け表示に戻す",
                        size="sm",
                        elem_id=f"sam3_anime_showcolor_{tab}",
                    )

            # Hidden store: processed image for export / re-postprocess
            store_image = gr.State(None)
            store_masks = gr.State({})  # {id: PIL L} after last generate (pre postprocess)
            store_labels = gr.State({})  # {concept_id: display label} after last generate
            # JS → Python transfer for Forge Canvas image (base64 data URL)
            canvas_b64 = gr.Textbox(value="", visible=False, elem_id=f"sam3_anime_canvas_b64_{tab}")

            # Export always targets img2img → Inpaint upload, even from txt2img.
            if is_img2img:
                export_label = "Export to inpaint (img2img)"
                download_label = "Download mask PNG"
                export_hint = "Export は **img2img → Inpaint upload** に画像+マスクを送ります。"
                export_btn_kwargs = {"variant": "primary"}
                download_btn_kwargs = {"variant": "secondary"}
            else:
                export_label = "Export to img2img Inpaint"
                download_label = "Download mask PNG"
                export_hint = (
                    "このタブでは **マスク PNG の Download** が主です。"
                    " Export を押すと **img2img → Inpaint upload** へ自動で切り替わります。"
                )
                export_btn_kwargs = {"variant": "secondary"}
                download_btn_kwargs = {"variant": "primary"}

            gr.Markdown(export_hint, elem_id=f"sam3_anime_export_hint_{tab}")
            with gr.Row():
                if is_img2img:
                    export_btn = gr.Button(export_label, elem_id=f"sam3_anime_export_{tab}", **export_btn_kwargs)
                    download_btn = gr.DownloadButton(
                        download_label, elem_id=f"sam3_anime_dl_{tab}", **download_btn_kwargs
                    )
                else:
                    download_btn = gr.DownloadButton(
                        download_label, elem_id=f"sam3_anime_dl_{tab}", **download_btn_kwargs
                    )
                    export_btn = gr.Button(export_label, elem_id=f"sam3_anime_export_{tab}", **export_btn_kwargs)
            export_status = gr.Markdown("", elem_id=f"sam3_anime_export_status_{tab}")
            export_payload = gr.Textbox(visible=False, elem_id=f"sam3_anime_export_payload_{tab}")

            # Button lockout while SAM3 is busy. Generate's first .click
            # handler fires immediately (queue=False) to disable
            # generate/load/unload/free; the real work runs in the second .click
            # (JS grabs the Forge Canvas image first, since a _js return replaces
            # that event's own inputs). On completion the buttons re-enable via
            # _generate's return values. Cancel is NOT locked so a long
            # generation can always be stopped.

            def _do_load(name: str):
                from sam3_anime import model_manager, vram

                if not name:
                    return model_manager.missing_checkpoint_hint()

                ok, msg = model_manager.load(name)
                _log(msg)
                base = vram.status_text(
                    model_manager.is_loaded(),
                    Path(model_manager.current_ckpt()).name if model_manager.current_ckpt() else None,
                )
                warn = model_manager.checkpoint_warning(name)
                if ok:
                    parts = [base, msg]
                    if warn:
                        parts.append(warn)
                    return " | ".join(parts)
                return f"LOAD FAILED: {msg}\n{base}" + (f"\n{warn}" if warn else "")

            def _do_unload():
                from sam3_anime import model_manager, vram

                ok, msg = model_manager.unload()
                _log(msg)
                if not ok:
                    return f"{vram.status_text(model_manager.is_loaded())} | {msg}"
                return vram.status_text(False)

            def _do_free():
                from sam3_anime import model_manager, vram

                ok, msg = model_manager.free_vram()
                _log(msg)
                if not ok:
                    return f"{vram.status_text(model_manager.is_loaded())} | {msg}"
                return vram.status_text(False)

            def _refresh_ckpts():
                choices = _safe_list_checkpoints()
                preferred = _preferred_ckpt(choices)
                return gr.update(choices=choices, value=preferred)

            def _on_ckpt_change(name: str):
                return _ckpt_warning_text(name)

            def _do_cancel():
                from sam3_anime import model_manager

                return model_manager.request_cancel()

            def _lock_buttons():
                from sam3_anime import ui_state

                st = ui_state.button_interactivity(running=True)
                return (
                    gr.update(interactive=st["generate"]),
                    gr.update(interactive=st["load"]),
                    gr.update(interactive=st["unload"]),
                    gr.update(interactive=st["free"]),
                )

            def _unlock_buttons():
                from sam3_anime import ui_state

                st = ui_state.button_interactivity(running=False)
                return (
                    gr.update(interactive=st["generate"]),
                    gr.update(interactive=st["load"]),
                    gr.update(interactive=st["unload"]),
                    gr.update(interactive=st["free"]),
                )

            load_btn.click(fn=_do_load, inputs=[ckpt], outputs=[status], queue=False)
            unload_btn.click(fn=_do_unload, inputs=[], outputs=[status], queue=False)
            free_btn.click(fn=_do_free, inputs=[], outputs=[status], queue=False)
            cancel_btn.click(fn=_do_cancel, inputs=[], outputs=[status], queue=False)
            refresh_ckpt_btn.click(fn=_refresh_ckpts, inputs=[], outputs=[ckpt], queue=False)
            ckpt.change(fn=_on_ckpt_change, inputs=[ckpt], outputs=[ckpt_note], queue=False)

            # Fill 参照画像 from img2img so the user can verify the source
            # before Generate (same strict source rules as generate).
            def _do_load_ref(canvas_data):
                image, notes = _resolve_reference_image(None, canvas_data)
                if image is None:
                    return (
                        gr.update(),
                        "参照画像が見つかりません。img2img に画像を置くか、参照画像へドロップしてください。",
                    )
                note = " ".join(notes) if notes else "参照画像を設定しました"
                return image.convert("RGB"), f"参照画像を読み込み | {note}"

            # JS only fills the hidden canvas textbox; Python resolves sources.
            _js_canvas_only = (
                "(canvas) => {"
                " try {"
                "  const root = (typeof gradioApp === 'function') ? gradioApp() : document;"
                "  const EXCLUDE = /controlnet|control_net|control-net|alwayson|t2i_adapter"
                "|ip[_-]?adapter|reference_only|animatediff|reactor|adetailer/i;"
                "  function blobOf(n) {"
                "    const id = n.id || '';"
                "    let cls = '';"
                "    if (typeof n.className === 'string') cls = n.className;"
                "    else if (n.className && n.className.baseVal) cls = n.className.baseVal;"
                "    const label = (n.getAttribute && n.getAttribute('aria-label')) || '';"
                "    return (id + ' ' + cls + ' ' + label).toLowerCase();"
                "  }"
                "  function excluded(el) {"
                "    let n = el;"
                "    while (n && n !== root) {"
                "      if (EXCLUDE.test(blobOf(n))) return true;"
                "      n = n.parentElement;"
                "    }"
                "    return false;"
                "  }"
                "  function inImg2img(el) {"
                "    let n = el;"
                "    let saw = false;"
                "    while (n && n !== root) {"
                "      const id = (n.id || '').toLowerCase();"
                "      if (id.indexOf('txt2img') !== -1) return false;"
                "      if (id.indexOf('img_inpaint') === 0) return true;"
                "      if (id.indexOf('img2img') !== -1 || id === 'mode_img2img') saw = true;"
                "      n = n.parentElement;"
                "    }"
                "    return saw;"
                "  }"
                "  const areas = root.querySelectorAll('.logical_image_background textarea');"
                "  let preferred = '';"
                "  for (const ta of areas) {"
                "    if (!ta.value || ta.value.indexOf('data:image/') !== 0) continue;"
                "    if (excluded(ta)) continue;"
                "    if (!inImg2img(ta)) continue;"
                "    preferred = ta.value;"
                "    break;"
                "  }"
                "  return [preferred || canvas || ''];"
                " } catch (e) { return [canvas || '']; }"
                "}"
            )
            load_ref_btn.click(
                fn=_do_load_ref,
                inputs=[canvas_b64],
                outputs=[input_image, status],
                _js=_js_canvas_only,
            )

            def _generate(
                image,
                canvas_data,
                ckpt_name,
                selected,
                free,
                thr,
                mside,
                do_invert,
                dil_amt,
                feather_amt,
            ):
                from sam3_anime import infer, model_manager, postprocess, ui_state

                # Failure outputs keep the previous generate's state:
                # gr.update() is a no-op so gallery/combined/overlay/store/active
                # stay as-is and the user can retry without losing masks.
                no_change = gr.update()
                buttons_back = _unlock_buttons()

                def _fail(status_message: str):
                    return (
                        no_change,
                        no_change,
                        no_change,
                        no_change,
                        no_change,
                        no_change,
                        no_change,
                        no_change,
                        no_change,
                        status_message,
                        *buttons_back,
                    )

                gallery_out = []
                combined = None
                overlay = None
                masks_state: dict = {}
                used_image = None
                notes: list[str] = []
                labels_state: dict = {}
                explicit_image = image is not None

                image, resolve_notes = _resolve_reference_image(image, canvas_data)
                notes.extend(resolve_notes)

                # Shared pure-logic resolver (raises ValueError with a
                # UI-safe message when there is no image or no prompts).
                try:
                    resolved, prompts, labels = ui_state.resolve_generation_inputs(image, canvas_data, selected, free)
                except ValueError as e:
                    return _fail(str(e))
                if resolved == "CANVAS":
                    # Canvas held a data URL we failed to decode earlier —
                    # resolver marked it so we surface the same guidance.
                    return _fail(
                        "No image. 拡張内の「参照画像」か img2img の init image を使ってください。"
                        "（ControlNet など他拡張の画像は自動参照しません）"
                    )
                image = resolved

                used_image = image.convert("RGB")

                if not model_manager.is_loaded():
                    if not ckpt_name:
                        return _fail("SAM3 checkpoint が未選択です。ドロップダウンから選んでください。")
                    _log(f"auto-loading SAM3: {ckpt_name}")
                    ok, load_msg = model_manager.load(ckpt_name)
                    if not ok:
                        return _fail(f"SAM3 load failed: {load_msg}")
                    notes.append(f"auto-loaded: {Path(ckpt_name).name}")

                try:
                    raw, skipped, cancelled = infer.generate_masks(
                        used_image,
                        prompts,
                        max_side=int(mside),
                        threshold=float(thr),
                    )
                    notes.extend(skipped)
                except Exception as e:
                    last = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
                    _log(traceback.format_exc())
                    return _fail(f"Generate failed: {last}")

                if not raw:
                    return _fail("Prompt produced no mask. " + " ".join(notes))

                masks_state = raw
                try:
                    processed, combined = postprocess.process_pipeline(
                        raw,
                        dilate=int(dil_amt),
                        feather=int(feather_amt),
                        invert=bool(do_invert),
                    )
                except Exception as e:
                    # Unlock buttons even on unexpected postprocess errors.
                    last = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
                    _log(traceback.format_exc())
                    return _fail(f"Postprocess failed: {last}")

                for cid, mask in processed.items():
                    gallery_out.append((mask, f"{constants.label_for_id(cid, labels)} ({cid})"))  # noqa: E501

                active_choices = ui_state.build_active_choices(processed, labels)
                active_value = list(processed)
                labels_state = labels

                # Multi-color overlay: each concept keeps its palette color.
                overlay = postprocess.make_multi_overlay(used_image, processed, alpha=0.45)

                status_msg = ui_state.build_status_message(len(processed), notes)
                if cancelled:
                    status_msg += " | CANCELLED — 完了分のみ表示しています"

                if _unload_after_generate_enabled():
                    try:
                        ok, msg = model_manager.unload()
                        if ok:
                            status_msg += " | SAM3 unloaded (settings)"
                        else:
                            status_msg += f" | auto-unload skipped: {msg}"
                        _log(msg)
                    except Exception as e:
                        _log(f"auto-unload failed: {e}")

                return (
                    gallery_out,
                    combined,
                    gr.update(value=overlay, label=ui_state.overlay_label("multi")),
                    used_image if not explicit_image else no_change,
                    used_image,
                    masks_state,
                    labels_state,
                    gr.update(choices=active_choices, value=active_value),
                    ui_state.build_color_legend_html(list(processed), labels),
                    status_msg,
                    *buttons_back,
                )

            # JS runs first: extract Forge Canvas base64, return as extra input
            # Gradio 4: _js return value replaces the inputs passed to fn.
            # Only accept canvases positively identified as img2img init
            # sources. Never fall back to "first data URL anywhere" — that
            # used to pull ControlNet / other extension images.
            _js_with_canvas = (
                "(img, canvas, ckpt, presets, free, thr, mside, invert, dilate, feather) => {"
                " try {"
                "  const root = (typeof gradioApp === 'function') ? gradioApp() : document;"
                "  const EXCLUDE = /controlnet|control_net|control-net|alwayson|t2i_adapter"
                "|ip[_-]?adapter|reference_only|animatediff|reactor|adetailer/i;"
                "  function blobOf(n) {"
                "    const id = n.id || '';"
                "    let cls = '';"
                "    if (typeof n.className === 'string') cls = n.className;"
                "    else if (n.className && n.className.baseVal) cls = n.className.baseVal;"
                "    const label = (n.getAttribute && n.getAttribute('aria-label')) || '';"
                "    return (id + ' ' + cls + ' ' + label).toLowerCase();"
                "  }"
                "  function excluded(el) {"
                "    let n = el;"
                "    while (n && n !== root) {"
                "      if (EXCLUDE.test(blobOf(n))) return true;"
                "      n = n.parentElement;"
                "    }"
                "    return false;"
                "  }"
                "  function inImg2img(el) {"
                "    let n = el;"
                "    let saw = false;"
                "    while (n && n !== root) {"
                "      const id = (n.id || '').toLowerCase();"
                "      if (id.indexOf('txt2img') !== -1) return false;"
                "      if (id.indexOf('img_inpaint') === 0) return true;"
                "      if (id.indexOf('img2img') !== -1 || id === 'mode_img2img') saw = true;"
                "      n = n.parentElement;"
                "    }"
                "    return saw;"
                "  }"
                "  const areas = root.querySelectorAll('.logical_image_background textarea');"
                "  let preferred = '';"
                "  for (const ta of areas) {"
                "    if (!ta.value || ta.value.indexOf('data:image/') !== 0) continue;"
                "    if (excluded(ta)) continue;"
                "    if (!inImg2img(ta)) continue;"
                "    preferred = ta.value;"
                "    break;"
                "  }"
                "  if (!img && preferred) canvas = preferred;"
                " } catch (e) {}"
                " return [img, canvas, ckpt, presets, free, thr, mside, invert, dilate, feather];"
                "}"
            )

            def _generate_js(img, canvas, ckpt_name, sel, free_txt, thr, mside, do_invert, dil_amt, feather_amt):
                """JS runs first on the generate event and its return replaces
                the inputs passed to _generate (Gradio 4)."""
                return [img, canvas, ckpt_name, sel, free_txt, thr, mside, do_invert, dil_amt, feather_amt]

            # 1) disable generate/load/unload/free immediately, 2) run generate
            # (JS re-reads the canvas for this event, buttons re-enable via
            # _generate's return).
            generate_btn.click(
                fn=_lock_buttons,
                inputs=[],
                outputs=[generate_btn, load_btn, unload_btn, free_btn],
                queue=False,
            )
            generate_btn.click(
                fn=_generate,
                inputs=[
                    input_image,
                    canvas_b64,
                    ckpt,
                    presets,
                    free_text,
                    threshold,
                    max_side,
                    invert,
                    dilate,
                    feather,
                ],
                outputs=[
                    gallery,
                    combined_img,
                    overlay_img,
                    input_image,
                    store_image,
                    store_masks,
                    store_labels,
                    active_masks,
                    legend_md,
                    status,
                    generate_btn,
                    load_btn,
                    unload_btn,
                    free_btn,
                ],
                _js=_js_with_canvas,
            )

            # Re-run postprocess when sliders / active mask selection change
            def _multi_overlay_update(store_img, store_m, active, labels, dil_amt, feather_amt, do_invert):
                from sam3_anime import postprocess, ui_state

                selected = ui_state.gallery_visible_ids(store_m, active)
                if store_img is None or not selected:
                    return (
                        gr.update(value=None, label=ui_state.overlay_label("multi")),
                        ui_state.build_color_legend_html([]),
                    )
                subset = {cid: store_m[cid] for cid in selected}
                processed, _combined = postprocess.process_pipeline(
                    subset,
                    dilate=int(dil_amt),
                    feather=int(feather_amt),
                    invert=bool(do_invert),
                )
                overlay = postprocess.make_multi_overlay(store_img, processed, alpha=0.45)
                return (
                    gr.update(value=overlay, label=ui_state.overlay_label("multi")),
                    ui_state.build_color_legend_html(list(processed), labels),
                )

            def _repost(store_img, store_m, active, dil_amt, feather_amt, do_invert, labels):
                from sam3_anime import postprocess, ui_state

                if not store_m:
                    return [], None, gr.update(value=None, label=ui_state.overlay_label("multi")), ""

                selected = ui_state.gallery_visible_ids(store_m, active)
                if not selected:
                    return [], None, gr.update(value=None, label=ui_state.overlay_label("multi")), ""
                subset = {cid: store_m[cid] for cid in selected}
                processed, combined = postprocess.process_pipeline(
                    subset,
                    dilate=int(dil_amt),
                    feather=int(feather_amt),
                    invert=bool(do_invert),
                )
                gallery_out = [
                    (m, f"{constants.label_for_id(cid, labels)} ({cid})") for cid, m in processed.items()
                ]
                overlay_update, legend = _multi_overlay_update(
                    store_img, store_m, active, labels, dil_amt, feather_amt, do_invert
                )
                return gallery_out, combined, overlay_update, legend

            def _repost_inputs():
                return [store_image, store_masks, active_masks, dilate, feather, invert, store_labels]

            def _make_quick_select(mode: str):
                def _fn(store_img, store_m, current, dil_amt, feather_amt, do_invert, labels):
                    from sam3_anime import ui_state

                    available = list(store_m.keys()) if store_m else []
                    selected = ui_state.quick_select_active(mode, available, current or [])
                    gallery_out, combined, overlay_update, legend = _repost(
                        store_img, store_m, selected, dil_amt, feather_amt, do_invert, labels
                    )
                    return gr.update(value=selected), gallery_out, combined, overlay_update, legend

                return _fn

            for _mode, _btn in (
                ("all", quick_all_btn),
                ("none", quick_none_btn),
                ("face", quick_face_btn),
                ("person", quick_person_btn),
                ("invert", quick_invert_btn),
            ):
                _btn.click(
                    fn=_make_quick_select(_mode),
                    inputs=_repost_inputs(),
                    outputs=[active_masks, gallery, combined_img, overlay_img, legend_md],
                    show_progress="hidden",
                )

            # Live update while dragging (not only on release)
            dilate.input(
                fn=_repost,
                inputs=_repost_inputs(),
                outputs=[gallery, combined_img, overlay_img, legend_md],
                show_progress="hidden",
            )
            feather.input(
                fn=_repost,
                inputs=_repost_inputs(),
                outputs=[gallery, combined_img, overlay_img, legend_md],
                show_progress="hidden",
            )
            invert.change(
                fn=_repost,
                inputs=_repost_inputs(),
                outputs=[gallery, combined_img, overlay_img, legend_md],
                show_progress="hidden",
            )
            # .input (not .change): user clicks only. Prevents the double
            # postprocess right after Generate, where _generate's gr.update()
            # on active_masks would otherwise trigger this handler again.
            active_masks.input(
                fn=_repost,
                inputs=_repost_inputs(),
                outputs=[gallery, combined_img, overlay_img, legend_md],
                show_progress="hidden",
            )

            def _on_gallery_select(
                evt: gr.SelectData,
                store_img,
                store_m,
                active,
                labels,
                dil_amt,
                feather_amt,
                do_invert,
            ):
                from sam3_anime import postprocess, ui_state

                no_change = gr.update()
                cid = ui_state.gallery_index_at(evt.index, store_m, active, columns=5)
                if cid is None or store_img is None or not store_m or cid not in store_m:
                    return no_change, no_change
                processed, _combined = postprocess.process_pipeline(
                    {cid: store_m[cid]},
                    dilate=int(dil_amt),
                    feather=int(feather_amt),
                    invert=bool(do_invert),
                )
                # Slightly stronger alpha so a single concept reads clearly.
                overlay = postprocess.make_overlay(store_img, processed[cid], cid, alpha=0.55)
                return (
                    gr.update(value=overlay, label=ui_state.overlay_label("single", cid, labels)),
                    ui_state.build_color_legend_html([cid], labels),
                )

            def _show_multi(store_img, store_m, active, labels, dil_amt, feather_amt, do_invert):
                return _multi_overlay_update(
                    store_img, store_m, active, labels, dil_amt, feather_amt, do_invert
                )

            gallery.select(
                fn=_on_gallery_select,
                inputs=[store_image, store_masks, active_masks, store_labels, dilate, feather, invert],
                outputs=[overlay_img, legend_md],
                show_progress="hidden",
            )
            show_color_btn.click(
                fn=_show_multi,
                inputs=_repost_inputs(),
                outputs=[overlay_img, legend_md],
                show_progress="hidden",
            )

            def _on_free_text_change(text, current_presets):
                free = (text or "").strip()
                presets = set(current_presets or [])
                if free:
                    presets.add("extra")
                else:
                    presets.discard("extra")
                return gr.update(value=sorted(presets))

            free_text.change(
                fn=_on_free_text_change,
                inputs=[free_text, presets],
                outputs=[presets],
                show_progress="hidden",
            )

            def _prepare_export(store_img, combined, store_m, active):
                from sam3_anime import forge_export, postprocess, ui_state

                has_canvas = _INPAINT_BASE is not None and _INPAINT_MASK is not None

                if store_img is None:
                    msg = "Export failed: No image"
                    return (msg, "", gr.update(), gr.update()) if has_canvas else (msg, "")

                # Prefer the displayed combined mask (already filtered by active + dilate/invert)
                mask = combined
                if mask is None and store_m:
                    selected = [cid for cid in (active or []) if cid in store_m] or list(store_m.keys())
                    if selected:
                        mask = postprocess.combine_masks([store_m[cid] for cid in selected])
                if mask is None:
                    msg = "Export failed: No mask（使用するマスクが未選択の可能性があります）"
                    return (msg, "", gr.update(), gr.update()) if has_canvas else (msg, "")

                ok, msg, payload, png_path = forge_export.export_to_inpaint(store_img, mask)
                if not ok:
                    _log(msg)
                    return (msg, "", gr.update(), gr.update()) if has_canvas else (msg, "")

                if has_canvas:
                    try:
                        init_pil, mask_pil = forge_export.build_inpaint_pair(store_img, mask)
                        note = ui_state.build_export_note(str(png_path))
                        return note, payload, gr.update(value=init_pil), gr.update(value=mask_pil)
                    except Exception as e:
                        _log(f"build_inpaint_pair failed: {e}")
                        return (
                            ui_state.build_export_note(str(png_path), set_error=str(e)),
                            payload,
                            gr.update(),
                            gr.update(),
                        )
                note = ui_state.build_export_note(str(png_path), image_set=False)
                return note, payload

            def _dl_mask(combined):
                from sam3_anime import forge_export

                if combined is None:
                    return None
                try:
                    return str(forge_export.save_combined_mask(combined))
                except Exception as e:
                    _log(f"download save failed: {e}")
                    return None

            export_outputs = [export_status, export_payload]
            if _INPAINT_BASE is not None and _INPAINT_MASK is not None:
                export_outputs.extend([_INPAINT_BASE, _INPAINT_MASK])
                _log("export wired to img_inpaint_base / img_inpaint_mask")
            else:
                _log("inpaint components not captured; export falls back to PNG + JS")

            # Python first: write image+mask into inpaint components / PNG
            export_event = export_btn.click(
                fn=_prepare_export,
                inputs=[store_image, combined_img, store_masks, active_masks],
                outputs=export_outputs,
            )

            # Separate click (WebUI copy-button pattern): tab switch is independent of image set
            export_btn.click(
                fn=None,
                _js="sam3AnimeFocusInpaintUpload",
                inputs=[],
                outputs=[],
            )

            # JS inject only when component capture failed — must run AFTER payload is written
            if not (_INPAINT_BASE is not None and _INPAINT_MASK is not None):
                export_event.then(
                    fn=None,
                    _js=(
                        "() => {"
                        " try {"
                        "  const root = (typeof gradioApp === 'function') ? gradioApp() : document;"
                        "  const payloadEl = root.querySelector("
                        '    \'textarea[id*="sam3_anime_export_payload"],'
                        ' [id*="sam3_anime_export_payload"] textarea\''
                        "  );"
                        "  const payload = payloadEl ? payloadEl.value : '';"
                        "  if (window.sam3AnimeInjectInpaint && payload) {"
                        "    window.sam3AnimeInjectInpaint(payload);"
                        "  }"
                        " } catch (e) {}"
                        " return [];"
                        "}"
                    ),
                    inputs=[],
                    outputs=[],
                )

            download_btn.click(
                fn=_dl_mask,
                inputs=[combined_img],
                outputs=[download_btn],
            )

        # Button-only script: do not return processing args
        return []

    # Do not touch generation pipeline
    def process(self, p, *args):
        return


def _safe_list_checkpoints():
    try:
        from sam3_anime import model_manager

        return model_manager.list_checkpoints()
    except Exception as e:
        _log(f"list_checkpoints failed: {e}")
        return []


def _preferred_ckpt(names: list[str] | None) -> str | None:
    try:
        from sam3_anime import model_manager

        return model_manager.preferred_checkpoint_name(names or [])
    except Exception:
        return names[0] if names else None


def _ckpt_warning_text(name: str | None) -> str:
    try:
        from sam3_anime import model_manager

        if not name:
            return model_manager.missing_checkpoint_hint()
        return model_manager.checkpoint_warning(name)
    except Exception as e:
        return f"checkpoint note unavailable: {e}" if not name else ""


def _decode_data_url_to_pil(value: str) -> Image.Image | None:
    """Decode a base64 data URL (Forge Canvas LogicalImage value) to PIL."""
    if not value or not isinstance(value, str):
        return None
    if not value.startswith("data:image/"):
        return None
    try:
        import base64
        import io

        # data:image/png;base64,xxxx
        _, _, b64 = value.partition(",")
        if not b64:
            return None
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        return img.convert("RGB")
    except Exception as e:
        _log(f"data-url decode failed: {e}")
        return None


def _resolve_reference_image(image, canvas_data) -> tuple[Image.Image | None, list[str]]:
    """Resolve the SAM3 reference image and note where it came from.

    Order: extension 参照画像 → img2img canvas data URL → img2img init /
    Inpaint upload. Never third-party images.
    """
    notes: list[str] = []
    if image is not None:
        return image, notes
    if canvas_data:
        decoded = _decode_data_url_to_pil(canvas_data)
        if decoded is not None:
            notes.append("img2img canvas を使用")
            return decoded, notes
    img, init_note = _try_img2img_init()
    if img is not None and init_note:
        notes.append(init_note)
    return img, notes


def _value_to_pil(value) -> Image.Image | None:
    """Best-effort convert a Gradio component value to RGB PIL."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        img = _decode_data_url_to_pil(value)
        if img is not None:
            return img
        # Gradio Image may hold a filesystem path.
        try:
            path = Path(value)
            if path.is_file():
                with Image.open(path) as im:
                    return im.convert("RGB")
        except Exception:
            pass
        return None
    if isinstance(value, dict):
        for key in ("image", "value", "composite", "background"):
            img = _value_to_pil(value.get(key))
            if img is not None:
                return img
        return None
    try:
        import numpy as np

        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            return Image.fromarray(arr[..., :3], mode="RGB")
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
    except Exception:
        pass
    return None


def _try_img2img_init() -> tuple[Image.Image | None, str | None]:
    """Best-effort: read img2img init if extension Image is empty.

    Sources, in order (never ControlNet / third-party / txt2img):
    1. Captured img2img canvases (img2img / sketch / inpaint / inpaint_sketch)
    2. Inpaint upload base (`img_inpaint_base`)
    Returns (image_or_None, note_or_None).
    """
    for comp in _CANVAS_BACKGROUNDS:
        try:
            img = _value_to_pil(getattr(comp, "value", None))
            if img is not None:
                return img, "img2img init image を使用"
        except Exception as e:
            _log(f"canvas background read failed: {e}")

    if _INPAINT_BASE is not None:
        try:
            img = _value_to_pil(getattr(_INPAINT_BASE, "value", None))
            if img is not None:
                return img, "Inpaint upload 画像を使用"
        except Exception as e:
            _log(f"img_inpaint_base read failed: {e}")

    return None, None
