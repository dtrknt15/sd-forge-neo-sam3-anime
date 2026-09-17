"""scripts.Script entry for SAM3 Anime Mask (Forge Neo).

Heavy imports stay inside functions so a missing sam3/torch never kills the UI.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from modules import scripts, script_callbacks, shared

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
        _CANVAS_BACKGROUNDS.append(component)
        _log(f"captured canvas background #{len(_CANVAS_BACKGROUNDS)} ({eid})")


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
                "v1 はアニメ用プリセットを SAM3 のテキストプロンプトに変換して切るだけです。"
                "物体の自動認識一覧はありません。"
                "Image が空なら img2img の画像を自動参照します。"
                "未ロード時は Generate が自動ロードします。"
                " 自由入力はカンマ区切りで複数コンセプトに分割されます（`extra` チェック不要）。"
            )

            # --- model row ---
            ckpts = _safe_list_checkpoints()
            default_ckpt = _preferred_ckpt(ckpts)
            with gr.Row():
                ckpt = gr.Dropdown(
                    label="SAM3 checkpoint",
                    choices=ckpts,
                    value=default_ckpt,
                    interactive=True,
                    elem_id=f"sam3_anime_ckpt_{tab}",
                )
            ckpt_note = gr.Markdown(
                _ckpt_warning_text(default_ckpt),
                elem_id=f"sam3_anime_ckpt_note_{tab}",
            )
            with gr.Row():
                load_btn = gr.Button("Load SAM3", elem_id=f"sam3_anime_load_{tab}")
                unload_btn = gr.Button("Unload SAM3", elem_id=f"sam3_anime_unload_{tab}")
                free_btn = gr.Button("Free VRAM", elem_id=f"sam3_anime_free_{tab}")
                refresh_ckpt_btn = gr.Button("Refresh checkpoints", elem_id=f"sam3_anime_refckpt_{tab}")
            status = gr.Textbox(
                label="Status",
                value="SAM3: not loaded",
                interactive=False,
                elem_id=f"sam3_anime_status_{tab}",
            )

            # --- input ---
            input_image = gr.Image(
                label="参照画像",
                type="pil",
                image_mode="RGB",
                elem_id=f"sam3_anime_input_{tab}",
            )

            # --- presets ---
            preset_ids = [p[0] for p in constants.PRESETS]
            preset_labels = [f"{p[1]} ({p[0]})" if p[0] != "extra" else p[1] for p in constants.PRESETS]
            # CheckboxGroup values are ids for backend clarity
            presets = gr.CheckboxGroup(
                choices=list(zip(preset_labels, preset_ids)),
                value=list(constants.DEFAULT_SELECTED),
                label="アニメプリセット",
                elem_id=f"sam3_anime_presets_{tab}",
            )
            free_text = gr.Textbox(
                label="自由入力（カンマ区切り可・extra チェック不要）",
                placeholder="例: cat ears, ribbon, sword",
                elem_id=f"sam3_anime_freetext_{tab}",
            )
            gr.Markdown("注意: `background` は前景人物を除外しきれないことがあります。")

            # --- generate params ---
            with gr.Row():
                generate_btn = gr.Button("Generate masks", variant="primary", elem_id=f"sam3_anime_gen_{tab}")
            with gr.Row():
                threshold = gr.Slider(
                    label="mask threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.01
                )
                max_side = gr.Slider(
                    label="max side", minimum=512, maximum=1536, value=1024, step=64
                )
            with gr.Row():
                combine = gr.Checkbox(label="combine selected into one mask", value=True)
                invert = gr.Checkbox(label="invert mask", value=False)
            dilate = gr.Slider(
                label="dilate / erode", minimum=-32, maximum=32, value=0, step=1
            )

            # --- outputs ---
            gallery = gr.Gallery(
                label="個別マスク", columns=4, height=240, elem_id=f"sam3_anime_gallery_{tab}"
            )
            active_masks = gr.CheckboxGroup(
                label="使用するマスク（生成後に個別ON/OFF）",
                choices=[],
                value=[],
                elem_id=f"sam3_anime_active_{tab}",
            )
            combined_img = gr.Image(label="合成マスク", type="pil", elem_id=f"sam3_anime_combined_{tab}")
            overlay_img = gr.Image(label="オーバーレイ", type="pil", elem_id=f"sam3_anime_overlay_{tab}")

            # Hidden store: processed image for export / re-postprocess
            store_image = gr.State(None)
            store_masks = gr.State({})  # {id: PIL L} after last generate (pre postprocess)
            # JS → Python transfer for Forge Canvas image (base64 data URL)
            canvas_b64 = gr.Textbox(value="", visible=False, elem_id=f"sam3_anime_canvas_b64_{tab}")

            with gr.Row():
                export_btn = gr.Button("Export to inpaint", elem_id=f"sam3_anime_export_{tab}")
                download_btn = gr.DownloadButton(
                    "Download combined mask", visible=True, elem_id=f"sam3_anime_dl_{tab}"
                )
            export_status = gr.Markdown("", elem_id=f"sam3_anime_export_status_{tab}")
            export_payload = gr.Textbox(visible=False, elem_id=f"sam3_anime_export_payload_{tab}")

            def _refresh_status():
                from sam3_anime import model_manager, vram

                return vram.status_text(
                    model_manager.is_loaded(),
                    Path(model_manager.current_ckpt()).name if model_manager.current_ckpt() else None,
                )

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

                model_manager.unload()
                return vram.status_text(False)

            def _do_free():
                from sam3_anime import model_manager, vram

                model_manager.free_vram()
                return vram.status_text(False)

            def _refresh_ckpts():
                choices = _safe_list_checkpoints()
                preferred = _preferred_ckpt(choices)
                return gr.update(choices=choices, value=preferred)

            def _on_ckpt_change(name: str):
                return _ckpt_warning_text(name)

            load_btn.click(fn=_do_load, inputs=[ckpt], outputs=[status], queue=False)
            unload_btn.click(fn=_do_unload, inputs=[], outputs=[status], queue=False)
            free_btn.click(fn=_do_free, inputs=[], outputs=[status], queue=False)
            refresh_ckpt_btn.click(fn=_refresh_ckpts, inputs=[], outputs=[ckpt], queue=False)
            ckpt.change(fn=_on_ckpt_change, inputs=[ckpt], outputs=[ckpt_note], queue=False)

            def _generate(
                image,
                canvas_data,
                ckpt_name,
                selected,
                free,
                thr,
                mside,
                do_combine,
                do_invert,
                dil_amt,
            ):
                from sam3_anime import infer, model_manager, postprocess, vram

                gallery_out = []
                combined = None
                overlay = None
                masks_state: dict = {}
                used_image = None
                notes: list[str] = []

                if image is None and canvas_data:
                    image = _decode_data_url_to_pil(canvas_data)
                    if image is not None:
                        notes.append("img2img canvas を使用")
                if image is None:
                    image = _try_img2img_init()
                if image is None:
                    return (
                        [], None, None, None, {}, gr.update(choices=[], value=[]),
                        "No image. 拡張内の Image か img2img の init image を使ってください。",
                    )

                used_image = image.convert("RGB")
                prompt_rows = constants.resolve_prompts(selected or [], free or "")
                prompts = [(cid, prompt) for cid, prompt, _label in prompt_rows]
                labels = {cid: label for cid, _prompt, label in prompt_rows}
                if not prompts:
                    return (
                        [], None, None, used_image, {}, gr.update(choices=[], value=[]),
                        "プリセットか自由入力を1つ以上指定してください。"
                        " 自由入力のみなら extra チェックは不要です。",
                    )

                use_dummy = False
                empty_active = gr.update(choices=[], value=[])
                if not model_manager.is_loaded():
                    if not ckpt_name:
                        return (
                            [], None, None, used_image, {}, empty_active,
                            "SAM3 checkpoint が未選択です。ドロップダウンから選んでください。",
                        )
                    _log(f"auto-loading SAM3: {ckpt_name}")
                    ok, load_msg = model_manager.load(ckpt_name)
                    if not ok:
                        return (
                            [], None, None, used_image, {}, empty_active,
                            f"SAM3 load failed: {load_msg}",
                        )
                    notes.append(f"auto-loaded: {Path(ckpt_name).name}")

                try:
                    raw, skipped = infer.generate_masks(
                        used_image,
                        prompts,
                        max_side=int(mside),
                        threshold=float(thr),
                        use_dummy=use_dummy,
                    )
                    notes.extend(skipped)
                except Exception as e:
                    last = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
                    _log(traceback.format_exc())
                    return (
                        [], None, None, used_image, {}, empty_active,
                        f"Generate failed: {last}",
                    )

                if not raw:
                    return (
                        [], None, None, used_image, {}, empty_active,
                        "Prompt produced no mask. " + " ".join(notes),
                    )

                masks_state = raw
                processed, combined = postprocess.process_pipeline(
                    raw, dilate=int(dil_amt), invert=bool(do_invert)
                )

                for cid, mask in processed.items():
                    gallery_out.append((mask, f"{constants.label_for_id(cid, labels)} ({cid})"))

                active_choices = [
                    (f"{labels.get(cid, constants.label_for_id(cid, labels))} ({cid})", cid)
                    for cid in processed.keys()
                ]
                active_value = list(processed.keys())

                # combined preview: use combined even if combine checkbox off (display)
                if combined is not None:
                    overlay = postprocess.make_overlay(
                        used_image,
                        combined,
                        concept_id="extra",
                        alpha=0.45,
                    )

                status_msg = f"Generated {len(processed)} mask(s)."
                if notes:
                    status_msg += " " + " ".join(notes)
                if not do_combine and len(processed) == 1:
                    status_msg += " (single mask; combine OFF uses this one on export)"

                if _unload_after_generate_enabled():
                    try:
                        model_manager.unload()
                        status_msg += " | SAM3 unloaded (settings)"
                    except Exception as e:
                        _log(f"auto-unload failed: {e}")

                return (
                    gallery_out,
                    combined,
                    overlay,
                    used_image,
                    masks_state,
                    gr.update(choices=active_choices, value=active_value),
                    status_msg,
                )

            # JS runs first: extract Forge Canvas base64, return as extra input
            # Gradio 4: _js return value replaces the inputs passed to fn
            _js_with_canvas = (
                "(img, canvas, ckpt, presets, free, thr, mside, combine, invert, dilate) => {"
                " try {"
                "  const root = (typeof gradioApp === 'function') ? gradioApp() : document;"
                "  const areas = root.querySelectorAll('.logical_image_background textarea');"
                "  let data = '';"
                "  for (const ta of areas) {"
                "    if (ta.value && ta.value.startsWith('data:image/')) { data = ta.value; break; }"
                "  }"
                "  if (!img && data) canvas = data;"
                " } catch (e) {}"
                " return [img, canvas, ckpt, presets, free, thr, mside, combine, invert, dilate];"
                "}"
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
                    combine,
                    invert,
                    dilate,
                ],
                outputs=[
                    gallery,
                    combined_img,
                    overlay_img,
                    store_image,
                    store_masks,
                    active_masks,
                    status,
                ],
                _js=_js_with_canvas,
            )

            # Re-run postprocess when sliders / active mask selection change
            def _repost(store_img, store_m, active, dil_amt, do_invert, do_combine):
                if not store_m:
                    return None, None, None
                from sam3_anime import postprocess

                selected = [cid for cid in (active or []) if cid in store_m]
                if not selected:
                    return [], None, None
                subset = {cid: store_m[cid] for cid in selected}
                processed, combined = postprocess.process_pipeline(
                    subset, dilate=int(dil_amt), invert=bool(do_invert)
                )
                gallery_out = [(m, f"{constants.label_for_id(cid)} ({cid})") for cid, m in processed.items()]
                overlay = None
                if combined is not None and store_img is not None:
                    overlay = postprocess.make_overlay(store_img, combined, "extra", 0.45)
                return gallery_out, combined, overlay

            def _repost_inputs():
                return [store_image, store_masks, active_masks, dilate, invert, combine]

            dilate.release(
                fn=_repost,
                inputs=_repost_inputs(),
                outputs=[gallery, combined_img, overlay_img],
                show_progress="hidden",
            )
            invert.change(
                fn=_repost,
                inputs=_repost_inputs(),
                outputs=[gallery, combined_img, overlay_img],
                show_progress="hidden",
            )
            active_masks.change(
                fn=_repost,
                inputs=_repost_inputs(),
                outputs=[gallery, combined_img, overlay_img],
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

            def _prepare_export(store_img, combined, store_m, do_combine, active):
                from sam3_anime import forge_export, postprocess

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
                    init_pil, mask_pil = forge_export.build_inpaint_pair(store_img, mask)
                    note = f"Exported to img2img → Inpaint upload. PNG: {png_path}"
                    return note, payload, init_pil, mask_pil
                return msg, payload

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

            export_btn.click(
                fn=_prepare_export,
                inputs=[store_image, combined_img, store_masks, combine, active_masks],
                outputs=export_outputs,
            ).then(
                fn=None,
                _js="() => { try { if (window.sam3AnimeFocusInpaintUpload) window.sam3AnimeFocusInpaintUpload(); } catch (e) {} return true; }",
                inputs=[],
                outputs=[],
            )

            # JS fallback only when component capture failed
            if not (_INPAINT_BASE is not None and _INPAINT_MASK is not None):
                export_btn.click(
                    fn=None,
                    _js=(
                        "() => {"
                        " try {"
                        "  const root = (typeof gradioApp === 'function') ? gradioApp() : document;"
                        "  const payloadEl = root.querySelector('textarea[id*=\"sam3_anime_export_payload\"], [id*=\"sam3_anime_export_payload\"] textarea');"
                        "  const payload = payloadEl ? payloadEl.value : '';"
                        "  if (window.sam3AnimeInjectInpaint && payload) return window.sam3AnimeInjectInpaint(payload);"
                        " } catch (e) { return 'js_error'; }"
                        " return 'no_payload';"
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


def _try_img2img_init() -> Image.Image | None:
    """Best-effort: read img2img / sketch / inpaint canvas image if extension Image is empty.

    Forge Canvas LogicalImage components hold base64 data URLs.
    Creation order: img2img, sketch, inpaint, inpaint_sketch — first non-empty wins.
    """
    for comp in _CANVAS_BACKGROUNDS:
        try:
            value = getattr(comp, "value", None)
            if value is None:
                continue
            if isinstance(value, Image.Image):
                return value.convert("RGB")
            if isinstance(value, str):
                img = _decode_data_url_to_pil(value)
                if img is not None:
                    return img
            # numpy path (LogicalImage(numpy=True) not used by default)
            try:
                import numpy as np

                arr = np.asarray(value)
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    return Image.fromarray(arr[..., :3], mode="RGB")
            except Exception:
                pass
        except Exception as e:
            _log(f"canvas background read failed: {e}")
    return None
