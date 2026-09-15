/**
 * Best-effort inject of init image + mask into Forge Neo img2img Inpaint upload.
 * Payload: JSON string {"image": "<b64 png>", "mask": "<b64 png>"}
 */
export function sam3AnimeInjectInpaint(payload) {
  if (!payload) return "no_payload";
  let data = payload;
  if (typeof payload === "string") {
    try {
      data = JSON.parse(payload);
    } catch (e) {
      return "bad_json";
    }
  }
  const imgB64 = data && data.image;
  const maskB64 = data && data.mask;
  if (!imgB64 || !maskB64) return "missing_b64";

  const root = typeof gradioApp === "function" ? gradioApp() : document;

  function b64ToBlob(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: "image/png" });
  }

  function setFileInput(container, file) {
    if (!container) return false;
    const input = container.querySelector("input[type=file]");
    if (!input) return false;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  // Activate img2img tab
  for (const btn of root.querySelectorAll("button")) {
    const t = (btn.textContent || "").trim();
    if (t === "img2img") {
      try { btn.click(); break; } catch (e) {}
    }
  }

  // Prefer Inpaint upload components (known IDs in Forge Neo)
  const baseEl = root.querySelector("#img_inpaint_base");
  const maskEl = root.querySelector("#img_inpaint_mask");
  if (!baseEl || !maskEl) return "missing_components";

  try {
    const tabBtn = root.querySelector("#img2img_inpaint_upload_tab button")
      || root.querySelector("#img2img_inpaint_upload_tab");
    if (tabBtn && tabBtn.click) tabBtn.click();
  } catch (e) {}

  const okBase = setFileInput(
    baseEl,
    new File([b64ToBlob(imgB64)], "sam3_init.png", { type: "image/png" })
  );
  const okMask = setFileInput(
    maskEl,
    new File([b64ToBlob(maskB64)], "sam3_mask.png", { type: "image/png" })
  );
  return okBase && okMask ? "ok" : "file_input_missing";
}

// Expose on window for Gradio _js callbacks
if (typeof window !== "undefined") {
  window.sam3AnimeInjectInpaint = sam3AnimeInjectInpaint;
}
