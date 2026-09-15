/**
 * Switch UI to img2img → Inpaint upload after Export.
 * Safe no-op if tabs are missing.
 */
export function sam3AnimeFocusInpaintUpload() {
  const root = typeof gradioApp === "function" ? gradioApp() : document;

  for (const btn of root.querySelectorAll("button")) {
    const t = (btn.textContent || "").trim();
    if (t === "img2img") {
      try { btn.click(); break; } catch (e) {}
    }
  }

  try {
    const tabBtn =
      root.querySelector("#img2img_inpaint_upload_tab button") ||
      root.querySelector("#img2img_inpaint_upload_tab");
    if (tabBtn && tabBtn.click) tabBtn.click();
  } catch (e) {}

  // Also try Gradio tab API when available
  try {
    const tabs = root.querySelector("#mode_img2img");
    if (tabs) {
      const labels = tabs.querySelectorAll("button[role='tab'], .tab-nav button");
      for (const b of labels) {
        const t = (b.textContent || "").toLowerCase();
        if (t.includes("inpaint upload")) {
          b.click();
          break;
        }
      }
    }
  } catch (e) {}

  return true;
}

if (typeof window !== "undefined") {
  window.sam3AnimeFocusInpaintUpload = sam3AnimeFocusInpaintUpload;
}
