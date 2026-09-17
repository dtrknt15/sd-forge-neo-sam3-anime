/**
 * Switch UI to img2img → Inpaint upload after Export.
 * Classic script (not a module) — Forge Neo loads extension .js as text/javascript.
 *
 * Gradio 4 `_js` protocol: return value replaces inputs.
 * Always return Array.from(arguments) (WebUI ui.js convention) — never a boolean.
 */
function sam3AnimeFocusInpaintUpload() {
  const root = typeof gradioApp === "function" ? gradioApp() : document;

  function doSwitch() {
    // 1) Prefer WebUI helper: #tabs[1] = img2img, mode_img2img[4] = Inpaint upload
    if (typeof switch_to_img2img_tab === "function") {
      try {
        switch_to_img2img_tab(4);
        return true;
      } catch (e) {
        // fall through
      }
    }

    // 2) Mirror WebUI switch_to_img2img_tab without relying on the global
    try {
      const topTabs = root.querySelector("#tabs");
      if (topTabs) {
        const topBtns = topTabs.querySelectorAll("button");
        if (topBtns.length > 1) topBtns[1].click();
      }
    } catch (e) {}

    try {
      const mode = root.querySelector("#mode_img2img") || root.getElementById("mode_img2img");
      if (mode) {
        const modeBtns = mode.querySelectorAll("button");
        // img2img, Sketch, Inpaint, Inpaint sketch, Inpaint upload, Batch
        if (modeBtns.length > 4) {
          modeBtns[4].click();
          return true;
        }
        for (const b of modeBtns) {
          const t = (b.textContent || "").toLowerCase();
          if (t.includes("inpaint upload")) {
            b.click();
            return true;
          }
        }
      }
    } catch (e) {}

    try {
      const tab = root.querySelector("#img2img_inpaint_upload_tab");
      if (tab) {
        const clickable =
          tab.querySelector("button") || (tab.tagName === "BUTTON" ? tab : null);
        if (clickable) clickable.click();
      }
    } catch (e) {}
    return false;
  }

  doSwitch();
  // Retry once on next frame — Gradio may still be settling after export
  requestAnimationFrame(function () {
    try {
      doSwitch();
    } catch (e) {}
  });

  // Scroll page top so Inpaint upload is visible after tab switch
  try {
    window.scrollTo(0, 0);
    if (typeof root.scrollTo === "function") root.scrollTo(0, 0);
    if (root.scrollerElement && root.scrollerElement.scrollTo) {
      root.scrollerElement.scrollTo(0, 0);
    }
    const main = root.querySelector("#tabs") || root.body || document.documentElement;
    if (main && main.scrollIntoView) main.scrollIntoView({ block: "start", behavior: "instant" });
  } catch (e) {}

  // Required by Gradio 4 _js: pass-through inputs (do not return boolean)
  return Array.from(arguments);
}

if (typeof window !== "undefined") {
  window.sam3AnimeFocusInpaintUpload = sam3AnimeFocusInpaintUpload;
}
