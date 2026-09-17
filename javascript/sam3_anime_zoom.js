/**
 * Click preview images (input / combined / overlay) to zoom.
 * Classic script — Forge Neo loads extension .js as text/javascript.
 */
(function () {
  function ensureModal() {
    let modal = document.getElementById("sam3_zoom_modal");
    if (modal) return modal;
    modal = document.createElement("div");
    modal.id = "sam3_zoom_modal";
    modal.innerHTML = "<img alt=''>";
    modal.addEventListener("click", function () {
      modal.classList.remove("sam3-zoom-open");
    });
    document.body.appendChild(modal);
    return modal;
  }

  function openZoom(src) {
    if (!src) return;
    const modal = ensureModal();
    const img = modal.querySelector("img");
    img.src = src;
    modal.classList.add("sam3-zoom-open");
  }

  function onDocClick(e) {
    const t = e.target;
    if (!t || t.tagName !== "IMG") return;
    const wrap =
      t.closest("[id^='sam3_anime_input_']") ||
      t.closest("[id^='sam3_anime_combined_']") ||
      t.closest("[id^='sam3_anime_overlay_']");
    if (!wrap) return;
    // Avoid stealing clicks on gallery thumbs if selector is too broad
    if (t.closest(".gallery")) return;
    e.preventDefault();
    e.stopPropagation();
    openZoom(t.src);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      document.addEventListener("click", onDocClick, true);
    });
  } else {
    document.addEventListener("click", onDocClick, true);
  }

  // Re-bind after Gradio re-renders
  const obs = new MutationObserver(function () {
    ensureModal();
  });
  obs.observe(document.documentElement, { childList: true, subtree: false });
})();
