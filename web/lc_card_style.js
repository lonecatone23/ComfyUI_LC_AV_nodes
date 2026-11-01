/**
 * One look for every card (the Proof Sheet's): the picture on top, then a caption with a yellow header line and the
 * details under it. Used by the Assembler card nodes and 📥 Import; the Proof Sheet and Storyboard Sheet match it.
 */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export function ensureCardStyle() {
  if (document.getElementById("lc-card-css")) return;
  const st = document.createElement("style");
  st.id = "lc-card-css";
  st.textContent = `
  .lccard{background:#181818;border:1px solid #333;border-radius:6px;overflow:hidden;display:flex;flex-direction:column;font:12px/1.35 system-ui,sans-serif;color:#ddd}
  .lccard.on{border-color:#3a7d44}
  .lccard-pic{position:relative;aspect-ratio:16/9;background:#262626;display:flex;align-items:center;justify-content:center;color:#777;font-size:28px}
  .lccard-pic img{width:100%;height:100%;object-fit:contain;display:block}
  .lccard-pic .lccard-act{position:absolute;top:4px;right:4px;display:flex;gap:4px}
  .lccard-cap{padding:4px 6px;display:flex;flex-direction:column;gap:2px}
  .lccard-cap .h{color:#ffc850;font-weight:600;display:flex;gap:6px;align-items:center}
  .lccard-cap .c{opacity:.75}.lccard-cap .q{color:#aac8ff}
  .lccard-cap .r{display:flex;flex-wrap:wrap;gap:3px}.lccard-cap .r i{font-style:normal;background:#24343f;border-radius:4px;padding:0 4px;font-size:11px}`;
  document.head.appendChild(st);
}

/** A card: {img (url) or icon, header, lines: [text], quote, chips: [text], lead (html before the header), on} */
export function cardHTML({ img = "", icon = "🖼️", header = "", lines = [], quote = "", chips = [], lead = "", on = false, attrs = "" }) {
  return `<div class="lccard${on ? " on" : ""}" ${attrs}>
    <div class="lccard-pic">${img ? `<img src="${esc(img)}" loading="lazy">` : icon}</div>
    <div class="lccard-cap"><div class="h">${lead}${esc(header)}</div>
      ${lines.filter(Boolean).map((l) => `<div class="c">${esc(l)}</div>`).join("")}
      ${quote ? `<div class="q">${esc(quote)}</div>` : ""}
      ${chips.length ? `<div class="r">${chips.map((c) => `<i>${esc(c)}</i>`).join("")}</div>` : ""}</div></div>`;
}
