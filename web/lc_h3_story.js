/**
 * LC H3 Story: the whole movie on one node. Project row, look, cast cards, place cards and shot cards, all real HTML
 * in one addDOMWidget (works on the classic canvas and in Nodes 2.0). The data lives in the hidden `story` STRING
 * widget as JSON. Pictures load like Load Image: click the slot to pick from input/ or output/ (or upload), or drop a
 * file or a picture from the canvas onto it. "H3 gets" on each shot shows the exact text the Segment sends, built by
 * the server from the card (POST /lc_h3/story/preview).
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "LCH3Story";
const MIN_W = 620;
const MAX_AUTO = 900;
const MIN_PANEL = 200;
const FRAMINGS = ["wide shot", "side view", "two-shot", "medium shot", "close-up", "extreme close-up", "over the shoulder", "from behind", "custom"];
const MOVES = ["still", "slow push-in", "slow pull back", "pan", "tracking", "orbit", "handheld"];
const WHO = ["woman", "man", "girl", "boy", "person", "cat", "dog"];

const CSS = `
.lcst{font:12px Arial,sans-serif;color:#ddd;box-sizing:border-box;padding:8px;display:flex;flex-direction:column;gap:10px;background:#0a0a0a;min-height:100%}
.lcst-part{border:1px solid #1a5a7a;border-radius:8px;background:#0f0f0f;overflow:hidden}
.lcst-pb{padding:8px;display:flex;flex-direction:column;gap:6px}
.lcst-part.min .lcst-pb{display:none}
.lcst-part:not(.lcst-top) .lcst-h{cursor:pointer;user-select:none}
.lcst-fold{width:10px;color:#9fd3ff}
.lcst-menu{position:fixed;z-index:10001;min-width:200px;max-height:320px;overflow:auto;background:#1b1f24;border:1px solid #3a4250;border-radius:6px;box-shadow:0 10px 30px #0009;padding:4px;font:12px Arial,sans-serif;color:#ddd;display:flex;flex-direction:column;gap:4px}
.lcst-menu .it{padding:5px 8px;border-radius:4px;cursor:pointer}
.lcst-menu .it:hover{background:#2b3542}
.lcst-menu .it.on{color:#eb4}
.lcst-top{padding:8px;display:flex;flex-direction:column;gap:6px;background:linear-gradient(180deg,#10232c,#0f0f0f);position:sticky;top:0;z-index:5;box-shadow:0 6px 12px #000c}
.lcst-min .lcst-sum{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#bbb}
.lcst *{box-sizing:border-box}
.lcst-h{font-weight:700;font-size:13px;color:#fff;display:flex;align-items:center;gap:6px;padding:6px 10px;background:linear-gradient(90deg,#1a5a7a,#123f56 70%,#0f0f0f);letter-spacing:.3px}
.lcst-h .lcst-tip{color:#bfe3f2}
.lcst-h .lcst-b{background:#0d2f3f;border-color:#2a7aa0}
.lcst-h span{flex:1}
.lcst input,.lcst select,.lcst textarea{background:#060606;color:#e6e6e6;border:1px solid #2a2a2a;border-radius:5px;padding:4px 6px;font:12px Arial,sans-serif;min-width:0}
.lcst input:focus,.lcst select:focus,.lcst textarea:focus{outline:none;border-color:#2a8ab5;box-shadow:0 0 0 1px #1a5a7a}
.lcst textarea{resize:vertical;width:100%}
.lcst-row{display:flex;gap:4px;align-items:center}
.lcst-col{display:flex;flex-direction:column;gap:4px;flex:1;min-width:0}
.lcst-card{border:1px solid #232323;border-left:3px solid #1a5a7a;border-radius:6px;padding:7px;display:flex;flex-direction:column;gap:5px;background:#151515}
.lcst-pc{flex-direction:row;align-items:flex-start;gap:6px}
.lcst-n{font-weight:700;color:#6cc4e6;white-space:nowrap}
.lcst-card>.lcst-row>.lcst-n:first-child{background:#1a5a7a;color:#fff;border-radius:10px;min-width:22px;text-align:center;padding:1px 6px}
.lcst-b{background:#1a1a1a;color:#d6d6d6;border:1px solid #333;border-radius:5px;cursor:pointer;padding:3px 8px;font:12px Arial,sans-serif;white-space:nowrap}
.lcst-b:hover{background:#12384a;border-color:#2a7aa0;color:#fff}
.lcst-b.pri{background:#1a5a7a;border-color:#2a7aa0;color:#fff}
.lcst-b.pri:hover{background:#23759c}
.lcst-b[data-del],.lcst-b[data-bx]{color:#ff5c5c}
.lcst-b[data-del] svg,.lcst-b[data-bx] svg{width:12px;height:12px;vertical-align:middle}
.lcst-b:disabled{opacity:.3;cursor:default}
.lcst-add{width:100%;padding:5px;background:transparent;border:1px dashed #1a5a7a;color:#6cc4e6}
.lcst-add:hover{background:#0d2f3f}
.lcst-slot{width:84px;height:84px;flex:0 0 84px;border:1px dashed #2f4f5c;border-radius:6px;background:#080808 center/cover no-repeat;cursor:pointer;display:flex;align-items:center;justify-content:center;color:#777;font-size:11px;text-align:center;position:relative}
.lcst-slot.drop{border-color:#eb4;background-color:#3a3220}
.lcst-slot .x{position:absolute;top:1px;right:1px;background:#000a;border:none;color:#ddd;border-radius:3px;cursor:pointer;font-size:11px;padding:0 4px;display:none}
.lcst-slot:hover .x{display:block}
.lcst-slot .x.cr{right:auto;left:1px}
.lcst-slot .mk,.lcst-slot .un{position:absolute;bottom:1px;background:#000a;border:none;color:#ddd;border-radius:3px;cursor:pointer;font-size:11px;padding:0 4px;display:none}
.lcst-slot .mk{right:1px}
.lcst-slot .un{left:1px}
.lcst-slot:hover .mk,.lcst-slot:hover .un{display:block}
.lcst-crop{position:fixed;inset:5% 8%;background:#202020;border:1px solid #555;border-radius:8px;z-index:10000;display:flex;flex-direction:column;gap:8px;padding:8px;font:12px Arial,sans-serif;color:#ddd;box-shadow:0 10px 40px #000}
.lcst-crop .on{background:#3a3220;border-color:#eb4;color:#eb4}
.lcst-crop-stage{position:relative;flex:1;min-height:0;display:flex;align-items:center;justify-content:center}
.lcst-crop-wrap{position:relative;display:inline-block;line-height:0}
.lcst-crop-wrap img{max-width:80vw;max-height:76vh;display:block;user-select:none}
.lcst-crop-box{position:absolute;box-shadow:0 0 0 9999px #000a;border:1px solid #eb4;cursor:move;touch-action:none}
.lcst-crop-box i{position:absolute;width:12px;height:12px;background:#eb4;border-radius:2px;margin:-6px 0 0 -6px}
.lcst-crop-box i[data-h=nw]{left:0;top:0;cursor:nwse-resize}.lcst-crop-box i[data-h=n]{left:50%;top:0;cursor:ns-resize}
.lcst-crop-box i[data-h=ne]{left:100%;top:0;cursor:nesw-resize}.lcst-crop-box i[data-h=e]{left:100%;top:50%;cursor:ew-resize}
.lcst-crop-box i[data-h=se]{left:100%;top:100%;cursor:nwse-resize}.lcst-crop-box i[data-h=s]{left:50%;top:100%;cursor:ns-resize}
.lcst-crop-box i[data-h=sw]{left:0;top:100%;cursor:nesw-resize}.lcst-crop-box i[data-h=w]{left:0;top:50%;cursor:ew-resize}
.lcst-anch{display:flex;gap:4px;flex-wrap:wrap;align-items:center}
.lcst-anch .lcst-slot{width:52px;height:52px;flex-basis:52px}
.lcst-tip{font-size:11px;color:#8a9aa2}
.lcst-warn{font-size:11px;color:#ffd860}
.lcst-stop{font-size:11px;color:#ff8c8c}
.lcst-sec{width:62px}
.lcst-pv summary{cursor:pointer;color:#6cc4e6;font-size:11px}
.lcst-pv summary .lcst-b{margin-left:6px;padding:0 6px}
.lcst-pvh{color:#6cc4e6;font-size:11px;margin:5px 0 2px}
.lcst-pvh b{color:#ffd860}
.lcst-pv textarea{width:100%;box-sizing:border-box;min-height:150px;resize:vertical;background:#050505;color:#e8e8e8;border:1px solid #ffd860;border-radius:4px;padding:5px;font:11px Consolas,monospace}
.lcst-pv pre{white-space:pre-wrap;background:#050505;border:1px solid #1d3a47;border-radius:4px;padding:5px;margin:3px 0 0;font:11px Consolas,monospace;color:#cfcfcf}
.lcst-status{font-size:11px;color:#9fd3ff;flex:1;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lcst-pick{position:fixed;inset:6% 10%;background:#202020;border:1px solid #555;border-radius:8px;z-index:10000;display:flex;flex-direction:column;font:12px Arial,sans-serif;color:#ddd;box-shadow:0 10px 40px #000}
.lcst-pick-top{display:flex;gap:6px;padding:8px;border-bottom:1px solid #333;align-items:center}
.lcst-pick-grid{flex:1;overflow:auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:6px;padding:8px}
.lcst-pick-grid div{aspect-ratio:1;background:#111 center/cover no-repeat;border-radius:4px;cursor:pointer;border:2px solid transparent;position:relative}
.lcst-pick-grid div:hover{border-color:#eb4}
.lcst-pick-grid span{position:absolute;bottom:0;left:0;right:0;background:#000a;font-size:10px;padding:1px 3px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.lcst-pick .on{background:#3a3220;border-color:#eb4;color:#eb4}
.lcst-slot.pano{width:84px;height:42px;flex-basis:auto}
.lcst-board{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}
.lcst-tile{flex:0 0 132px;border:1px solid #262626;border-radius:6px;background:#111;padding:3px;display:flex;flex-direction:column;gap:3px}
.lcst-tile .th{height:70px;border-radius:3px;background:#111 center/cover no-repeat;position:relative;cursor:pointer;display:flex;align-items:center;justify-content:center;color:#666;font-size:11px}
.lcst-tile .th b{position:absolute;top:2px;left:3px;background:#1a5a7a;color:#fff;border-radius:3px;padding:0 4px;font-size:11px}
.lcst-tile .st{position:absolute;bottom:2px;left:3px;right:3px;background:#000b;border-radius:3px;padding:0 3px;font-size:10px;color:#ddd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:left}
.lcst-tile .lcst-row{gap:2px;justify-content:space-between}
.lcst-tile .lcst-b{padding:1px 5px;font-size:11px}
.lcst-tile.chg{border-color:#c9a227}
.lcst-tile.lck{border-color:#5a8fd8}
.lcst-play{position:fixed;inset:8% 12%;background:#181818;border:1px solid #555;border-radius:8px;z-index:10000;display:flex;flex-direction:column;gap:6px;padding:8px;font:12px Arial,sans-serif;color:#ddd;box-shadow:0 10px 40px #000}
.lcst-play video{flex:1;min-height:0;width:100%;background:#000;border-radius:4px}
.lcst-slot.sm{width:54px;height:30px;flex:0 0 54px;font-size:10px}
.lcst-lock{opacity:.35;pointer-events:none}
select.lcst-lock,select:disabled{opacity:.4;cursor:default}
.lcst-b.on{background:#3a3220;border-color:#eb4;color:#eb4}
.lcst-vw{display:flex;gap:6px;align-items:flex-start}
.lcst-bk{display:flex;flex-direction:column;gap:4px;margin:4px 0}
.lcst-bk canvas{border-radius:4px;cursor:crosshair;touch-action:none;max-width:100%}
.lcst-vw canvas{display:block;border-radius:4px;background:#111;touch-action:none}
.lcst-vw [data-vv]{flex:3;min-width:0;aspect-ratio:16/9;cursor:grab}
.lcst-vw [data-vs]{flex:2;min-width:0;aspect-ratio:2/1;cursor:crosshair}
.lcst-vn{width:54px}
`;

function ensureStyle() {
  if (document.getElementById("lcst-style")) return;
  const s = document.createElement("style");
  s.id = "lcst-style";
  s.textContent = CSS;
  document.head.appendChild(s);
}

const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const uid = (p) => p + Math.random().toString(36).slice(2, 8);
const opts = (list, value, labels = {}) => list.map((v) => `<option value="${esc(v)}"${v === value ? " selected" : ""}>${esc(labels[v] ?? v)}</option>`).join("");
const words = (t) => (String(t || "").match(/[A-Za-z0-9']+/g) || []).length;
const TRASH = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 3h6l1 2h4v2H4V5h4zM6 9h12l-1 12H7z"/></svg>`;
const viewURL = (pic) => (pic?.filename ? api.apiURL(`/view?${new URLSearchParams({ filename: pic.filename, subfolder: pic.subfolder || "", type: pic.type || "input" })}`) : "");

function blank() {
  return { version: 1, project: "my_movie", look: "", cast: [], outfits: [], places: [], props: [], shots: [], pitch: "", model: "", saved: 0 };
}

function getData(node) {
  try {
    const d = JSON.parse(widget(node, "story")?.value || "{}");
    const b = blank();
    for (const k of Object.keys(b)) if (d[k] !== undefined) b[k] = d[k];
    for (const k of ["cast", "outfits", "places", "props", "shots"]) if (!Array.isArray(b[k])) b[k] = [];
    return b;
  } catch (_) {
    return blank();
  }
}

function setData(node, data, rerender) {
  const w = widget(node, "story");
  if (w) w.value = JSON.stringify(data);
  if (rerender) render(node);
  schedulePreview(node);
  fit(node);
  try {
    app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.();
  } catch (_) {}
}

function hideJson(node) {
  const w = widget(node, "story");
  if (!w || w._lcHidden) return;
  w._lcHidden = true;
  w.type = "hidden";
  w.computeSize = () => [0, -4];
  try {
    w.hidden = true;
  } catch (_) {}
  if (w.options) w.options.hidden = true;
}

function height(node) {
  const h = node.__lcst?.inner.offsetHeight || 0;
  if (h > 0) node.__lcstH = Math.ceil(h) + 8;
  return node.__lcstH || 500;
}

function fit(node) {
  if (!node.__lcst) return;
  requestAnimationFrame(() => {
    if (!node.__lcst.inner.offsetHeight) return;
    const contentH = height(node);
    const grew = contentH > (node.__lcstPrev || 0) + 1;
    node.__lcstPrev = contentH;
    const want = node.computeSize()[1] - Math.min(contentH, MIN_PANEL) + Math.min(contentH, MAX_AUTO);
    if (grew && node.size[1] < want - 0.5) node.setSize([Math.max(node.size[0], MIN_W), want]);
    node.setDirtyCanvas?.(true, true);
  });
}

/** A section folded or opened: the node's bottom edge moves by exactly the height the panel lost or gained, so
 * opening it again gives back what folding took. */
function shrink(node, before) {
  const inner = node.__lcst?.inner;
  if (!inner || !before) return;
  const delta = inner.offsetHeight - before;
  node.__lcstPrev = height(node);
  node.setSize([node.size[0], Math.max(200, node.size[1] + delta)]);
  node.setDirtyCanvas?.(true, true);
}

function status(node, text, err, quiet) {
  const el = node.__lcst?.wrap.querySelector("[data-status]");
  // the shot count (quiet) never covers a message from the last 10 s
  if (quiet && Date.now() - (node.__lcstMsgT || 0) < 10000) return;
  if (!quiet) node.__lcstMsgT = Date.now();
  if (el) {
    el.textContent = text;
    el.style.color = err ? "#ff8c8c" : "#9fd3ff";
    el.title = text;
  }
}

function ensureFace(node) {
  if (node.__lcst) return node.__lcst;
  ensureStyle();
  const wrap = document.createElement("div");
  wrap.style.cssText = "overflow-y:auto;overflow-x:hidden;height:100%;box-sizing:border-box;background:#0a0a0a;border-radius:6px";
  const inner = document.createElement("div");
  inner.className = "lcst";
  wrap.appendChild(inner);
  wrap.addEventListener("keydown", (e) => e.stopPropagation()); // typing must not trigger canvas shortcuts
  const dom = node.addDOMWidget("lc_story_face", "LC_STORY_FACE", wrap, { getMinHeight: () => Math.min(height(node), MIN_PANEL), serialize: false });
  dom.serializeValue = () => undefined;
  // the frontend sizes the panel from widget.width when it is set, and LiteGraph can leave a stale width there (the
  // panel then stays narrow inside a wide node, even after a resize): the panel is always the node's width
  try {
    Object.defineProperty(dom, "width", { get: () => node.size?.[0], set: () => {}, configurable: true });
  } catch (_) {}
  node.__lcst = { wrap, inner, preview: null };
  try {
    new ResizeObserver(() => fit(node)).observe(inner);
  } catch (_) {}
  return node.__lcst;
}

// ------------------------------------------------------------------ pictures
async function upload(file, project) {
  const body = new FormData();
  body.append("image", file);
  const safe = String(project || "story").replace(/[^A-Za-z0-9 _.-]/g, "_");
  body.append("subfolder", `lc_h3_refs/${safe.replace(/[ .]/g, "") ? safe : "story"}`); // "." / ".." would leave lc_h3_refs/
  body.append("type", "input");
  const r = await api.fetchApi("/upload/image", { method: "POST", body });
  if (r.status !== 200) throw new Error(`upload failed (${r.status})`);
  const j = await r.json();
  return { type: "input", subfolder: j.subfolder || "", filename: j.name };
}

/** A story that uses library pictures (the sample story, a shared one): fetch the ones not on disk yet, then redraw. */
async function fetchLibraryPics(node) {
  const w = widget(node, "story");
  if (!String(w?.value || "").includes('"lc_h3_library')) return;
  try {
    const j = await api.fetchApi("/lc_h3/library/ensure", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ story: getData(node) }) }).then((r) => r.json());
    if (j.error) throw new Error(j.error);
    if (j.fetched) render(node);
  } catch (err) {
    status(node, `📚 Library pictures: ${err.message}`, true);
  }
}

/** A cast or prop picture onto plain grey (BiRefNet on the server): a card's own background leaks into every shot. */
async function onGrey(node, pic) {
  if (String(pic?.subfolder || "").startsWith("lc_h3_library")) return pic; // library pictures are on grey already
  status(node, "✂️ Taking the background out…");
  try {
    const j = await api.fetchApi("/lc_h3/story/cutout", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project: getData(node).project, pic }) }).then((r) => r.json());
    if (j.error) throw new Error(j.error);
    status(node, "✂️ Background taken out: the card is on plain grey.");
    return j.pic || pic;
  } catch (err) {
    status(node, `The background is still in: ${err.message}`, true);
    return pic;
  }
}

/** A picture dragged from the canvas (a preview) or a web page: its /view URL -> slot. */
function picFromDrop(dt) {
  const raw = dt.getData("text/uri-list") || dt.getData("text/plain") || (dt.getData("text/html").match(/src="([^"]+)"/) || [])[1] || "";
  try {
    const u = new URL(raw, location.href);
    if (!u.pathname.endsWith("/view")) return null;
    const p = u.searchParams;
    return p.get("filename") ? { type: p.get("type") || "output", subfolder: p.get("subfolder") || "", filename: p.get("filename") } : null;
  } catch (_) {
    return null;
  }
}

function openPicker(node, onPick, accept = "image/*", libCat = "") {
  document.querySelector(".lcst-pick")?.remove();
  const box = document.createElement("div");
  box.className = "lcst-pick";
  box.innerHTML = `<div class="lcst-pick-top"><b style="color:#eb4">Pick a picture</b>
      <button class="lcst-b on" data-w="input">input</button><button class="lcst-b" data-w="output">output</button>
      ${libCat ? `<button class="lcst-b" data-w="library" title="Ready-made ${libCat} pictures. A ⬇ one downloads on the click.">📚 Library</button>` : ""}
      <button class="lcst-b" data-up>⬆ Upload…</button>
      <input data-find placeholder="filter by name" style="background:#1e1e1e;color:#ddd;border:1px solid #444;border-radius:4px;padding:3px 5px;width:180px">
      <span style="flex:1"></span><button class="lcst-b" data-close>✕</button></div>
    <div class="lcst-pick-grid"></div>`;
  document.body.appendChild(box);
  const grid = box.querySelector(".lcst-pick-grid");
  const close = () => box.remove();
  const load = async (where) => {
    box.querySelectorAll("[data-w]").forEach((b) => b.classList.toggle("on", b.dataset.w === where));
    grid.innerHTML = `<span class="lcst-tip">loading…</span>`;
    if (where === "library") return loadLibrary();
    try {
      const j = await api.fetchApi(`/lc_h3/story/images?where=${where}&t=${Date.now()}`).then((r) => r.json());
      grid.innerHTML = "";
      const want = (box.querySelector("[data-find]").value || "").toLowerCase();
      for (const pic of j.images || []) {
        const path = `${pic.subfolder ? pic.subfolder + "/" : ""}${pic.filename}`;
        if (want && !path.toLowerCase().includes(want)) continue;
        const d = document.createElement("div");
        d.title = path;
        // an <img loading=lazy> per tile: a folder of hundreds only loads what is on screen
        d.innerHTML = `<img loading="lazy" src="${viewURL(pic)}&preview=webp;60" style="width:100%;height:100%;object-fit:cover;border-radius:3px"><span>${esc(pic.filename)}</span>`;
        d.onclick = () => {
          close();
          onPick(pic);
        };
        grid.appendChild(d);
      }
      if (!grid.children.length) grid.innerHTML = `<span class="lcst-tip">No pictures here yet.</span>`;
    } catch (e) {
      grid.innerHTML = `<span class="lcst-stop">${esc(e.message)}</span>`;
    }
  };
  // the library: thumbnails ship with the pack, the full picture downloads on the click (once) and lands in input/lc_h3_library/
  const loadLibrary = async () => {
    try {
      const j = await api.fetchApi(`/lc_h3/library?category=${encodeURIComponent(libCat)}&t=${Date.now()}`).then((r) => r.json());
      grid.innerHTML = "";
      const want = (box.querySelector("[data-find]").value || "").toLowerCase();
      for (const it of j.items || []) {
        const words = [it.name, it.text, it.file, it.season, it.gender, it.theme].join(" ").toLowerCase();
        if (want && !words.includes(want)) continue;
        const d = document.createElement("div");
        d.title = `${it.name}: ${it.text}${it.have ? "" : "\n⬇ downloads on the click"}`;
        d.innerHTML = `<img loading="lazy" src="${api.apiURL(it.thumb)}" style="width:100%;height:100%;object-fit:cover;border-radius:3px"><span>${it.have ? "" : "⬇ "}${esc(it.name)}</span>`;
        d.onclick = async () => {
          grid.innerHTML = `<span class="lcst-tip">${it.have ? "Loading" : "Downloading"} ${esc(it.name)}…</span>`;
          try {
            const r = await api.fetchApi("/lc_h3/library/get", { method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ stem: it.stem }) }).then((x) => x.json());
            if (r.error) throw new Error(r.error);
            close();
            onPick(r.pic, r.item);
          } catch (e) {
            grid.innerHTML = `<span class="lcst-stop">${esc(e.message)}</span>`;
          }
        };
        grid.appendChild(d);
      }
      if (!grid.children.length) grid.innerHTML = `<span class="lcst-tip">Nothing in the library matches.</span>`;
    } catch (e) {
      grid.innerHTML = `<span class="lcst-stop">${esc(e.message)}</span>`;
    }
  };
  box.querySelectorAll("[data-w]").forEach((b) => (b.onclick = () => load(b.dataset.w)));
  let findT = 0;
  box.querySelector("[data-find]").oninput = () => {
    clearTimeout(findT);
    findT = setTimeout(() => load(box.querySelector("[data-w].on")?.dataset.w || "input"), 250);
  };
  box.addEventListener("keydown", (e) => e.stopPropagation(), true); // typing in the filter must not hit canvas shortcuts
  box.querySelector("[data-close]").onclick = close;
  box.querySelector("[data-up]").onclick = () => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = accept;
    inp.onchange = async () => {
      if (!inp.files?.[0]) return;
      try {
        const pic = await upload(inp.files[0], getData(node).project);
        close();
        onPick(pic);
      } catch (e) {
        status(node, e.message, true);
      }
    };
    inp.click();
  };
  box.addEventListener("keydown", (e) => e.key === "Escape" && close());
  load("input");
}

/** Click = picker, drop = file upload or a canvas picture. */
function wireSlot(node, el, onPick, libCat = "") {
  el.onclick = (e) => {
    if (e.target.classList.contains("x")) return;
    openPicker(node, onPick, "image/*", libCat);
  };
  el.ondragover = (e) => {
    e.preventDefault();
    e.stopPropagation();
    el.classList.add("drop");
  };
  el.ondragleave = () => el.classList.remove("drop");
  el.ondrop = async (e) => {
    e.preventDefault();
    e.stopPropagation(); // the canvas would load a dropped picture as a new node
    el.classList.remove("drop");
    const f = e.dataTransfer?.files?.[0];
    try {
      if (f) onPick(await upload(f, getData(node).project));
      else {
        const pic = picFromDrop(e.dataTransfer);
        if (pic) onPick(pic);
        else status(node, "That isn't a picture I can use. Drop a file, or a picture from a preview.", true);
      }
    } catch (err) {
      status(node, err.message, true);
    }
  };
}

// a storyboard slot (shot start / end): ↻ empties it and the next LC H3 Shot Card run (shot blank) makes it again
const REDO = `<button class="x" data-clr title="Redo: empties this slot. The next LC H3 Shot Card run (shot left blank) makes it again; or drop in your own picture.">↻</button>`;
// a card picture's crop (fractions x, y, w, h) shown in the slot: the cropped part fills it
const cropCSS = (c) => {
  if (!Array.isArray(c) || c.length !== 4) return "";
  const [x, y, w, h] = c.map(Number);
  if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return "";
  const pos = (o, s) => (s >= 1 ? 0 : (o / (1 - s)) * 100);
  return `background-size:${100 / w}% ${100 / h}%;background-position:${pos(x, w)}% ${pos(y, h)}%;`;
};
const slotHTML = (pic, attrs, label, crop = false) =>
  `<div class="lcst-slot" ${attrs} style="${pic?.filename ? `background-image:url('${viewURL(pic)}&preview=webp;70');${cropCSS(pic.crop)}` : ""}" title="${esc(pic?.filename ? `${pic.filename}\nClick to change, or drop a picture here` : "Click to pick a picture, or drop one here")}">${pic?.filename ? `<button class="x" data-clr>×</button>${crop ? `<button class="x cr" data-crop title="Crop: what of this picture the card uses (a head-and-shoulders face out of a full-length photo, just the coat of an outfit)">✂</button>` : ""}` : esc(label)}</div>`;

// ------------------------------------------------------------------ the wardrobe
// {person: [{id, label, card, pic}]}, the same as the server: the cast card's own 👗 outfit is outfit 1, then the person's
// outfit cards in order. An outfit card's label is its own name ("the red dress"): a name that agrees with the picture
// held it 8 of 8, "outfit 2" 4 of 8 (the headshot's clothes won).
function robeOf(d) {
  const out = {};
  for (const c of d.cast || []) {
    const nm = String(c.name || "").trim();
    if (!nm) continue;
    out[nm] = c.outfit_picture?.filename ? [{ id: `${c.id}_dress`, label: "outfit 1", card: null, pic: c.outfit_picture }] : [];
  }
  for (const o of d.outfits || []) {
    const who = String(o.who || "").trim();
    if (!String(o.name || "").trim() || !out[who]) continue;
    const name = String(o.name).trim().replace(/^(the|a|an)\s+/i, "").replace(/\.$/, "");
    out[who].push({ id: `${o.id}_outfit`, label: name ? `the ${name}` : `outfit ${out[who].length + 1}`, card: o, pic: o.picture?.filename ? o.picture : null });
  }
  return out;
}

// ------------------------------------------------------------------ ✂ crop a card picture
// The box, its 8 handles, the minimum size and the aspect lock work like LC Image Crop (LC123). The crop is saved on
// the picture as fractions [x, y, w, h]; the server crops every picture it sends (H3 references, the Shot Card).
const CROP_ASPECTS = { free: 0, "1:1": 1, "4:5": 4 / 5, "3:4": 3 / 4, "16:9": 16 / 9 };
const CROP_MIN = 0.04;

function openCrop(pic, onDone) {
  document.querySelector(".lcst-crop")?.remove();
  const box = document.createElement("div");
  box.className = "lcst-crop";
  box.innerHTML = `<div class="lcst-row"><b style="color:#eb4">✂ Crop</b><span class="lcst-tip" style="flex:1">Drag the box or its corners. The card uses only what is inside.</span>
      ${Object.keys(CROP_ASPECTS).map((k) => `<button class="lcst-b" data-ar="${k}">${k}</button>`).join("")}
      <button class="lcst-b" data-cr-reset title="The whole picture">Reset</button><button class="lcst-b pri" data-cr-ok>Save</button><button class="lcst-b" data-cr-x>✕</button></div>
    <div class="lcst-crop-stage"><div class="lcst-crop-wrap"><img draggable="false"><div class="lcst-crop-box">${["nw", "n", "ne", "e", "se", "s", "sw", "w"].map((h) => `<i data-h="${h}"></i>`).join("")}</div></div></div>`;
  document.body.appendChild(box);
  const img = box.querySelector("img");
  const frame = box.querySelector(".lcst-crop-box");
  const ok = (v) => Array.isArray(v) && v.length === 4 && v.every((x) => Number.isFinite(Number(x))) && Number(v[2]) > 0 && Number(v[3]) > 0;
  let c = ok(pic.crop) ? pic.crop.map(Number) : [0, 0, 1, 1];
  let ar = 0; // width / height of the crop in pixels, 0 = free
  const draw = () => {
    const [x, y, w, h] = c;
    Object.assign(frame.style, { left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` });
    box.querySelectorAll("[data-ar]").forEach((b) => b.classList.toggle("on", CROP_ASPECTS[b.dataset.ar] === ar));
  };
  const clamp = () => {
    let [x, y, w, h] = c;
    w = Math.min(1, Math.max(CROP_MIN, w));
    h = Math.min(1, Math.max(CROP_MIN, h));
    x = Math.min(1 - w, Math.max(0, x));
    y = Math.min(1 - h, Math.max(0, y));
    c = [x, y, w, h];
  };
  // the picture's own shape turns a pixel aspect into fractions
  const fitAspect = (anchorBottom, anchorRight) => {
    if (!ar || !img.naturalWidth) return;
    const pa = img.naturalWidth / img.naturalHeight;
    let [x, y, w, h] = c;
    const want = (w * pa) / ar; // the height that gives the aspect at this width
    if (want <= 1) {
      if (anchorBottom) y += h - want;
      h = want;
    } else {
      const nw = (h * ar) / pa;
      if (anchorRight) x += w - nw;
      w = nw;
    }
    c = [x, y, w, h];
    clamp();
  };
  const close = () => box.remove();
  img.onload = draw;
  img.src = viewURL(pic);
  let drag = null;
  frame.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    frame.setPointerCapture(e.pointerId);
    drag = { h: e.target.dataset.h || "move", sx: e.clientX, sy: e.clientY, c0: [...c] };
  });
  frame.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const r = img.getBoundingClientRect();
    if (!r.width || !r.height) return; // not laid out yet: no size to measure the drag against
    const dx = (e.clientX - drag.sx) / r.width, dy = (e.clientY - drag.sy) / r.height;
    let [x, y, w, h] = drag.c0;
    const hd = drag.h;
    if (hd === "move") {
      x += dx;
      y += dy;
    } else {
      if (hd.includes("w")) { x += dx; w -= dx; }
      if (hd.includes("e")) w += dx;
      if (hd.includes("n")) { y += dy; h -= dy; }
      if (hd.includes("s")) h += dy;
    }
    if (w < CROP_MIN) { if (hd.includes("w")) x -= CROP_MIN - w; w = CROP_MIN; }
    if (h < CROP_MIN) { if (hd.includes("n")) y -= CROP_MIN - h; h = CROP_MIN; }
    c = [x, y, w, h];
    clamp();
    if (hd !== "move") fitAspect(hd.includes("n"), hd.includes("w"));
    draw();
  });
  frame.addEventListener("pointerup", () => (drag = null));
  box.querySelectorAll("[data-ar]").forEach((b) => (b.onclick = () => {
    ar = CROP_ASPECTS[b.dataset.ar];
    fitAspect(false, false);
    draw();
  }));
  box.querySelector("[data-cr-reset]").onclick = () => {
    c = [0, 0, 1, 1];
    ar = 0;
    draw();
  };
  box.querySelector("[data-cr-ok]").onclick = () => {
    if (!ok(c)) return close(); // never save a broken crop
    const whole = c.every((v, k) => Math.abs(v - [0, 0, 1, 1][k]) < 1e-3);
    onDone(whole ? null : c.map((v) => Math.round(v * 10000) / 10000));
    close();
  };
  box.querySelector("[data-cr-x]").onclick = close;
  box.addEventListener("keydown", (e) => (e.stopPropagation(), e.key === "Escape" && close()), true);
}

// ------------------------------------------------------------------ 360 angle per shot
// Same projection as LC 360 View (MaskMaker) and the server's cut_view, so what you aim is what H3 gets.
const PANO_MAX = 2048;
const panos = new Map(); // url -> {img, data} | "loading"
const VIEW_DEFAULT = { yaw: 0, pitch: -5, fov: 60 };

function panoFor(url, done) {
  const p = panos.get(url);
  if (p && p !== "loading") return p;
  if (!p) {
    panos.set(url, "loading");
    const img = new Image();
    img.onload = () => {
      const s = Math.min(1, PANO_MAX / img.width);
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * s);
      c.height = Math.round(img.height * s);
      const g = c.getContext("2d", { willReadFrequently: true });
      g.drawImage(img, 0, 0, c.width, c.height);
      panos.set(url, { img, data: g.getImageData(0, 0, c.width, c.height) });
      done();
    };
    img.onerror = () => panos.delete(url);
    img.src = url;
  }
  return null;
}

const TIGHT_FRAMINGS = ["medium shot", "close-up", "extreme close-up", "over the shoulder"];
/** The angle the server gives a close cut with no 🌐 angle of its own (reverse angle across from a person who has one). */
function autoView(d, i, place) {
  const s = d.shots[i];
  if (!TIGHT_FRAMINGS.includes(s.framing || "wide shot")) return null;
  const same = (k) => shotPlace(d, k) === place;
  const other = d.shots.find((o, k) => same(k) && (o.subject || "everyone") !== (s.subject || "everyone") && o.view?.on);
  if (other) return { yaw: ((((Number(other.view.yaw) || 0) + 360) % 360) - 180), pitch: other.view.pitch ?? -1, fov: other.view.fov ?? 55 };
  return { yaw: 0, pitch: -1, fov: 55 };
}

/**
 * Blocking on a shot card: boxes for people and props, x/y/w/h as 0..1 of the frame, saved as shot.boxes
 * [{name, x, y, w, h}]. The background is the shot's start frame, else its place's room picture.
 */
function blockingEditor(node, d, i, box, onChange) {
  const s = d.shots[i];
  s.boxes = Array.isArray(s.boxes) ? s.boxes : [];
  const cv = box.querySelector("[data-bkc]");
  const rows = box.querySelector("[data-bkr]");
  const names = d.cast.map((c) => String(c.name || "").trim()).filter(Boolean);
  const place = shotPlace(d, i);
  const angle = !s.start?.filename && place?.panorama?.filename ? s.view?.on ? s.view : autoView(d, i, place) : null;
  const panoURL = angle ? viewURL(place.panorama) : "";
  const bgPic = angle ? place.panorama : s.start?.filename ? s.start : place?.picture;
  let bgCache = null;
  const img = new Image();
  let sel = s.boxes.length ? 0 : -1;
  let drag = null;
  const colors = ["#e5b23a", "#4fc3f7", "#ef6f6c", "#8bc34a", "#ba68c8", "#ff9800"];
  const paint = () => {
    const W = Math.max(160, box.clientWidth || 320), H = Math.round(W * 9 / 16);
    cv.width = W;
    cv.height = H;
    cv.style.width = W + "px";
    cv.style.height = H + "px";
    const g = cv.getContext("2d");
    g.fillStyle = "#111";
    g.fillRect(0, 0, W, H);
    const pano = angle ? panoFor(panoURL, paint) : null;
    if (angle && pano) {
      const key = `${W}x${H}|${angle.yaw}|${angle.pitch}|${angle.fov}|${place.front?.yaw}`;
      if (bgCache?.key === key) g.putImageData(bgCache.data, 0, 0);
      else {
        renderView(g, W, H, pano, { ...angle, yaw: (Number(angle.yaw) || 0) + (Number(place.front?.yaw) || 0) });
        bgCache = { key, data: g.getImageData(0, 0, W, H) };
      }
      g.fillStyle = "rgba(0,0,0,.55)";
      g.fillRect(0, H - 16, W, 16);
      g.fillStyle = "#ddd";
      g.font = "11px sans-serif";
      g.fillText(s.view?.on ? "background: this shot's 🌐 angle" : "background: the angle this shot gets by itself", 6, H - 4);
    } else if (!angle && img.complete && img.naturalWidth) g.drawImage(img, 0, 0, W, H);
    else {
      g.fillStyle = "#555";
      g.font = "12px sans-serif";
      g.fillText(bgPic ? "loading…" : "no picture yet: boxes still work", 10, 20);
    }
    s.boxes.forEach((b, k) => {
      const c = colors[k % colors.length];
      g.strokeStyle = c;
      g.lineWidth = k === sel ? 3 : 2;
      g.strokeRect(b.x * W, b.y * H, b.w * W, b.h * H);
      g.fillStyle = c;
      g.font = "bold 12px sans-serif";
      g.fillText(`${k + 1} ${b.name || "?"}`, b.x * W + 4, b.y * H + 14);
      g.fillRect((b.x + b.w) * W - 6, (b.y + b.h) * H - 6, 6, 6);
    });
  };
  const list = () => {
    rows.innerHTML = s.boxes.map((b, k) => `<div class="lcst-row" style="gap:4px">
        <span class="lcst-tip" style="color:${colors[k % colors.length]}">${k + 1}</span>
        <select data-bn="${k}" style="width:120px">${["", ...names].map((n) => `<option value="${esc(n)}"${n === b.name ? " selected" : ""}>${n ? esc(n) : "prop / other…"}</option>`).join("")}</select>
        <input data-bt="${k}" value="${esc(names.includes(b.name) ? "" : b.name || "")}" placeholder="a prop: the red umbrella" style="flex:1"${names.includes(b.name) ? " disabled" : ""}>
        <button class="lcst-b" data-bx="${k}" title="Remove this box">${TRASH}</button>
      </div>`).join("") || `<span class="lcst-tip">Drag on the picture to draw a box for a person or a prop.</span>`;
    rows.querySelectorAll("[data-bn]").forEach((el) => (el.onchange = () => {
      s.boxes[Number(el.dataset.bn)].name = el.value;
      onChange();
      list();
      paint();
    }));
    rows.querySelectorAll("[data-bt]").forEach((el) => (el.oninput = () => {
      s.boxes[Number(el.dataset.bt)].name = el.value;
      onChange();
      paint();
    }));
    rows.querySelectorAll("[data-bx]").forEach((el) => (el.onclick = () => {
      s.boxes.splice(Number(el.dataset.bx), 1);
      sel = Math.min(sel, s.boxes.length - 1);
      onChange();
      list();
      paint();
    }));
  };
  const at = (e) => {
    const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  };
  cv.onpointerdown = (e) => {
    e.stopPropagation();
    const [x, y] = at(e);
    const k = s.boxes.findIndex((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h);
    if (k >= 0) {
      const b = s.boxes[k];
      const corner = Math.abs(x - (b.x + b.w)) < 0.03 && Math.abs(y - (b.y + b.h)) < 0.05;
      drag = { k, mode: corner ? "size" : "move", x, y, b0: { ...b } };
      sel = k;
    } else {
      s.boxes.push({ name: names.find((n) => !s.boxes.some((b) => b.name === n)) || "", x, y, w: 0.01, h: 0.01 });
      sel = s.boxes.length - 1;
      drag = { k: sel, mode: "size", x, y, b0: { ...s.boxes[sel] } };
    }
    cv.setPointerCapture(e.pointerId);
    paint();
  };
  cv.onpointermove = (e) => {
    if (!drag) return;
    const [x, y] = at(e);
    const b = s.boxes[drag.k];
    const cl = (v) => Math.min(1, Math.max(0, v));
    if (drag.mode === "move") {
      b.x = Math.min(cl(drag.b0.x + x - drag.x), 1 - b.w);
      b.y = Math.min(cl(drag.b0.y + y - drag.y), 1 - b.h);
    } else {
      b.w = Math.max(0.02, cl(x) - b.x);
      b.h = Math.max(0.02, cl(y) - b.y);
    }
    paint();
  };
  cv.onpointerup = () => {
    if (!drag) return;
    drag = null;
    s.boxes = s.boxes.filter((b) => b.w > 0.02 || b.h > 0.02).map((b) => ({ ...b, x: +b.x.toFixed(4), y: +b.y.toFixed(4), w: +b.w.toFixed(4), h: +b.h.toFixed(4) }));
    onChange();
    list();
    paint();
  };
  if (bgPic && !angle) {
    img.onload = paint;
    img.src = viewURL(bgPic);
  }
  list();
  requestAnimationFrame(paint);
}

/** A keep-rolling shot's source: [the cut that started the run, its number], or null for a cut. Keep rolling carries
 * the framing, subject, place, 🌐 angle and clothes straight on from it, with no start frame or blocking of its own
 * (the server's inherit_rolls does the same), so those controls are greyed out on the card. */
function rollSrc(d, i) {
  if (!i || d.shots[i]?.join !== "roll") return null;
  let k = i;
  while (k > 0 && d.shots[k]?.join === "roll") k--;
  return [d.shots[k], k + 1];
}

/** The place a shot is in: its own pick, else the shot before's, else the first place (as the server works it out). */
function shotPlace(d, i) {
  let where = "";
  for (let k = 0; k <= i; k++) where = String(d.shots[k]?.where || "").trim() || where || String(d.places[0]?.name || "");
  const key = where.toLowerCase();
  return d.places.find((p) => String(p.name || "").trim().toLowerCase() === key) || null;
}

/** A flat view out of the panorama (pano = panoFor's result) drawn onto g, W x H. */
function renderView(g, W, H, pano, v) {
  const src = pano.data, PW = src.width, PH = src.height, sd = src.data;
  const out = g.createImageData(W, H), od = out.data;
  const f = W / 2 / Math.tan((v.fov * Math.PI) / 360);
  const p = (v.pitch * Math.PI) / 180, t = (v.yaw * Math.PI) / 180;
  const cp = Math.cos(p), sp = Math.sin(p), ct = Math.cos(t), st = Math.sin(t);
  for (let j = 0; j < H; j++) {
    const y = -(j - H / 2 + 0.5);
    for (let i = 0; i < W; i++) {
      const x = i - W / 2 + 0.5;
      const n = Math.hypot(x, y, f);
      const dx = x / n, dy = y / n, dz = f / n;
      const dy2 = dy * cp + dz * sp, dz2 = -dy * sp + dz * cp;
      const dx3 = dx * ct + dz2 * st, dz3 = -dx * st + dz2 * ct;
      let u = Math.floor((Math.atan2(dx3, dz3) / (2 * Math.PI) + 0.5) * PW);
      const vv2 = Math.min(PH - 1, Math.max(0, Math.floor((0.5 - Math.asin(Math.max(-1, Math.min(1, dy2))) / Math.PI) * PH)));
      u = ((u % PW) + PW) % PW;
      const si = (vv2 * PW + u) * 4, oi = (j * W + i) * 4;
      od[oi] = sd[si];
      od[oi + 1] = sd[si + 1];
      od[oi + 2] = sd[si + 2];
      od[oi + 3] = 255;
    }
  }
  g.putImageData(out, 0, 0);
}
function paintView(vv, vs, pano, v) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = Math.max(16, Math.round(vv.clientWidth * dpr)), H = Math.max(9, Math.round(vv.clientHeight * dpr));
  vv.width = W;
  vv.height = H;
  const g = vv.getContext("2d");
  if (!pano) {
    g.fillStyle = "#888";
    g.font = `${12 * dpr}px Arial`;
    g.textAlign = "center";
    g.fillText("loading the panorama…", W / 2, H / 2);
    return;
  }
  renderView(g, W, H, pano, v);
  // the strip: the whole panorama with the view's spot marked
  const SW = Math.max(16, Math.round(vs.clientWidth * dpr)), SH = Math.max(8, Math.round(vs.clientHeight * dpr));
  vs.width = SW;
  vs.height = SH;
  const s = vs.getContext("2d");
  s.drawImage(pano.img, 0, 0, SW, SH);
  const vfov = (2 * Math.atan(Math.tan((v.fov * Math.PI) / 360) * (H / W)) * 180) / Math.PI;
  const cx = (v.yaw / 360 + 0.5) * SW, cy = (0.5 - v.pitch / 180) * SH, bw = (v.fov / 360) * SW, bh = (vfov / 180) * SH;
  s.strokeStyle = "#eb4";
  s.lineWidth = 1.5 * dpr;
  for (const off of [-SW, 0, SW]) s.strokeRect(cx - bw / 2 + off, cy - bh / 2, bw, bh);
}

/** Drag the view to look around, the wheel (or Shift + drag) zooms, click the strip to jump. v.yaw counts from
 * `front` (the place's wide-shot direction in the panorama), so 90 is always the right side of the room. */
function wireView(box, v, url, front, save) {
  const off = typeof front === "function" ? front : () => front;
  const vv = box.querySelector("[data-vv]"), vs = box.querySelector("[data-vs]");
  const row = box.previousElementSibling; // the row with the numbers
  const nums = { yaw: row?.querySelector('[data-vn="yaw"]'), pitch: row?.querySelector('[data-vn="pitch"]'), fov: row?.querySelector('[data-vn="fov"]') };
  const clamp = () => {
    v.yaw = Math.round((((((Number(v.yaw) || 0) + 180) % 360) + 360) % 360 - 180) * 2) / 2;
    v.pitch = Math.round(Math.max(-85, Math.min(85, Number(v.pitch) || 0)) * 2) / 2;
    v.fov = Math.round(Math.max(10, Math.min(150, Number(v.fov) || 60)) * 2) / 2;
  };
  const draw = () => {
    clamp();
    for (const k of Object.keys(nums)) if (nums[k] && document.activeElement !== nums[k]) nums[k].value = v[k];
    paintView(vv, vs, panoFor(url, draw), { ...v, yaw: v.yaw + off() });
    box._lcDraw = draw;
  };
  let drag = null, saveT = 0;
  const later = () => {
    clearTimeout(saveT);
    saveT = setTimeout(() => save(false), 250);
  };
  vv.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    vv.setPointerCapture(e.pointerId);
    vv.style.cursor = "grabbing";
    drag = { x: e.clientX, y: e.clientY, yaw: v.yaw, pitch: v.pitch, fov: v.fov, w: vv.getBoundingClientRect().width };
  });
  vv.addEventListener("pointermove", (e) => {
    if (!drag) return;
    e.stopPropagation();
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (e.shiftKey) v.fov = drag.fov * Math.exp(dy * 0.01);
    else {
      v.yaw = drag.yaw - (dx * drag.fov) / drag.w; // grab and pull: the picture follows the pointer
      v.pitch = drag.pitch + (dy * drag.fov) / drag.w;
    }
    draw();
  });
  const end = (e) => {
    if (!drag) return;
    drag = null;
    vv.style.cursor = "grab";
    try {
      vv.releasePointerCapture(e.pointerId);
    } catch (_) {}
    save(false);
  };
  vv.addEventListener("pointerup", end);
  vv.addEventListener("pointercancel", end);
  vv.addEventListener("wheel", (e) => {
    e.preventDefault();
    e.stopPropagation();
    v.fov = v.fov * Math.exp(e.deltaY * 0.001);
    draw();
    later();
  }, { passive: false });
  vs.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const r = vs.getBoundingClientRect();
    v.yaw = ((e.clientX - r.left) / r.width - 0.5) * 360 - off();
    v.pitch = (0.5 - (e.clientY - r.top) / r.height) * 180;
    draw();
    save(false);
  });
  for (const k of Object.keys(nums)) {
    if (!nums[k]) continue;
    nums[k].oninput = () => {
      if (nums[k].value === "" || isNaN(Number(nums[k].value))) return;
      v[k] = Number(nums[k].value);
      draw();
      later();
    };
  }
  try {
    new ResizeObserver(() => draw()).observe(vv);
  } catch (_) {}
  draw();
}

// ------------------------------------------------------------------ preview (what H3 gets)
function schedulePreview(node) {
  clearTimeout(node.__lcstT);
  node.__lcstT = setTimeout(() => preview(node), 500);
}

async function preview(node) {
  const d = getData(node);
  try {
    const j = await api.fetchApi("/lc_h3/story/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ story: d, max_shot_seconds: widget(node, "max_shot_seconds")?.value || 20 }),
    }).then((r) => r.json());
    if (j.error) throw new Error(j.error);
    node.__lcst.preview = j;
    paintPreview(node);
    refreshBoard(node); // an edited shot shows ✏️ changed since made
  } catch (e) {
    status(node, `Preview: ${e.message}`, true);
  }
}

function paintPreview(node) {
  const j = node.__lcst?.preview;
  if (!j) return;
  const total = (j.shots || []).reduce((a, s) => a + (s.seconds || 0), 0);
  status(node, `${(j.shots || []).length} shots · about ${Math.round(total)} s`, false, true);
  for (const s of j.shots || []) {
    const i = s.shot - 1;
    const sec = node.__lcst.wrap.querySelector(`[data-s="${i}"][data-f="seconds"]`);
    if (sec) sec.placeholder = `auto ${s.seconds.toFixed(1)} s`;
    const w = node.__lcst.wrap.querySelector(`[data-warn="${i}"]`);
    if (w) w.innerHTML = (s.warnings || []).map((x) => `<div class="${x.startsWith("🛑") ? "lcst-stop" : "lcst-warn"}">${esc(x)}</div>`).join("");
    const pv = node.__lcst.wrap.querySelector(`[data-pv="${i}"]`);
    if (pv && !node.__lcstEditing?.has(getData(node).shots[i]?.id)) paintPrompt(node, pv, s, i);
  }
}

/**
 * The H3 prompt of one shot: the built text per part, or a text box when ✎ Edit is on. A saved hand edit is
 * shot.prompt_edit {"1": text}; the server sends it as that part's text and gives back the automatic one (`auto`).
 */
function paintPrompt(node, pv, s, i) {
  const d = getData(node);
  const shot = d.shots[i];
  if (!shot) return;
  const editing = node.__lcstEditing?.has(shot.id);
  const segs = s.segments || [];
  const head = (g) => `${segs.length > 1 ? `part ${g.index} · ` : ""}${g.mode} · ${g.seconds} s${g.edited ? " · <b>✎ your text: the cards no longer change it</b>" : ""}`;
  if (!editing) {
    pv.innerHTML = segs.map((g) => `<div class="lcst-pvh">${head(g)}</div><pre>${esc(g.text)}</pre>`).join("");
    return;
  }
  pv.innerHTML = segs.map((g, k) => `<div class="lcst-pvh">${head(g)}</div><textarea data-pvt="${k + 1}" spellcheck="false">${esc(g.text)}</textarea>`).join("")
    + `<div class="lcst-row"><button class="lcst-b" data-pvsave>Use this text</button><button class="lcst-b" data-pvcancel>Cancel</button>`
    + `<button class="lcst-b" data-pvauto title="Throw your text away and build it from the cards again">↶ Back to automatic</button></div>`;
  const done = () => {
    node.__lcstEditing.delete(shot.id);
    paintPrompt(node, pv, s, i);
    fit(node);
  };
  pv.querySelector("[data-pvcancel]").onclick = done;
  pv.querySelector("[data-pvauto]").onclick = () => {
    const d2 = getData(node);
    delete d2.shots[i].prompt_edit;
    node.__lcstEditing.delete(shot.id);
    setData(node, d2, true);
  };
  pv.querySelector("[data-pvsave]").onclick = () => {
    const d2 = getData(node);
    const hand = {};
    pv.querySelectorAll("[data-pvt]").forEach((ta) => {
      const g = segs[Number(ta.dataset.pvt) - 1];
      const built = g.edited ? g.auto : g.text;
      if (ta.value.trim() && ta.value.trim() !== built.trim()) hand[ta.dataset.pvt] = ta.value.trim();
    });
    if (Object.keys(hand).length) d2.shots[i].prompt_edit = hand;
    else delete d2.shots[i].prompt_edit;
    node.__lcstEditing.delete(shot.id);
    setData(node, d2, true);
  };
}

// ------------------------------------------------------------------ the board (what is made)
const fileURL = (f) => (f ? `${viewURL(f)}&t=${f.t || 0}` : "");

async function refreshBoard(node) {
  const d = getData(node);
  try {
    const j = await api.fetchApi("/lc_h3/story/board", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ story: d, max_shot_seconds: widget(node, "max_shot_seconds")?.value || 20 }),
    }).then((r) => r.json());
    if (j.error) throw new Error(j.error);
    node.__lcst.board = j;
    paintBoard(node);
  } catch (e) {
    console.warn("[LC H3 Story] board:", e.message);
  }
}

function paintBoard(node) {
  const el = node.__lcst?.wrap.querySelector("[data-board]");
  const j = node.__lcst?.board;
  if (!el || !j) return;
  const d = getData(node);
  const by = Object.fromEntries((j.shots || []).map((s) => [s.shot_id, s]));
  el.innerHTML = d.shots.map((s, i) => {
    const b = by[s.id];
    const segs = b?.segments || [];
    const made = segs.length && segs.every((g) => g.made);
    const some = segs.some((g) => g.made);
    const changed = segs.some((g) => g.changed);
    const broken = segs.some((g) => g.broken);
    const locked = segs.some((g) => g.locked);
    const thumb = segs.find((g) => g.thumb)?.thumb;
    // ↻ queued it: queued until its first segment has a take newer than the click
    const qd = node.__lcst.queued?.[s.id];
    if (qd && (segs[0]?.video?.t || 0) >= qd) delete node.__lcst.queued[s.id];
    const queued = !!node.__lcst.queued?.[s.id];
    const st = queued ? "⏳ queued" : locked ? "🔒 locked" : broken ? "⚠ join broken: ↻ it" : changed ? "✏️ changed since made" : made ? `✅ made${segs.length > 1 ? ` · ${segs.length} parts` : ""}` : some ? "◐ part made" : "⬜ not made";
    return `<div class="lcst-tile${changed || broken ? " chg" : ""}${locked ? " lck" : ""}"${broken ? ` title="This shot rolls on from shot ${i}, which was remade after this take: the join jumps. ↻ this shot to make it carry on from the new ending."` : ""}>
      <div class="th" data-bplay="${i}" style="${thumb ? `background-image:url('${fileURL(thumb)}')` : ""}" title="${some ? "▶ Play" : "Not made yet"}">
        <b>${i + 1}</b>${thumb ? "" : "not made"}<span class="st">${st}</span></div>
      <div class="lcst-row">
        <button class="lcst-b" data-bplay="${i}" ${some ? "" : "disabled"} title="Play this shot (📌 grab a frame as an anchor from the player)">▶</button>
        <button class="lcst-b" data-brepair="${i}" ${locked || !made || segs.length > 1 ? "disabled" : ""} title="${segs.length > 1 ? "Repair works on one-part shots: this one is made in parts" : "Repair: a new take of this shot that keeps its last moment, so a keep-rolling shot after it still joins (the old take is kept in takes/)"}">🩹</button>
        <button class="lcst-b" data-bredo="${i}" ${locked ? "disabled" : ""} title="${made ? "Redo: a whole new take, new ending too. A keep-rolling shot after it then needs ↻ as well (the old take is kept in takes/)" : "Make this shot"}">↻</button>
        <button class="lcst-b" data-block="${i}" ${some ? "" : "disabled"} title="${locked ? "Unlock" : "Lock: ↻ and redo leave it alone"}">${locked ? "🔓" : "🔒"}</button>
      </div></div>`;
  }).join("") || `<span class="lcst-tip">Add a shot to start.</span>`;
  const mv = node.__lcst.wrap.querySelector("[data-bmovie]");
  if (mv) mv.disabled = !j.movie;
  el.querySelectorAll("[data-bplay]").forEach((b) => (b.onclick = () => {
    const s = d.shots[Number(b.dataset.bplay)];
    const segs = (by[s.id]?.segments || []).filter((g) => g.made);
    if (segs.length) openPlayer(node, segs, s, Number(b.dataset.bplay));
  }));
  el.querySelectorAll("[data-bredo]").forEach((b) => (b.onclick = () => makeShot(node, Number(b.dataset.bredo), "redo")));
  el.querySelectorAll("[data-brepair]").forEach((b) => (b.onclick = () => makeShot(node, Number(b.dataset.brepair), "repair")));
  el.querySelectorAll("[data-block]").forEach((b) => (b.onclick = async () => {
    const s = d.shots[Number(b.dataset.block)];
    const locked = (by[s.id]?.segments || []).some((g) => g.locked);
    try {
      const r = await api.fetchApi("/lc_h3/story/lock", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project: d.project, shot_id: s.id, locked: !locked }) }).then((x) => x.json());
      if (r.error) throw new Error(r.error);
    } catch (e) {
      status(node, `Lock: ${e.message}`, true);
    }
    refreshBoard(node);
  }));
}

/** ▶: the shot's parts one after the other (the parts the movie uses); 📌 grabs the frame on screen as an anchor of
 * the shot's place, 🖼 makes it the next shot's start frame. */
function openPlayer(node, segs, shot, i, movie) {
  document.querySelector(".lcst-play")?.remove();
  const box = document.createElement("div");
  box.className = "lcst-play";
  box.innerHTML = `<div class="lcst-row"><b style="color:#eb4">${movie ? "🎬 The movie" : `Shot ${i + 1}`}</b><span data-pinfo class="lcst-tip" style="flex:1"></span>
      ${movie ? "" : `<button class="lcst-b" data-pin title="Room anchor: this frame is shown to H3 as a reference for the shot's place (who sits where, the furniture), so later shots there keep the room. It does not start any shot.">📌 Room anchor</button>`}
      ${movie || !getData(node).shots[i + 1] ? "" : `<button class="lcst-b" data-next title="Shot ${i + 2} opens exactly on this frame (its 🖼 start). A keep-rolling shot ${i + 2} becomes a cut, since rolling already starts from the end of this one.">🖼 Start shot ${i + 2} here</button>`}
      <button class="lcst-b" data-close>✕</button></div><video controls autoplay></video>`;
  document.body.appendChild(box);
  const v = box.querySelector("video");
  let k = 0;
  const play = () => {
    v.src = fileURL(segs[k].video);
    box.querySelector("[data-pinfo]").textContent = segs.length > 1 ? `part ${k + 1} of ${segs.length}` : "";
  };
  v.onended = () => {
    if (k < segs.length - 1) {
      k += 1;
      play();
    }
  };
  play();
  const close = () => box.remove();
  box.querySelector("[data-close]").onclick = close;
  box.addEventListener("keydown", (e) => (e.stopPropagation(), e.key === "Escape" && close()), true);
  const grab = async () => {
    v.pause();
    const d = getData(node);
    const j = await api.fetchApi("/lc_h3/anchor", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project: d.project, index: segs[k].index, time: v.currentTime }) }).then((r) => r.json());
    if (j.error) throw new Error(j.error);
    const [sub, name] = [j.file.slice(0, j.file.lastIndexOf("/")), j.file.slice(j.file.lastIndexOf("/") + 1)];
    return { d, pic: { type: "input", subfolder: sub, filename: name } };
  };
  // the buttons answer on themselves (the Story node's status line is often off screen behind the player): working,
  // then green with what happened, or red with why; plus a toast. The player stays open for the room anchor (add a few)
  const feedback = (btn, state, text) => {
    const label = btn.dataset.label || (btn.dataset.label = btn.textContent);
    btn.disabled = state === "busy";
    btn.textContent = state === "busy" ? "⏳ grabbing the frame…" : text;
    btn.style.background = state === "ok" ? "#1e6b3a" : state === "err" ? "#7a2222" : "";
    btn.style.borderColor = state === "ok" ? "#3fbf6f" : state === "err" ? "#e05555" : "";
    if (state !== "busy") {
      clearTimeout(btn.__lcReset);
      btn.__lcReset = setTimeout(() => ((btn.textContent = label), (btn.style.background = ""), (btn.style.borderColor = ""), (btn.disabled = false)), 2500);
      try {
        app.extensionManager?.toast?.add({ severity: state === "ok" ? "success" : "error", summary: "LC H3 Story", detail: text, life: 4000 });
      } catch (_e) {}
    }
  };
  box.querySelector("[data-pin]")?.addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    feedback(btn, "busy");
    try {
      const { d, pic } = await grab();
      const place = shotPlace(d, i);
      if (!place) throw new Error("This shot has no place to add the anchor to.");
      place.anchors = [...(place.anchors || []), pic];
      setData(node, d, true);
      const msg = `📌 Room anchor ${place.anchors.length} added to ${place.name} (${v.currentTime.toFixed(2)} s of shot ${i + 1})`;
      status(node, msg + ".");
      feedback(btn, "ok", `✅ ${msg}`);
    } catch (err) {
      status(node, err.message, true);
      feedback(btn, "err", `⚠ ${err.message}`);
    }
  });
  // the next shot opens exactly on this frame: its 🖼 start slot (a pinned frame 0 is followed exactly)
  box.querySelector("[data-next]")?.addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    feedback(btn, "busy");
    try {
      const { d, pic } = await grab();
      const nxt = d.shots[i + 1];
      if (!nxt) throw new Error("There is no next shot.");
      nxt.start = pic;
      if ((nxt.join || "cut") === "roll") nxt.join = "cut"; // a keep-rolling shot starts from the shot before: a start frame needs a cut
      setData(node, d, true);
      const msg = `🖼 Shot ${i + 2} now starts on this frame (${v.currentTime.toFixed(2)} s of shot ${i + 1}). Run to make it again`;
      status(node, msg + ".");
      feedback(btn, "ok", `✅ ${msg}`);
      setTimeout(close, 1500);
    } catch (err) {
      status(node, err.message, true);
      feedback(btn, "err", `⚠ ${err.message}`);
    }
  });
}

function findSegment() {
  const all = [];
  const walk = (g) => (g?._nodes || g?.nodes || []).forEach((n) => (all.push(n), n.subgraph && walk(n.subgraph)));
  walk(app.graph);
  return all.filter((n) => (n.type === "LCH3Segment" || n.type === "LCH3Make") && n.mode !== 2 && n.mode !== 4);
}

/** One run of the Segment / Make node on segment `index` as a redo of that segment only (redo_after = just this
 * segment); every widget goes back right after the run is queued. */
async function redoSegment(index, redoAfter = "just this segment") {
  const segNodes = findSegment();
  if (segNodes.length !== 1) throw new Error(segNodes.length ? "More than one LC H3 Segment is on: set its index by hand." : "No LC H3 Segment in this workflow.");
  const seg = segNodes[0];
  const get = (name) => seg.widgets?.find((x) => x.name === name)?.value;
  const set = (name, val) => {
    const w = seg.widgets?.find((x) => x.name === name);
    if (w && val !== undefined) (w.value = val), w.callback?.(val);
  };
  const before = { index: get("index"), make: get("make"), redo_after: get("redo_after") };
  set("index", index);
  set("make", "redo");
  set("redo_after", redoAfter);
  try {
    await app.queuePrompt(0, 1);
  } finally {
    // the queued run keeps redo; the node goes back, so the next Run makes all remaining / your changes again
    set("make", before.make === "redo" ? "all remaining" : before.make);
    set("index", before.index);
    set("redo_after", before.redo_after);
  }
}
window.__lcH3RedoSegment = redoSegment; // the segment chain makes a long shot's later parts with it

/** ↻: remakes this shot and nothing after it (its own parts, when it is longer than max_shot_seconds). The shots after
 * it are kept as they are and the movie is joined again with the new take; a keep-rolling shot after it no longer
 * joins smoothly until it is remade too (user rule 2026-11-07: "If I hit redo on one segment, it does not keep going"). */
async function makeShot(node, i, how = "redo") {
  const n = findSegment().length;
  if (n !== 1) return status(node, n ? "More than one LC H3 Segment is on: set its index by hand." : "No LC H3 Segment in this workflow.", true);
  await saveProject(node);
  await refreshBoard(node);
  const d = getData(node);
  const shotsB = node.__lcst.board?.shots || [];
  const b = shotsB.find((s) => s.shot_id === d.shots[i].id);
  const segs = (b?.segments || []).map((x) => x.index).filter(Boolean);
  if (!segs.length) return status(node, "Could not find that shot in the saved plan. 💾 Save and try again.", true);
  const nb = d.shots[i + 1] && shotsB.find((s) => s.shot_id === d.shots[i + 1].id);
  const rollsOn = !!(nb?.segments?.[0]?.made && nb.segments[0].mode === "CONTINUE");
  const repair = how === "repair" && segs.length === 1 && (b?.segments || []).every((g) => g.made);
  status(node, repair ? `🩹 Repairing shot ${i + 1}: new take, same ending${rollsOn ? `, so shot ${i + 2} still joins` : ""}.`
    : `↻ Making shot ${i + 1} (segment ${segs[0]}${segs.length > 1 ? `-${segs[segs.length - 1]}` : ""}).${rollsOn ? ` ⚠ Shot ${i + 2} will need ↻ too.` : " Nothing after it is touched."}`);
  try {
    const at = Math.floor(Date.now() / 1000);
    window.__lcH3Only = { project: d.project, rest: segs.slice(1) }; // the chain stops once these are made
    await redoSegment(segs[0], repair ? "repair (keep its ending)" : "just this segment");
    node.__lcst.queued = { ...(node.__lcst.queued || {}), [d.shots[i].id]: at };
    paintBoard(node);
  } catch (e) {
    window.__lcH3Only = null;
    status(node, e.message, true);
  }
}

// ------------------------------------------------------------------ 🎛 shot LoRAs
let loraList = null; // every file under models/loras, asked once
function loadLoras(node) {
  if (loraList) return;
  loraList = [];
  api.fetchApi("/models/loras").then((r) => r.json()).then((l) => { loraList = Array.isArray(l) ? l : []; render(node); }).catch(() => {});
}

const loraBase = (n) => String(n || "").split(/[\\/]/).pop().replace(/\.safetensors$/, "");

/** The shot's own LoRAs: on top of the base LoRAs in the pipe, for this shot only. A keep-rolling shot uses its cut's.
 * Picked the same way as in the LC LoRA Loader (LC123's folder picker, ◀ strength ▶ and ℹ Info, via window.LC123LoraUI);
 * the controls are mounted after the card is drawn (mountLoraControls). */
function loraRows(s, i, rs) {
  const src = rs ? rs[0] : s;
  const list = src.loras || [];
  if (rs) return list.length ? `<div class="lcst-row"><span class="lcst-tip" title="Keep rolling uses the LoRAs of shot ${rs[1]}: a new LoRA mid-roll jumps at the join. Change them on shot ${rs[1]}, or make this a cut.">🎛 ${list.map((x) => `${esc(loraBase(x.lora))} ${x.strength}`).join(" · ")} 🔒</span></div>` : "";
  const ui = window.LC123LoraUI;
  return `<div class="lcst-row" style="flex-wrap:wrap;gap:4px">
    ${list.map((x, k) => `<span class="lcst-lora" style="display:inline-flex;align-items:center;gap:4px">
      <button class="lcst-b" data-slp="${i}|${k}" style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(x.lora ? x.lora.replace(/\\/g, "/") : "Pick a LoRA")}">🎛 ${x.lora ? esc(loraBase(x.lora)) : "pick a LoRA…"}</button>
      ${ui ? `<span data-slstr="${i}|${k}"></span>` : `<input class="lcst-sec" data-sls="${i}|${k}" value="${esc(x.strength)}" title="Strength">`}
      ${ui ? `<button class="lcst-b" data-sli="${i}|${k}" title="${x.lora ? "Trigger words (from the file's own metadata)" : "Pick a LoRA first, then ℹ shows its trigger words"}" style="color:#9cf;${x.lora ? "" : "opacity:.5"}">ℹ</button>` : ""}
      <button class="lcst-b" data-sld="${i}|${k}" title="Take this LoRA off the shot" style="color:#e88">✕</button></span>`).join("")}
    <button class="lcst-b" data-sla="${i}" title="Add a LoRA to this shot only (character, outfit, backdrop, style, action...). It goes on top of the base LoRAs and comes off for the next cut.">🎛 ${list.length ? "+" : "LoRA for this shot"}</button>
  </div>`;
}

/** After render: LC123's folder picker on each 🎛 button, its ◀ strength ▶ control and ℹ Info popover. */
function mountLoraControls(wrap, d, save) {
  const ui = window.LC123LoraUI;
  const at = (v) => v.split("|").map(Number);
  wrap.querySelectorAll("[data-slp]").forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    const [i, k] = at(b.dataset.slp);
    const row = d.shots[i].loras[k];
    if (ui) ui.pickLora(b, () => row.lora, (full) => { row.lora = full; save(true); });
  }));
  if (!ui) return;
  wrap.querySelectorAll("[data-slstr]").forEach((sp) => {
    const [i, k] = at(sp.dataset.slstr);
    const row = d.shots[i].loras[k];
    const ctl = ui.makeStrength(() => row.strength ?? 1, (v) => { row.strength = v; save(false); });
    sp.replaceWith(ctl.wrap);
  });
  wrap.querySelectorAll("[data-sli]").forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    const [i, k] = at(b.dataset.sli);
    const name = d.shots[i].loras[k].lora;
    if (name) ui.showInfoPopover(b, name); // always shown: the trigger words are what a character LoRA needs
    else b.parentElement.querySelector("[data-slp]")?.click(); // no LoRA yet: open the picker
  }));
}

// ------------------------------------------------------------------ ✨ Write the story
let modelList = null; // {models, default}, asked once

async function fillModels(sel, d) {
  try {
    modelList = modelList || (await api.fetchApi("/lc_h3/story/models").then((r) => r.json()));
  } catch (_) {
    modelList = { models: [], default: "" };
  }
  const list = modelList.models || [];
  const cur = d.model && list.includes(d.model) ? d.model : modelList.default;
  sel.innerHTML = list.length ? opts(list, cur, Object.fromEntries(list.map((m) => [m, m.split("/").pop()]))) : `<option value="">no models in models/LLM</option>`;
  sel.title = `${cur || ""}\nThe model ✨ writes with (any text model under models/LLM). It loads for the one answer and unloads after.`;
}

function openWriter(node) {
  document.querySelector(".lcst-pick")?.remove();
  const d = getData(node);
  const has = d.shots.some((s) => String(s.action || s.line || "").trim());
  const box = document.createElement("div");
  box.className = "lcst-pick";
  box.style.inset = "12% 20%";
  box.innerHTML = `<div class="lcst-pick-top"><b style="color:#eb4">✨ Write the story</b><span style="flex:1"></span><button class="lcst-b" data-close>✕</button></div>
    <div style="padding:10px;display:flex;flex-direction:column;gap:8px;flex:1;min-height:0">
      <span class="lcst-tip">Who, where, what happens, how it ends. Add a length if you want one ("6 shots", "about 40 seconds").</span>
      <textarea data-pitch style="flex:1;min-height:120px;background:#1e1e1e;color:#ddd;border:1px solid #444;border-radius:4px;padding:6px;font:13px Arial,sans-serif;resize:none"
        placeholder="Nova and Anika meet for coffee in a small French cafe. Anika tells a story that goes too far, Nova laughs it off, then gets up and leaves her sitting there.">${esc(d.pitch || "")}</textarea>
      ${has ? `<span class="lcst-warn">⚠️ This replaces the cast, places and shots. Pictures, voices and 360 panoramas on cards with the same name stay.</span>` : ""}
      <div class="lcst-row"><label class="lcst-tip" title="Off: the writer as always. On: adult scenes are allowed, the same rule as LC Vision's NSFW style tag. Saved with the project."><input type="checkbox" data-nsfw${d.nsfw ? " checked" : ""}> 🔞 NSFW</label>
        <span class="lcst-tip" data-wstatus style="flex:1"></span><button class="lcst-b" data-go style="color:#eb4;padding:4px 14px">✨ Write</button></div>
    </div>`;
  document.body.appendChild(box);
  const ta = box.querySelector("[data-pitch]");
  ta.focus();
  const close = () => box.remove();
  box.querySelector("[data-close]").onclick = close;
  box.addEventListener("keydown", (e) => (e.stopPropagation(), e.key === "Escape" && close()), true);
  box.querySelector("[data-go]").onclick = async () => {
    const go = box.querySelector("[data-go]");
    const st = box.querySelector("[data-wstatus]");
    const model = node.__lcst.wrap.querySelector("[data-model]")?.value || d.model || modelList?.default || "";
    if (!ta.value.trim()) return (st.textContent = "Write a pitch first.");
    go.disabled = true;
    const t0 = Date.now();
    const tick = setInterval(() => (st.textContent = `✨ Writing with ${model.split("/").pop()}… ${Math.round((Date.now() - t0) / 1000)} s`), 500);
    try {
      const j = await api.fetchApi("/lc_h3/story/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pitch: ta.value, model, story: getData(node), max_shot_seconds: widget(node, "max_shot_seconds")?.value || 20,
                               nsfw: !!box.querySelector("[data-nsfw]")?.checked }),
      }).then((r) => r.json());
      if (j.error) throw new Error(j.error);
      setData(node, { ...blank(), outfits: getData(node).outfits, props: getData(node).props, ...j.story, pitch: ta.value, model, nsfw: !!box.querySelector("[data-nsfw]")?.checked }, true); // the writer doesn't make props: keep yours
      status(node, `✨ Wrote ${j.story.shots.length} shots in ${Math.round((Date.now() - t0) / 1000)} s. Everything is yours to edit now.`);
      close();
    } catch (e) {
      st.textContent = e.message;
      st.className = "lcst-stop";
      go.disabled = false;
    } finally {
      clearInterval(tick);
    }
  };
}

// ------------------------------------------------------------------ project row
/** A small dropdown under a button: closes on a click outside or Escape. */
function openMenu(anchor, html) {
  document.querySelector(".lcst-menu")?.remove();
  const menu = document.createElement("div");
  menu.className = "lcst-menu";
  menu.innerHTML = html;
  document.body.appendChild(menu);
  const r = anchor.getBoundingClientRect();
  menu.style.left = Math.min(r.left, innerWidth - menu.offsetWidth - 8) + "px";
  menu.style.top = Math.min(r.bottom + 4, innerHeight - menu.offsetHeight - 8) + "px";
  const close = () => {
    menu.remove();
    document.removeEventListener("pointerdown", outside, true);
  };
  const outside = (e) => !menu.contains(e.target) && close();
  setTimeout(() => document.addEventListener("pointerdown", outside, true), 0);
  menu.addEventListener("keydown", (e) => (e.stopPropagation(), e.key === "Escape" && close()), true);
  return { menu, close };
}

/** 📂 Load: the saved projects, newest first, as a dropdown. */
async function loadProject(node, anchor) {
  let list = [];
  try {
    list = (await api.fetchApi(`/lc_h3/story/projects?t=${Date.now()}`).then((r) => r.json())).projects || [];
  } catch (_) {}
  const cur = getData(node).project;
  const { menu, close } = openMenu(anchor, list.length
    ? list.map((n) => `<div class="it${n === cur ? " on" : ""}" data-n="${esc(n)}">${esc(n)}</div>`).join("")
    : `<span class="lcst-tip">No saved projects yet.</span>`);
  menu.querySelectorAll("[data-n]").forEach((it) => (it.onclick = () => (close(), openProject(node, it.dataset.n))));
}

/** 💾 Save as: a variant of the project under a new name (its made shots come along); this node switches to it. */
function saveProjectAs(node, anchor) {
  const d = getData(node);
  const { menu, close } = openMenu(anchor, `<div class="lcst-row"><input data-as value="${esc(`${d.project || "project"} v2`)}" style="width:190px" placeholder="new project name">
    <button class="lcst-b pri" data-asgo>Save</button></div>
    <div class="lcst-tip">Copies the whole project (made shots too). The original stays as it is.</div>`);
  const input = menu.querySelector("[data-as]");
  input.focus();
  input.select();
  const go = async () => {
    const name = input.value.trim();
    if (!name) return;
    close();
    status(node, `Saving a copy as "${name}"…`);
    try {
      const j = await api.fetchApi("/lc_h3/story/save_as", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ story: d, name, max_shot_seconds: widget(node, "max_shot_seconds")?.value || 20 }) }).then((r) => r.json());
      if (j.error) throw new Error(j.error);
      setData(node, { ...d, project: j.project }, true);
      markSaved(node, j.saved);
      status(node, `Saved as "${j.project}"${j.copied ? " (made shots copied)" : ""}. You're working on the copy now.`);
      refreshBoard(node);
    } catch (e) {
      status(node, e.message, true);
    }
  };
  menu.querySelector("[data-asgo]").onclick = go;
  input.addEventListener("keydown", (e) => e.key === "Enter" && go());
}

/** 1248 x 864 -> "13:9"; the usual shapes keep their names (1024 x 576 -> "16:9"). */
function ratioName(w, h) {
  const common = [[16, 9], [9, 16], [4, 3], [3, 4], [3, 2], [2, 3], [1, 1], [21, 9], [5, 4], [4, 5]];
  const hit = common.find(([a, b]) => Math.abs(w / h - a / b) < 0.01);
  if (hit) return `${hit[0]}:${hit[1]}`;
  const g = (a, b) => (b ? g(b, a % b) : a);
  const k = g(w, h);
  return `${w / k}:${h / k}`;
}

/** The size line under the project name: the size the project was first made at, and a warning when made shots differ
 * (a resize that went wrong shows up here). It reads the project folder, so a reloaded workflow can't lose it. */
function paintSize(node) {
  const el = node.__lcst?.wrap.querySelector("[data-size]");
  if (!el) return;
  const info = node.__lcstSize;
  const project = getData(node).project;
  const paint = () => {
    const i = node.__lcstSize;
    if (!i || i.project !== project || !i.first) {
      el.textContent = "";
      return;
    }
    const f = i.first;
    const odd = i.shots.filter((s) => s.width !== f.width || s.height !== f.height);
    let text = `📐 ${f.width} × ${f.height} · ${ratioName(f.width, f.height)}${f.guess ? " (from the oldest shot)" : ""}`;
    let tip = "The size this project is made at (the size of the last Run). Change the size and Run: every shot made at the old size is made again, in order.";
    if (odd.length) {
      const groups = new Map();
      odd.forEach((s) => groups.set(`${s.width} × ${s.height} · ${ratioName(s.width, s.height)}`, [...(groups.get(`${s.width} × ${s.height} · ${ratioName(s.width, s.height)}`) || []), s.index]));
      text += ` ⚠ differs: ${[...groups].map(([k, v]) => `shot${v.length > 1 ? "s" : ""} ${v.join(", ")} ${k}`).join("; ")}`;
      tip += " Shots made at another size are made again on the next Run (they break the anchors and distort the joins).";
    }
    el.textContent = text;
    el.title = tip;
    el.style.color = odd.length ? "#ffcf7a" : "#9fd3ff";
  };
  if (info && info.project === project) paint();
  if (!project || (info && info.project === project && Date.now() - info.t < 3000)) return;
  node.__lcstSize = { ...(info && info.project === project ? info : {}), project, t: Date.now(), shots: info?.project === project ? info.shots : [] };
  api.fetchApi(`/lc_h3/story/size?project=${encodeURIComponent(project)}&t=${Date.now()}`).then((r) => r.json()).then((j) => {
    node.__lcstSize = { project, t: Date.now(), first: j.first || null, shots: j.shots || [] };
    paint();
  }).catch(() => {});
}

async function openProject(node, name) {
  try {
    const j = await api.fetchApi(`/lc_h3/story/load?project=${encodeURIComponent(name)}&t=${Date.now()}`).then((r) => r.json());
    if (j.error) throw new Error(j.error);
    if (!j.story) return status(node, `No story saved for "${name}".`, true);
    const s = { ...blank(), ...j.story, project: name };
    setData(node, s, true);
    status(node, j.story.imported ? `Imported "${name}" from the old Shot List. Check it, then 💾 Save.` : `Loaded "${name}".`);
  } catch (e) {
    status(node, e.message, true);
  }
}

/** 🆕 New: clears the board. When the board differs from the project's last save, the first click only warns (toast)
 * and a second click within 6 s clears anyway. Nothing on disk is touched. */
const boardSig = (d) => JSON.stringify(["look", "cast", "outfits", "places", "props", "shots"].map((k) => d[k] ?? null));
const hasContent = (d) => !!(String(d.look || "").trim() || d.cast.length || d.outfits.length || d.places.length || d.props.length || d.shots.length);
function toast(severity, detail, life = 6000) {
  try {
    app.extensionManager?.toast?.add({ severity, summary: "LC H3 Story", detail, life });
  } catch (_) {}
}
async function newProject(node) {
  const d = getData(node);
  if (hasContent(d) && !(node.__lcstNewArmed > Date.now())) {
    let saved = null;
    try {
      const j = await api.fetchApi(`/lc_h3/story/load?project=${encodeURIComponent(d.project || "")}&t=${Date.now()}`).then((r) => r.json());
      if (j?.story) saved = { ...blank(), ...j.story };
    } catch (_) {}
    if (!saved || boardSig(saved) !== boardSig(d)) {
      node.__lcstNewArmed = Date.now() + 6000;
      toast("warn", `Progress on "${d.project || "this project"}" has not been saved yet. 💾 Save first, or click 🆕 New again to clear the board anyway.`);
      status(node, "⚠ Not saved yet: 🆕 New again clears the board anyway.", true);
      return;
    }
  }
  node.__lcstNewArmed = 0;
  // exactly what a brand-new Story node starts with (only the ✨ writer's model is kept: it is a setting, not the story)
  if (node.properties) node.properties.lcstMin = [];
  node.__lcstBox?.clear?.();
  setData(node, { ...blank(), model: d.model }, true);
  toast("info", "Board cleared. Name the project and 💾 Save it.", 5000);
  status(node, "New board. ✋ Name the project, then 💾 Save.");
  const name = node.__lcst?.wrap?.querySelector('[data-k="project"]');
  if (name) {
    name.focus();
    name.select();
  }
}

async function saveProject(node) {
  const d = getData(node);
  try {
    const j = await api.fetchApi("/lc_h3/story/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ story: d, max_shot_seconds: widget(node, "max_shot_seconds")?.value || 20 }),
    }).then((r) => r.json());
    if (j.error) throw new Error(j.error);
    markSaved(node, j.saved);
    status(node, `Saved "${j.project}".` + (j.moved?.length ? ` Takes of shots that moved went to takes/: ${j.moved.join(", ")}.` : ""));
  } catch (e) {
    status(node, e.message, true);
  }
}

// ------------------------------------------------------------------ ✨ make this, ↶ the picture before
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** The picture a slot had goes on the card's list of earlier ones (newest first, 8 kept): ↶ brings it back. The files stay on disk. */
function keepOld(card, slot) {
  const old = card?.[slot];
  if (!old?.filename) return;
  const prev = ((card.prev = card.prev || {})[slot] = (card.prev[slot] || []).filter((p) => p.filename !== old.filename));
  prev.unshift({ type: old.type, subfolder: old.subfolder, filename: old.filename, ...(old.crop ? { crop: old.crop } : {}) });
  prev.length = Math.min(prev.length, 8);
}

/** ↶: the slot goes back one picture. The one it has now goes to the end of the list, so repeated clicks go round them all. */
function earlier(card, slot) {
  const cur = card[slot];
  const list = (card.prev?.[slot] || []).filter((p) => p.filename !== cur?.filename);
  if (!list.length) return false;
  const pick = list.shift();
  if (cur?.filename) list.push({ type: cur.type, subfolder: cur.subfolder, filename: cur.filename, ...(cur.crop ? { crop: cur.crop } : {}) });
  card.prev = { ...(card.prev || {}), [slot]: list };
  card[slot] = pick;
  return true;
}

/** What a picture slot is: which card and field it fills, and the job that makes it (kind + name for the Story Card Prompt, shot + slot for the Shot Card). */
function slotInfo(d, el) {
  const ds = el.dataset;
  if (ds.pic) {
    const [k, i] = ds.pic.split("|");
    const n = Number(i);
    if (k === "c") return { find: (x) => x.cast[n], slot: "picture", kind: "cast card", what: "headshot" };
    if (k === "w") return { find: (x) => x.cast[n], slot: "outfit_picture", kind: "cast outfit", what: "outfit picture" };
    if (k === "o") return { find: (x) => x.outfits[n], slot: "picture", kind: "outfit", what: "outfit picture" };
    if (k === "r") return { find: (x) => x.props[n], slot: "picture", kind: "prop", what: "prop picture" };
    if (k === "p") return { find: (x) => x.places[n], slot: "picture", kind: "place", what: "room picture" };
  }
  if (ds.pano) return { find: (x) => x.places[Number(ds.pano)], slot: "panorama", kind: "360 panorama", what: "360 panorama" };
  if (ds.start) return { find: (x) => x.shots[Number(ds.start)], slot: "start", shot: Number(ds.start) + 1, what: "start frame" };
  if (ds.end) return { find: (x) => x.shots[Number(ds.end)], slot: "end", shot: Number(ds.end) + 1, what: "end frame" };
  return null;
}

const setW = (n, w, v) => {
  w.value = v;
  w.callback?.(v, app.canvas, n, null, null);
};

/**
 * ✨ on a slot: aims the workflow's LC H3 Story Card Prompt (or Shot Card) at exactly this card, switches its section on, runs, and
 * puts every widget back as it was. Nothing to type, so nothing can aim at the wrong card.
 */
/** Every node of the workflow, the ones inside subgraph nodes too (the Cards & Storyboard node holds the card makers). */
function allNodes(graph) {
  const out = [];
  const walk = (g, seen) => {
    for (const n of g?._nodes || []) {
      out.push(n);
      if (n.subgraph && !seen.has(n.subgraph)) {
        seen.add(n.subgraph);
        walk(n.subgraph, seen);
      }
    }
  };
  walk(graph, new Set());
  return out;
}

/** The nodes of one section of a subgraph: the ones inside the group whose title matches (by position, like the Section Bypasser). */
function sectionNodes(maker, re) {
  const g = maker.graph;
  if (!g || g === app.graph) return [];
  const boxes = (g._groups || []).filter((x) => re.test(String(x.title || ""))).map((x) => Array.from(x._bounding || x.bounding || []));
  return (g._nodes || []).filter((n) => boxes.some(([x, y, w, h]) => n.pos[0] >= x && n.pos[0] <= x + w && n.pos[1] >= y && n.pos[1] <= y + h));
}

async function makeThis(node, info) {
  if (node.__lcstMaking) return status(node, "Still making the last picture: wait for it to finish.", true);
  const card = info.find(getData(node));
  const isShot = info.shot !== undefined;
  const nodes = allNodes(node.graph || app.graph);
  const maker = nodes.find((n) => n.type === (isShot ? "LCH3ShotCard" : "LCH3StoryCardPrompt"));
  if (!maker) return status(node, `Add ${isShot ? "LC H3 Shot Card" : "LC H3 Story Card Prompt"} and LC H3 Save to Story to this workflow first.`, true);
  const mw = (nm) => maker.widgets?.find((w) => w.name === nm);
  const want = isShot ? /storyboard/i : /story card prompt|cards? (&|and) storyboard/i;
  let bypasser = null;
  let target = null;
  for (const n of nodes) for (const w of n.widgets || []) if (w.type === "toggle" && want.test(String(w.label || w.name))) (bypasser = n), (target = w);
  const inSub = !!maker.graph && maker.graph !== app.graph;
  if (!bypasser && inSub) for (const n of nodes) for (const w of n.widgets || []) if (w.type === "toggle" && /\bmake\b/i.test(String(w.label || w.name))) bypasser = n; // Make goes off for the run
  // no switch: the nodes inside the Make group are bypassed for the run (the ✨ run once also rendered the movie, 2026-11-03)
  let makeNodes = [];
  if (inSub && !bypasser) {
    const boxes = (app.graph._groups || []).filter((x) => /\bmake\b/i.test(String(x.title || ""))).map((x) => Array.from(x._bounding || x.bounding || []));
    makeNodes = (app.graph._nodes || []).filter((n) => n.mode === 0 && boxes.some(([x, y, w, h]) => n.pos[0] >= x && n.pos[0] <= x + w && n.pos[1] >= y && n.pos[1] <= y + h));
    if (!makeNodes.some((n) => n.type === "LCH3Make")) return status(node, "✨ Could not find the Make node: it has to stay off while a picture is made, or the run would also render the movie.", true);
  }
  if (!bypasser && !inSub && (maker.mode === 2 || maker.mode === 4)) return status(node, `${isShot ? "Storyboard Maker" : "Story Card Prompt"} is switched off in the workflow: turn it on first.`, true);
  const aim = isShot ? { shot: String(info.shot), slot: info.slot } : { kind: info.kind, name: String(card?.name || "").trim() };
  const savers = nodes.filter((n) => n.type === "LCH3SaveToStory");
  const was = {
    maker: Object.keys(aim).map((k) => [mw(k), mw(k)?.value]),
    toggles: bypasser ? bypasser.widgets.map((w) => w.value) : null,
    savers: savers.map((n) => n.widgets.map((w) => w.value)),
  };
  const sect = sectionNodes(maker, isShot ? /storyboard maker/i : /story card prompt/i);
  was.modes = sect.map((n) => [n, n.mode]);
  was.make = makeNodes.map((n) => [n, n.mode]);
  const who = isShot ? `shot ${info.shot}` : aim.name;
  const before = card?.[info.slot]?.filename || "";
  const sidx = (n, nm) => n.widgets.find((w) => w.name === nm);
  try {
    for (const [k, v] of Object.entries(aim)) if (mw(k)) setW(maker, mw(k), v);
    for (const n of savers) {
      if (sidx(n, "kind")) setW(n, sidx(n, "kind"), "from card");
      if (sidx(n, "name")) setW(n, sidx(n, "name"), "");
    }
    if (bypasser) bypasser.widgets.forEach((w) => setW(bypasser, w, w === target));
    for (const [n] of was.modes) n.mode = 0;
    for (const [n] of was.make) n.mode = 4;
    await sleep(300); // the section switch settles before the graph is read
    node.__lcstMaking = true;
    status(node, `✨ Making ${who}'s ${info.what}...`);
    await app.queuePrompt(0, 1);
  } catch (e) {
    node.__lcstMaking = false;
    status(node, `✨ ${e.message || e}`, true);
  } finally {
    // the job is read at the moment it is queued: the workflow goes back exactly as it was
    for (const [w, v] of was.maker) if (w) setW(maker, w, v);
    savers.forEach((n, k) => n.widgets.forEach((w, j) => w.value !== was.savers[k][j] && setW(n, w, was.savers[k][j])));
    if (bypasser) bypasser.widgets.forEach((w, k) => w.value !== was.toggles[k] && setW(bypasser, w, was.toggles[k]));
    for (const [n, m] of was.modes) n.mode = m;
    for (const [n, m] of was.make) n.mode = m;
  }
  if (!node.__lcstMaking) return;
  for (let i = 0; i < 900; i++) {
    await sleep(2000);
    const q = await api.fetchApi("/queue").then((r) => r.json()).catch(() => null);
    if (q && !q.queue_running?.length && !q.queue_pending?.length) break;
  }
  node.__lcstMaking = false;
  const now = info.find(getData(node))?.[info.slot]?.filename || "";
  status(node, now && now !== before ? `✨ New ${info.what} for ${who}. The old one is kept: ↶ brings it back.` : `✨ ${who}'s ${info.what} did not change: check the run for an error.`, !(now && now !== before));
}

/** 🌐 360 from card: LC Vision reads the room picture and writes the whole place out (pano_prompt on the place card), then the
 * Card Prompt's ✨ 360 draws from that. */
async function pano360FromRoom(node, i) {
  if (node.__lcstMaking) return status(node, "Still making the last picture: wait for it to finish.", true);
  const d = getData(node);
  const p = d.places[i];
  if (!p?.picture?.filename) return status(node, "Put a room picture on the card first.", true);
  if (!String(p.name || "").trim()) return status(node, "Name the place first.", true);
  if (p.panorama?.filename && !window.confirm("Make a new 360 from the room picture?\nThe 360 on the card now is kept: ↶ brings it back.")) return;
  try {
    status(node, "🌐 LC Vision is reading the room picture...");
    const m = await api.fetchApi("/lc_vision/boxes/models").then((r) => (r.ok ? r.json() : null)).catch(() => null);
    if (!m) throw new Error("🌐 360 from card needs the LC Vision pack (ComfyUI_LC_Vision_nodes).");
    const q = new URLSearchParams({ filename: p.picture.filename, subfolder: p.picture.subfolder || "", type: p.picture.type || "input" });
    const blob = await api.fetchApi(`/view?${q}`).then((r) => {
      if (!r.ok) throw new Error("Could not open the room picture.");
      return r.blob();
    });
    const picture = await new Promise((res, rej) => {
      const fr = new FileReader();
      fr.onload = () => res(fr.result);
      fr.onerror = () => rej(new Error("Could not read the room picture."));
      fr.readAsDataURL(blob);
    });
    const j = await api.fetchApi("/lc_vision/place/describe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: m.default || "", picture }),
    }).then((r) => r.json());
    if (j.error) throw new Error(j.error);
    const now = getData(node);
    now.places[i].pano_prompt = j.text;
    setData(node, now, true);
  } catch (e) {
    return status(node, e.message || String(e), true);
  }
  makeThis(node, { find: (x) => x.places[i], slot: "panorama", kind: "360 panorama", what: "360 panorama" });
}

/** The two small buttons on a picture slot (shown on hover): ✨ make this picture, ↶ the picture before. */
function decorateSlot(node, d, el, save) {
  const info = slotInfo(d, el);
  const card = info?.find(d);
  if (!card) return;
  const mk = document.createElement("button");
  mk.className = "mk";
  mk.textContent = "✨";
  mk.title = `Make this ${info.what}: aims the workflow's ${info.shot !== undefined ? "Shot Card" : "Story Card Prompt"} at this card and runs it. The picture on it now is kept (↶).`;
  mk.onclick = (e) => {
    e.stopPropagation();
    if (info.shot === undefined && !String(card.name || "").trim()) return status(node, "Name the card first, then ✨.", true);
    if (card[info.slot]?.filename && !window.confirm(`Make a new ${info.what}?\nThe one on the card is kept: ↶ brings it back.`)) return;
    makeThis(node, info);
  };
  el.appendChild(mk);
  if ((card.prev?.[info.slot] || []).some((p) => p.filename !== card[info.slot]?.filename)) {
    const un = document.createElement("button");
    un.className = "un";
    un.textContent = "↶";
    un.title = "The picture before this one (click again to go round them all)";
    un.onclick = (e) => {
      e.stopPropagation();
      if (earlier(card, info.slot)) save(true);
    };
    el.appendChild(un);
  }
}

// ------------------------------------------------------------------ the form
function render(node) {
  loadLoras(node);
  const face = ensureFace(node);
  const d = getData(node);
  const names = d.cast.map((c) => String(c.name || "").trim()).filter(Boolean);
  const places = d.places.map((p) => String(p.name || "").trim()).filter(Boolean);
  const focus = document.activeElement && face.wrap.contains(document.activeElement) ? document.activeElement : null;
  const fkey = focus ? [focus.dataset.k, focus.dataset.c, focus.dataset.p, focus.dataset.s, focus.dataset.f].join("|") : "";

  face.inner.innerHTML = `
    <section class="lcst-part lcst-top">
    <div class="lcst-row">
      <span class="lcst-n">🎬</span>
      <input data-k="project" value="${esc(d.project)}" placeholder="project name" title="The project folder: output/LC_H3/<name>/" style="width:170px">
      <button class="lcst-b" data-load title="Open a saved project (an old Shot List project is imported)">📂 Load</button>
      <button class="lcst-b" data-save title="Save story.json and plan.json now (Run saves too)">💾 Save</button>
      <button class="lcst-b" data-saveas title="Save a variant under a new name: the made shots come along, the original stays as it is">💾 Save as</button>
      <button class="lcst-b" data-new title="Clear the board for a new project. Saved projects stay on disk (📂 Load brings them back)">🆕 New</button>
      <span class="lcst-status" data-status></span>
    </div>
    <div class="lcst-row"><span class="lcst-tip" data-size style="flex:1"></span></div>
    <div class="lcst-row">
      <button class="lcst-b pri" data-write title="Type a short pitch: the model fills in the look, cast, places and shots once. After that it is all plain text you edit." >✨ Write the story</button>
      <select data-model style="flex:1" title="The model ✨ writes with (any text model under models/LLM). It loads for the one answer and unloads after."></select>
    </div>
    </section>
    <section class="lcst-part"><div class="lcst-h"><span>🎨 Look</span></div><div class="lcst-pb">
    <input data-k="look" value="${esc(d.look)}" placeholder="The look of the whole movie: cinematic, soft natural light, shallow depth of field">
    </div></section>

    <section class="lcst-part"><div class="lcst-h"><span>🎭 Cast</span><span class="lcst-tip" style="flex:0;white-space:nowrap;font-weight:400">headshot = who they are, 👗 outfit = what they wear</span></div><div class="lcst-pb">
    ${d.cast.map((c, i) => `<div class="lcst-card lcst-pc">
        <div class="lcst-col" style="flex:0 0 84px">
          ${slotHTML(c.picture, `data-pic="c|${i}" title="Headshot: the face and hair. H3 takes who they are from it."`, "🙂 headshot", true)}
          ${slotHTML(c.outfit_picture, `data-pic="w|${i}" title="Outfit 1: the whole outfit, head to toe (✂ to crop). H3 takes the clothes from this picture, so the look words stay on the face."`, "👗 outfit", true)}
        </div>
        <div class="lcst-col">
          <div class="lcst-row">
            <input data-c="${i}" data-f="name" value="${esc(c.name)}" placeholder="Name" style="width:110px">
            <select data-c="${i}" data-f="who" title="Tells people apart in the prompt">${opts(WHO, c.who || "woman")}</select>
            <span style="flex:1"></span>
            <button class="lcst-b" data-voice="${i}" title="${esc(c.voice?.filename ? `Voice: ${c.voice.filename}. Click to change.` : "Add a voice clip (2-15 s): H3 copies the timbre, not the words")}">${c.voice?.filename ? "🎤 " + esc(c.voice.filename.slice(0, 14)) : "🎤 voice"}</button>
            ${c.voice?.filename ? `<button class="lcst-b" data-novoice="${i}" title="Remove the voice">×</button>` : ""}
            <button class="lcst-b" data-copy="${i}" title="Copy name + look, to paste into any image workflow">📋</button>
            <button class="lcst-b" data-del="cast|${i}" title="Remove this person">${TRASH}</button>
          </div>
          <textarea rows="2" data-c="${i}" data-f="look" placeholder="mid twenties, a soft oval face, green eyes, light freckles, fair skin, long wavy blonde hair" title="Describe the head and hair in detail, then stop: age, face, eyes, freckles, skin, hair. Never clothes. The headshot is made from this, and clothes in a headshot beat every outfit picture.">${esc(c.look)}</textarea>
          <input data-c="${i}" data-f="outfit_notes" value="${esc(c.outfit_notes)}" placeholder="👗 outfit, in full detail: a cream cable knit sweater, a black pleated skirt, knee-high black boots" title="Describe the whole outfit head to toe in detail. Only the 👗 outfit picture is made from this (LC H3 Story Card Prompt: cast card + outfit). H3 never reads these words: it sees the picture.">
        </div></div>`).join("")}
    <button class="lcst-b lcst-add" data-add="cast">➕ Add person</button>
    </div></section>

    <section class="lcst-part"><div class="lcst-h"><span>👗 Outfits</span><span class="lcst-tip" style="flex:0;white-space:nowrap;font-weight:400">a change of clothes: pick it on a shot (👗), or name it in the action</span></div><div class="lcst-pb">
    ${d.outfits.map((o, i) => `<div class="lcst-card lcst-pc">
        ${slotHTML(o.picture, `data-pic="o|${i}"`, "outfit picture", true)}
        <div class="lcst-col">
          <div class="lcst-row">
            <input data-o="${i}" data-f="name" value="${esc(o.name)}" placeholder="a simple name: gym clothes" style="flex:1" title="A simple name: red cocktail dress, yoga outfit, gym clothes. Never a list of pieces (shoes and pants pull close-ups wide). With a picture, this name is all H3 reads; the picture carries the rest. Pick it on a shot with 👗, or name it in the action.">
            <select data-o="${i}" data-f="who" title="Who wears it">${opts(["", ...names], names.includes(o.who) ? o.who : "", { "": "who wears it…" })}</select>
            <button class="lcst-b" data-del="outfits|${i}" title="Remove this outfit">${TRASH}</button>
          </div>
          <textarea rows="2" data-o="${i}" data-f="description" placeholder="in full detail, head to toe: an off the shoulder grey cutoff sweatshirt, high waisted black leggings, white gym shoes" title="Describe the whole outfit head to toe in detail. The outfit picture is made from this. With a picture, H3 never reads these words; with no picture, they are all there is.">${esc(o.description)}</textarea>
        </div></div>`).join("")}
    <button class="lcst-b lcst-add" data-add="outfits">➕ Add outfit</button>
    </div></section>

    <section class="lcst-part"><div class="lcst-h"><span>📍 Places</span></div><div class="lcst-pb">
    ${d.places.map((p, i) => `<div class="lcst-card lcst-pc">
        <div class="lcst-col" style="flex:0 0 84px">
          ${slotHTML(p.picture, `data-pic="p|${i}"`, "room picture", true)}
          ${slotHTML(p.panorama, `data-pano="${i}" title="A 360 panorama of this place (2:1, e.g. 2048 x 1024). Each shot then aims its own camera angle in it."`, "🌐 360").replace('class="lcst-slot"', 'class="lcst-slot pano"')}
          ${p.picture?.filename ? `<button class="lcst-b" data-p360="${i}" style="padding:2px 4px;font-size:11px" title="Make the 🌐 360 from the room picture: LC Vision reads the picture and writes the whole place out, then the Card Prompt draws the 360 from that. The 360 on the card now is kept (↶).">🌐 360 from card</button>` : ""}
        </div>
        <div class="lcst-col">
          <div class="lcst-row">
            <input data-p="${i}" data-f="name" value="${esc(p.name)}" placeholder="the cafe" style="width:150px" title="Shots pick their place by this name">
            <input data-p="${i}" data-f="sound" value="${esc(p.sound)}" placeholder="sounds only: rain on the glass, cups clinking, footsteps (never who speaks)" title="Sounds you can hear. Do not write who speaks or 'nobody speaks': naming speech makes H3 speak." style="flex:1">
            <button class="lcst-b" data-del="places|${i}" title="Remove this place">${TRASH}</button>
          </div>
          <textarea rows="2" data-p="${i}" data-f="description" placeholder="What is there: the room, the furniture, the light">${esc(p.description)}</textarea>
          <div class="lcst-anch"><span class="lcst-tip" title="Frames of this room that keep it the same across cuts (📌 on a made shot adds one)">📌 anchors</span>
            ${(p.anchors || []).map((a, k) => slotHTML(a, `data-anch="${i}|${k}"`, "")).join("")}
            ${slotHTML(null, `data-anch="${i}|new"`, "+")}
          </div>
          ${p.panorama?.filename ? `<div class="lcst-row" title="Turn the view until it shows what your wide shot shows. Every shot's angle then counts from here: 90 = the right side of the room, -90 = the left, 180 = behind the camera.">
            <span class="lcst-tip">📷 the wide shot faces</span><input class="lcst-vn" type="number" step="0.5" data-vn="yaw">
            <span class="lcst-tip">pitch</span><input class="lcst-vn" type="number" step="0.5" data-vn="pitch">
            <span class="lcst-tip">fov</span><input class="lcst-vn" type="number" step="0.5" data-vn="fov">
            <span class="lcst-tip" style="flex:1">aim it at what the wide shot shows</span>
          </div>
          <div class="lcst-vw" data-front="${i}"><canvas data-vv title="Drag to look around. Wheel or Shift + drag zooms."></canvas><canvas data-vs title="Click to jump there"></canvas></div>` : ""}
        </div></div>`).join("")}
    <button class="lcst-b lcst-add" data-add="places">➕ Add place</button>
    </div></section>

    <section class="lcst-part"><div class="lcst-h"><span>👜 Props</span><span class="lcst-tip" style="flex:0;white-space:nowrap;font-weight:400">a shot gets the picture when it names the prop</span></div><div class="lcst-pb">
    ${d.props.map((r, i) => `<div class="lcst-card lcst-pc">
        ${slotHTML(r.picture, `data-pic="r|${i}"`, "prop picture", true)}
        <div class="lcst-col">
          <div class="lcst-row">
            <input data-r="${i}" data-f="name" value="${esc(r.name)}" placeholder="the red umbrella" style="flex:1" title="Name it in a shot's action with these exact words and the shot gets its picture">
            <button class="lcst-b" data-del="props|${i}" title="Remove this prop">${TRASH}</button>
          </div>
          <textarea rows="2" data-r="${i}" data-f="description" placeholder="what it looks like: a red umbrella with a curved wooden handle">${esc(r.description)}</textarea>
        </div></div>`).join("")}
    <button class="lcst-b lcst-add" data-add="props">➕ Add prop</button>
    </div></section>

    <section class="lcst-part"><div class="lcst-h"><span>🎞 Shots</span><span class="lcst-tip" style="flex:0;white-space:nowrap;font-weight:400">one shot = one segment</span>
      <button class="lcst-b" data-bmovie title="Play the whole movie (joined after every made shot)">🎬 movie</button>
      <button class="lcst-b" data-brefresh title="Look for new takes">⟳</button>
      <button class="lcst-b" data-minall="1" title="Minimize every shot to one line">▸ all</button>
      <button class="lcst-b" data-minall="0" title="Show every shot in full">▾ all</button></div><div class="lcst-pb">
    <div class="lcst-board" data-board></div>
    ${d.shots.map((s, i) => { const rs = rollSrc(d, i); const src = rs ? rs[0] : s; const lk = rs ? " disabled" : "";
      if ((node.properties?.lcstMin || []).includes(s.id)) {
        const sum = `${rs ? "🎞" : "✂"} ${src.framing || "wide shot"} · ${s.subject || "everyone"}${s.line ? ` · ${s.speaker || "?"}: \u201c${s.line}\u201d` : ""}${s.action ? ` · ${s.action}` : ""}`;
        return `<div class="lcst-card lcst-min"><div class="lcst-row">
          <span class="lcst-n">${i + 1}</span>
          <button class="lcst-b" data-min="${i}" title="Show this shot">▸</button>
          <span class="lcst-sum" title="${esc(sum)}">${esc(sum)}</span>
          <button class="lcst-b" data-mv="${i}|-1" title="Move up" ${i ? "" : "disabled"}>↑</button>
          <button class="lcst-b" data-mv="${i}|1" title="Move down" ${i < d.shots.length - 1 ? "" : "disabled"}>↓</button>
          <button class="lcst-b" data-del="shots|${i}" title="Remove this shot">${TRASH}</button>
        </div></div>`;
      }
      return `<div class="lcst-card">
        <div class="lcst-row">
          <span class="lcst-n">${i + 1}</span>
          <button class="lcst-b" data-min="${i}" title="Minimize this shot to one line">▾</button>
          <select data-s="${i}" data-f="framing"${lk} title="How the shot is framed">${opts(FRAMINGS, src.framing || "wide shot")}</select>
          <select data-s="${i}" data-f="subject" title="Who the shot is on. nobody: no people in it, the place alone (only a voiceover can speak)">${opts(["everyone", "nobody", ...names], names.includes(s.subject) || s.subject === "nobody" ? s.subject : "everyone", { nobody: "nobody (empty)" })}</select>
          <select data-s="${i}" data-f="move" title="Camera move">${opts(MOVES, s.move || "still")}</select>
          ${i ? `<select data-s="${i}" data-f="join" title="cut: a new shot. keep rolling: no cut, the camera carries straight on from the shot before">${opts(["cut", "roll"], s.join === "roll" ? "roll" : "cut", { cut: "✂ cut", roll: "🎞 keep rolling" })}</select>` : ""}
          ${rs ? `<span class="lcst-tip" title="Keep rolling carries the camera straight on: the framing, the place, the 🌐 angle and the clothes come from shot ${rs[1]} (who is in it stays yours: someone can walk in), and the opening frames are the shot before's last moment (no start frame or blocking). Change them on shot ${rs[1]}, or make this a cut.">🔒 from shot ${rs[1]}</span>` : ""}
          <select data-s="${i}" data-f="where"${lk} title="Where this shot is">${opts(["", ...places], !rs && places.includes(s.where) ? s.where : "", { "": i ? "same place" : places[0] || "place…" })}</select>
          <input class="lcst-sec" data-s="${i}" data-f="seconds" value="${esc(s.seconds)}" placeholder="auto" title="Length in seconds. Empty = auto (fits the line). Your number wins; a line that can't fit stretches it.">
          ${String(s.seconds ?? "").trim() ? `<button class="lcst-b" data-auto="${i}" title="Back to auto: the length fits the line again">auto</button>` : ""}
          <span style="flex:1"></span>
          <button class="lcst-b" data-mv="${i}|-1" title="Move up" ${i ? "" : "disabled"}>↑</button>
          <button class="lcst-b" data-mv="${i}|1" title="Move down" ${i < d.shots.length - 1 ? "" : "disabled"}>↓</button>
          <button class="lcst-b" data-del="shots|${i}" title="Remove this shot">${TRASH}</button>
        </div>
        <textarea rows="2" data-s="${i}" data-f="action" placeholder="What happens (before the line). Or write timed beats: At 0.00 seconds, ... At 2.40 seconds, ...">${esc(s.action)}</textarea>
        ${Object.entries(robeOf(d)).filter(([, items]) => items.length > 1 || items.some((x) => x.card)).length ? `<div class="lcst-row">
          ${Object.entries(robeOf(d)).filter(([, items]) => items.length > 1 || items.some((x) => x.card)).map(([nm, items]) =>
            `<span class="lcst-tip">👗 ${esc(nm)}</span><select data-so="${i}|${esc(nm)}"${lk} title="${esc(nm)}'s outfit from this shot on (blank = as before)">
              <option value="">as before</option>${items.map((x) => `<option value="${esc(x.id)}"${s.outfit?.[nm] === x.id ? " selected" : ""}>${esc(x.label)}${x.card ? "" : " · cast card"}${x.pic ? "" : " (no picture)"}</option>`).join("")}</select>`).join("")}
        </div>` : ""}
        <div class="lcst-row">
          <select data-s="${i}" data-f="speaker" style="width:100px" title="Who says the line. (V.O.) = voiceover: heard off screen, lips closed">${opts(["", ...names, ...names.map((n) => n + " (V.O.)")], s.speaker || "", { "": "no line" })}</select>
          <input data-s="${i}" data-f="line" value="${esc(s.line)}" placeholder="The line, word for word" style="flex:1">
          <span class="lcst-tip" data-wc="${i}">${words(s.line) ? words(s.line) + " words" : ""}</span>
        </div>
        <div class="lcst-row">
          ${slotHTML(s.start, `data-start="${i}" title="${esc(s.start?.filename ? `Start frame: ${s.start.filename}\nClick to change, or drop a picture here` : "Start frame (cuts only): the first frame, kept exactly. The one way to get a background exactly: e.g. the person in front of this shot's 360 view (saved in the project's views folder).")}"`, "🖼 start").replace('class="lcst-slot"', `class="lcst-slot sm${rs ? " lcst-lock" : ""}"`).replace('<button class="x" data-clr>×</button>', REDO)}
          <input data-s="${i}" data-f="after" value="${esc(s.after)}" placeholder="What happens after the line (optional)" style="flex:1">
          ${slotHTML(s.end, `data-end="${i}" title="${esc(s.end?.filename ? `End frame: ${s.end.filename}\nClick to change, or drop a picture here` : "End frame (optional): the shot lands on this picture. Render it with a storyboard module (LC H3 Shot Card) or import one.")}"`, "🏁 end").replace('class="lcst-slot"', 'class="lcst-slot sm"').replace('<button class="x" data-clr>×</button>', REDO)}
          <button class="lcst-b${node.__lcstBox?.has(s.id) ? " on" : ""}" data-box="${i}"${lk} title="Blocking: draw where each person and prop is. The storyboard module (LC H3 Shot Card) places them there.">📐${(s.boxes || []).length ? " " + s.boxes.length : ""}</button>
        </div>
        ${loraRows(s, i, rs)}
        ${!rs && node.__lcstBox?.has(s.id) ? `<div class="lcst-bk" data-bk="${i}"><canvas data-bkc title="Drag on empty space for a new box, drag a box to move it, its corner to resize."></canvas><div data-bkr></div></div>` : ""}
        ${!rs && shotPlace(d, i)?.panorama?.filename ? `<div class="lcst-row">
          <button class="lcst-b${s.view?.on ? " on" : ""}" data-von="${i}" title="Aim this shot's camera in the place's 360 panorama. The view becomes the shot's background picture.">🌐 angle${s.view?.on ? "" : "…"}</button>
          ${s.view?.on ? `${[["◀", -90, "the left side of the room"], ["▲", 0, "what the wide shot shows"], ["▶", 90, "the right side of the room"], ["▼", 180, "behind the wide shot's camera"]]
            .map(([t, y, tip]) => `<button class="lcst-b" data-vp="${i}|${y}" title="Look at ${tip} (${y}°)">${t}</button>`).join("")}
          <span class="lcst-tip" title="0 = what the wide shot shows, 90 = right side, -90 = left side, 180 = behind the camera">angle</span><input class="lcst-vn" type="number" step="0.5" data-vn="yaw">
          <span class="lcst-tip">pitch</span><input class="lcst-vn" type="number" step="0.5" data-vn="pitch">
          <span class="lcst-tip">fov</span><input class="lcst-vn" type="number" step="0.5" data-vn="fov">
          <input data-vb="${i}" value="${esc(s.view.behind)}" placeholder="what is behind them: the arched windows" style="flex:1" title="Said in the Camera line too: H3 follows words over pictures">` : `<span class="lcst-tip">pick the camera angle in the 360 panorama</span>`}
        </div>
        ${s.view?.on ? `<div class="lcst-vw" data-view="${i}"><canvas data-vv title="Drag to look around. Wheel or Shift + drag zooms."></canvas><canvas data-vs title="Click to jump there"></canvas></div>` : ""}` : ""}
        <div data-warn="${i}"></div>
        <details class="lcst-pv"${node.__lcstOpen?.has(s.id) ? " open" : ""} data-pvd="${esc(s.id)}"><summary>H3 prompt ▾<button class="lcst-b" data-pvedit="${i}" title="Write this shot's prompt by hand. What you save is exactly what H3 gets for this shot; changes to the cards stop changing it until you go back to automatic.">✎ Edit</button><button class="lcst-b" data-pvref="${i}" title="Build it from the cards again now and show exactly what will be sent to the model.">↻</button></summary><div data-pv="${i}">…</div></details>
      </div>`; }).join("")}
    <button class="lcst-b lcst-add" data-add="shots">➕ Add shot</button>
    </div></section>`;

  const q = (sel) => face.wrap.querySelectorAll(sel);
  const save = (re) => setData(node, d, re);
  q("[data-k]").forEach((el) => (el.oninput = () => {
    d[el.dataset.k] = el.value;
    save(false);
  }));
  const bind = (attr, list) =>
    q(`[data-${attr}]`).forEach((el) => {
      const i = Number(el.dataset[attr]);
      const f = el.dataset.f;
      if (!f) return;
      const set = () => {
        d[list][i][f] = el.value;
        if (list === "shots" && f === "line") {
          const wc = face.wrap.querySelector(`[data-wc="${i}"]`);
          if (wc) wc.textContent = words(el.value) ? `${words(el.value)} words` : "";
          if (el.value && !d.shots[i].speaker) {
            const pick = names.length === 1 ? names[0] : names.includes(d.shots[i].subject) ? d.shots[i].subject : "";
            if (pick) {
              d.shots[i].speaker = pick;
              const sp = face.wrap.querySelector(`[data-s="${i}"][data-f="speaker"]`);
              if (sp) sp.value = pick;
            }
          }
        }
        // names and place names feed the dropdowns: redraw when one is finished
        save(el.tagName === "SELECT");
      };
      if (el.tagName === "SELECT") el.onchange = set;
      else {
        el.oninput = set;
        if (f === "name" || f === "seconds") el.onchange = () => save(true); // seconds: shows / hides the auto button
      }
    });
  bind("c", "cast");
  bind("o", "outfits");
  bind("p", "places");
  bind("r", "props");
  bind("s", "shots");

  q("[data-pic]").forEach((el) => {
    const [kind, i] = el.dataset.pic.split("|");
    const item = kind === "c" || kind === "w" ? d.cast[Number(i)] : kind === "o" ? d.outfits[Number(i)] : kind === "r" ? d.props[Number(i)] : d.places[Number(i)];
    const field = kind === "w" ? "outfit_picture" : "picture"; // w: the cast card's 👗 outfit
    wireSlot(node, el, async (pic, lib) => {
      keepOld(item, field);
      item[field] = kind === "p" ? pic : await onGrey(node, pic); // cast, outfits and props go on plain grey
      if (kind === "w") {
        if (lib && !String(item.outfit_notes || "").trim()) item.outfit_notes = lib.text || ""; // the library card text
        return save(true);
      }
      if (lib) { // a library pick fills what the card doesn't say yet
        if (!String(item.name || "").trim()) item.name = lib.name || "";
        if (kind === "c" && !String(item.look || "").trim()) item.look = lib.text || "";
        if (kind === "c" && lib.who) item.who = lib.who;
        if (kind !== "c" && !String(item.description || "").trim()) item.description = lib.text || "";
      }
      save(true);
    }, { c: "cast", w: "outfits", o: "outfits", r: "props" }[kind] || "");
    el.querySelector("[data-clr]")?.addEventListener("click", (e) => {
      e.stopPropagation();
      keepOld(item, field);
      item[field] = null;
      save(true);
    });
    el.querySelector("[data-crop]")?.addEventListener("click", (e) => {
      e.stopPropagation();
      openCrop(item[field], (crop) => {
        item[field] = { ...item[field] };
        if (crop) item[field].crop = crop;
        else delete item[field].crop;
        save(true);
      });
    });
  });
  q("[data-sla]").forEach((b) => (b.onclick = () => {
    const s = d.shots[Number(b.dataset.sla)];
    s.loras = [...(s.loras || []), { lora: "", strength: 1 }]; // blank until picked: a LoRA you did not choose never goes on
    save(true);
  }));
  mountLoraControls(face.wrap, d, save); // LC123's picker, strength and Info on the 🎛 rows
  q("[data-sls]").forEach((el) => (el.onchange = () => {
    const [i, k] = el.dataset.sls.split("|").map(Number);
    const v = parseFloat(el.value);
    d.shots[i].loras[k].strength = Number.isFinite(v) ? v : 1;
    save(true);
  }));
  q("[data-sld]").forEach((b) => (b.onclick = () => {
    const [i, k] = b.dataset.sld.split("|").map(Number);
    d.shots[i].loras.splice(k, 1);
    if (!d.shots[i].loras.length) delete d.shots[i].loras;
    save(true);
  }));
  q("[data-so]").forEach((el) => (el.onchange = () => {
    const [i, nm] = [Number(el.dataset.so.split("|")[0]), el.dataset.so.slice(el.dataset.so.indexOf("|") + 1)];
    const s = d.shots[i];
    s.outfit = { ...(s.outfit || {}) };
    if (el.value) s.outfit[nm] = el.value;
    else delete s.outfit[nm];
    if (!Object.keys(s.outfit).length) delete s.outfit;
    save(true);
  }));
  q("[data-anch]").forEach((el) => {
    const [i, k] = el.dataset.anch.split("|");
    const p = d.places[Number(i)];
    p.anchors = Array.isArray(p.anchors) ? p.anchors : [];
    wireSlot(node, el, (pic) => {
      if (k === "new") p.anchors.push(pic);
      else p.anchors[Number(k)] = pic;
      save(true);
    });
    el.querySelector("[data-clr]")?.addEventListener("click", (e) => {
      e.stopPropagation();
      p.anchors.splice(Number(k), 1);
      save(true);
    });
  });
  q("[data-pano]").forEach((el) => {
    const p = d.places[Number(el.dataset.pano)];
    wireSlot(node, el, (pic, lib) => {
      keepOld(p, "panorama");
      p.panorama = pic;
      if (lib && !String(p.description || "").trim()) p.description = lib.text || "";
      if (lib && !String(p.name || "").trim()) p.name = lib.name || "";
      save(true);
    }, "360");
    el.querySelector("[data-clr]")?.addEventListener("click", (e) => {
      e.stopPropagation();
      keepOld(p, "panorama");
      p.panorama = null;
      save(true);
    });
  });
  q("[data-von]").forEach((b) => (b.onclick = () => {
    const i = Number(b.dataset.von);
    const s = d.shots[i];
    if (s.view?.on) s.view.on = false;
    else {
      // start from the last angle this shot had, else the angle of the shot before in the same place
      const place = shotPlace(d, i);
      const before = d.shots.slice(0, i).reverse().find((x, k) => x.view?.on && shotPlace(d, i - 1 - k) === place);
      s.view = { ...VIEW_DEFAULT, ...(before?.view || {}), ...(s.view || {}), behind: s.view?.behind ?? "", on: true };
    }
    save(true);
  }));
  q("[data-auto]").forEach((b) => (b.onclick = () => {
    d.shots[Number(b.dataset.auto)].seconds = "";
    save(true);
  }));
  q("[data-vb]").forEach((el) => (el.oninput = () => {
    d.shots[Number(el.dataset.vb)].view.behind = el.value;
    save(false);
  }));
  q("[data-start]").forEach((el) => {
    const s = d.shots[Number(el.dataset.start)];
    wireSlot(node, el, (pic) => {
      keepOld(s, "start");
      s.start = pic;
      save(true);
    });
    el.querySelector("[data-clr]")?.addEventListener("click", (e) => {
      e.stopPropagation();
      keepOld(s, "start");
      s.start = null;
      save(true);
    });
  });
  q("[data-end]").forEach((el) => {
    const s = d.shots[Number(el.dataset.end)];
    wireSlot(node, el, (pic) => {
      keepOld(s, "end");
      s.end = pic;
      save(true);
    });
    el.querySelector("[data-clr]")?.addEventListener("click", (e) => {
      e.stopPropagation();
      keepOld(s, "end");
      s.end = null;
      save(true);
    });
  });
  // minimized shots: kept with the workflow (node properties), not in the story, so nothing counts as edited
  q("[data-min]").forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    const id = d.shots[Number(b.dataset.min)]?.id;
    node.properties = node.properties || {};
    const m = new Set(node.properties.lcstMin || []);
    if (m.has(id)) m.delete(id);
    else m.add(id);
    node.properties.lcstMin = [...m];
    save(true);
  }));
  q("[data-minall]").forEach((b) => (b.onclick = (e) => {
    e.stopPropagation();
    node.properties = node.properties || {};
    node.properties.lcstMin = b.dataset.minall === "1" ? d.shots.map((x) => x.id) : [];
    save(true);
  }));
  q("[data-box]").forEach((b) => (b.onclick = () => {
    const s = d.shots[Number(b.dataset.box)];
    node.__lcstBox = node.__lcstBox || new Set();
    if (node.__lcstBox.has(s.id)) node.__lcstBox.delete(s.id);
    else node.__lcstBox.add(s.id);
    save(true);
  }));
  q("[data-bk]").forEach((box) => blockingEditor(node, d, Number(box.dataset.bk), box, () => save(false)));
  q("[data-vp]").forEach((b) => (b.onclick = () => {
    const [i, y] = b.dataset.vp.split("|").map(Number);
    d.shots[i].view.yaw = y;
    save(true);
  }));
  q("[data-view]").forEach((box) => {
    const i = Number(box.dataset.view);
    const place = shotPlace(d, i);
    wireView(box, d.shots[i].view, viewURL(place?.panorama), () => Number(place?.front?.yaw) || 0, save);
  });
  q("[data-front]").forEach((box) => {
    const p = d.places[Number(box.dataset.front)];
    p.front = { yaw: 0, pitch: 0, fov: 90, ...(p.front || {}) };
    // the shots in this place count their angle from here: redraw them as it turns
    wireView(box, p.front, viewURL(p.panorama), 0, (re) => {
      save(re);
      q("[data-view]").forEach((b) => b._lcDraw?.());
    });
  });
  q("[data-voice]").forEach((b) => (b.onclick = () => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "audio/*";
    inp.onchange = async () => {
      if (!inp.files?.[0]) return;
      try {
        d.cast[Number(b.dataset.voice)].voice = await upload(inp.files[0], d.project);
        save(true);
      } catch (e) {
        status(node, e.message, true);
      }
    };
    inp.click();
  }));
  q("[data-novoice]").forEach((b) => (b.onclick = () => {
    d.cast[Number(b.dataset.novoice)].voice = null;
    save(true);
  }));
  q("[data-copy]").forEach((b) => (b.onclick = async () => {
    const c = d.cast[Number(b.dataset.copy)];
    const text = `${c.name}, a ${c.who || "person"} ${c.look || ""}`.trim();
    try {
      await navigator.clipboard.writeText(text);
      status(node, `Copied: ${text.slice(0, 60)}`);
    } catch (_) {
      window.prompt("Copy this:", text);
    }
  }));
  q("[data-del]").forEach((b) => (b.onclick = () => {
    const [list, i] = b.dataset.del.split("|");
    if (b.dataset.sure !== "1") {
      b.dataset.sure = "1";
      b.textContent = "sure?";
      setTimeout(() => {
        if (b.isConnected) {
          b.dataset.sure = "";
          b.innerHTML = TRASH;
        }
      }, 2500);
      return;
    }
    d[list].splice(Number(i), 1);
    save(true);
  }));
  q("[data-mv]").forEach((b) => (b.onclick = () => {
    const [i, dir] = b.dataset.mv.split("|").map(Number);
    const [s] = d.shots.splice(i, 1);
    d.shots.splice(i + dir, 0, s);
    save(true);
  }));
  q("[data-add]").forEach((b) => (b.onclick = () => {
    const list = b.dataset.add;
    if (list === "cast") d.cast.push({ id: uid("c"), name: "", who: d.cast.length % 2 ? "man" : "woman", look: "", picture: null, voice: null });
    else if (list === "places") d.places.push({ id: uid("p"), name: d.places.length ? "" : "the place", description: "", sound: "", picture: null, anchors: [] });
    else if (list === "props") d.props.push({ id: uid("r"), name: "", description: "", picture: null });
    else if (list === "outfits") d.outfits.push({ id: uid("o"), name: "", who: names.length === 1 ? names[0] : "", description: "", picture: null });
    else {
      const first = !d.shots.length;
      d.shots.push({ id: uid("s"), framing: first ? "wide shot" : "close-up", subject: first ? "everyone" : names[0] || "everyone", move: "still",
        join: "cut", where: "", action: "", speaker: "", line: "", after: "", seconds: "" });
    }
    save(true);
    const sel = list === "cast" ? `[data-c="${d.cast.length - 1}"][data-f="name"]` : list === "places" ? `[data-p="${d.places.length - 1}"][data-f="name"]`
      : list === "props" ? `[data-r="${d.props.length - 1}"][data-f="name"]` : list === "outfits" ? `[data-o="${d.outfits.length - 1}"][data-f="name"]`
      : `[data-s="${d.shots.length - 1}"][data-f="action"]`;
    face.wrap.querySelector(sel)?.focus();
  }));
  q("[data-pvd]").forEach((el) => el.addEventListener("toggle", () => {
    node.__lcstOpen = node.__lcstOpen || new Set();
    if (el.open) node.__lcstOpen.add(el.dataset.pvd);
    else node.__lcstOpen.delete(el.dataset.pvd);
    fit(node);
  }));
  q("[data-pvedit]").forEach((b) => (b.onclick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const i = Number(b.dataset.pvedit);
    const id = getData(node).shots[i]?.id;
    node.__lcstEditing = node.__lcstEditing || new Set();
    node.__lcstOpen = node.__lcstOpen || new Set();
    node.__lcstEditing.add(id);
    node.__lcstOpen.add(id);
    const det = b.closest("details");
    if (det) det.open = true;
    const sh = node.__lcst.preview?.shots?.find((x) => x.shot === i + 1);
    const pv = face.wrap.querySelector(`[data-pv="${i}"]`);
    if (sh && pv) paintPrompt(node, pv, sh, i);
    fit(node);
  }));
  q("[data-pvref]").forEach((b) => (b.onclick = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    clearTimeout(node.__lcstT);
    b.textContent = "…";
    await preview(node);
    b.textContent = "✓";
    setTimeout(() => (b.textContent = "↻"), 900);
  }));
  q("[data-model]").forEach((sel) => {
    fillModels(sel, d);
    sel.onchange = () => {
      d.model = sel.value;
      save(false);
    };
  });
  q("[data-write]").forEach((b) => (b.onclick = () => openWriter(node)));
  q("[data-brefresh]").forEach((b) => (b.onclick = () => refreshBoard(node)));
  q("[data-bmovie]").forEach((b) => (b.onclick = () => {
    const m = node.__lcst.board?.movie;
    if (m) openPlayer(node, [{ video: m }], null, 0, true);
  }));
  q("[data-load]").forEach((b) => (b.onclick = () => loadProject(node, b)));
  q("[data-save]").forEach((b) => (b.onclick = () => saveProject(node)));
  q("[data-saveas]").forEach((b) => (b.onclick = () => saveProjectAs(node, b)));
  q("[data-new]").forEach((b) => (b.onclick = () => newProject(node)));
  q("[data-p360]").forEach((b) => (b.onclick = () => pano360FromRoom(node, Number(b.dataset.p360))));

  q("[data-pic],[data-pano],[data-start],[data-end]").forEach((el) => decorateSlot(node, d, el, save));

  // a click on a section's title bar folds it away (kept with the workflow); the bar's own buttons still work
  const folded = new Set(node.properties?.lc_story_folded || []);
  face.inner.querySelectorAll("section.lcst-part:not(.lcst-top)").forEach((sec) => {
    const h = sec.querySelector(".lcst-h");
    const key = (h?.querySelector("span")?.textContent || "").replace(/[^\p{L}\p{N} ]/gu, "").trim().toLowerCase();
    if (!h || !key) return;
    h.insertAdjacentHTML("afterbegin", `<span class="lcst-fold">${folded.has(key) ? "▸" : "▾"}</span>`);
    sec.classList.toggle("min", folded.has(key));
    h.onclick = (e) => {
      if (e.target.closest("button,input,select")) return;
      const set = new Set(node.properties?.lc_story_folded || []);
      set.has(key) ? set.delete(key) : set.add(key);
      node.properties = { ...(node.properties || {}), lc_story_folded: [...set] };
      const before = face.inner.offsetHeight;
      sec.classList.toggle("min", set.has(key));
      h.querySelector(".lcst-fold").textContent = set.has(key) ? "▸" : "▾";
      shrink(node, before);
    };
  });

  // keep typing where it was after a redraw
  if (fkey) {
    const [k, c, p, s, f] = fkey.split("|");
    const sel = k ? `[data-k="${k}"]` : c ? `[data-c="${c}"][data-f="${f}"]` : p ? `[data-p="${p}"][data-f="${f}"]` : s ? `[data-s="${s}"][data-f="${f}"]` : "";
    if (sel) face.wrap.querySelector(sel)?.focus();
  }
  paintPreview(node);
  paintBoard(node);
  paintSize(node);
  fit(node);
}

/** Old stories (or the example) without ids: give every card one, so takes stay with their shot. */
function ensureIds(node) {
  const d = getData(node);
  let changed = false;
  for (const [list, p] of [["cast", "c"], ["outfits", "o"], ["places", "p"], ["props", "r"], ["shots", "s"]])
    for (const it of d[list]) if (!it.id) (it.id = uid(p)), (changed = true);
  if (changed) widget(node, "story").value = JSON.stringify(d);
}

app.registerExtension({
  name: "LC.AV.H3Story",
  setup() {
    // a finished run may have made a shot: every Story node looks for new takes
    api.addEventListener("execution_success", () => {
      for (const n of app.graph?._nodes || []) if (n.type === NODE && n.__lcst) refreshBoard(n);
    });
    // a stopped or failed run made nothing: ⏳ queued goes back to what the shot was
    for (const ev of ["execution_interrupted", "execution_error"])
      api.addEventListener(ev, () => {
        for (const n of app.graph?._nodes || []) if (n.type === NODE && n.__lcst?.queued) (n.__lcst.queued = {}), paintBoard(n);
      });
    // LC H3 Save to Story put a picture on a card (story.json is saved already): the open Story node takes it too
    api.addEventListener("executed", (e) => {
      const p = e.detail?.output?.lc_story_card?.[0];
      if (!p) return;
      for (const n of app.graph?._nodes || []) if (n.type === NODE && getData(n).project === p.project) applyCard(n, p);
    });
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      hideJson(this);
      ensureIds(this);
      render(this);
      schedulePreview(this);
      if (!this.size || this.size[0] < MIN_W) this.setSize([MIN_W, this.computeSize()[1]]);
      return r;
    };
    // never override configure() or touch node.pos: the default restore path runs untouched
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (!this.color && !this.bgcolor) (this.color = "#1a5a7a"), (this.bgcolor = "#0a0a0a"); // the pack color (saved colors win)
      hideJson(this);
      ensureIds(this);
      render(this);
      schedulePreview(this);
      pickUpNewerSave(this);
      fetchLibraryPics(this);
      return r;
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      markSaved(this, message?.lc_saved?.[0]); // Run saved story.json
      return r;
    };
  },
});

/** A picture from LC H3 Save to Story: into its slot on the card (a new card when the name had none). */
function applyCard(node, p) {
  const d = getData(node);
  const kind = p.kind === "character" ? "cast card" : p.kind; // saved before the rename
  // "cast outfit" (the cast card's 👗 slot) was missing here: the picture saved, the node never took it, and its next
  // Run wrote the card back without it (2026-11-07, "Uber run": outfit cards never populated)
  const list = kind === "cast card" || kind === "cast outfit" ? d.cast : kind === "outfit" ? d.outfits : kind === "prop" ? d.props
    : kind === "shot start" || kind === "shot end" ? d.shots : d.places;
  const slot = { "cast card": "picture", "cast outfit": "outfit_picture", outfit: "picture", place: "picture", prop: "picture",
    "360 panorama": "panorama", "shot start": "start", "shot end": "end" }[kind];
  if (!slot) return status(node, `A picture was saved to the story as "${kind}", which this Story node doesn't know. 📂 Load the project to see it.`, true);
  const card = list.find((x) => x.id === p.id);
  if (card) {
    card[slot] = p.pic;
    if (p.card?.prev?.[slot]) card.prev = { ...(card.prev || {}), [slot]: p.card.prev[slot] }; // ↶ brings the old one back
  }
  else if (p.new) list.push(p.card);
  else return status(node, "A picture was saved to the story, but this node has no such card. 📂 Load the project to see it.", true);
  d.saved = p.saved;
  setData(node, d, true);
  status(node, `🃏 ${p.card?.name || "Shot"}: new ${slot === "panorama" ? "360" : slot === "start" ? "start frame" : slot === "end" ? "end frame" : slot === "outfit_picture" ? "👗 outfit picture" : "picture"} saved.`);
}

/** This copy matches the save on disk from time t (so only a save made elsewhere is "newer"). No redraw. */
function markSaved(node, t) {
  if (!t) return;
  const w = widget(node, "story");
  try {
    const d = JSON.parse(w.value || "{}");
    d.saved = t;
    w.value = JSON.stringify(d);
  } catch (_) {}
}

/** A workflow carries the story as it was when the workflow was saved. The project folder may hold a newer save
 * (edited and run since, maybe from another copy of the workflow): that one wins, and the status says so. */
async function pickUpNewerSave(node) {
  const d = getData(node);
  if (!d.project) return;
  try {
    const j = await api.fetchApi(`/lc_h3/story/load?project=${encodeURIComponent(d.project)}&t=${Date.now()}`).then((r) => r.json());
    const disk = j.story;
    if (!disk || disk.imported || !(Number(disk.saved) > (Number(d.saved) || 0) + 1)) return;
    setData(node, { ...blank(), ...disk, project: d.project }, true);
    status(node, `Loaded the latest save of "${d.project}" (${new Date(disk.saved * 1000).toLocaleString()}), newer than the copy in this workflow.`);
  } catch (_) {}
}
