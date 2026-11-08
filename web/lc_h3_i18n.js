/**
 * LC H3 Story hover tips in the reader's language. Follows LC123's Note language, else ComfyUI's language (the same
 * rule as LC Vision's assistant). i18n/<code>.json is keyed by the English tip; {name} parts are filled from the
 * English one (file names, shot numbers). Anything without a translation stays English. Only elements inside the
 * Story panel and its pop-ups (lcst-* classes) are touched.
 */
import { app } from "../../scripts/app.js";

const CODES = ["zh", "ja", "ko"];
const get = (id) => app.extensionManager?.setting?.get?.(id) ?? app.ui?.settings?.getSettingValue?.(id);

function lang() {
  for (const v of [get("LC123.Notes.Language"), get("Comfy.Locale")]) {
    if (!v || v === "auto") continue;
    const base = String(v).split(/[-_]/)[0];
    if (v === "zh-TW") return ""; // Traditional Chinese: no file yet, stay English
    if (CODES.includes(base)) return base;
    return "";
  }
  return "";
}

const dicts = {}; // code -> {exact, patterns}
async function dict(code) {
  if (!code) return null;
  if (code in dicts) return dicts[code];
  dicts[code] = null;
  try {
    const raw = await fetch(new URL(`./i18n/${code}.json`, import.meta.url)).then((r) => r.json());
    const exact = new Map();
    const patterns = [];
    for (const [en, tr] of Object.entries(raw)) {
      if (!/\{\w+\}/.test(en)) {
        exact.set(en, tr);
        continue;
      }
      const names = [];
      const re = en
        .split(/(\{\w+\})/)
        .map((part) => {
          const m = part.match(/^\{(\w+)\}$/);
          if (!m) return part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
          if (names.includes(m[1])) return `\\k<${m[1]}>`;
          names.push(m[1]);
          return `(?<${m[1]}>[\\s\\S]+?)`;
        })
        .join("");
      patterns.push([new RegExp(`^${re}$`), tr]);
    }
    dicts[code] = { exact, patterns };
  } catch (_) {
    dicts[code] = { exact: new Map(), patterns: [] };
  }
  return dicts[code];
}

function translate(d, en) {
  const hit = d.exact.get(en);
  if (hit) return hit;
  for (const [re, tr] of d.patterns) {
    const m = en.match(re);
    if (m) return tr.replace(/\{(\w+)\}/g, (_, k) => m.groups?.[k] ?? "");
  }
  return null;
}

const ATTR = "data-lc-en"; // the English tip, kept so a language change or a re-render reads the original

function apply(el, d) {
  const cur = el.getAttribute("title");
  if (cur == null) return;
  const done = el.getAttribute("data-lc-tr");
  if (done !== null && cur === done) return; // already ours
  el.setAttribute(ATTR, cur);
  const tr = d ? translate(d, cur) : null;
  if (tr && tr !== cur) {
    el.setAttribute("data-lc-tr", tr);
    el.setAttribute("title", tr);
  } else {
    el.removeAttribute("data-lc-tr");
  }
}

const inStory = (el) => el.closest?.('[class*="lcst"]');
let current = null;

async function sweep(root) {
  const d = await dict(lang());
  current = d;
  if (!d) return;
  const els = root.querySelectorAll ? root.querySelectorAll("[title]") : [];
  if (root.getAttribute?.("title") != null && inStory(root)) apply(root, d);
  for (const el of els) if (inStory(el)) apply(el, d);
}

app.registerExtension({
  name: "LC.H3Story.i18n",
  async setup() {
    if (!lang()) return; // English: nothing to do (a language change needs a reload, like the frontend itself)
    await dict(lang());
    const obs = new MutationObserver((muts) => {
      if (!current) return;
      for (const m of muts) {
        if (m.type === "attributes") {
          if (inStory(m.target)) apply(m.target, current);
        } else {
          for (const n of m.addedNodes) if (n.nodeType === 1 && (inStory(n) || n.querySelector?.('[class*="lcst"]'))) sweep(n);
        }
      }
    });
    obs.observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["title"] });
    sweep(document.body);
  },
});
