/**
 * LC H3 queue guard: one LC submission at a time, each one checked before it leaves and tracked by its prompt id.
 * Card pictures (✨), shot redos (↻ / 🩹) and the segment chain all queue through here, so a card made while a movie
 * renders can't change the graph under the chain's next shot (or the other way round), and each one waits for and
 * reacts to its own job only.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MAKERS = new Set(["LCH3Make", "LCH3Segment"]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let chain = Promise.resolve();

/** true when an API prompt (prompt.output, or the prompt of a /queue or /history item) makes shots */
export function makesShots(output) {
  return Object.values(output || {}).some((n) => MAKERS.has(n?.class_type));
}

/** Run fn (change the graph, submit, put it back) with no other LC submission in between. A Run the user pressed that
 * the frontend is still sending goes first, so its graph is read before ours changes anything. */
export function withQueueLock(fn) {
  const run = chain.then(async () => {
    for (let i = 0; i < 150 && app.processingQueue; i++) await sleep(100);
    return fn();
  });
  chain = run.catch(() => {});
  return run;
}

/** Queue the workflow as it is now; check(output) returns a reason to refuse it (nothing is sent). Returns the prompt
 * id, or throws. Call it inside withQueueLock. */
export async function submitChecked(check) {
  const prev = api.queuePrompt;
  let id = null;
  let refused = null;
  const wrapped = async function (number, prompt, ...rest) {
    const why = check?.(prompt?.output || {});
    if (why) {
      refused = why;
      throw new Error(why);
    }
    const r = await prev.call(this, number, prompt, ...rest);
    if (r?.prompt_id) id = String(r.prompt_id);
    return r;
  };
  api.queuePrompt = wrapped;
  try {
    await app.queuePrompt(0, 1);
  } finally {
    if (api.queuePrompt === wrapped) api.queuePrompt = prev;
  }
  if (refused) throw new Error(refused);
  if (!id) throw new Error("ComfyUI didn't take the job (no prompt id came back).");
  return id;
}

/** The raw queue: [number, id, prompt, ...] items. api.getQueue() is no use here: on frontend 1.58 it only gives job
 * summaries (id, status), without the prompt, so it can't tell a shot from a card picture. */
async function rawQueue() {
  const q = await api.fetchApi("/queue").then((r) => r.json());
  return { running: q?.queue_running || [], pending: q?.queue_pending || [] };
}

/** Prompt ids waiting in the queue that make shots (card jobs and other workflows don't count) */
export async function pendingShotIds() {
  try {
    const q = await rawQueue();
    return q.pending.filter((it) => makesShots(it[2])).map((it) => String(it[1]));
  } catch (_e) {
    return [];
  }
}

/** Did job `id` make shots? Read from the queue or the history (null when it can't be told) */
export async function isShotJob(id) {
  if (!id) return null;
  try {
    const h = await api.fetchApi(`/history/${encodeURIComponent(id)}`).then((r) => r.json());
    const rec = h?.[id];
    if (rec?.prompt) return makesShots(rec.prompt[2]);
    const q = await rawQueue();
    const it = [...q.running, ...q.pending].find((x) => String(x[1]) === String(id));
    return it ? makesShots(it[2]) : null;
  } catch (_e) {
    return null;
  }
}

/** Wait until job `id` is finished (true: success, false: error or stopped; null: gave up after maxMs) */
export async function waitForJob(id, maxMs = 30 * 60 * 1000) {
  const end = Date.now() + maxMs;
  while (Date.now() < end) {
    await sleep(1500);
    try {
      const h = await api.fetchApi(`/history/${encodeURIComponent(id)}`).then((r) => r.json());
      if (h?.[id]) return h[id].status?.status_str === "success"; // the history only holds finished jobs
      const q = await rawQueue();
      const there = [...q.running, ...q.pending].some((x) => String(x[1]) === String(id));
      if (!there) {
        await sleep(1500); // a finished job reaches the history a moment after it leaves the queue
        const h2 = await api.fetchApi(`/history/${encodeURIComponent(id)}`).then((r) => r.json());
        if (!h2?.[id]) return false; // taken out of the queue before it ran
      }
    } catch (_e) {}
  }
  return null;
}

export async function runningIds() {
  try {
    return (await rawQueue()).running.map((x) => String(x[1]));
  } catch (_e) {
    return [];
  }
}

window.LCH3Queue = { runningIds,  withQueueLock, submitChecked, makesShots, pendingShotIds, isShotJob, waitForJob };
