/**
 * LC H3 Segment, make: all remaining.
 * The moment Segment Save reports a segment saved, the next one is queued (while this run is still finishing), so
 * the queue never sits empty between segments and "on empty queue" sounds ring once, at the end.
 * Stop (or an error) ends the chain: the segment queued ahead is taken back out, nothing carries on.
 * Only shot jobs count: a ✨ card picture queued (or failing, or stopped) mid-movie leaves the chain alone, and the
 * queue-ahead goes through the LC queue lock, so it can't pick up a card's half-switched graph (lc_h3_queue.js).
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { withQueueLock, submitChecked, makesShots, pendingShotIds, isShotJob } from "./lc_h3_queue.js";

let ahead = null; // prompt id of the segment queued ahead of time
let lastMade = 0; // the segment the chain saved last: the same one twice means index isn't moving, so stop
let queuing = false; // a queue-ahead in progress: a second save message arriving meanwhile must not queue another
window.__lcH3ChainNext = false; // read by LC Notify: a chained segment is lined up, the movie isn't done

function toast(detail, severity = "info") {
  try {
    app.extensionManager?.toast?.add({ severity, summary: "LC H3", detail, life: 5000 });
  } catch (_e) {}
}

function chainNodes() {
  return (app.graph?._nodes || []).filter(
    (n) => (n.type === "LCH3Segment" || n.type === "LCH3Make") && n.mode === 0 && n.widgets?.find((w) => w.name === "make")?.value === "all remaining"
  );
}

const pendingIds = pendingShotIds; // waiting shot jobs only: a card picture in the queue doesn't hold the chain up

// segments still to make, read fresh from the project (null when it can't be read: then the message decides)
async function leftNow(project) {
  try {
    const d = await api.fetchApi(`/lc_vision/movie/status?project=${encodeURIComponent(project)}&t=${Date.now()}`, { cache: "no-store" })
      .then((r) => r.json());
    if (!Array.isArray(d?.segments)) return null;
    return d.segments.filter((x) => !x.done).length;
  } catch (_e) {
    return null;
  }
}

async function takeBack() {
  if (!ahead) return;
  const id = ahead;
  ahead = null;
  window.__lcH3ChainNext = false;
  try {
    await api.fetchApi("/queue", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ delete: [id] }) });
  } catch (_e) {}
}

// the join failed but the chain goes on: a sticky error toast so it is noticed
api.addEventListener("lc_h3_join_failed", (e) => {
  const s = e.detail || {};
  try {
    app.extensionManager?.toast?.add({ severity: "error", summary: `LC H3: ${s.project || "movie"} not joined`, detail: s.message || "" });
  } catch (_e) {}
});

api.addEventListener("lc_h3_segment_saved", async (e) => {
  const s = e.detail || {};
  // ↻ on one shot (LC H3 Story): make its own parts, then stop. The shots after it stay as they are; the save joined
  // the movie again with the new take (user rule 2026-11-07: "If I hit redo on one segment, it does not keep going")
  const only = window.__lcH3Only;
  if (only && only.project === s.project) {
    lastMade = 0;
    if (only.rest.length) {
      const nxt = only.rest.shift();
      try {
        await window.__lcH3RedoSegment?.(nxt);
        toast(`Segment ${s.made} saved. Part ${nxt} of this shot is next.`);
      } catch (err) {
        window.__lcH3Only = null;
        toast(`The rest of the shot wasn't queued: ${err.message || err}`, "warn");
      }
      return;
    }
    window.__lcH3Only = null;
    window.__lcH3ChainNext = false;
    toast(`Segment ${s.made} remade and joined into the movie. Nothing after it was touched.`, "success");
    return;
  }
  if (!chainNodes().length) return;
  const waiting = await pendingIds();
  if (!s.left) {
    lastMade = 0;
    window.__lcH3ChainNext = false;
    if (!waiting.length) toast(`All segments of ${s.project} are made and joined into the finished movie.`, "success");
    return;
  }
  if (waiting.length || queuing) return; // a run already waiting (a second redo, etc.): don't stack more on top
  if (s.made === lastMade) {
    lastMade = 0;
    toast(`Segment ${s.made} was made twice in a row, so the chain stopped. Set make to all remaining.`, "warn");
    return;
  }
  queuing = true;
  try {
    // check the project itself right before queuing: a run that would find nothing to make is never queued
    const left = s.changed ? s.left : await leftNow(s.project); // changed: edited shots LC H3 Make remakes (all made already)
    if (left === 0 || (await pendingIds()).length) return;
    lastMade = s.made;
    window.__lcH3ChainNext = true;
    // all remaining ignores index: this is always the next one not made yet
    ahead = await withQueueLock(async () => {
      if ((await pendingIds()).length) return null; // a redo got in while we waited for the lock
      return submitChecked((out) => (makesShots(out) ? null : "The next segment wasn't queued: Make is switched off in the workflow."));
    });
    if (ahead) toast(`Segment ${s.made} saved. Segment ${s.next} is next (${s.left} left). Stop ends the chain.`);
    else window.__lcH3ChainNext = false;
  } catch (err) {
    window.__lcH3ChainNext = false;
    lastMade = 0;
    toast(`The chain stopped: ${err.message || err}`, "warn");
  } finally {
    queuing = false;
  }
});

api.addEventListener("execution_start", (e) => {
  if (ahead && e.detail?.prompt_id === ahead) {
    ahead = null; // the queued-ahead segment is running now
    window.__lcH3ChainNext = false;
  }
});

for (const ev of ["execution_interrupted", "execution_error"]) {
  api.addEventListener(ev, async (e) => {
    // a card picture (or another workflow) stopped or failed: the movie carries on
    if ((await isShotJob(e.detail?.prompt_id)) === false) return;
    const had = !!ahead;
    await takeBack();
    lastMade = 0;
    window.__lcH3Only = null; // a stopped ↻ leaves nothing behind for the next run
    if (had && ev === "execution_interrupted") toast("Stopped. The next segment was taken back out of the queue. Run again to carry on.");
  });
}

app.registerExtension({ name: "LC.AV.H3SegmentChain" });
