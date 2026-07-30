// Replace the uploaded clip set with a freshly generated one, carrying every
// existing label across.
//
// Needed because the clip boundaries themselves changed: the first cut split a
// single shot into two clips (wind-up in one, the ball arriving in the other),
// so clips are now padded and merged, and 31 became 20. That renumbers
// everything, and the labels are the expensive part -- they cannot be dropped
// just because the media around them moved.
//
// A label carries over when the new clip is at the SAME hoop and its window
// contains the old clip's moment. If two different outcomes land in one new
// clip, it is left unlabelled with a note rather than guessed at: merging can
// genuinely put two shots in one clip, and inventing an answer for that is the
// mistake this whole file exists to avoid.
//
//   PB_EM=... PB_PW=... node scripts/migrate-clips.mjs [--apply]

import fs from "fs";
import path from "path";

const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const APPLY = process.argv.includes("--apply");
const VIDEO = "IMG_2403.MOV";
const CLIPS = "out/clips";

async function api(method, p, token, body) {
  const isForm = body instanceof FormData;
  const r = await fetch(PB + p, {
    method,
    headers: { ...(token ? { Authorization: token } : {}), ...(body && !isForm ? { "Content-Type": "application/json" } : {}) },
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  });
  const t = await r.text();
  let d; try { d = JSON.parse(t); } catch { d = t; }
  return { ok: r.ok, status: r.status, d };
}

const auth = await api("POST", "/api/collections/_superusers/auth-with-password", null,
  { identity: process.env.PB_EM, password: process.env.PB_PW });
if (!auth.ok) { console.error("auth failed"); process.exit(1); }
const token = auth.d.token;

// 1. capture what's there, labels included
const old = (await api("GET", `/api/collections/vision_shots/records?perPage=500&filter=(video='${VIDEO}')&sort=n`, token)).d.items || [];
const carried = old.filter((r) => r.label).map((r) => ({ t: r.t, hoop: r.hoop, label: r.label, by: r.labeledBy, clock: r.clock }));
console.log(`${old.length} existing records, ${carried.length} carry a label`);

const index = JSON.parse(fs.readFileSync(path.join(CLIPS, "index.json"), "utf8"));
console.log(`${index.length} new clips\n`);

// How many separate ball arcs are in each clip? Merging windows so a shot isn't
// cut in half can also pull two nearby shots into one clip, and one label can't
// describe two outcomes. A shot is a rise then a fall, so counting the times the
// ball peaks (a minimum in image y, since y grows downward) counts the shots.
// This doesn't split them -- it tells review when there's more than one to
// account for, which beats silently pretending a clip holds a single shot.
const rim = JSON.parse(fs.readFileSync("out/rimwatch_full.json", "utf8"));
function arcsIn(clip) {
  const pts = (rim.hits[clip.hoop] || [])
    .filter((h) => h.t >= clip.clip_start && h.t <= clip.clip_stop)
    .sort((a, b) => a.t - b.t);
  if (pts.length < 6) return pts.length ? 1 : 0;
  // smooth y a little so detector jitter isn't mistaken for an arc
  const y = pts.map((_, i) => {
    const w = pts.slice(Math.max(0, i - 2), i + 3);
    return w.reduce((s, p) => s + p.y, 0) / w.length;
  });
  let arcs = 0, lastPeakT = -99;
  for (let i = 2; i < y.length - 2; i++) {
    const isPeak = y[i] < y[i - 2] && y[i] < y[i + 2];
    // a real peak rises meaningfully above where the ball sits, and two peaks
    // inside half a second are the same one
    if (isPeak && pts[i].t - lastPeakT > 0.6) { arcs++; lastPeakT = pts[i].t; }
  }
  return Math.max(1, arcs);
}
for (const c of index) c.arcs = arcsIn(c);
const multi = index.filter((c) => c.arcs > 1);
console.log(`${multi.length} clips hold more than one arc: ${multi.map((c) => `#${c.n} ${c.clock} (${c.arcs})`).join(", ")}\n`);

// 2. work out where each label lands, before touching anything
const assign = new Map();
for (const L of carried) {
  // a call belongs to the clip whose DESCENT window contains it
  const hit = index.find((c) => c.hoop === L.hoop && L.t >= c.t - 1.5 && L.t <= c.t_end + 1.5)
    || index.find((c) => c.hoop === L.hoop && L.t >= c.clip_start && L.t <= c.clip_stop);
  if (!hit) { console.log(`  ${L.clock} ${L.hoop} ${L.label}: no new clip covers it — DROPPED`); continue; }
  const prev = assign.get(hit.n);
  if (prev && prev.label !== L.label) {
    console.log(`  clip ${hit.n} (${hit.clock}): ${prev.label} and ${L.label} both land here — left unlabelled`);
    assign.set(hit.n, { label: "", by: "", clash: `${prev.label} + ${L.label}` });
  } else if (!prev) {
    assign.set(hit.n, L);
  }
}
console.log(`\n${assign.size} of ${index.length} new clips inherit a label`);
if (!APPLY) { console.log("\ndry run — pass --apply"); process.exit(0); }

// 3. swap
for (const r of old) await api("DELETE", `/api/collections/vision_shots/records/${r.id}`, token);
console.log(`deleted ${old.length} old records`);

let added = 0;
for (const c of index) {
  const L = assign.get(c.n);
  const fd = new FormData();
  fd.set("video", VIDEO);
  fd.set("n", String(c.n));
  fd.set("t", String(c.t));
  fd.set("tEnd", String(c.t_end));
  fd.set("durationS", (c.frames / 30).toFixed(2));
  fd.set("clock", c.clock);
  fd.set("hoop", c.hoop);
  fd.set("dets", String(c.dets));
  fd.set("peakConf", String(c.peak_conf));
  fd.set("arcs", String(c.arcs || 1));
  fd.set("label", L && !L.clash ? L.label : "");
  fd.set("labeledBy", L && !L.clash ? L.by : "");
  fd.set("labeledAt", L && !L.clash ? String(Date.now()) : "0");
  fd.set("notes", L && L.clash ? `two different calls (${L.clash}) fall in this clip — probably more than one shot` : "");
  fd.set("clip", new Blob([fs.readFileSync(path.join(CLIPS, c.file))], { type: "video/mp4" }), c.file);
  const r = await api("POST", "/api/collections/vision_shots/records", token, fd);
  if (!r.ok) { console.error(`  FAILED ${c.file}`, r.status, JSON.stringify(r.d).slice(0, 200)); continue; }
  added++;
}
console.log(`uploaded ${added} new records`);

const now = (await api("GET", `/api/collections/vision_shots/records?perPage=500&filter=(video='${VIDEO}')`, token)).d.items || [];
const tally = (a) => a.reduce((m, v) => ((m[v] = (m[v] || 0) + 1), m), {});
console.log("labels:", tally(now.map((r) => r.label || "(unjudged)")));
console.log("by:    ", tally(now.map((r) => r.labeledBy || "(none)")));
