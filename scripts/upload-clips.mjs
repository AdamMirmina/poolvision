// Push candidate shot clips into poolean's PocketBase so they can be reviewed in
// the app. Idempotent: a clip already present (same video + candidate number) is
// skipped, so re-running after generating more clips only adds the new ones.
//
// Media deliberately does NOT go into either git repo. poolvision was split out
// of poolean precisely to keep model weights and training media from bloating a
// repo that uploads to EAS on every iOS build, and review clips are exactly that
// kind of media.
//
// Labels review already called by hand are prefilled, so review isn't asked to judge
// the same shot twice.
//
//   PB_EM=... PB_PW=... node scripts/upload-clips.mjs [--clips out/clips] [--labels labels/shots.csv]

import fs from "fs";
import path from "path";

import { creds } from "./env.mjs";
const { EM, PW, URL: PB } = creds();

const arg = (name, dflt) => {
  const i = process.argv.indexOf("--" + name);
  return i > -1 ? process.argv[i + 1] : dflt;
};
const CLIPS = arg("clips", "out/clips");
const LABELS = arg("labels", "labels/shots.csv");

// Which recording these clips belong to. This used to default to IMG_2403.MOV,
// which is the kind of default that does real damage quietly: running with
// --clips out/clips_IMG_2528 and no --video filed 22 shootout clips under
// IMG_2403, where they collided by shot number with that video's already-judged
// shots and read as part of it. Nothing errored.
//
// So: infer it from the clips directory, which already names the recording, and
// refuse to run when it cannot be inferred rather than guessing.
const VIDEO = (() => {
  const explicit = arg("video", null);
  if (explicit) return explicit.endsWith(".MOV") ? explicit : explicit + ".MOV";
  const m = /clips[_-](.+)$/.exec(CLIPS.replace(/[\\/]+$/, "").split(/[\\/]/).pop());
  if (m) return m[1].endsWith(".MOV") ? m[1] : m[1] + ".MOV";
  console.error(`Cannot tell which recording "${CLIPS}" belongs to.`);
  console.error("Pass --video IMG_XXXX.MOV explicitly.");
  process.exit(1);
})();

async function api(method, p, token, body) {
  const r = await fetch(PB + p, {
    method,
    headers: { ...(token ? { Authorization: token } : {}), ...(body && !(body instanceof FormData) ? { "Content-Type": "application/json" } : {}) },
    body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  let d; try { d = JSON.parse(text); } catch { d = text; }
  return { ok: r.ok, status: r.status, d };
}

// the hand calls, so a clip review's already judged arrives pre-labeled.
function readLabels(file) {
  if (!fs.existsSync(file)) return [];
  const lines = fs.readFileSync(file, "utf8").split(/\r?\n/).filter((l) => l && !l.startsWith("#"));
  const head = lines.shift().split(",");
  return lines.map((l) => {
    // naive split is fine: only the notes column is ever quoted, and it's last
    const parts = l.split(",");
    const row = {};
    head.forEach((h, i) => (row[h] = parts[i] ?? ""));
    const [m, s] = row.time.split(":").map(Number);
    return { ...row, secs: m * 60 + s };
  });
}

const auth = await api("POST", "/api/collections/_superusers/auth-with-password", null, { identity: EM, password: PW });
if (!auth.ok) { console.error("auth failed", auth.status, auth.d); process.exit(1); }
const token = auth.d.token;

const index = JSON.parse(fs.readFileSync(path.join(CLIPS, "index.json"), "utf8"));
const labels = readLabels(LABELS);
console.log(`${index.length} clips, ${labels.length} hand labels`);

const existing = await api("GET", `/api/collections/vision_shots/records?perPage=500&filter=(video='${VIDEO}')`, token);
const have = new Set((existing.d.items || []).map((r) => r.n));
console.log(`${have.size} already uploaded`);

let added = 0, prefilled = 0;
for (const c of index) {
  if (have.has(c.n)) continue;

  // a hand label counts for this clip if it falls inside the clip's window
  const hit = labels.find((l) => l.secs >= c.t - 3 && l.secs <= c.t_end + 3);
  const others = labels.filter((l) => l.secs >= c.t - 3 && l.secs <= c.t_end + 3);

  const fd = new FormData();
  fd.set("video", VIDEO);
  fd.set("n", String(c.n));
  fd.set("t", String(c.t));
  // tEnd was missing here until 2026-07-31, so all 128 of IMG_2481's records
  // stored 0. Nothing broke visibly -- review only reads `t` -- but every
  // later attempt to pull a clip's detections back out of the rim pass searched
  // the window [t, 0] and found nothing, which stranded 119 of the labels
  // when they were first used for training.
  fd.set("tEnd", String(c.t_end));
  fd.set("durationS", (c.frames / 30).toFixed(2));
  fd.set("clock", c.clock);
  fd.set("hoop", c.hoop);
  fd.set("dets", String(c.dets));
  fd.set("peakConf", String(c.peak_conf));
  fd.set("label", hit ? hit.outcome : "");
  fd.set("labeledBy", hit ? "hand" : "");
  fd.set("labeledAt", hit ? String(Date.now()) : "0");
  fd.set("notes", others.length > 1
    ? `${others.length} hand labels fall in this clip (${others.map((o) => o.time).join(", ")}) — may be more than one shot`
    : "");
  const buf = fs.readFileSync(path.join(CLIPS, c.file));
  fd.set("clip", new Blob([buf], { type: "video/mp4" }), c.file);

  const r = await api("POST", "/api/collections/vision_shots/records", token, fd);
  if (!r.ok) { console.error(`  FAILED ${c.file}`, r.status, JSON.stringify(r.d).slice(0, 300)); continue; }
  added++;
  if (hit) prefilled++;
  console.log(`  + ${c.file}  ${c.clock} ${c.hoop}${hit ? `  [${hit.outcome}]` : ""}`);
}

console.log(`\nuploaded ${added}, of which ${prefilled} arrived pre-labeled from the calls`);
console.log(`${index.length - added - have.size} skipped`);
