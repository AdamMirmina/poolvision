// Turn the marking panel's notes into an answer-key file.
//
// Parsed rather than hand-copied. Twenty-three entries transcribed by eye is a
// transcription error waiting to happen, and an answer key with a wrong row in
// it sends the whole project chasing a failure that never occurred.
//
// It merges entries sharing a timestamp. marked several shots twice --
// once for hoop/outcome/cap and again to add "dunk" -- and told me so directly:
// "ones like these are the same and i'm just specifying that it was a dunk."
// Two rows, one shot.
//
//   node scripts/key-from-notes.mjs --video IMG_2770.MOV --out labels/answerkey_IMG_2770.json
import fs from "fs";
import path from "path";
import { hydrate } from "./env.mjs";

hydrate();

const arg = (n, d) => {
  const i = process.argv.indexOf("--" + n);
  return i > -1 ? process.argv[i + 1] : d;
};
const VIDEO = arg("video", "IMG_2770.MOV");
const OUT = arg("out", "");
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";

const CAPS = ["pink", "white", "blue", "purple", "green", "yellow", "orange", "red", "black"];
const OUTCOMES = ["make", "miss", "airball"];

const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
const res = await fetch(PB + "/api/collections/vision_tapes/records?filter="
  + encodeURIComponent("active=true"), { headers: { Authorization: auth.token } });
const rec = ((await res.json()).items || [])[0];
if (!rec) { console.error("no active tape"); process.exit(1); }

const shots = new Map();
const notes = [];
for (const raw of String(rec.notes || "").split("\n")) {
  const line = raw.trim();
  if (!line) continue;
  const m = /^(\d{1,2}):(\d{2})/.exec(line);
  if (!m) { notes.push(line); continue; }
  const t = Number(m[1]) * 60 + Number(m[2]);
  // Split on anything that is not a word, so "yellow, dunk" and "orange dunk"
  // and "yellow  dunk" all read the same.
  const words = line.slice(m[0].length).toLowerCase().split(/[^a-z]+/).filter(Boolean);
  const prev = shots.get(t) || { t, at: m[0], hoop: "", outcome: "", cap: "", dunk: false, notes: [] };
  for (const w of words) {
    if (w === "left" || w === "right") prev.hoop = w;
    else if (OUTCOMES.includes(w)) prev.outcome = w;
    else if (CAPS.includes(w)) prev.cap = w;
    else if (w === "dunk") prev.dunk = true;
    else if (w === "missed" || w === "attempted") prev.notes.push(w);
  }
  shots.set(t, prev);
}

const list = [...shots.values()].sort((a, b) => a.t - b.t).map((s) => {
  const o = { t: s.t, at: s.at, hoop: s.hoop, outcome: s.outcome, shooter_cap: s.cap };
  if (s.dunk) o.dunk = true;
  if (s.notes.length) o.note = s.notes.join(" ") + " dunk";
  return o;
});

// Loud about anything incomplete. A row missing a hoop or an outcome would be
// scored as though the model got it wrong.
const bad = list.filter((s) => !s.hoop || !s.outcome);
for (const s of bad) console.error(`INCOMPLETE ${s.at}: ${JSON.stringify(s)}`);

const key = {
  video: VIDEO,
  marked_by: "review",
  marked_on: "2026-08-11",
  covers: [Number(rec.startAt) || 0, (Number(rec.startAt) || 0) + 360],
  how: "Marked blind in the panel's pause-to-mark UI, against a tape carrying no model output.",
  why_it_matters: "Third camera position, nine players, and the first footage where dunks are a third of the play. Nothing in the pipeline has seen any of it.",
  timing_caveat: notes.join(" ") || null,
  shots: list,
  not_shots: [],
};
const dest = OUT || `labels/answerkey_${path.basename(VIDEO, ".MOV")}.json`;
fs.writeFileSync(dest, JSON.stringify(key, null, 1) + "\n", "utf8");

const by = (f) => list.reduce((a, s) => (a[s[f]] = (a[s[f]] || 0) + 1, a), {});
console.log(`${list.length} shots -> ${dest}`);
console.log("  outcome", JSON.stringify(by("outcome")));
console.log("  hoop   ", JSON.stringify(by("hoop")));
console.log("  cap    ", JSON.stringify(by("shooter_cap")));
console.log("  dunks  ", list.filter((s) => s.dunk).length);
if (bad.length) { console.error(`\n${bad.length} INCOMPLETE row(s) above -- fix before scoring`); process.exit(1); }
