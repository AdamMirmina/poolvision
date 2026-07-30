// Recompute which hand label belongs to which clip, and correct every record.
//
// The first prefill was wrong in two ways: it matched on time alone, so a label
// could land on a clip at the OTHER hoop, and it allowed one hand call to be
// copied onto several overlapping clips. Both inflate the label set with
// fabricated entries, which is worse than having no labels at all, since the
// whole point of this file is to be the thing a model is measured against.
//
// Correct rule: a hand label belongs to exactly one clip -- same hoop, and the
// nearest clip whose window contains it. Every clip that doesn't win a label is
// explicitly cleared, so a bad prefill can't survive a re-run.
//
//   PB_EM=... PB_PW=... node scripts/fix-prefill.mjs [--apply]

import fs from "fs";

const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const EM = process.env.PB_EM, PW = process.env.PB_PW;
const APPLY = process.argv.includes("--apply");
const VIDEO = "IMG_2403.MOV";
const SLACK = 3.0; // seconds of tolerance on a hand-called time

async function api(method, p, token, body) {
  const r = await fetch(PB + p, {
    method,
    headers: { ...(token ? { Authorization: token } : {}), ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const t = await r.text();
  let d; try { d = JSON.parse(t); } catch { d = t; }
  return { ok: r.ok, status: r.status, d };
}

const labels = fs.readFileSync("labels/shots.csv", "utf8").split(/\r?\n/)
  .filter((l) => l && !l.startsWith("#")).slice(1)
  .map((l) => {
    const p = l.split(",");
    const [m, s] = p[0].split(":").map(Number);
    return { time: p[0], secs: m * 60 + s, hoop: p[2], outcome: p[3] };
  });

const auth = await api("POST", "/api/collections/_superusers/auth-with-password", null, { identity: EM, password: PW });
if (!auth.ok) { console.error("auth failed", auth.d); process.exit(1); }
const token = auth.d.token;

const res = await api("GET", `/api/collections/vision_shots/records?perPage=500&filter=(video='${VIDEO}')&sort=n`, token);
const clips = res.d.items;
console.log(`${clips.length} clips, ${labels.length} hand labels\n`);

// Each label claims the nearest same-hoop clip that could contain it.
const assign = new Map(); // clip id -> label
for (const L of labels) {
  // t_end isn't stored on the record, so match on the clip's start time within a
  // tolerance rather than on its window. Same hoop is the part that matters.
  const pool = clips.filter((c) => c.hoop === L.hoop && Math.abs(c.t - L.secs) <= 6);
  if (!pool.length) { console.log(`  ${L.time} ${L.hoop} ${L.outcome}: NO clip at that hoop`); continue; }
  const best = pool.reduce((a, b) => (Math.abs(a.t - L.secs) <= Math.abs(b.t - L.secs) ? a : b));
  const prev = assign.get(best.id);
  if (prev && prev.outcome !== L.outcome) {
    console.log(`  clash on clip ${best.n} (${best.clock} ${best.hoop}): ${prev.time} ${prev.outcome} vs ${L.time} ${L.outcome} -> left unlabelled`);
    assign.set(best.id, { outcome: "", time: `${prev.time}+${L.time}`, clash: true });
  } else if (!prev) {
    assign.set(best.id, L);
  } else {
    console.log(`  ${L.time} and ${prev.time} both map to clip ${best.n}, same outcome (${L.outcome}) -- likely two shots in one clip`);
  }
}

let changed = 0;
console.log("\n%s", "clip  clock  hoop   was      ->  now");
for (const c of clips) {
  const L = assign.get(c.id);
  const want = L && !L.clash ? L.outcome : "";
  if ((c.label || "") === want) continue;
  changed++;
  console.log(`  ${String(c.n).padStart(3)}  ${c.clock.padEnd(6)} ${c.hoop.padEnd(6)} ${(c.label || "(none)").padEnd(8)} ->  ${want || "(cleared)"}`);
  if (APPLY) {
    const r = await api("PATCH", `/api/collections/vision_shots/records/${c.id}`, token, {
      label: want,
      // "hand-called" = derived from his timestamps by an automatic mapping that has
      // already been wrong once, so the UI treats it as a suggestion needing one tap
      // to confirm. "hand" is reserved for a label he actually set in the app.
      labeledBy: want ? "hand-called" : "",
      labeledAt: want ? Date.now() : 0,
      notes: L && L.clash ? `two conflicting hand calls (${L.time}) fall here; needs a human look` : "",
    });
    if (!r.ok) console.error("    PATCH failed", r.status, JSON.stringify(r.d).slice(0, 200));
  }
}
const labeled = clips.filter((c) => { const L = assign.get(c.id); return L && !L.clash; }).length;
console.log(`\n${changed} records ${APPLY ? "corrected" : "would change (dry run, pass --apply)"}`);
console.log(`${labeled} of ${clips.length} clips carry a hand label; ${clips.length - labeled} still need judging`);
