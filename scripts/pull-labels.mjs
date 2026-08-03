// the judgments live in PocketBase. labels/allshots.json is a CACHE, and a
// stale one is worse than none: every measurement in this repo reads it, so when
// it lags, every reported number is computed on a fraction of his work and looks
// completely normal.
//
// It lagged by 187 shots -- he had judged 415 and the file held 228 -- and the
// drop-zone sweeps, recall figures and veto precision quoted all afternoon were
// all computed on 55% of the data. Caught in review: That was right.
//
// Run this before ANY measurement that reads labels.
import fs from "fs";
import { hydrate } from "./env.mjs";
hydrate();
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const t = (await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json()).token;
let page = 1, all = [];
while (true) {
  const r = await (await fetch(`${PB}/api/collections/vision_shots/records?perPage=200&page=${page}`,
    { headers: { Authorization: t } })).json();
  all = all.concat(r.items);
  if (all.length >= r.totalItems) break;
  page++;
}
const rows = all.map((r) => ({
  video: r.video, n: r.n, t: r.t, tEnd: r.tEnd, clock: r.clock, hoop: r.hoop,
  label: r.label || "", clipStart: r.clipStart, durationS: r.durationS,
  shooterOk: r.shooterOk || "", shooterName: r.shooterName || "",
})).sort((a, b) => String(a.video).localeCompare(String(b.video)) || a.n - b.n);
fs.writeFileSync("labels/allshots.json", JSON.stringify(rows, null, 1));
const judged = rows.filter((r) => r.label).length;
console.log(`pulled ${rows.length} shots, ${judged} judged`);
const c = {};
for (const r of rows) c[r.label || "(none)"] = (c[r.label || "(none)"] || 0) + 1;
console.log(JSON.stringify(c));
