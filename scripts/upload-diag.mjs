// Move the per-shot review frames out of poolean's git and into PocketBase,
// where the clips already live.
//
// They should never have been committed. Each frame is around 800 KB, there are
// 83 of them, and they were renamed once when the naming scheme changed to be
// keyed by recording -- so every version is still in poolean's history. That
// repo is uploaded to EAS on every iOS build, and keeping exactly this kind of
// media out of it is the entire reason poolvision is a separate repo.
//
// Attaching them to the vision_shots record they belong to also fixes a smaller
// problem: review was reconstructing a filename from the record, so a
// missing frame showed as a broken image rather than as "no frame for this shot".
//
// The one thing that DOES still belong in poolean's repo is the manifest: 81 KB
// of JSON review reads to know which shots have a frame and what the model
// was thinking. It is small and it is site data, so it gets copied over rather
// than uploaded.
//
//   node scripts/upload-diag.mjs [--frames out/diag] [--dry]
import fs from "fs";
import path from "path";
import { hydrate } from "./env.mjs";

hydrate();

const arg = (n, d) => {
  const i = process.argv.indexOf("--" + n);
  return i > -1 ? process.argv[i + 1] : d;
};
const DIR = arg("frames", path.resolve("out/diag"));
const SITE = arg("site", path.resolve("../poolean/web/public/assets"));
const DRY = process.argv.includes("--dry");
const FORCE = process.argv.includes("--force");
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";

const token = (await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json()).token;
if (!token) { console.error("auth failed"); process.exit(1); }

// pose-diag-<VIDEO>-<n>.jpg, and the same name with -3pt for the three-point view.
const files = fs.readdirSync(DIR).filter((f) => /^pose-diag-.+\.jpg$/.test(f));
const want = new Map();
for (const f of files) {
  const m = /^pose-diag-(.+)-(\d+)(-3pt)?\.jpg$/.exec(f);
  if (!m) { console.log(`  ? cannot parse ${f}`); continue; }
  const key = `${m[1]}.MOV#${m[2]}`;
  const e = want.get(key) || { video: `${m[1]}.MOV`, n: Number(m[2]) };
  e[m[3] ? "diag3pt" : "diag"] = path.join(DIR, f);
  want.set(key, e);
}
console.log(`${files.length} frames covering ${want.size} shots`);

let page = 1, all = [];
while (true) {
  const r = await (await fetch(`${PB}/api/collections/vision_shots/records?perPage=200&page=${page}`,
    { headers: { Authorization: token } })).json();
  all = all.concat(r.items);
  if (all.length >= r.totalItems) break;
  page++;
}
const byKey = new Map(all.map((r) => [`${r.video}#${r.n}`, r]));

let sent = 0, skipped = 0, missing = 0;
for (const [key, e] of want) {
  const rec = byKey.get(key);
  if (!rec) { missing++; console.log(`  ! no record for ${key}`); continue; }
  // Skip only when NOT re-rendering. This 'already has one' check silently
  // threw away 57 freshly-rendered frames carrying the corrected drop zones,
  // because the records already held the old ones -- so a rebuild that took
  // all night changed nothing review could see, and reported success.
  if (!FORCE && rec.diag && (!e.diag3pt || rec.diag3pt)) { skipped++; continue; }
  if (DRY) { sent++; continue; }
  const fd = new FormData();
  for (const f of ["diag", "diag3pt"]) {
    if (!e[f]) continue;
    const buf = fs.readFileSync(e[f]);
    fd.set(f, new Blob([buf], { type: "image/jpeg" }), path.basename(e[f]));
  }
  const r = await fetch(`${PB}/api/collections/vision_shots/records/${rec.id}`, {
    method: "PATCH", headers: { Authorization: token }, body: fd,
  });
  if (r.ok) { sent++; process.stdout.write(`\r  uploaded ${sent}`); }
  else console.log(`\n  x ${key}: ${(await r.text()).slice(0, 140)}`);
}
console.log(`\n\n${sent} shots given their frames, ${skipped} already had them, ${missing} had no record`);

const manifest = path.join(DIR, "pose-diag.json");
if (fs.existsSync(manifest) && fs.existsSync(SITE)) {
  if (!DRY) fs.copyFileSync(manifest, path.join(SITE, "pose-diag.json"));
  console.log(`manifest copied to ${SITE} (commit it in poolean)`);
}
