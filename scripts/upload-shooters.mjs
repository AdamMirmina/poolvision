// Attach each shot's release clip and measured cap color to its record, so
// review can confirm the attribution by watching rather than by trusting a number.
//
// The shot clips already in review are cropped to the rim and cannot show
// who threw the ball -- the shooter is thirty feet outside that frame. These are
// a separate, wider crop centered on the release.
//
//   PB_EM=... PB_PW=... node scripts/upload-shooters.mjs --video IMG_2481
import fs from "fs";
import path from "path";

const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const argOf = (n, d) => { const i = process.argv.indexOf("--" + n); return i > -1 ? process.argv[i + 1] : d; };
const VIDEO = argOf("video", "IMG_2481");
const CLIPS = argOf("clips", "out/shooter_clips");

const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
const token = auth.token;
if (!token) { console.error("auth failed"); process.exit(1); }

const results = JSON.parse(fs.readFileSync(`out/attribute_${VIDEO}.json`, "utf8"));
const byN = new Map(results.filter((r) => r.stage === "attributed").map((r) => [Number(r.n), r]));

let page = 1, all = [];
while (true) {
  const j = await (await fetch(`${PB}/api/collections/vision_shots/records?perPage=200&page=${page}`,
    { headers: { Authorization: token } })).json();
  all = all.concat(j.items);
  if (page >= j.totalPages) break;
  page++;
}
const mine = all.filter((r) => r.video === VIDEO + ".MOV");

let sent = 0, skipped = 0;
for (const rec of mine) {
  const a = byN.get(Number(rec.n));
  const file = path.join(CLIPS, `${VIDEO}_${rec.n}.mp4`);
  if (!a || !fs.existsSync(file)) { skipped++; continue; }
  const fd = new FormData();
  fd.set("capHue", String(a.hue));
  // Never overwrite an answer review has already given. Re-running this after a
  // pipeline change would otherwise silently wipe his confirmations, which are
  // the only ground truth attribution has.
  if (!rec.shooterOk) fd.set("shooterOk", "");
  fd.set("shooterClip", new Blob([fs.readFileSync(file)], { type: "video/mp4" }), path.basename(file));
  const r = await fetch(`${PB}/api/collections/vision_shots/records/${rec.id}`,
    { method: "PATCH", headers: { Authorization: token }, body: fd });
  if (!r.ok) { console.error(`  FAILED #${rec.n}`, r.status, (await r.text()).slice(0, 160)); continue; }
  sent++;
}
console.log(`${sent} shots now carry a release clip; ${skipped} had no attribution to attach`);
const hues = [...byN.values()].map((r) => r.hue);
const bands = hues.reduce((m, h) => { const k = h < 0 ? "white" : Math.floor(h / 30) * 30; m[k] = (m[k] || 0) + 1; return m; }, {});
console.log("cap colors seen:", bands);
