// Attach the whole-frame clip to each attributed shot.
//
// The tight shooter crop cannot show whether a play was a shot at all, which is
// what the marked shot hit: "some of these are not shots." This is the same moment seen
// from the whole frame, release through the ball reaching the rim.
import fs from "fs";
import path from "path";
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const argOf = (n, d) => { const i = process.argv.indexOf("--" + n); return i > -1 ? process.argv[i + 1] : d; };
const VIDEO = argOf("video", "IMG_2481");
const DIR = argOf("dir", "out/wide_clips");
const token = (await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json()).token;
let page = 1, all = [];
while (true) {
  const j = await (await fetch(`${PB}/api/collections/vision_shots/records?perPage=200&page=${page}`,
    { headers: { Authorization: token } })).json();
  all = all.concat(j.items);
  if (page >= j.totalPages) break;
  page++;
}
let sent = 0;
for (const r of all.filter((x) => x.video === VIDEO + ".MOV")) {
  const f = path.join(DIR, `${VIDEO}_${r.n}.mp4`);
  if (!fs.existsSync(f)) continue;
  if (r.wideClip) { continue; }              // already up, don't re-send
  const fd = new FormData();
  fd.set("wideClip", new Blob([fs.readFileSync(f)], { type: "video/mp4" }), path.basename(f));
  const res = await fetch(`${PB}/api/collections/vision_shots/records/${r.id}`,
    { method: "PATCH", headers: { Authorization: token }, body: fd });
  if (res.ok) sent++;
  else console.error(`  FAILED #${r.n}`, res.status);
}
console.log(`${sent} whole-frame clips uploaded`);
