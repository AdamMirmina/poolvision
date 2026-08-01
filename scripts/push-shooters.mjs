// Push the shooter box and the release-derived in-air window.
//
// Replaces the crop-upload step: the sidebar magnifies the wide clip onto this
// box rather than playing a separately cut file, so there is nothing to encode
// or upload -- just four numbers.
import fs from "fs";
const PB = "https://poolean-api.adammirmina.com";
const argOf = (n, d) => { const i = process.argv.indexOf("--" + n); return i > -1 ? process.argv[i + 1] : d; };
const VIDEO = argOf("video", "IMG_2481");
const token = (await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json()).token;
const res = JSON.parse(fs.readFileSync(`out/attribute_${VIDEO}.json`, "utf8"));
const byN = new Map(res.filter((r) => r.n !== undefined && r.shooterPath).map((r) => [Number(r.n), r]));
let page = 1, all = [];
while (true) {
  const j = await (await fetch(`${PB}/api/collections/vision_shots/records?perPage=200&page=${page}`,
    { headers: { Authorization: token } })).json();
  all = all.concat(j.items);
  if (page >= j.totalPages) break;
  page++;
}
let n = 0;
for (const r of all.filter((x) => x.video === VIDEO + ".MOV")) {
  const a = byN.get(Number(r.n));
  if (!a) continue;
  // The path, not a point: a swimmer moves through the clip and a fixed box
  // leaves them behind. Stored as JSON text, one row per sampled moment.
  const body = { shooterPath: JSON.stringify(a.shooterPath) };
  if (a.hue !== undefined) body.capHue = a.hue;
  await fetch(`${PB}/api/collections/vision_shots/records/${r.id}`, {
    method: "PATCH", headers: { Authorization: token, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  n++;
}
console.log(`shooter box written on ${n} records`);
