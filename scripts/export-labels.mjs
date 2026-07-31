// Freeze the judged clips to disk so training never depends on a live database.
//
// the calls are the only ground truth this project has and they cost real
// time to produce; a training run that silently trains on whatever the API
// happened to return that minute is not reproducible. Committed alongside the
// code that consumes it.
//
//   PB_EM=... PB_PW=... node scripts/export-labels.mjs
import fs from "fs";
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const r = await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
});
const token = (await r.json()).token;
let page = 1, all = [];
while (true) {
  const j = await (await fetch(`${PB}/api/collections/vision_shots/records?perPage=200&page=${page}`,
    { headers: { Authorization: token } })).json();
  all = all.concat(j.items);
  if (page >= j.totalPages) break;
  page++;
}
const out = all.filter((x) => x.label).map((x) => ({
  video: x.video, n: x.n, t: Number(x.t), tEnd: Number(x.tEnd), hoop: x.hoop,
  label: x.label, by: x.labeledBy, notes: x.notes || "",
}));
fs.mkdirSync("labels", { recursive: true });
fs.writeFileSync("labels/judged.json", JSON.stringify(out, null, 1));
console.log(`${out.length} judged clips -> labels/judged.json`);
