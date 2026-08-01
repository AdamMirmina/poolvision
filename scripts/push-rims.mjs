// Push the hoop box for every shot that has one.
import fs from "fs";
const PB = "https://poolean-api.adammirmina.com";
const token = (await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json()).token;
const ov = JSON.parse(fs.readFileSync("out/rimoverlay.json", "utf8"));
const byKey = new Map(ov.map((a) => [a.video + "#" + a.n, a]));
let page = 1, all = [];
while (true) {
  const j = await (await fetch(`${PB}/api/collections/vision_shots/records?perPage=200&page=${page}`,
    { headers: { Authorization: token } })).json();
  all = all.concat(j.items);
  if (page >= j.totalPages) break;
  page++;
}
let n = 0;
for (const r of all) {
  const a = byKey.get(r.video + "#" + r.n);
  if (!a) continue;
  await fetch(`${PB}/api/collections/vision_shots/records/${r.id}`, {
    method: "PATCH", headers: { Authorization: token, "Content-Type": "application/json" },
    body: JSON.stringify({ rimPctWide: a.rimPctWide, rimPctRim: a.rimPctRim }),
  });
  n++;
}
console.log(`hoop box on ${n} records`);
