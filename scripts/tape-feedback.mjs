import { hydrate } from "./env.mjs";
hydrate();
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
if (!auth.token) { console.log("AUTH FAILED", JSON.stringify(auth).slice(0, 200)); process.exit(1); }
const res = await fetch(PB + "/api/collections/vision_tapes/records",
  { headers: { Authorization: auth.token } });
const r = await res.json();
console.log("status", res.status, "items", (r.items || []).length);
if (!res.ok) console.log(JSON.stringify(r).slice(0, 500));
for (const x of r.items || []) {
  console.log(`=== ${x.title || x.id}   active=${x.active}   updated=${x.updated}`);
  console.log("FIELDS: " + Object.keys(x).join(", "));
  for (const k of ["notes", "feedback", "marks", "comment", "comments"]) {
    if (x[k]) console.log(`${k}:\n${x[k]}\n`);
  }
}
