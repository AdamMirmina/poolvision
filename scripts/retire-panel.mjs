// Take down panels that have already been answered.
//
// A panel left up after it has been judged is not neutral: it reads as still
// waiting, so review goes back to a question that is already closed and
// cannot tell it apart from a new one. The uploaders retire the PREVIOUS set
// when they push a new one, which means an answered panel stays live until
// something else happens to replace it, and if nothing does it stays forever.
//
//   node scripts/retire-panel.mjs           # retire anything already answered
//   node scripts/retire-panel.mjs --all     # retire everything active
//   node scripts/retire-panel.mjs --list    # show what is live, change nothing
import { hydrate } from "./env.mjs";
hydrate();

const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const ALL = process.argv.includes("--all");
const LIST = process.argv.includes("--list");

const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
if (!auth.token) { console.error("auth failed"); process.exit(1); }
const A = { Authorization: auth.token };

let live = 0, retired = 0;
for (const coll of ["vision_pics", "vision_tapes"]) {
  const res = await fetch(
    `${PB}/api/collections/${coll}/records?filter=` + encodeURIComponent("active=true"),
    { headers: A });
  if (!res.ok) { console.error(`${coll}: HTTP ${res.status}`); process.exit(1); }
  const items = (await res.json()).items || [];
  for (const it of items) {
    const answered = !!(it.notes && String(it.notes).trim());
    live++;
    if (LIST || (!answered && !ALL)) {
      console.log(`  live  ${coll}  ${it.title}  ${answered ? "(ANSWERED)" : "(waiting)"}`);
      continue;
    }
    const up = await fetch(`${PB}/api/collections/${coll}/records/${it.id}`, {
      method: "PATCH", headers: { ...A, "Content-Type": "application/json" },
      body: JSON.stringify({ active: false }),
    });
    if (!up.ok) { console.error(`  FAILED to retire ${it.id}: HTTP ${up.status}`); process.exit(1); }
    console.log(`  retired  ${coll}  ${it.title}`);
    retired++;
  }
}
console.log(`${live} active, ${retired} retired`);
