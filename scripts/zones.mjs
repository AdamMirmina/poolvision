// Move the drop zone's along/into offsets between hoops.py and PocketBase, so
// review can nudge a zone from review instead of telling me "a little more
// south" and waiting for a round trip.
//
// Everything else about a zone is measured automatically from new footage --
// the waterline, the wall direction, the perpendicular, the size in rim widths.
// The one thing no script gets right is where along the wall the square sits:
// deriving it from where balls were last seen made recall worse (67% to 29%),
// because that measurement is polluted by sightings that are still at the rim or
// on the deck. Until that is fixed, a person's eye is the better instrument, and
// this is what makes using it cheap.
//
//   node scripts/zones.mjs push    # seed PocketBase from hoops.py
//   node scripts/zones.mjs pull    # write out/zones.json for the pipeline
import fs from "fs";
import { hydrate } from "./env.mjs";

hydrate();
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const mode = process.argv[2] || "pull";

const token = (await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json()).token;
if (!token) { console.error("auth failed"); process.exit(1); }

const list = async () => {
  const r = await (await fetch(`${PB}/api/collections/vision_zones/records?perPage=200`,
    { headers: { Authorization: token } })).json();
  return r.items || [];
};

if (mode === "push") {
  // hoops.py stays the source of the MEASURED axes; this only seeds the two
  // offsets so there is a row to drag.
  const seed = JSON.parse(fs.readFileSync("out/zones_seed.json", "utf8"));
  const have = new Map((await list()).map((r) => [`${r.video}#${r.hoop}`, r]));
  for (const s of seed) {
    const key = `${s.video}#${s.hoop}`;
    const cur = have.get(key);
    const body = { video: s.video, hoop: s.hoop, alongOff: s.alongOff, intoOff: s.intoOff };
    const r = cur
      ? await fetch(`${PB}/api/collections/vision_zones/records/${cur.id}`,
          { method: "PATCH", headers: { Authorization: token, "Content-Type": "application/json" }, body: JSON.stringify(body) })
      : await fetch(`${PB}/api/collections/vision_zones/records`,
          { method: "POST", headers: { Authorization: token, "Content-Type": "application/json" }, body: JSON.stringify(body) });
    console.log(`${r.ok ? (cur ? "updated" : "created") : "FAILED"}  ${key}`);
  }
} else {
  const rows = await list();
  const out = {};
  for (const r of rows) out[`${r.video}#${r.hoop}`] = { alongOff: r.alongOff, intoOff: r.intoOff };
  fs.mkdirSync("out", { recursive: true });
  fs.writeFileSync("out/zones.json", JSON.stringify(out, null, 1));
  console.log(`wrote out/zones.json with ${rows.length} zone(s)`);
}
