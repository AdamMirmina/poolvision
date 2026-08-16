// Prove the frames the PANEL serves are the ones just regenerated.
//
// The frames had been
// regenerated locally and never uploaded, so the panel kept serving the old
// ones. Checking the local file is not the check; the served file is.
import fs from "fs";
import path from "path";
import { hydrate } from "./env.mjs";

hydrate();
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
if (!auth.token) { console.error("auth failed"); process.exit(1); }

const res = await fetch(PB + "/api/collections/vision_shots/records?perPage=300",
  { headers: { Authorization: auth.token } });
const items = ((await res.json()).items || []).filter((x) => x.diag);
console.log(`${items.length} shot records carry a diag frame`);

let checked = 0, mismatched = 0;
for (const x of items.slice(0, 6)) {
  const name = x.diag;
  const url = `${PB}/api/files/vision_shots/${x.id}/${name}`;
  const head = await fetch(url, { method: "HEAD" });
  const served = Number(head.headers.get("content-length") || 0);

  // PocketBase SANITISES the stored filename: pose-diag-IMG_2482-1.jpg is kept
  // as pose_diag_img_2482_1.jpg. Matching on the raw name finds nothing and
  // reports every frame as missing, which is a false alarm dressed as a finding
  // -- the same shape as every other silent-mismatch on this project, except it
  // fails loud in the wrong direction.
  // Two transformations, not one: sanitised AND given a random suffix.
  const bare = name.replace(/_[a-z0-9]{10}(\.\w+)$/, "$1").toLowerCase();
  const local = fs.readdirSync("out/diag")
    .map((f) => path.resolve("out/diag", f))
    .find((f) => path.basename(f).replace(/-/g, "_").toLowerCase() === bare);
  const size = local && fs.existsSync(local) ? fs.statSync(local).size : 0;
  const same = size > 0 && Math.abs(size - served) < 2048;
  checked++;
  if (!same) mismatched++;
  console.log(`  ${name}  served ${served}B  local ${size}B  ${same ? "match" : "DIFFERENT"}`);
}
console.log(mismatched === 0
  ? `\nall ${checked} sampled frames match the regenerated files`
  : `\n${mismatched} of ${checked} do NOT match -- the panel is serving something else`);
