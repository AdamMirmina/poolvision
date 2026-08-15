// vision_pics needs to hold short clips too. A still asks the viewer to
// reconstruct motion from a frozen path; a clip just shows what happened.
import { hydrate } from "./env.mjs";
hydrate();
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
const H = { Authorization: auth.token, "Content-Type": "application/json" };
const col = await (await fetch(PB + "/api/collections/vision_pics", { headers: H })).json();
if ((col.fields || []).some((f) => f.name === "clips")) { console.log("already there"); process.exit(0); }
const fields = [...col.fields, {
  name: "clips", type: "file", maxSelect: 16, maxSize: 60000000,
  mimeTypes: ["video/mp4", "video/quicktime"],
}];
const res = await fetch(PB + "/api/collections/vision_pics", {
  method: "PATCH", headers: H, body: JSON.stringify({ fields }),
});
console.log(res.ok ? "added clips field" : "failed " + JSON.stringify(await res.json()).slice(0, 300));
