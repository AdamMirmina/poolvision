// A home on the panel for still images, so a visual review needs to look at lands
// where he already is instead of in a chat message he has to scroll back to.
import { hydrate } from "./env.mjs";
hydrate();
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
if (!auth.token) { console.error("auth failed"); process.exit(1); }
const H = { Authorization: auth.token, "Content-Type": "application/json" };

const existing = await (await fetch(PB + "/api/collections/vision_pics", { headers: H })).json();
if (existing && existing.id) { console.log("vision_pics already exists"); process.exit(0); }

const res = await fetch(PB + "/api/collections", {
  method: "POST", headers: H,
  body: JSON.stringify({
    name: "vision_pics",
    type: "base",
    // Same access shape as vision_tapes: the panel reads it, only a superuser
    // writes it, and the notes come back through an update rule.
    listRule: "", viewRule: "", createRule: null, updateRule: "", deleteRule: null,
    fields: [
      { name: "title",   type: "text" },
      { name: "caption", type: "text", max: 2000 },
      { name: "images",  type: "file", maxSelect: 12, maxSize: 12000000,
        mimeTypes: ["image/jpeg", "image/png", "image/webp"] },
      { name: "notes",   type: "text", max: 20000 },
      { name: "active",  type: "bool" },
    ],
  }),
});
const out = await res.json();
console.log(res.ok ? "created vision_pics" : "failed: " + JSON.stringify(out).slice(0, 400));
