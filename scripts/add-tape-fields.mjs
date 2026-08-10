// A marking UI needs two facts the tape record does not carry yet.
//
// startAt: the video element's currentTime is relative to the CLIP, but every
// timestamp on this project is the position in the original recording -- that is
// the whole reason the clock is burned in. Without the offset the app would
// record 0:07 for a shot review and the pipeline both call 5:07.
//
// caps: the cap colors in play, which differ per party. IMG_2482 has five
// players; the drill footage has three. A fixed list would be wrong for one of
// them, and guessing is what a per-party calibration exists to avoid.
import { hydrate } from "./env.mjs";
hydrate();
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";
const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
const H = { Authorization: auth.token, "Content-Type": "application/json" };
const col = await (await fetch(PB + "/api/collections/vision_tapes", { headers: H })).json();
const have = new Set((col.fields || []).map((f) => f.name));
const add = [];
if (!have.has("startAt")) add.push({ name: "startAt", type: "number" });
if (!have.has("caps")) add.push({ name: "caps", type: "json" });
if (add.length) {
  const res = await fetch(PB + "/api/collections/vision_tapes", {
    method: "PATCH", headers: H, body: JSON.stringify({ fields: [...col.fields, ...add] }),
  });
  console.log(res.ok ? "added " + add.map((f) => f.name).join(", ")
    : "failed " + JSON.stringify(await res.json()).slice(0, 300));
} else console.log("fields already present");

// Fill them in on whatever is active.
const cur = await (await fetch(PB + "/api/collections/vision_tapes/records?filter="
  + encodeURIComponent("active=true"), { headers: H })).json();
for (const r of cur.items || []) {
  const m = /IMG_(\d+),\s*(\d+):(\d\d)/.exec(r.title || "");
  const startAt = m ? Number(m[2]) * 60 + Number(m[3]) : 0;
  const caps = /2482/.test(r.title || "")
    ? ["pink", "yellow", "blue", "white", "lime"]
    : ["blue", "pink", "white"];
  await fetch(PB + "/api/collections/vision_tapes/records/" + r.id, {
    method: "PATCH", headers: H, body: JSON.stringify({ startAt, caps }),
  });
  console.log(`${r.title}\n  startAt=${startAt}s  caps=${caps.join(", ")}`);
}
