// Put stills on the judging panel, and take the previous set down.
//
// Standing instruction: if a visual is built for him to review, it belongs on
// poolean temporarily, next to the box he types feedback into -- not sent as a
// file and not previewed inline in chat. The panel is where he already is when
// he judges the model, and a picture living anywhere else is a second place he
// has to go.
//
//   node scripts/upload-pics.mjs --title "..." --caption "..." out/trace_*.jpg
import fs from "fs";
import path from "path";
import { hydrate } from "./env.mjs";

hydrate();

const arg = (n, d) => {
  const i = process.argv.indexOf("--" + n);
  return i > -1 ? process.argv[i + 1] : d;
};
const TITLE = arg("title", "");
const CAPTION = arg("caption", "");
const files = process.argv.slice(2).filter((a) => /\.(jpe?g|png|webp)$/i.test(a));
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";

if (!files.length) { console.error("no images given"); process.exit(1); }
for (const f of files) {
  if (!fs.existsSync(f)) { console.error(`no ${f}`); process.exit(1); }
}

const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
if (!auth.token) { console.error("auth failed"); process.exit(1); }
const A = { Authorization: auth.token };

// One active set at a time, so his feedback is never ambiguous about which
// picture it refers to.
const old = await (await fetch(
  PB + "/api/collections/vision_pics/records?filter=" + encodeURIComponent("active=true"),
  { headers: A })).json();
for (const r of old.items || []) {
  await fetch(PB + "/api/collections/vision_pics/records/" + r.id, {
    method: "PATCH", headers: { ...A, "Content-Type": "application/json" },
    body: JSON.stringify({ active: false }),
  });
  console.log(`retired  ${r.title || r.id}`);
}

const fd = new FormData();
fd.append("title", TITLE);
fd.append("caption", CAPTION);
fd.append("active", "true");
for (const f of files) {
  const type = /\.png$/i.test(f) ? "image/png" : /\.webp$/i.test(f) ? "image/webp" : "image/jpeg";
  fd.append("images", new Blob([fs.readFileSync(f)], { type }), path.basename(f));
}

const res = await fetch(PB + "/api/collections/vision_pics/records", {
  method: "POST", headers: A, body: fd,
});
const rec = await res.json();
if (!res.ok || !rec.id) {
  console.error("upload failed", JSON.stringify(rec).slice(0, 400));
  process.exit(1);
}

// Read it back and fetch each image. A 200 on the POST is not proof the panel
// can render them -- that check has been skipped before and cost a round trip.
const back = await (await fetch(PB + "/api/collections/vision_pics/records/" + rec.id,
  { headers: A })).json();
console.log(`\nuploaded ${(back.images || []).length} image(s) as "${TITLE}"`);
for (const img of back.images || []) {
  const url = `${PB}/api/files/vision_pics/${back.id}/${img}`;
  const head = await fetch(url, { method: "HEAD" });
  console.log(`  ${head.status === 200 ? "ok " : "BAD"} ${img}`);
}
