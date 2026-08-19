// Put a marking tape on the judging panel, and take the previous one down.
//
// Without this the tape sits in out/ and nobody sees it. That has already been
// the failure mode twice: a step reports success, the artifact exists, and it
// never reaches the person who was waiting for it.
//
// The tape carries ONLY the model's calls. review watched one that had its own
// marks burned in, said "all shots are correctly identified", then caught the
// circularity himself: "i was under the impression that the model identified
// those shots when really you were displaying what i had already claimed were
// shots." The answer key stays on the marker's side of the test, so what is marked against
// this tape is real evidence about the model.
//
//   node scripts/upload-tape.mjs --file out/marktape_fresh.mp4 \
//        --title "600-900s, model calls only" --calls out/clips_fresh/index.json
import fs from "fs";
import path from "path";
import { hydrate } from "./env.mjs";

hydrate();

const arg = (n, d) => {
  const i = process.argv.indexOf("--" + n);
  return i > -1 ? process.argv[i + 1] : d;
};
const FILE = path.resolve(arg("file", "out/marktape_fresh.mp4"));
const TITLE = arg("title", "");
const CALLS = arg("calls", "");
// Where this tape sits in the ORIGINAL recording, in seconds.
//
// The marking panel reads rec.startAt and adds it to the video position, so
// every timestamp review writes is a position in the source file rather than in
// the excerpt. Without it a tape cut from 15:00 records its first shot as 0:07,
// and nothing downstream can line that up with a scan. It was never passed
// here, so it defaulted to zero on every tape not cut from the start of its
// recording -- silently, since a plausible-looking timestamp is exactly what a
// wrong offset produces.
const START = Number(arg("start", "0")) || 0;
// Which cap colors are actually in this recording, comma separated.
//
// The panel falls back to blue,pink,white when this is absent, which is a
// sensible default and silently wrong for any session with a different set: the
// marking card then offers colors nobody is wearing and omits ones that are in
// the water. IMG_2932 is black, white, purple and yellow.
const CAPS = arg("caps", "").split(",").map((x) => x.trim()).filter(Boolean);
const KEEP = process.argv.includes("--keep-others");
const PB = process.env.PB_URL || "https://poolean-api.adammirmina.com";

if (!fs.existsSync(FILE)) {
  console.error(`no ${FILE}`);
  process.exit(1);
}

const token = (await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json()).token;
if (!token) { console.error("auth failed"); process.exit(1); }
const auth = { Authorization: token };

// Calls go up alongside the video so the panel can list them, and so a tape can
// be tied to the run that produced it later. A tape with no record of what it
// claimed is unjudgeable after the fact.
let calls = [];
if (CALLS && fs.existsSync(CALLS)) {
  calls = JSON.parse(fs.readFileSync(CALLS, "utf8"))
    .map((r) => ({ t: r.t, hoop: r.hoop }))
    .filter((r) => typeof r.t === "number")
    .sort((a, b) => a.t - b.t);
}

// Retire whatever is up now. Two active tapes means review does not know which one
// his feedback is about.
if (!KEEP) {
  const old = await (await fetch(
    PB + "/api/collections/vision_tapes/records?filter=" + encodeURIComponent("active=true"),
    { headers: auth })).json();
  for (const r of old.items || []) {
    await fetch(PB + "/api/collections/vision_tapes/records/" + r.id, {
      method: "PATCH",
      headers: { ...auth, "Content-Type": "application/json" },
      body: JSON.stringify({ active: false }),
    });
    console.log(`retired  ${r.title || r.id}`);
  }
}

const fd = new FormData();
fd.append("video", new Blob([fs.readFileSync(FILE)], { type: "video/mp4" }), path.basename(FILE));
fd.append("title", TITLE);
fd.append("startAt", String(START));
if (CAPS.length) fd.append("caps", JSON.stringify(CAPS));
fd.append("active", "true");
fd.append("calls", JSON.stringify(calls));

const res = await fetch(PB + "/api/collections/vision_tapes/records", {
  method: "POST", headers: auth, body: fd,
});
const rec = await res.json();
if (!res.ok || !rec.id) {
  console.error("upload failed", JSON.stringify(rec).slice(0, 400));
  process.exit(1);
}

// Read it back. "The POST returned 200" is not proof the panel can play it --
// verify the file field is actually populated and the URL resolves.
const back = await (await fetch(
  PB + "/api/collections/vision_tapes/records/" + rec.id, { headers: auth })).json();
const url = `${PB}/api/files/vision_tapes/${back.id}/${back.video}`;
const head = await fetch(url, { method: "HEAD" });
console.log(`\nuploaded ${path.basename(FILE)}  (${(fs.statSync(FILE).size / 1e6).toFixed(0)} MB)`);
console.log(`title    ${TITLE}`);
console.log(`calls    ${calls.length}`);
console.log(`playable ${head.status === 200 ? "yes" : "NO -- " + head.status}`);
console.log(`url      ${url}`);
