// Register the uploaded review copies as session recordings.
//
// A session recording is one file covering several back-to-back games, with
// each game holding its own offset into it. Until now those resolved only from
// a blob in whoever uploaded the file's own browser storage, so the mechanism
// worked on exactly one machine. Giving the record a url is what lets a second
// person open the same recording.
//
// Run after scripts/upload-film.sh:
//   node scripts/register-film.mjs          list what would be written
//   node scripts/register-film.mjs --write  write it
import fs from "fs";
import path from "path";
import { hydrate } from "./env.mjs";

hydrate();
const PB = process.env.POOLEAN_PB_URL || "https://poolean-api.adammirmina.com";
const SITE = process.env.POOLEAN_SITE_URL || "https://poolean.adammirmina.com";
const WRITE = process.argv.includes("--write");

const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
if (!auth.token) { console.error("auth failed against " + PB); process.exit(1); }
const H = { Authorization: auth.token, "Content-Type": "application/json" };

async function api(url, init) {
  const res = await fetch(url, { headers: H, ...init });
  if (!res.ok) {
    // A non-2xx has to read as a failure. Registering nothing and reporting
    // success would leave a reviewer looking for recordings that were never
    // written, with the upload itself looking fine.
    console.error(`${init?.method || "GET"} ${url}: HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
    process.exit(1);
  }
  return res.json();
}

const files = fs.existsSync("out/film")
  ? fs.readdirSync("out/film").filter((f) => f.endsWith(".mp4")).sort()
  : [];
if (!files.length) {
  console.error("Nothing in out/film. Run scripts/make-review-copies.sh, then scripts/upload-film.sh.");
  process.exit(1);
}

// Every recording gets its date, because the file timestamp is the time it was
// filmed rather than the time it was copied.
//
// That was worth establishing rather than assuming, and the durations settle it.
// Read each timestamp as the moment recording ENDED and subtract the length: the
// four from one evening chain back to back, 18:30 to 18:38, 18:37 to 18:46,
// 18:59 to 19:15, with gaps around a minute. Copying multi-gigabyte files does
// not produce a contiguous chain like that, and it would not land every
// recording in the evening.
//
// An earlier version showed a date only when it matched a game date, on the
// theory that these were copy times. That left three recordings looking
// undated when the real story is the opposite: they are dated fine and it is
// the GAMES that are missing, because nobody logged those two sessions.
const games = (await api(`${PB}/api/collections/games/records?perPage=500`)).items || [];
const partyDates = new Set(games.map((g) => String(g.date || "")).filter(Boolean));

const existing = (await api(`${PB}/api/collections/session_videos/records?perPage=500`)).items || [];
const byLabelStem = new Map(existing.map((r) => [String(r.label).split(" · ")[0], r]));

const rows = files.map((file) => {
  const stem = path.basename(file, ".mp4");
  const src = path.join("footage", stem + ".MOV");
  let day = "";
  let unlogged = false;
  if (fs.existsSync(src)) {
    const d = fs.statSync(src).mtime;
    day = d.toDateString().replace(/^(\w{3}) (\w{3}) (\d+).*/, (_, w, m, n) => `${w} ${m} ${Number(n)}`);
    // A recording whose night has no games logged. Said on the label rather than
    // left as a silent mismatch, because it is the one case where picking this
    // recording will not line up with any game in the list.
    unlogged = !partyDates.has(day);
  }
  return {
    stem,
    unlogged,
    label: day ? `${stem} · ${day}${unlogged ? " · no games logged" : ""}` : stem,
    url: `${SITE}/film/${file}`,
    mb: Math.round(fs.statSync(path.join("out/film", file)).size / 1e6),
  };
});

for (const r of rows) {
  const found = byLabelStem.get(r.stem);
  const verb = !found ? "create" : (found.url === r.url && found.label === r.label ? "unchanged" : "update");
  console.log(`  ${verb.padEnd(9)} ${r.label.padEnd(40)} ${String(r.mb).padStart(4)} MB  ${r.url}`);
  if (!WRITE || verb === "unchanged") continue;
  await api(
    found ? `${PB}/api/collections/session_videos/records/${found.id}`
          : `${PB}/api/collections/session_videos/records`,
    { method: found ? "PATCH" : "POST", body: JSON.stringify({ label: r.label, url: r.url }) });
}

const unlogged = rows.filter((r) => r.unlogged);
console.log(`\n${rows.length} recording(s), dated from when each was filmed.`);
if (unlogged.length) {
  console.log(`${unlogged.length} cover a session with no games logged in poolean:`);
  unlogged.forEach((r) => console.log(`  ${r.label}`));
}
if (!WRITE) console.log("Nothing written. Re-run with --write.");
