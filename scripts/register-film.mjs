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

// A date goes on the label only when it matches a real party exactly.
//
// The only date available for a recording is its own file timestamp, which is
// when it was copied off the phone rather than when it was filmed. Checked
// against the actual game dates, most line up and some do not, and a label
// asserting the wrong night is worse than one asserting nothing: the recording
// is still identifiable by name, and a wrong date sends someone looking for the
// wrong game.
const games = (await api(`${PB}/api/collections/games/records?perPage=500`)).items || [];
const partyDates = new Set(games.map((g) => String(g.date || "")).filter(Boolean));

const existing = (await api(`${PB}/api/collections/session_videos/records?perPage=500`)).items || [];
const byLabelStem = new Map(existing.map((r) => [String(r.label).split(" · ")[0], r]));

const rows = files.map((file) => {
  const stem = path.basename(file, ".mp4");
  const src = path.join("footage", stem + ".MOV");
  let day = "";
  if (fs.existsSync(src)) {
    const d = fs.statSync(src).mtime;
    const guess = d.toDateString().replace(/^(\w{3}) (\w{3}) (\d+).*/, (_, w, m, n) => `${w} ${m} ${Number(n)}`);
    if (partyDates.has(guess)) day = guess;
  }
  return {
    stem,
    label: day ? `${stem} · ${day}` : stem,
    url: `${SITE}/film/${file}`,
    mb: Math.round(fs.statSync(path.join("out/film", file)).size / 1e6),
  };
});

for (const r of rows) {
  const found = byLabelStem.get(r.stem);
  const verb = !found ? "create" : (found.url === r.url && found.label === r.label ? "unchanged" : "update");
  console.log(`  ${verb.padEnd(9)} ${r.label.padEnd(28)} ${String(r.mb).padStart(4)} MB  ${r.url}`);
  if (!WRITE || verb === "unchanged") continue;
  await api(
    found ? `${PB}/api/collections/session_videos/records/${found.id}`
          : `${PB}/api/collections/session_videos/records`,
    { method: found ? "PATCH" : "POST", body: JSON.stringify({ label: r.label, url: r.url }) });
}

const dated = rows.filter((r) => r.label.includes(" · ")).length;
console.log(`\n${rows.length} recording(s), ${dated} matched to a party date by the file's own timestamp.`);
if (!WRITE) console.log("Nothing written. Re-run with --write.");
