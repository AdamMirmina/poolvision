// Build an answer key from reviewed game film.
//
// Every keyed window so far was marked by hand for this project alone: someone
// sat with a clock-only tape and wrote down each shot, which is slow and is the
// reason there are five keys rather than fifty. Meanwhile the league's own
// reviewer is already watching the same footage shot by shot for the box score.
//
// That review carries everything a key needs -- when the shot went up, which
// hoop, and whether it went in -- so this turns it into one, and reviewing film
// starts producing evaluation data as a side effect of work already happening.
//
// It is NOT a blind key, and the file says so. A blind pass is worth more,
// because a tape showing no model output cannot bias what gets marked. This is
// worth having anyway: it is free, it is far larger, and it is marked by someone
// with no idea what the detector called.
//
//   node scripts/key-from-review.mjs --game <gameId> --video IMG_2932.MOV
//   node scripts/key-from-review.mjs --list
import fs from "fs";
import path from "path";
import { hydrate } from "./env.mjs";

hydrate();
const PB = process.env.POOLEAN_PB_URL || "https://poolean-api.adammirmina.com";

const arg = (n, d) => {
  const i = process.argv.indexOf("--" + n);
  return i > -1 ? process.argv[i + 1] : d;
};

const auth = await (await fetch(PB + "/api/collections/_superusers/auth-with-password", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ identity: process.env.PB_EM, password: process.env.PB_PW }),
})).json();
if (!auth.token) { console.error("auth failed against " + PB); process.exit(1); }
const H = { Authorization: auth.token };

const get = async (coll, q = "") => {
  const res = await fetch(`${PB}/api/collections/${coll}/records?perPage=500${q}`, { headers: H });
  if (!res.ok) {
    // A non-2xx has to read as a failure, not as an empty result. An empty key
    // silently written from a failed request is worse than no key at all.
    console.error(`${coll}: HTTP ${res.status} ${(await res.text()).slice(0, 160)}`);
    process.exit(1);
  }
  return (await res.json()).items || [];
};

const games = await get("games");
const shots = await get("shot_events");

if (process.argv.includes("--list")) {
  const byGame = {};
  shots.forEach((s) => { byGame[s.game] = (byGame[s.game] || 0) + 1; });
  const rows = games
    .map((g) => ({ id: g.id, date: g.date, n: byGame[g.id] || 0, video: g.videoUrl || "" }))
    .filter((r) => r.n > 0)
    .sort((a, b) => b.n - a.n);
  if (!rows.length) {
    console.log("No reviewed games yet: no shot_events rows carry a game.");
    process.exit(0);
  }
  console.log("reviewed games, most shots first:");
  rows.forEach((r) => console.log(`  ${r.id}  ${String(r.date).padEnd(14)} ${String(r.n).padStart(3)} shots  ${r.video}`));
  process.exit(0);
}

const gameId = arg("game");
const video = arg("video");
if (!gameId || !video) {
  console.error("need --game <gameId> and --video <FILE.MOV>. Use --list to see what has been reviewed.");
  process.exit(1);
}

const game = games.find((g) => g.id === gameId);
if (!game) { console.error(`no game ${gameId}`); process.exit(1); }

const mine = shots.filter((s) => s.game === gameId);
if (!mine.length) { console.error(`game ${gameId} has no reviewed shots`); process.exit(1); }

// A shot with no timestamp cannot be scored against a call, because there is
// nothing to line it up with. Counted and reported rather than dropped quietly:
// a key that is short by a third with no explanation is how a recall number
// silently becomes wrong.
const timed = mine.filter((s) => typeof s.at === "number" && s.at > 0);
const untimed = mine.length - timed.length;

// The tracker's timestamps live in the SOURCE recording's own timeline, which
// for a session video covering several games is the shared master. videoStart
// is this game's offset into that recording, in the same space, so a key for one
// video file needs the raw time, not a time relative to the game.
const shotsOut = timed
  .sort((a, b) => a.at - b.at)
  .map((s) => ({
    t: Math.round(s.at * 100) / 100,
    at: `${Math.floor(s.at / 60)}:${String(Math.floor(s.at % 60)).padStart(2, "0")}`,
    // Which hoop is not recorded by the review: it tracks who shot, not where.
    // Left absent rather than guessed, and the scorer treats a missing hoop as
    // "either", so a real call at the right moment still matches.
    outcome: s.made === false ? "miss" : "make",
    points: s.points,
    shooter_cap: "",
    shooter_slug: s.shooter,
  }));

const out = {
  video,
  game: gameId,
  date: game.date || "",
  shots: shotsOut,
  note_on_provenance:
    "Derived from reviewed game film, not marked blind for this project. The " +
    "reviewer logs every shot for the box score without seeing any model " +
    "output, so it is not biased by the detector; but it was marked to answer " +
    "a different question, so treat a disagreement as worth checking rather " +
    "than as a definite miss." +
    (untimed ? ` ${untimed} of ${mine.length} reviewed shots carry no video timestamp and are NOT in this key.` : ""),
};

const dest = path.resolve(`labels/answerkey_${path.basename(video, path.extname(video))}_review.json`);
fs.writeFileSync(dest, JSON.stringify(out, null, 1));
console.log(`${shotsOut.length} shots -> ${path.relative(process.cwd(), dest)}`);
if (untimed) {
  console.log(`  ${untimed} reviewed shot(s) had no timestamp and were left out.`);
  console.log("  Those were logged before the tracker stamped playback time, or with no video loaded.");
}
const makes = shotsOut.filter((s) => s.outcome === "make").length;
console.log(`  ${makes} makes, ${shotsOut.length - makes} misses`);
