// Read credentials from .env so they never have to be typed, pasted into a chat,
// or left in shell history.
//
// review asked whether pasting them in chat would be acceptable. It would not: the
// conversation is written to a transcript that persists, and these are live
// credentials for real data. A gitignored file gives the same convenience with
// none of that, and matches how cloud containers already get them.
import fs from "fs";
import path from "path";

export function loadEnv(dir = process.cwd()) {
  const f = path.join(dir, ".env");
  if (!fs.existsSync(f)) return {};
  const out = {};
  const text = fs.readFileSync(f, "utf8");
  for (const raw of text.split("\n")) {
    const t = raw.trim();
    if (!t || t.startsWith("#")) continue;
    const i = t.indexOf("=");
    if (i < 0) continue;
    let v = t.slice(i + 1).trim();
    const quoted = (v.startsWith(String.fromCharCode(34)) && v.endsWith(String.fromCharCode(34)))
      || (v.startsWith(String.fromCharCode(39)) && v.endsWith(String.fromCharCode(39)));
    if (quoted) v = v.slice(1, -1);
    out[t.slice(0, i).trim()] = v;
  }
  return out;
}

// Put .env into process.env without disturbing anything already set there, so a
// script that reads process.env directly works with no edit beyond importing
// this. The environment still wins, which is what keeps CI and containers
// behaving as before.
export function hydrate(dir = process.cwd()) {
  const e = loadEnv(dir);
  for (const [k, v] of Object.entries(e)) {
    if (v && !process.env[k]) process.env[k] = v;
  }
  return e;
}

// Environment first, since containers and CI set it there, then .env.
export function creds() {
  const e = loadEnv();
  const EM = process.env.PB_EM || e.PB_EM;
  const PW = process.env.PB_PW || e.PB_PW;
  const URL = process.env.PB_URL || e.PB_URL || "https://poolean-api.adammirmina.com";
  if (!EM || !PW) {
    console.error("No PocketBase credentials found.");
    console.error("Copy .env.example to .env and fill it in, or set PB_EM and PB_PW");
    console.error("in the environment. See docs/CREDENTIALS.md.");
    process.exit(1);
  }
  return { EM, PW, URL };
}
