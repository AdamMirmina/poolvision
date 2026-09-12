// Multipart-upload one recording through the film-handoff Worker, resumably.
//
//   node scripts/handoff/upload.mjs footage/IMG_2932.MOV
//
// Needs HANDOFF_URL and UPLOAD_KEY in the environment. Progress (uploadId and
// every etag so far) is kept in out/handoff/<name>.json, so a killed run picks
// up at the next part rather than from zero. Parts are 95 MB: the free plan
// caps a request body at 100 MB.
import { statSync, existsSync, readFileSync, writeFileSync, mkdirSync, openSync, readSync, closeSync } from "node:fs";
import { basename, join } from "node:path";

const [, , file] = process.argv;
const BASE = process.env.HANDOFF_URL;
const KEY = process.env.UPLOAD_KEY;
if (!file || !BASE || !KEY) {
  console.error("usage: HANDOFF_URL=.. UPLOAD_KEY=.. node upload.mjs <file>");
  process.exit(2);
}

const PART = 95 * 1024 * 1024;
const name = basename(file);
const size = statSync(file).size;
const total = Math.ceil(size / PART);
mkdirSync("out/handoff", { recursive: true });
const stateFile = join("out/handoff", name + ".json");
let st = existsSync(stateFile) ? JSON.parse(readFileSync(stateFile, "utf8")) : null;
const H = { "x-upload-key": KEY };
const q = (op, extra = "") => `${BASE}/mp/${op}?name=${encodeURIComponent(name)}${extra}`;

async function call(url, init) {
  const r = await fetch(url, { ...init, headers: { ...H, ...(init && init.headers ? init.headers : {}) } });
  const t = await r.text();
  if (!r.ok) throw new Error(`${(init && init.method) || "GET"} ${url.replace(BASE, "")} -> ${r.status} ${t.slice(0, 200)}`);
  return JSON.parse(t);
}

// Already there at the right size: nothing to do.
const existing = await call(q("head"));
if (existing.size === size) {
  console.log(`${name}: already uploaded (${size} bytes)`);
  process.exit(0);
}

if (!st || st.size !== size) {
  const { uploadId } = await call(q("create"), { method: "POST" });
  st = { uploadId, size, parts: {} };
  writeFileSync(stateFile, JSON.stringify(st));
  console.log(`${name}: new upload, ${total} parts`);
} else {
  console.log(`${name}: resuming, ${Object.keys(st.parts).length}/${total} parts done`);
}

const fd = openSync(file, "r");
const t0 = Date.now();
let sent = 0;
for (let n = 1; n <= total; n++) {
  if (st.parts[n]) continue;
  const off = (n - 1) * PART;
  const len = Math.min(PART, size - off);
  const buf = Buffer.alloc(len);
  readSync(fd, buf, 0, len, off);
  let last = null;
  for (let attempt = 1; attempt <= 4; attempt++) {
    try {
      const res = await call(q("part", `&uploadId=${st.uploadId}&n=${n}`), { method: "PUT", body: buf });
      st.parts[n] = res.etag;
      writeFileSync(stateFile, JSON.stringify(st));
      last = null;
      break;
    } catch (e) {
      last = e;
      console.error(`  part ${n} attempt ${attempt} failed: ${e.message}`);
      await new Promise((r) => setTimeout(r, 3000 * attempt));
    }
  }
  if (last) {
    closeSync(fd);
    console.error(`${name}: giving up at part ${n}`);
    process.exit(1);
  }
  sent += len;
  const mbps = sent / 1048576 / ((Date.now() - t0) / 1000);
  const doneBytes = Object.keys(st.parts).length * PART;
  console.log(`  ${name} ${n}/${total}  ${(Math.min(doneBytes, size) / 1e9).toFixed(2)}/${(size / 1e9).toFixed(2)} GB  ${mbps.toFixed(1)} MB/s`);
}
closeSync(fd);

const parts = Object.entries(st.parts)
  .map(([k, etag]) => ({ partNumber: Number(k), etag }))
  .sort((a, b) => a.partNumber - b.partNumber);
const done = await call(q("complete", `&uploadId=${st.uploadId}`), {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(parts),
});
// The far side's own size is the verification, not the client's tally.
if (done.size !== size) {
  console.error(`${name}: SIZE MISMATCH server=${done.size} local=${size}`);
  process.exit(1);
}
console.log(`${name}: complete, ${done.size} bytes on R2`);
