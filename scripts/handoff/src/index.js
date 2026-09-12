// Two jobs, both scoped to the raw/ prefix of one bucket.
//
// Upload: the REST API caps a single object PUT at about 300 MiB, so multi-GB
// recordings go up as R2 multipart uploads driven through this Worker, one part
// per request, under the free plan's 100 MB request-body ceiling.
//
// Download: GET /f/<name>?k=<token>, with Range support so a 20 GB pull can be
// resumed rather than restarted. R2 egress is free, so the only cost of the
// whole arrangement is a few days of storage.

const PREFIX = "raw/";

const json = (o, status = 200) =>
  new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json" } });

const timingSafeEq = (a, b) => {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
};

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname.startsWith("/f/")) return download(req, env, url);
    if (url.pathname.startsWith("/mp/")) return upload(req, env, url);
    return new Response("not found", { status: 404 });
  },
};

async function download(req, env, url) {
  if (!timingSafeEq(url.searchParams.get("k") || "", env.DL_KEY || "")) return new Response("forbidden", { status: 403 });
  const name = decodeURIComponent(url.pathname.slice(3));
  if (!/^[A-Za-z0-9_.-]+$/.test(name)) return new Response("bad name", { status: 400 });
  const key = PREFIX + name;

  const head = await env.FILM.head(key);
  if (!head) return new Response("no such file", { status: 404 });
  const size = head.size;

  const common = {
    "accept-ranges": "bytes",
    "content-type": "video/quicktime",
    "content-disposition": `attachment; filename="${name}"`,
    "cache-control": "private, no-store",
    etag: head.httpEtag,
  };
  if (req.method === "HEAD") return new Response(null, { headers: { ...common, "content-length": String(size) } });

  const rangeH = req.headers.get("range");
  if (rangeH) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(rangeH.trim());
    if (!m) return new Response("bad range", { status: 416, headers: { "content-range": `bytes */${size}` } });
    let start = m[1] === "" ? null : Number(m[1]);
    let end = m[2] === "" ? null : Number(m[2]);
    if (start === null) { start = Math.max(0, size - end); end = size - 1; }
    else if (end === null || end >= size) end = size - 1;
    if (start > end || start >= size) return new Response("bad range", { status: 416, headers: { "content-range": `bytes */${size}` } });
    const obj = await env.FILM.get(key, { range: { offset: start, length: end - start + 1 } });
    if (!obj) return new Response("no such file", { status: 404 });
    return new Response(obj.body, {
      status: 206,
      headers: { ...common, "content-length": String(end - start + 1), "content-range": `bytes ${start}-${end}/${size}` },
    });
  }
  const obj = await env.FILM.get(key);
  if (!obj) return new Response("no such file", { status: 404 });
  return new Response(obj.body, { headers: { ...common, "content-length": String(size) } });
}

async function upload(req, env, url) {
  if (!timingSafeEq(req.headers.get("x-upload-key") || "", env.UPLOAD_KEY || "")) return new Response("forbidden", { status: 403 });
  const name = url.searchParams.get("name") || "";
  if (!/^[A-Za-z0-9_.-]+$/.test(name)) return json({ error: "bad name" }, 400);
  const key = PREFIX + name;
  const op = url.pathname.slice(4);

  if (op === "create" && req.method === "POST") {
    const mp = await env.FILM.createMultipartUpload(key, { httpMetadata: { contentType: "video/quicktime" } });
    return json({ uploadId: mp.uploadId });
  }
  if (op === "part" && req.method === "PUT") {
    const uploadId = url.searchParams.get("uploadId");
    const n = Number(url.searchParams.get("n"));
    if (!uploadId || !Number.isInteger(n) || n < 1) return json({ error: "bad part" }, 400);
    const mp = env.FILM.resumeMultipartUpload(key, uploadId);
    const part = await mp.uploadPart(n, req.body);
    return json({ partNumber: part.partNumber, etag: part.etag });
  }
  if (op === "complete" && req.method === "POST") {
    const uploadId = url.searchParams.get("uploadId");
    const parts = await req.json();
    const mp = env.FILM.resumeMultipartUpload(key, uploadId);
    const obj = await mp.complete(parts);
    return json({ key: obj.key, size: obj.size, etag: obj.httpEtag });
  }
  if (op === "abort" && req.method === "POST") {
    const uploadId = url.searchParams.get("uploadId");
    await env.FILM.resumeMultipartUpload(key, uploadId).abort();
    return json({ aborted: true });
  }
  if (op === "head" && req.method === "GET") {
    const h = await env.FILM.head(key);
    return json(h ? { size: h.size, etag: h.httpEtag } : { size: null });
  }
  return json({ error: "no such op" }, 404);
}
