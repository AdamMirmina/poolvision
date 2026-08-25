#!/usr/bin/env bash
# Push the review copies to R2 and print the URLs they are served from.
#
# The bucket has no public access. Every object is reachable only through
# poolean's /film/ route, which requires an admin session, because the footage
# is of identifiable people and an unlisted URL is not a permission.
#
# Run after scripts/make-review-copies.sh finishes:
#   bash scripts/upload-film.sh
set -u
ACC=7add713aa8b1453972d1181241b4f88c
BUCKET=poolean-film
WEB=/c/dev/poolean/web

shopt -s nullglob
files=(out/film/*.mp4)
if [ ${#files[@]} -eq 0 ]; then
  echo "Nothing in out/film. Run scripts/make-review-copies.sh first." >&2
  exit 1
fi

for f in "${files[@]}"; do
  name=$(basename "$f")
  mb=$(du -m "$f" | cut -f1)
  echo "uploading $name (${mb} MB) ..."
  # --remote, or wrangler writes to the LOCAL simulated bucket and reports
  # success having uploaded nothing anyone can reach.
  ( cd "$WEB" && CLOUDFLARE_ACCOUNT_ID=$ACC npx wrangler r2 object put \
      "$BUCKET/$name" --file "$(cd - >/dev/null && pwd)/$f" \
      --content-type video/mp4 --remote >/dev/null 2>&1 ) \
    && echo "  https://poolean.adammirmina.com/film/$name" \
    || echo "  $name FAILED"
done

echo
echo "Total in bucket:"
( cd "$WEB" && CLOUDFLARE_ACCOUNT_ID=$ACC npx wrangler r2 bucket info "$BUCKET" --remote 2>/dev/null | grep -iE "size|objects" )
