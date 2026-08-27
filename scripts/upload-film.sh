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
SRC=${SRC:-out/film}
files=($SRC/*.mp4)
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
  #
  # The repo's own wrangler rather than npx. npx re-resolves the package on
  # every run and will reinstall it into a shared cache; when that cache is
  # locked by another wrangler process the install dies with EBUSY and takes the
  # upload down with it. That happened to all ten files at once.
  #
  # And the error is shown rather than sent to /dev/null. It was hidden, so ten
  # uploads failed for one reason and this printed FAILED ten times without ever
  # saying why. Stops on the first failure instead of carrying on: a bucket half
  # updated is worse than one not updated at all, because nothing afterwards
  # says which half.
  abs=$(cd "$(dirname "$f")" && pwd)/$(basename "$f")
  if ( cd "$WEB" && CLOUDFLARE_ACCOUNT_ID=$ACC ./node_modules/.bin/wrangler r2 object put \
       "$BUCKET/$name" --file "$abs" --content-type video/mp4 --remote 2>&1 | tail -2 ); then
    echo "  https://poolean.adammirmina.com/film/$name"
  else
    echo "  $name FAILED, stopping"; exit 1
  fi
done

echo
echo "Total in bucket:"
( cd "$WEB" && CLOUDFLARE_ACCOUNT_ID=$ACC npx wrangler r2 bucket info "$BUCKET" --remote 2>/dev/null | grep -iE "size|objects" )
