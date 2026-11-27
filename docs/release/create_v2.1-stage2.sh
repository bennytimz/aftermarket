#!/usr/bin/env bash
# Create the GitHub release v2.1-stage2 with the resubmitted paper PDF attached (v2.0-stage2 is left as it is):
#   bash docs/release/create_v2.1-stage2.sh [COMMIT] [--dry-run]     (COMMIT defaults to HEAD)
# Refuses unless COMMIT is on origin/main, the tag does not exist yet, tracked files are clean, and the PDF in the
# working tree is the one committed at COMMIT. The PDF is attached from COMMIT, not from the working tree; the notes
# are read from origin/main and must name COMMIT and the PDF's sha256.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REPO=bennytimz/aftermarket
TAG=v2.1-stage2
PDF=docs/aftermarket_stage2.pdf
NOTES=docs/release/v2.1-stage2.md
DRY=0
SHA=HEAD
for arg in "$@"; do
  case "$arg" in --dry-run) DRY=1 ;; *) SHA=$arg ;; esac
done
SHA=$(git rev-parse --verify "$SHA^{commit}")

die() { echo "error: $*" >&2; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked files have uncommitted changes"
git fetch -q origin main --tags
git merge-base --is-ancestor "$SHA" origin/main || die "$SHA is not on origin/main (push first)"
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || git ls-remote --exit-code --tags origin "$TAG" >/dev/null; then
  die "tag $TAG already exists"
fi
git cat-file -e "$SHA:$PDF" 2>/dev/null || die "$PDF is not committed at $SHA"
# The notes come from origin/main (they may be finished after the submitted commit, which the tag still points at);
# they must state that commit and the PDF's sha256, so the release text cannot disagree with what it tags.
git cat-file -e "origin/main:$NOTES" 2>/dev/null || die "$NOTES is not on origin/main (push it first)"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git show "$SHA:$PDF" > "$TMP/aftermarket_stage2.pdf"
cmp -s "$TMP/aftermarket_stage2.pdf" "$PDF" || die "working-tree $PDF differs from the one committed at $SHA"
HASH=$(sha256sum "$TMP/aftermarket_stage2.pdf" | cut -d' ' -f1)
git show "origin/main:$NOTES" > "$TMP/notes.md"
grep -q "$SHA" "$TMP/notes.md" || die "$NOTES does not state the tagged commit $SHA"
grep -q "$HASH" "$TMP/notes.md" || die "$NOTES does not state the PDF sha256 $HASH"
echo "tag $TAG -> $SHA; PDF sha256 $HASH"

CMD=(gh release create "$TAG" "$TMP/aftermarket_stage2.pdf" --repo "$REPO" --target "$SHA"
     --title "v2.1-stage2: Stage 2 resubmission" --notes-file "$TMP/notes.md")
if [ "$DRY" = 1 ]; then
  echo "dry run: would run:"; printf ' %q' "${CMD[@]}"; echo; echo "--- notes ---"; cat "$TMP/notes.md"
else
  "${CMD[@]}"
fi
