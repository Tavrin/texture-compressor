#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
private='mo''ss'
user='et''ienne'
planning='rf''c'
pattern="/home/[^/[:space:]]+/|/mnt/|${user}|\\b${private}\\b|\\blane\\b|\\bapollo\\b|\\b[[:alnum:]_]+-([a-z0-9]{4}[0-9]|[a-z0-9]{3}[0-9][a-z0-9]|[a-z0-9]{2}[0-9][a-z0-9]{2}|[a-z0-9][0-9][a-z0-9]{3}|[0-9][a-z0-9]{4})\\b|${planning}[[:space:]]*[0-9]+|S1[[:space:]]+oracle|phase[[:space:]]+0\\.[0-9]+"
set +e
git grep --untracked -niIE "$pattern" -- . ':!tools/check_public_text.sh'
status=$?
set -e
case "$status" in
  0) echo 'FAIL: private text found' >&2; exit 1 ;;
  1) echo 'PASS: public text guard' ;;
  *) exit "$status" ;;
esac
