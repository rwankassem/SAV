#!/usr/bin/env bash
cd /c/Users/rwank/OneDrive/Desktop/SAV
GIT_DIR=/c/Users/rwank/OneDrive/Desktop/SAV/.git git filter-branch --env-filter 'source /c/Users/rwank/OneDrive/Desktop/SAV/rewrite-author.sh' -- --all
if [ $? -ne 0 ]; then
  echo FILTER_BRANCH_FAILED
  exit 1
fi
 echo FILTER_DONE
git shortlog -sne main | grep -E 'mosayyyed1|rwankassem68@gmail.com'
git push --force-with-lease origin main
