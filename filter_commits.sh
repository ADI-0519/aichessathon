#!/bin/bash
git fetch
git checkout yaseen
git reset --hard origin/yaseen
git filter-branch -f --msg-filter 'sed -e "/Co-Authored-By:.*[Cc]laude/d" -e "/Co-Authored-By:.*Claude/d"' HEAD~20..HEAD
