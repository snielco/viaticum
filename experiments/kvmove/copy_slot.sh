#!/bin/bash
# Copy a slot file between two llama.cpp hosts, relayed through a third host that can SSH to both
# (the two GPU hosts need no SSH trust between them).
# Usage: copy_slot.sh strix|observer strix|observer FILE   -> prints copy seconds and both sha256 (direct reads).
# Env: RELAY (ssh target of the relay host), STRIX_SSH, OBSERVER_SSH (ssh targets as seen from the relay).
set -euo pipefail
: "${RELAY:?set RELAY}" "${STRIX_SSH:?set STRIX_SSH}" "${OBSERVER_SSH:?set OBSERVER_SSH}"
declare -A H=([strix]=$STRIX_SSH [observer]=$OBSERVER_SSH)
declare -A D=([strix]=llamacpp-vk/slot-cache [observer]=llm/slot-cache)
src=$1 dst=$2 f=$3
t0=$(date +%s.%N)
ssh "$RELAY" "ssh ${H[$src]} 'cat ${D[$src]}/$f' | ssh ${H[$dst]} 'cat > ${D[$dst]}/$f.part && mv ${D[$dst]}/$f.part ${D[$dst]}/$f'"
t1=$(date +%s.%N)
size=$(ssh ${H[$dst]} "stat -c %s ${D[$dst]}/$f")
s1=$(ssh ${H[$src]} "dd if=${D[$src]}/$f iflag=direct bs=16M status=none | sha256sum | cut -c1-64")
s2=$(ssh ${H[$dst]} "sync; dd if=${D[$dst]}/$f iflag=direct bs=16M status=none | sha256sum | cut -c1-64")
python3 -c "import sys; b,t=int(sys.argv[1]),float(sys.argv[2]); print('copy %s -> %s %s: %d bytes in %.1f s = %.0f MB/s' % (sys.argv[3],sys.argv[4],sys.argv[5],b,t,b/t/1e6))" "$size" "$(echo "$t1 - $t0" | bc)" $src $dst $f
echo "sha256 $src $s1"; echo "sha256 $dst $s2"; [ "$s1" = "$s2" ] && echo MATCH || { echo MISMATCH; exit 1; }
