#!/bin/bash
# Wait for seed 137 to finish (already running as PID passed as $1)
wait $1 2>/dev/null

# Run seed 256
echo "=== Starting seed 256 ==="
cd /Users/renzobalcazar/Desktop/experiment/sym-jepa
python3 -u -m symbolic.train --size large --seed 256 --max-epochs 20 > logs/seed256.txt 2>&1
echo "=== Seed 256 complete ==="
echo "All seeds done."
