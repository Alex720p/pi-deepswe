#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
reward=0
if git -c safe.directory='*' apply /logs/artifacts/model.patch \
    && [ "$(cat answer.txt)" = solved ] \
    && [ "$(cat new.txt)" = new ] \
    && ! timeout 2 bash -c '</dev/tcp/1.1.1.1/443' 2>/dev/null; then
    reward=1
fi
printf '{"reward":%s,"reward_partial":%s}\n' "$reward" "$reward" > /logs/verifier/reward.json
printf '{"results":{"tool":{"name":"pi-fixture"},"summary":{"tests":1,"passed":%s,"failed":%s,"pending":0,"skipped":0,"other":0},"tests":[]}}\n' "$reward" "$((1-reward))" > /logs/verifier/ctrf.json
