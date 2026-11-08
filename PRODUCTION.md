# Qualified production build

- Runtime source commit: `09c2b6cd504ba7909524cd5e6b2acca533d606f1`
- Runtime source tree: `5dcc325acd9b6b60956edfb285b5ab17e4d89f83`
- Image: `tensorfold-dsv41:graph-fallback-09c2b6cd5`
- Image source label: `tf.source-ref=09c2b6cd504ba7909524cd5e6b2acca533d606f1`
- Checkpoint revision: `02d2ab346129098f9b3546143bb69944d88c7f977218627b916128cb9c1edfc9`
- Required switch: `TF_DS_PREFILL_YIELD=1`

Verified identical on all four production ranks:

- `tensorfold-native`: `dccf5d8dd9f772a28d46097f18c66a0938f393ddda467eac2c72208e015ea684`
- `tf-dsv41-lanes`: `c50416083939ef3c380ea9ab00f397105ca7a6b4b0fa491dd67659c9b951768c`
- `tf-dsv41-rank-cache`: `8055e5ade1c206f477191fa7840c60e9766def5be2f6511c02e9f427f04c343e`

## CUDA-graph recovery

Both DSv4.1 backbone-round and drafter-pass graph misses now execute eagerly before attempting capture. If any rank fails capture, every rank keeps the already-computed eager result and postpones future capture attempts for 64 misses. Repeated failures double the delay up to 4096 misses. A successful capture resets the backoff.

This is a DSv4.1 adaptation of MiaAI-Lab's eager-result/backoff design from `Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` release 1.0.0. See `PROVENANCE.md`; it is not claimed as an original design from this repository.

## Qualification, 2026-11-09

- Forced rank-3 backbone-round capture failure: HTTP 200, eager result retained, all ranks agreed, exact 768-token oracle hash preserved.
- Forced rank-3 drafter-pass capture failure: same result and exact oracle preservation.
- Clean production: three of three 768-token oracle runs matched `9bea36391f3d8a34a439c938d0738f364cdd984dbc0636caeab54433bfe1108c`.
- Fresh 18,958-token prompt: HTTP 200, correct retrieval result, no cached prompt tokens.
- Six-image regression during a 4096-token coding stream: all five vision assertions passed; no decode gap exceeded 1.104 seconds.
- All four containers: running, zero restarts, no OOM; `/health` green.
- Host suite: 251 passed, one skipped, and two unrelated/environmental failures (overlay-filesystem `O_DIRECT` and the pre-existing stochastic lane-probability test).

The DGX02 launcher pins this image with `PREPARE=0`. The immediate rollback image is `tensorfold-dsv41:image-yield-reviewed-c1592dc1`; the older baseline rollback is `tensorfold-dsv41:native-6b3711e64978`.
