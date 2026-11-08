# Production provenance

This branch is a production snapshot, not a claim that its complete TensorFold engine was authored here.

The imported baseline at `7950dd7422b9552a6052a7e3c58af40cf92f29ea` materializes the four-DGX-Spark engine built from:

- TensorFold and its contributors;
- BertholomusAI's DeepSeek-V4.1 TP4 engine at `59402086381a42c74d53053443224eec2790b4c5`; and
- the ordered MiaAI-Lab/community patch stack from `DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold`.

The recipe's `NOTICE` and `CREDITS.md` are the authoritative detailed attribution for that imported stack. It includes work by MiaAI-Lab, Hector Isaza, Mauvis Ledford, Thomas Kempka, knapcio, Hironori Uchikawa, Manuel, SvangenStudios, shalk3, BobClawblaw, Ash Hart, BertholomusAI and the other contributors named there. DeepSeek supplies the model arithmetic and weights; the weights are not included in this repository.

The production-specific work added in this repository is:

- `245be183036c27a3b64c1dd630453e24e77b5527`: per-image vision yielding, implemented by Codex from the operator's requested design on top of the existing prefill-yield scheduler;
- `c1592dc1fb97776057e4f68852cadce346596808`: lifecycle, validation and RDMA-wake hardening implemented by Codex after a read-only Claude Code Opus review.

No upstream pull request is intended from this flattened production branch.
