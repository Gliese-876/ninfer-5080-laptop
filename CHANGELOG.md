# Changes to [toddballinger/ninfer-5080 v1.5](https://github.com/toddballinger/ninfer-5080/releases/tag/qwen3.8-27b-rtx5080-128k-vision-v1.5)

This fork starts at `e4edd6d5c5f9f7996de0f3d9f6c311e883452580`.
The inference implementation remains NInfer; the repository is named `ninfer-5080-laptop`.

- Reuse the official indexed draft-head weights without a duplicate allocation.
- Optimize qualified Q3/Q4/Q5 CUDA paths, long Q4 attention and packed transfers.
- Share the qualified T4/T5 paths across MTP-3 and MTP-4.
- Cache exact tokenization and repair the temporary hidden-state buffer used during exact-hit MTP replay.
- Publish portable tooling, bilingual documentation and reproducible benchmark data.

The modified source and test files carry a change notice. The Apache-2.0 `LICENSE` is retained.
Runtime numerical formats, error criteria and the official model bytes remain the basis of validation.
