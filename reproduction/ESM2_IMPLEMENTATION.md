# ESM-2 implementation

Source: author `Model_development/ESM_embedding_task-320.ipynb`, cells 3, 5, 6.

- Checkpoint `esm2_t6_8M_UR50D`, 6 layers, ~8M parameters, 320 dimensions.
- Representation layer 6; mean over residues `1:token_len-1`, excluding BOS, EOS and padding.
- No normalization or sequence transformation; both tuple fields contain sequence.
- `model.eval()`, `torch.no_grad()`, CPU, `return_contacts=True`.
- Author uses one batch for the entire input. Primary reconstruction uses batch 32 to bound memory. A separately executed path-only-repaired author notebook also computes all 420 in one batch.
- Notebook recorded pandas 1.5.3, numpy 1.26.4, fair-esm 2.0.0, torch 2.0.0 and Python 3.9.16. This run uses isolated Python 3.12 and locked modern wheels; ESM stays 2.0.0.
- Cleaned CSV has 339 × 320 float64 decimal-read values. `No` retains raw row identity. Regenerated vectors are float32. No nearest-neighbor matching or relabeling is used.

{
  "shape": [
    420,
    320
  ],
  "dtype": "float32",
  "mean_cosine": 0.9999999999992677,
  "median_cosine": 0.9999999999993984,
  "minimum_cosine": 0.9999999999945851,
  "mean_MAE": 3.2831077883273516e-07,
  "maximum_absolute_difference": 3.7847042465219416e-06,
  "checkpoint": "esm2_t6_8M_UR50D",
  "layer": 6,
  "pooling": "residue mean, exclude BOS/EOS/padding",
  "device": "cpu",
  "batch_size": 32,
  "author_batch_size": "all sequences in one batch",
  "normalization": "none",
  "eval": true,
  "no_grad": true,
  "cache_sequence_order_verified": true
}
