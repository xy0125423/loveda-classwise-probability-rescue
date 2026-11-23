# Release checks

Checked on 2026-11-23 before publication. The public tree is a new, isolated release directory.

## Executable checks

- Imported every `loveda_rescue` module, parsed all 22 Python files, and invoked `--help` on all six command entry points.
- Ran all 6 unit tests successfully.
- Ran the synthetic no-data demo and confirmed `B=2, H=2, W=2, M=6, net=0`.
- In a copied release tree, ran the calibration, evaluation, and five-fold CLI paths on 10 generated probability/label records. These are interface checks only, not LoveDA results.
- Regenerated all 5 statistical figures from packaged aggregate statistics and visually checked titles, labels, legends, and value annotations. No input imagery or spatial case images are included.

## Content and preservation checks

- Compared 20,070 historical file size and modification-time records captured before release assembly with the current G1 history; all matched. The seven code-source relationships are documented with source and release SHA-256 values in [`source_provenance.json`](source_provenance.json).
- Scanned the release for model weights, NumPy probability/logit caches, raster data, QGIS projects, archives, and PDFs; none were present. The five PNGs are aggregate-statistics figures.
- Scanned text/configuration files for Windows/POSIX home paths and common GitHub/cloud credential patterns; no matches were found. Local Markdown links resolve.
- Confirmed English and Chinese result tables carry the same two transfer rows, metrics, recalls, and net pixel counts. Attribution contains one author, Mengle Xue, and the author-provided contact address only.
- [`file_manifest.csv`](file_manifest.csv) lists the final release files with byte sizes and SHA-256 digests; it excludes itself to avoid a recursive hash.

## Scope boundary

No LoveDA training, model inference, threshold changes, case reselection, or spatial-case interpretation was performed for this release. The README distinguishes executable full-data reproduction steps from the statistical-figure redraw. The published figures use saved aggregate records and do not substitute for model reproduction.
