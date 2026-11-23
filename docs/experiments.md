# Experiment protocol and scope

| Direction | Train | Source model selection | Source calibration | Historical target |
|---|---:|---:|---:|---:|
| Urban→Rural | 1156 | 338 | 339 | Rural992; confirmation981 |
| Rural→Urban | 1366 | 496 | 496 | Urban677 |

Both models trained for20 epochs; source-selected checkpoints were epochs18 and20. Seeds are20261114/20261115, source split seeds20261124/20261125. AdamW lr1e-4, decay.01, crop256, batch2; ReduceLROnPlateau factor.5/patience2/absolute threshold0/min_lr1e-6; early stopping after5 consecutive non-improving epochs. See configs for full frozen values. Source selection and later sliding-window metrics are not interchangeable.

The packaged training loop calls the original train_epoch and validate functions. It omits internal monitoring/report generation and packages checkpoints under portable filenames. A new environment/run has not been tested for full training trajectory equality; it is an executable reproduction route, not an already completed rerun.

Development was retrospective: Rural992 was diagnosed before the11-pilot exclusion; Urban677 had earlier exploration; source five-fold analysis happened after fixed Building target evaluation. Cross-direction data reuse means source-selection separation within one model is not project-wide blindness. Source image folds (seed20261122) and image bootstrap do not block spatial dependence. No new geographic validation or official leaderboard result is claimed. External-data experiments remain out of scope.

Historical Water/Building fixed-target paired image bootstraps used10,000/2,000 resamples. Existing intervals are redrawn, not recomputed during this release. The source OOF rows are supplied in results/source_cv.csv; source calibration ablation in results/ablation.csv. Full precision and recall remain visible alongside gains; negative source control and harmed image evidence are retained in the statistical figures.

Release validation environment: Python3.11, torch2.15.0+cpu, torchvision0.29.2+cpu, numpy2.5.4, scipy1.18.1, matplotlib3.11.2, Pillow13.0.0. These are the installed packaging-test versions, not a claim about the exact original training environment. requirements.txt lists installable ranges through pyproject.toml. On the Windows packaging host, Matplotlib required the active conda environment's library directories to be available to the process. Activate the environment normally before running the redraw script; no host-specific absolute paths are embedded in the repository.

Main limitations: one architecture and one trained model per direction; limited training; historical target influence; different score workpoints and coverage; low rescued-class recall. A stronger baseline, other seeds or a new region could change the observed score ordering.
