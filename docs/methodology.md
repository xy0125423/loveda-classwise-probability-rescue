# Method and interpretation

The network is a torchvision ImageNet ResNet18 encoder with a project-authored compact U-Net decoder: skip concatenation, bilinear upsampling, double Conv3x3/BatchNorm/ReLU, decoder channels 128/64/32/16 and a seven-class head. See model.py. All encoder layers are trainable.

Source selection uses deterministic nonoverlapping 256px tiles. Later frozen inference uses a 256px window, stride128, reflect padding128, and nonperiodic Hann axis weights clamped to 0.001. Fuse raw logits in float32, then compute one softmax. Ties select the smallest index. These two evaluation protocols are intentionally distinct.

For rescue, candidates are original Background pixels with target class as highest nonbackground probability. Probability score is P_c. Ratio is ln((P_c+epsilon)/(P_0+epsilon)), epsilon=1e-8. Historical later ratio implementations use float64 intermediate arithmetic and cast scores to float32; an earlier Water pilot used float32 intermediates. Both modes are explicit in rescue.score; the frozen later branch config uses float64_then_float32. Do not silently exchange modes or retune frozen thresholds.

The source negative pool includes every valid GT!=c pixel, even outside candidates. Use the linear q=.999 quantile. This empirical source quantile does not guarantee target false-positive rate or false-discovery rate. Threshold calibration is decision-workpoint selection; no ECE or reliability-curve evidence establishes probability calibration here.

B2 is 5x5 original-label majority. C4 candidates come from 5x5 mean probability, with original-label support>=.6, probability gap<=.05, 4096-bin mixed-risk cutoff indexed by original predicted class, and strong RGB edge protection. Exact features/gate functions are extracted from the historical implementation; see spatial.py and rescue_thresholds.yaml. A globally missing original class cannot obtain C4 hard-label support.

B=wrong→right, H=right→wrong, W=wrong→another wrong class. Count only changed valid pixels; M=B+H+W and net=B−H. W does not alter the total correct-pixel count but can change class IoU. Zero modifications is not evidence of correctness. Undefined precision is null when a class has no predictions; absent-in-both classes are excluded from macro IoU. Metrics in API/JSON are fractions, README percentages are explicitly labeled.

The two main branches use separate models: Water ratio for Urban→Rural, Building probability for Rural→Urban. Building ratio remains a harmful source control and is refused for target evaluation by the CLI. No multi-class combined rescue claim or new segmentation architecture is made.
