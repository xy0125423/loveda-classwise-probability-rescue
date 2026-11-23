# Asset and dependency attribution

MIT applies to project-authored software, documentation and statistical figures, not to external datasets or dependencies. No source files from the official LoveDA repository were copied. Dataset loaders, compact U-Net decoder, training loop and postprocessing were authored within this research project. ResNet18 is instantiated from torchvision, not vendored here. PyTorch and torchvision retain their BSD-style licenses; NumPy, SciPy, Pillow and Matplotlib retain their respective licenses. Installing dependencies does not relicense them.

LoveDA images/annotations are owned by RSIDEA, Wuhan University. Its official repository limits use to academic purposes, prohibits commercial use and refers to Google Earth terms. Obtain data from the official source and follow those terms. The repository excludes RGB/GT images and crops, spatial case figures, manuscript PDFs, weights and probability caches. Statistical figures contain project-authored diagrams and aggregate statistics only; no third-party paper figures are embedded.

Official encoder weights are obtained separately by the user from torchvision's documented source and are not covered by this repository's MIT grant. See the torchvision documentation for weight-specific provenance and conditions. Literature is cited in references.bib, not redistributed as full-text PDFs.
