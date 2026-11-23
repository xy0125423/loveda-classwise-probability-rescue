# Dataset layout and access

Download LoveDA yourself: https://github.com/Junjue-Wang/LoveDA · https://doi.org/10.5281/zenodo.5706578 · https://rsidea.whu.edu.cn/LoveDA.htm . The official repository states academic use only, commercial use prohibited, and Google Earth imagery terms apply. This repository does not redistribute pixels or automatically download data.

```text
data/LoveDA/
  Train/Urban/{images_png,masks_png}/*.png
  Train/Rural/{images_png,masks_png}/*.png
  Val/Urban/{images_png,masks_png}/*.png
  Val/Rural/{images_png,masks_png}/*.png
```

The historical loader also accepts repeated Train/Train or Val/Val directories and read-only Train.zip/Val.zip under the given data root. Every image must have a same-name mask. Raw GT=0 maps to ignore=255. Raw 1..7 map to evaluated model classes 0..6: Background, Building, Road, Water, Barren, Forest, Agriculture. Background is not ignored. Test is not used by these commands.

Pass `--data-root data/LoveDA`; paths in commands are examples relative to your checkout. Name-only frozen source splits are in configs/splits.json. Water target confirmation uses configs/water_confirmation.json (981 names, excluding 11 pilot images); Building target uses the full Urban Val set (677). No image content is included in either manifest.
