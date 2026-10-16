# Problem 5: new-source images

This folder is populated by `uv run python src/collect_new_images.py`, which
searches Wikimedia Commons, downloads files, and records source URLs and license
metadata in `data/new_images/sources.csv`.

Target labels:
- `cartoon_animal/`: cartoon/illustrated animal images
- `plush_animal/`: animal-shaped plush/stuffed toys
- `real_animal/`: photographs of real animals

Important:
1. The script is a starting point, not a guarantee that every search result is
   correctly labelled. Open and inspect each downloaded file; replace mistakes.
2. Each Commons file has its own attribution and license requirements. Keep
   `sources.csv` with the submitted project.
3. These images are from an external source, but this does NOT prove they do not
   overlap with the downloaded dataset. Run `uv run python src/clean.py overlap`
   and inspect `results/cleaning/overlap.png` before claiming they are distinct.
4. The assignment requires at least five usable images per class. If the script
   cannot find five, add more manually from a documented source.
