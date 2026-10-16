"""Collect a small, documented external test set from Wikimedia Commons.

Run from the repository root:
    uv run python src/collect_new_images.py

Downloads at least five images per class when enough suitable results are
available. Read the generated source manifest and visually check every image.
This script records the source page and license metadata; it does NOT prove
that an image is absent from the training set. Run the overlap checker after.
"""
import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "new_images"
MANIFEST = OUT / "sources.csv"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "MAS1004-assignment1-student-project/1.0 (educational use)"}
TARGETS = {
    "cartoon_animal": [
        "animal cartoon drawing illustration filetype:bitmap",
        "cartoon animal illustration drawing",
    ],
    "plush_animal": [
        "plush animal toy photograph",
        "stuffed animal toy photograph",
    ],
    "real_animal": [
        "wild animal photograph",
        "real animal photograph wildlife",
    ],
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ROWS = []


def query(search):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": "filetype:bitmap " + search, "gsrnamespace": 6,
        "gsrlimit": 30, "prop": "imageinfo", "iiprop": "url|extmetadata",
        "iiurlwidth": 1000,
    }
    data = requests.get(API, params=params, headers=HEADERS, timeout=30)
    data.raise_for_status()
    return list(data.json().get("query", {}).get("pages", {}).values())


def metadata(page):
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {})
    def value(key):
        return re.sub("<[^>]+>", "", meta.get(key, {}).get("value", "")).strip()
    return info, value("Artist"), value("LicenseShortName"), value("LicenseUrl"), value("ImageDescription")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for label, searches in TARGETS.items():
        folder = OUT / label
        folder.mkdir(parents=True, exist_ok=True)
        existing = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
        if len(existing) >= 5:
            print(f"{label}: already has {len(existing)} images; leaving them untouched")
            continue
        seen = set()
        for search in searches:
            try:
                pages = query(search)
            except Exception as exc:
                print(f"Search failed for {label}: {exc}")
                continue
            for page in pages:
                title = page.get("title", "")
                info, artist, license_name, license_url, description = metadata(page)
                image_url = info.get("url", "")
                page_url = "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_")
                ext = Path(urlparse(image_url).path).suffix.lower()
                if not image_url or ext not in EXTENSIONS or title in seen:
                    continue
                if not license_name or not license_url:
                    continue
                # Wikimedia Commons accepts freely licensed/public-domain files,
                # but we still keep the specific license for attribution.
                seen.add(title)
                name = f"{len(existing)+1:02d}_{re.sub(r'[^a-zA-Z0-9_-]+','_',Path(title).stem)[:55]}{ext}"
                destination = folder / name
                if destination.exists():
                    continue
                try:
                    response = requests.get(image_url, headers=HEADERS, timeout=45)
                    response.raise_for_status()
                    destination.write_bytes(response.content)
                    # Verify Pillow can decode it before counting it.
                    from PIL import Image
                    with Image.open(destination) as im:
                        im.verify()
                    ROWS.append([label, str(destination.relative_to(ROOT)), page_url,
                                 image_url, artist, license_name, license_url, description])
                    existing.append(destination)
                    print(f"{label}: saved {destination.name} ({license_name})")
                except Exception as exc:
                    destination.unlink(missing_ok=True)
                    print(f"Skipping {title}: {exc}")
                if len(existing) >= 5:
                    break
                time.sleep(0.2)
            if len(existing) >= 5:
                break
        if len(existing) < 5:
            print(f"WARNING: {label} has {len(existing)} usable files; add more manually.")
    if ROWS:
        write_header = not MANIFEST.exists()
        with MANIFEST.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["class", "local_path", "source_page", "direct_image_url",
                                 "creator", "license", "license_url", "description"])
            writer.writerows(ROWS)
    print(f"Source manifest: {MANIFEST}")
    print("Review labels and licenses, then run: uv run python src/clean.py overlap")
    print("Do not claim non-overlap until the overlap output has been inspected.")


if __name__ == "__main__":
    main()
