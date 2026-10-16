
"""Collect a documented external test set from Wikimedia Commons.

Run from the repository root:
    uv run python src/collect_new_images.py
"""
import csv
import re
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "new_images"
MANIFEST = OUT / "sources.csv"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": (
        "MAS1004-assignment1-student-project/1.0 "
        "(educational use; contact: student project)"
    )
}
TARGETS = {
"cartoon_animal": [
    '"cartoon animal" isolated illustration',
    '"animal clipart" transparent background',
],
    "plush_animal": [
        "plush animal toy photograph",
        "stuffed animal toy photograph",
    ],
    "real_animal": [
        "wild animal photograph",
        "real animal wildlife photograph",
    ],
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FIELDS = [
    "class", "local_path", "source_page", "direct_image_url",
    "creator", "license", "license_url", "description",
]


def clean_html(value):
    return re.sub(r"<[^>]+>", "", value or "").strip()


def get_json(params, attempts=5):
    """Retry politely when Commons temporarily rate-limits requests."""
    for attempt in range(attempts):
        response = requests.get(
            API, params=params, headers=HEADERS, timeout=30
        )
        if response.status_code == 429 or response.status_code >= 500:
            wait = min(10 * (attempt + 1), 60)
            print(f"Commons HTTP {response.status_code}; waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        time.sleep(2)
        return response.json()
    raise RuntimeError("Commons kept rate-limiting requests after retries")


def query(search):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": "filetype:bitmap " + search,
        "gsrnamespace": 6,
        "gsrlimit": 30,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 800,
    }
    data = get_json(params)
    return list(data.get("query", {}).get("pages", {}).values())


def metadata(page):
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {})

    def value(key):
        return clean_html(meta.get(key, {}).get("value", ""))

    return (
        info,
        value("Artist"),
        value("LicenseShortName"),
        value("LicenseUrl"),
        value("ImageDescription"),
    )


def download_image(info, destination):
    """Prefer a Commons thumbnail to reduce bandwidth and rate-limit risk."""
    image_url = info.get("thumburl") or info.get("url")
    if not image_url:
        raise ValueError("No image URL provided by Commons")

    for attempt in range(4):
        response = requests.get(
            image_url, headers=HEADERS, timeout=45
        )
        if response.status_code == 429 or response.status_code >= 500:
            wait = 10 * (attempt + 1)
            print(f"Download HTTP {response.status_code}; waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        destination.write_bytes(response.content)

        try:
            with Image.open(destination) as image:
                image.verify()
            return image_url
        except Exception:
            destination.unlink(missing_ok=True)
            raise ValueError("Downloaded file is not a valid image")

    raise RuntimeError("Download repeatedly rate-limited")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Load existing manifest rows to avoid duplicate entries on reruns.
    old_rows = []
    if MANIFEST.exists():
        with MANIFEST.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            old_rows = list(reader)

    known_paths = {row.get("local_path", "") for row in old_rows}
    rows_to_add = []

    for label, searches in TARGETS.items():
        folder = OUT / label
        folder.mkdir(parents=True, exist_ok=True)
        existing = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        if len(existing) >= 5:
            print(f"{label}: already has {len(existing)} images")
            continue

        seen_titles = set()
        for search in searches:
            try:
                pages = query(search)
            except Exception as exc:
                print(f"Search failed for {label}: {exc}")
                continue

            for page in pages:
                title = page.get("title", "")
                if not title or title in seen_titles:
                    continue

                info, artist, license_name, license_url, description = metadata(page)
                original_url = info.get("url", "")
                ext = Path(unquote(urlparse(original_url).path)).suffix.lower()

                if ext not in EXTENSIONS:
                    continue
                if not license_name or not license_url:
                    continue

                seen_titles.add(title)
                safe_stem = re.sub(
                    r"[^a-zA-Z0-9_-]+", "_", Path(title).stem
                )[:55]
                destination = folder / f"{len(existing)+1:02d}_{safe_stem}{ext}"

                if destination.exists():
                    continue

                try:
                    used_url = download_image(info, destination)
                    relative_path = str(destination.relative_to(ROOT))

                    # Avoid adding a second manifest entry for the same path.
                    if relative_path not in known_paths:
                        page_url = (
                            "https://commons.wikimedia.org/wiki/"
                            + title.replace(" ", "_")
                    )
                        rows_to_add.append([
                            label, relative_path, page_url, used_url,
                            artist, license_name, license_url, description
                        ])
                        known_paths.add(relative_path)

                    existing.append(destination)
                    print(f"{label}: saved {destination.name} ({license_name})")

                except Exception as exc:
                    destination.unlink(missing_ok=True)
                    print(f"Skipping {title}: {exc}")

                if len(existing) >= 5:
                    break

                # Be considerate to Wikimedia Commons.
                time.sleep(3)

            if len(existing) >= 5:
                break

        if len(existing) < 5:
            print(
                f"WARNING: {label} has {len(existing)} usable files; "
                "more may need to be added manually."
            )

    if rows_to_add:
        write_header = not MANIFEST.exists() or MANIFEST.stat().st_size == 0
        with MANIFEST.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(FIELDS)
            writer.writerows(rows_to_add)

    print(f"Source manifest: {MANIFEST}")
    print("Review every image and its license before using it.")
    print("Then run: uv run python src/clean.py overlap")
    print("This script does not prove that images are absent from training data.")


if __name__ == "__main__":
    main()
