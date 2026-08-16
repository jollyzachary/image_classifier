from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DemoAsset:
    name: str
    filename: str
    source_page: str
    file_url: str
    sha256: str


ASSETS = (
    DemoAsset(
        name="Sunflower",
        filename="sunflower.jpg",
        source_page=(
            "https://commons.wikimedia.org/wiki/File:Sunflower_-a_close_up_view.jpg"
        ),
        file_url=(
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
            "Sunflower_-a_close_up_view.jpg"
        ),
        sha256="8c3dab486191534029e9cd867d5848bfb8161c665c705089b4b78504259c3671",
    ),
    DemoAsset(
        name="Tabby cat",
        filename="tabby-cat.jpg",
        source_page="https://commons.wikimedia.org/wiki/File:Tabby-cat.jpg",
        file_url=(
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/Tabby-cat.jpg"
        ),
        sha256="89556ebf24e8ec7a3f16f8624cb7be3be6cd236cb8a2abe96396ad64849a3702",
    ),
    DemoAsset(
        name="Coffee cup",
        filename="coffee-cup.jpg",
        source_page="https://commons.wikimedia.org/wiki/File:Cup_Coffee.jpg",
        file_url=(
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/Cup_Coffee.jpg"
        ),
        sha256="32680d80bea857cfa3c9822629d78c29a79737212ba6c719ae966c10f159edd4",
    ),
    DemoAsset(
        name="Vintage automobile",
        filename="vintage-car.jpg",
        source_page=(
            "https://commons.wikimedia.org/wiki/File:Retro_old_car_oldtimer.jpg"
        ),
        file_url=(
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
            "Retro_old_car_oldtimer.jpg"
        ),
        sha256="1b8b8e7017137fb615f3a72c8da10336c158a2fcbc3aede28c9edc7fabb753c8",
    ),
)
DESTINATION = Path("data/open-vocabulary-demo")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(asset: DemoAsset) -> Path:
    destination = DESTINATION / asset.filename
    if destination.is_file() and sha256(destination) == asset.sha256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(
        asset.file_url, headers={"User-Agent": "image-classifier-demo/1.0"}
    )
    with urlopen(request, timeout=90) as response, temporary.open("wb") as file:
        while block := response.read(1024 * 1024):
            file.write(block)
    if sha256(temporary) != asset.sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded asset failed verification: {asset.source_page}")
    temporary.replace(destination)
    return destination


def main() -> int:
    for asset in ASSETS:
        path = download(asset)
        print(f"Verified {asset.name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
