from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
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
        name="Fire truck",
        filename="fire-engine.jpg",
        source_page=(
            "https://commons.wikimedia.org/wiki/File:Air_Force_fire_truck.jpg"
        ),
        file_url=(
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
            "Air_Force_fire_truck.jpg"
        ),
        sha256="a7fd9c5809e73b024a06abdaa910f3308fad21831cdcb1212f8a33ca56cd24c7",
    ),
    DemoAsset(
        name="Espresso machine",
        filename="espresso-machine.jpg",
        source_page=(
            "https://commons.wikimedia.org/wiki/File:Espressso_machine_2014.JPG"
        ),
        file_url=(
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
            "Espressso_machine_2014.JPG"
        ),
        sha256="c531fbff004b636bbea8c57d7439bd13f22e77ccb74793cfb167390478f9f4a6",
    ),
    DemoAsset(
        name="Typewriter",
        filename="typewriter.jpg",
        source_page=(
            "https://commons.wikimedia.org/wiki/File:National_Typewriter_No5,_foto.JPG"
        ),
        file_url=(
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
            "National_Typewriter_No5,_foto.JPG"
        ),
        sha256="45ffde5fa7620bd747df10041473d0881fcd69ec4ab8fa4d0603d0b6819e9225",
    ),
    DemoAsset(
        name="Steam locomotive",
        filename="steam-locomotive.jpg",
        source_page=(
            "https://commons.wikimedia.org/wiki/File:Steam_locomotive_(1).jpg"
        ),
        file_url=(
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
            "Steam_locomotive_(1).jpg"
        ),
        sha256="b69ad9de037667004abaf26bc440e13e535fe86962b806376033d577b282b38f",
    ),
)
DESTINATION = Path("data/complex-object-demo")


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
        asset.file_url,
        headers={
            "User-Agent": (
                "image-classifier-engine/1.0 "
                "(+https://github.com/jollyzachary/image_classifier)"
            )
        },
    )
    for attempt, delay in enumerate((0, 5, 15), start=1):
        if delay:
            time.sleep(delay)
        try:
            with urlopen(request, timeout=90) as response, temporary.open("wb") as file:
                while block := response.read(1024 * 1024):
                    file.write(block)
            break
        except HTTPError as error:
            temporary.unlink(missing_ok=True)
            if error.code != 429 or attempt == 3:
                raise
    if sha256(temporary) != asset.sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded asset failed verification: {asset.source_page}")
    temporary.replace(destination)
    return destination


def main() -> int:
    for index, asset in enumerate(ASSETS):
        if index:
            time.sleep(1)
        path = download(asset)
        print(f"Verified {asset.name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
