from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

FILE_URL = (
    "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
    "Sunflower_-a_close_up_view.jpg"
)
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Sunflower_-a_close_up_view.jpg"
EXPECTED_SHA256 = "8c3dab486191534029e9cd867d5848bfb8161c665c705089b4b78504259c3671"
DESTINATION = Path("data/demo-input/sunflower.jpg")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if DESTINATION.is_file():
        if sha256(DESTINATION) != EXPECTED_SHA256:
            raise ValueError(
                f"existing demo image has an unexpected hash: {DESTINATION}"
            )
        print(f"Demo image already verified: {DESTINATION}")
        return 0

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    request = Request(FILE_URL, headers={"User-Agent": "image-classifier-demo/1.0"})
    with urlopen(request, timeout=60) as response, DESTINATION.open("wb") as file:
        while block := response.read(1024 * 1024):
            file.write(block)

    if sha256(DESTINATION) != EXPECTED_SHA256:
        raise ValueError(f"downloaded demo image failed verification: {SOURCE_PAGE}")
    print(f"Downloaded and verified demo image: {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
