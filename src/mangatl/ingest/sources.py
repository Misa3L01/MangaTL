"""Concrete page sources: folder of images, single image, zip/cbz archive, PDF."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

from mangatl.ingest.base import IMAGE_EXTENSIONS, PageRef, PageSource, natural_key

PDF_FALLBACK_DPI = 300


def _is_image_name(name: str) -> bool:
    path = PurePosixPath(name)
    hidden = any(part.startswith(".") or part == "__MACOSX" for part in path.parts)
    return not hidden and path.suffix.lower() in IMAGE_EXTENSIONS


class FolderSource(PageSource):
    kind = "folder"

    def pages(self) -> list[PageRef]:
        files = [
            p
            for p in self.path.rglob("*")
            if p.is_file() and _is_image_name(p.relative_to(self.path).as_posix())
        ]
        files.sort(key=lambda p: natural_key(p.relative_to(self.path).as_posix()))
        return [PageRef(i, p.stem, str(p)) for i, p in enumerate(files, start=1)]

    def load(self, ref: PageRef) -> Image.Image:
        with Image.open(ref.key) as img:
            img.load()
            return img


class ImageSource(PageSource):
    kind = "image"

    def pages(self) -> list[PageRef]:
        return [PageRef(1, self.path.stem, str(self.path))]

    def load(self, ref: PageRef) -> Image.Image:
        with Image.open(ref.key) as img:
            img.load()
            return img


class ArchiveSource(PageSource):
    kind = "archive"

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._zip = zipfile.ZipFile(path)

    def pages(self) -> list[PageRef]:
        names = [n for n in self._zip.namelist() if not n.endswith("/") and _is_image_name(n)]
        names.sort(key=natural_key)
        return [PageRef(i, PurePosixPath(n).stem, n) for i, n in enumerate(names, start=1)]

    def load(self, ref: PageRef) -> Image.Image:
        with Image.open(io.BytesIO(self._zip.read(ref.key))) as img:
            img.load()
            return img

    def close(self) -> None:
        self._zip.close()


class PdfSource(PageSource):
    """Renders each PDF page at the native resolution of its embedded scan."""

    kind = "pdf"

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        import pymupdf

        self._pymupdf = pymupdf
        self._doc = pymupdf.open(path)

    def pages(self) -> list[PageRef]:
        stem = self.path.stem
        return [PageRef(i + 1, f"{stem}_{i + 1:03d}", str(i)) for i in range(self._doc.page_count)]

    def load(self, ref: PageRef) -> Image.Image:
        pymupdf = self._pymupdf
        page = self._doc[int(ref.key)]
        zoom = PDF_FALLBACK_DPI / 72
        gray = False
        infos = page.get_image_info()
        if infos:
            # Use the largest embedded image to pick a zoom that keeps its native pixels.
            biggest = max(infos, key=lambda info: info["width"] * info["height"])
            bbox_w = biggest["bbox"][2] - biggest["bbox"][0]
            if bbox_w > 0:
                zoom = biggest["width"] / bbox_w
            gray = biggest.get("colorspace") == 1
        colorspace = pymupdf.csGRAY if gray else pymupdf.csRGB
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=colorspace, alpha=False)
        mode = "L" if pix.n == 1 else "RGB"
        return Image.frombytes(mode, (pix.width, pix.height), pix.samples)

    def close(self) -> None:
        self._doc.close()


def open_source(path: Path) -> PageSource:
    if path.is_dir():
        return FolderSource(path)
    suffix = path.suffix.lower()
    if suffix in (".zip", ".cbz"):
        return ArchiveSource(path)
    if suffix == ".pdf":
        return PdfSource(path)
    if suffix in IMAGE_EXTENSIONS:
        return ImageSource(path)
    raise ValueError(
        f"Formato de entrada no soportado: {path.name}. "
        "Usa una carpeta de imágenes, .zip, .cbz, .pdf o una imagen suelta."
    )
