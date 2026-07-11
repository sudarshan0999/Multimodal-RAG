"""PDF image and diagram extraction (user-provided algorithm, unchanged)."""

import io
import os
from typing import Any

import pymupdf as fitz
from PIL import Image


def fetch_images_from_pdf(pdfPath: str, output_dir: str) -> list[dict[str, Any]]:
    """Extract raster and vector figures from a PDF into PNG files.

    Returns a list of dicts with keys: path (str), page (int, 1-based), index (int).
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_meta: list[dict[str, Any]] = []
    pdf_file = fitz.open(pdfPath)
    output_format = "png"
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        page_rect = page.rect
        image_index = 1
        saved_bboxes = []

        # ── Method 1: Embedded raster images ───────────────────────────────────
        seen_xrefs = set()  # to Avoid Duplicate Images
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                rects = page.get_image_rects(xref)
                if not rects:
                    base_image = pdf_file.extract_image(xref)
                    image = Image.open(io.BytesIO(base_image["image"]))
                    if image.width >= 100 and image.height >= 100:
                        colors = image.getcolors(maxcolors=image.width * image.height)
                        if colors is None or len(colors) > 1:
                            path = os.path.join(
                                output_dir,
                                f"image{page_index + 1}_{image_index}.{output_format}",
                            )
                            image.save(path, format=output_format.upper())
                            saved_meta.append(
                                {"path": path, "page": page_index + 1, "index": image_index}
                            )
                            image_index += 1
                    continue

                for rect in rects:
                    if rect.width < 50 or rect.height < 50:
                        continue

                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    # Changed
                    char_count = len(
                        page.get_text("text", clip=rect).replace(" ", "").replace("\n", "")
                    )
                    area = rect.width * rect.height
                    # char_density = char_count / max(area, 1)

                    # Text block: high char count  (many chars packed into area)
                    # Figure with labels: may have high char count but very low density
                    if char_count > 100:
                        continue

                    # Till here

                    if _is_duplicate(bbox, saved_bboxes):
                        continue

                    expanded_rect = _expand_rect(
                        rect,
                        page_rect,
                        v_margin_pts=40,
                        h_margin_pts=40,
                    )

                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=expanded_rect)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))

                    colors = image.getcolors(maxcolors=image.width * image.height)
                    if colors is None or len(colors) > 1:
                        path = os.path.join(
                            output_dir,
                            f"image{page_index + 1}_{image_index}.{output_format}",
                        )
                        image.save(path, format=output_format.upper())
                        saved_bboxes.append(bbox)
                        saved_meta.append(
                            {"path": path, "page": page_index + 1, "index": image_index}
                        )
                        image_index += 1

            except Exception as e:
                print(
                    f"Error processing raster image on page {page_index + 1}, image {image_index}: {e}"
                )

        # ── Method 2: Vector graphics ───────────────────────────────────────────
        try:
            drawings = page.get_drawings()
            if not drawings:
                continue

            raw_rects = [fitz.Rect(d["rect"]) for d in drawings if d.get("rect")]

            # ── Step 1: Merge drawing primitives into figure regions ────────────
            # Large gap (30pts) on first pass to unite sub-drawings of one figure,
            # then a second pass to catch anything still fragmented.
            merged = _merge_rects_aggressive(raw_rects, gap=30)

            # ── Step 2: Remove rects that are fully contained inside a larger one.
            # This is what was causing sub-regions of a single figure to be saved
            # separately — each got detected as its own region before the full
            # bounding union was computed.
            merged = _remove_contained_rects(merged)

            for rect in merged:
                if rect.width < 50 or rect.height < 50:
                    continue

                bbox = (rect.x0, rect.y0, rect.x1, rect.y1)

                # Changed
                char_count = len(
                    page.get_text("text", clip=rect).replace(" ", "").replace("\n", "")
                )
                area = rect.width * rect.height

                # Text block: high char count AND high density (many chars packed into area)
                # Figure with labels: may have high char count but very low density
                if char_count > 100:
                    continue

                # Till here

                if _is_duplicate(bbox, saved_bboxes, tolerance=20):
                    continue

                # 40 PDF pts * zoom 2.0 = 80 rendered px top/bottom
                # 50 PDF pts * zoom 2.0 = 100 rendered px left/right
                expanded_rect = _expand_rect(
                    rect,
                    page_rect,
                    v_margin_pts=40,
                    h_margin_pts=50,
                )

                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=expanded_rect)
                image = Image.open(io.BytesIO(pix.tobytes("png")))

                colors = image.getcolors(maxcolors=image.width * image.height)
                if colors is None or len(colors) > 1:
                    path = os.path.join(
                        output_dir,
                        f"image{page_index + 1}_{image_index}.{output_format}",
                    )
                    image.save(path, format=output_format.upper())
                    saved_bboxes.append(bbox)
                    saved_meta.append(
                        {"path": path, "page": page_index + 1, "index": image_index}
                    )
                    image_index += 1

        except Exception as e:
            print(f"Error processing vector graphics on page {page_index + 1}: {e}")

    pdf_file.close()
    return saved_meta


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _expand_rect(
    rect: fitz.Rect,
    page_rect: fitz.Rect,
    v_margin_pts: float = 40,
    h_margin_pts: float = 50,
) -> fitz.Rect:
    """
    Expand rect by v_margin_pts top/bottom and h_margin_pts left/right,
    clamped to page boundary.

    At zoom=2.0:
        v_margin_pts=40  →  80 rendered pixels top and bottom
        h_margin_pts=50  → 100 rendered pixels left and right
    """
    return fitz.Rect(
        max(rect.x0 - h_margin_pts, page_rect.x0),  # max and min are for boundary protection if the page margins are over
        max(rect.y0 - v_margin_pts, page_rect.y0),
        min(rect.x1 + h_margin_pts, page_rect.x1),
        min(rect.y1 + v_margin_pts, page_rect.y1),
    )


def _is_duplicate(bbox, saved_bboxes, tolerance: float = 10) -> bool:
    x0, y0, x1, y1 = bbox
    for b in saved_bboxes:
        if abs(x0 - b[0]) < tolerance and abs(y0 - b[1]) < tolerance:
            return True
    return False


def _merge_rects_aggressive(rects: list, gap: float = 30) -> list:
    """
    Two-pass greedy merge. Gap of 30pts unites sub-drawings that are
    up to ~0.4 inches apart — enough for multi-panel technical figures
    like the DIP/IC installation diagram.
    """

    def one_pass(rects_in):
        merged = []
        for rect in rects_in:
            expanded = rect + (-gap, -gap, gap, gap)
            absorbed = False
            for i, existing in enumerate(merged):
                if existing.intersects(expanded):
                    merged[i] = existing | rect
                    absorbed = True
                    break
            if not absorbed:
                merged.append(fitz.Rect(rect))
        return merged

    result = one_pass(rects)
    result = one_pass(result)
    return result


def _remove_contained_rects(rects: list) -> list:
    """
    Remove any rect that is fully contained inside another rect.
    This prevents sub-regions of a merged figure from being saved
    as separate images alongside the full figure.

    Example: if the 5-panel DIP/IC figure merges into one large rect,
    but one sub-panel also survived as its own rect, the sub-panel
    gets dropped here because it's entirely inside the large one.
    """
    result = []
    for i, r in enumerate(rects):
        contained = False
        for j, other in enumerate(rects):
            if i == j:
                continue
            # r is contained in other if other fully covers r
            if (
                other.x0 <= r.x0
                and other.y0 <= r.y0
                and other.x1 >= r.x1
                and other.y1 >= r.y1
            ):
                contained = True
                break
        if not contained:
            result.append(r)
    return result
