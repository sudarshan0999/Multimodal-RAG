"""Create a small sample PDF under sample_data/pdfs/ for demos and tests."""

from pathlib import Path

import fitz


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "sample_data" / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sample_circuits.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100),
        "Sample: Series and parallel resistors.\n"
        "A series circuit has one current path.\n"
        "A parallel circuit splits current across branches.",
    )
    # Simple vector drawing (may yield one diagram region for extraction)
    shape = page.new_shape()
    shape.draw_line(fitz.Point(100, 400), fitz.Point(200, 400))
    shape.draw_line(fitz.Point(150, 350), fitz.Point(150, 450))
    shape.finish(width=1.5)
    shape.commit()

    page2 = doc.new_page()
    page2.insert_text((72, 100), "Page 2: Capacitors store charge.")

    doc.save(str(path))
    doc.close()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
