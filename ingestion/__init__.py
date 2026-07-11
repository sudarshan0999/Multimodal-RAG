"""PDF ingestion: text, images, and pipeline orchestration."""

__all__ = ["IngestionPipeline", "IngestionResult", "fetch_images_from_pdf"]

def __getattr__(name: str):
    if name in ("IngestionPipeline", "IngestionResult"):
        from ingestion.pipeline import IngestionPipeline, IngestionResult

        return {"IngestionPipeline": IngestionPipeline, "IngestionResult": IngestionResult}[name]
    if name == "fetch_images_from_pdf":
        from ingestion.pdf_images import fetch_images_from_pdf

        return fetch_images_from_pdf
    raise AttributeError(name)
