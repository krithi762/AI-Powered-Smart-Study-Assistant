"""
OCR Processor module for AI-Powered Smart Study Assistant.

Provides Tesseract OCR fallback for scanned PDFs that contain no extractable text.
Uses PyMuPDF for rendering PDF pages to images and pytesseract for OCR text extraction.

Architecture:
  1. check_tesseract_available()      → detect binary + version
  2. render_pdf_page_to_image()       → PyMuPDF → PIL Image
  3. run_tesseract_on_image()         → pytesseract → {text, confidence, word_count}
  4. extract_text_via_ocr()           → full document OCR, same shape as extract_text_from_pdf()
  5. should_use_ocr()                 → decision helper
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows Tesseract installation path candidates (searched in order)
# ---------------------------------------------------------------------------
_WINDOWS_TESSERACT_PATHS: list[str] = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\Public\Tesseract-OCR\tesseract.exe",
    r"C:\tools\Tesseract-OCR\tesseract.exe",
]

# Minimum words per page below which we consider a page "image-only"
MIN_WORDS_PER_PAGE_THRESHOLD: int = 10

# DPI used when rendering PDF pages to images for OCR (higher = better quality, slower)
OCR_RENDER_DPI: int = 300


# ---------------------------------------------------------------------------
# 1. Tesseract availability check
# ---------------------------------------------------------------------------

def _find_tesseract_binary() -> str | None:
    """
    Finds the Tesseract binary on the current system.

    Search order:
      1. PATH (cross-platform)
      2. Known Windows installation directories
    """
    # Check PATH first
    if shutil.which("tesseract"):
        return shutil.which("tesseract")

    # Fallback: probe known Windows paths
    for candidate in _WINDOWS_TESSERACT_PATHS:
        if Path(candidate).is_file():
            return candidate

    return None


def check_tesseract_available() -> dict[str, Any]:
    """
    Detects whether Tesseract OCR is installed and accessible.

    Returns:
        dict: {
            "available": bool,
            "path": str | None,
            "version": str | None,
            "error": str | None,
            "install_instructions": str | None   (Windows guide when missing)
        }
    """
    binary = _find_tesseract_binary()
    if not binary:
        return {
            "available": False,
            "path": None,
            "version": None,
            "error": "Tesseract binary not found on PATH or standard Windows locations.",
            "install_instructions": _windows_install_instructions(),
        }

    # Try to run tesseract --version
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (result.stdout + result.stderr).strip()
        version_line = output.splitlines()[0] if output else "unknown"
        return {
            "available": True,
            "path": binary,
            "version": version_line,
            "error": None,
            "install_instructions": None,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as exc:
        return {
            "available": False,
            "path": binary,
            "version": None,
            "error": f"Tesseract found at {binary!r} but failed to execute: {exc!s}",
            "install_instructions": _windows_install_instructions(),
        }


def _windows_install_instructions() -> str:
    return (
        "Tesseract OCR is not installed.\n\n"
        "Windows Installation Steps:\n"
        "  1. Download the installer from:\n"
        "     https://github.com/UB-Mannheim/tesseract/wiki\n"
        "     (Choose the latest stable .exe, e.g. tesseract-ocr-w64-setup-5.x.x.exe)\n"
        "  2. Run the installer → keep default path:\n"
        "     C:\\Program Files\\Tesseract-OCR\\\n"
        "  3. During install, check 'Add to PATH' (or add manually).\n"
        "  4. Restart your terminal / Flask server.\n"
        "  5. Verify: tesseract --version\n\n"
        "Without Tesseract, scanned PDFs cannot be processed by OCR.\n"
        "Normal (text-based) PDFs continue to work without Tesseract."
    )


# ---------------------------------------------------------------------------
# 2. Render one PDF page to a PIL Image
# ---------------------------------------------------------------------------

def render_pdf_page_to_image(
    pdf_path: str | Path,
    page_number: int,  # 1-indexed
    dpi: int = OCR_RENDER_DPI,
) -> Any:  # returns PIL.Image.Image or None
    """
    Renders a single PDF page to a PIL Image using PyMuPDF.

    Args:
        pdf_path:    Path to the PDF file.
        page_number: 1-indexed page number.
        dpi:         Rendering resolution (default 300 DPI for OCR quality).

    Returns:
        PIL.Image.Image on success, None on failure.
    """
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError:
        logger.error("Pillow is not installed. Run: pip install Pillow")
        return None

    path = Path(pdf_path)
    doc = None
    try:
        doc = pymupdf.open(str(path))
        page_idx = page_number - 1  # Convert to 0-indexed

        if page_idx < 0 or page_idx >= doc.page_count:
            logger.warning("Page %d is out of range for %s", page_number, path.name)
            return None

        page = doc.load_page(page_idx)

        # Scale matrix for the desired DPI (72 is PyMuPDF's base DPI)
        scale = dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)

        # Render to pixmap (RGB, no alpha)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert pixmap bytes to PIL Image
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return img

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to render page %d of %s: %s", page_number, pdf_path, exc)
        return None
    finally:
        if doc is not None:
            with contextlib.suppress(Exception):
                doc.close()


# ---------------------------------------------------------------------------
# 3. Run Tesseract on a PIL Image
# ---------------------------------------------------------------------------

def run_tesseract_on_image(
    image: Any,  # PIL.Image.Image
    lang: str = "eng",
    tesseract_cmd: str | None = None,
) -> dict[str, Any]:
    """
    Runs Tesseract OCR on a PIL Image and returns extracted text with metadata.

    Args:
        image:          PIL Image to process.
        lang:           Tesseract language code (default 'eng').
        tesseract_cmd:  Optional explicit path to tesseract binary.

    Returns:
        dict: {
            "success": bool,
            "text": str,
            "confidence": float | None,  (0.0–100.0, None if unavailable)
            "word_count": int,
            "error": str | None
        }
    """
    empty_result: dict[str, Any] = {
        "success": False,
        "text": "",
        "confidence": None,
        "word_count": 0,
        "error": None,
    }

    if image is None:
        empty_result["error"] = "No image provided."
        return empty_result

    try:
        import pytesseract  # type: ignore[import-untyped]
    except ImportError:
        empty_result["error"] = "pytesseract is not installed. Run: pip install pytesseract"
        return empty_result

    # Set tesseract_cmd if explicitly provided
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        # Auto-detect binary and configure
        binary = _find_tesseract_binary()
        if binary:
            pytesseract.pytesseract.tesseract_cmd = binary

    try:
        # Try to get per-word confidence data first
        try:
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                output_type=pytesseract.Output.DICT,
                config="--psm 3",
            )
            confidences = [
                int(c) for c in data.get("conf", [])
                if str(c).lstrip("-").isdigit() and int(c) >= 0
            ]
            avg_confidence = float(sum(confidences) / len(confidences)) if confidences else None
        except Exception:  # noqa: BLE001
            avg_confidence = None

        # Extract plain text
        raw_text: str = pytesseract.image_to_string(
            image,
            lang=lang,
            config="--psm 3",
        )
        clean_text = raw_text.strip()
        word_count = len(clean_text.split()) if clean_text else 0

        return {
            "success": True,
            "text": clean_text,
            "confidence": avg_confidence,
            "word_count": word_count,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("Tesseract OCR failed: %s", exc)
        empty_result["error"] = f"OCR failed: {exc!s}"
        return empty_result


# ---------------------------------------------------------------------------
# 4. Full document OCR (same return shape as extract_text_from_pdf)
# ---------------------------------------------------------------------------

def extract_text_via_ocr(
    pdf_path: str | Path,
    existing_pages: list[dict[str, Any]] | None = None,
    lang: str = "eng",
    dpi: int = OCR_RENDER_DPI,
) -> dict[str, Any]:
    """
    Extracts text from a PDF using Tesseract OCR as a fallback for scanned pages.

    For pages in ``existing_pages`` that already have sufficient text (≥ MIN_WORDS_PER_PAGE_THRESHOLD
    words), the existing extracted text is preserved and OCR is skipped for that page.
    Pages with little/no text are rendered to images and processed by Tesseract.

    Returns the same shape as ``extract_text_from_pdf()``:
        {
            "success": bool,
            "total_pages": int,
            "total_words": int,
            "has_text": bool,
            "is_scanned": bool,
            "pages": [{"page_number": int, "text": str, "word_count": int}],
            "metadata": dict,
            "extraction_method": str,       # "ocr", "mixed", "pymupdf"
            "ocr_applied": bool,
            "ocr_engine": str,
            "ocr_confidence": float | None,
            "error": str | None
        }
    """
    path = Path(pdf_path)

    base_result: dict[str, Any] = {
        "success": False,
        "total_pages": 0,
        "total_words": 0,
        "has_text": False,
        "is_scanned": True,
        "pages": [],
        "metadata": {},
        "extraction_method": "ocr",
        "ocr_applied": True,
        "ocr_engine": "tesseract",
        "ocr_confidence": None,
        "error": None,
    }

    # Verify Tesseract is available
    tess_info = check_tesseract_available()
    if not tess_info["available"]:
        base_result["error"] = (
            f"Tesseract OCR is not available: {tess_info['error']}\n\n"
            f"{tess_info.get('install_instructions', '')}"
        )
        return base_result

    # Open PDF to get page count
    doc = None
    try:
        doc = pymupdf.open(str(path))
        total_pages = doc.page_count
    except Exception as exc:  # noqa: BLE001
        base_result["error"] = f"Cannot open PDF for OCR: {exc!s}"
        return base_result
    finally:
        if doc is not None:
            with contextlib.suppress(Exception):
                doc.close()

    if total_pages == 0:
        base_result["error"] = "PDF has zero pages."
        return base_result

    # Build a lookup for existing page texts (keyed by 1-indexed page_number)
    existing_lookup: dict[int, str] = {}
    if existing_pages:
        for p in existing_pages:
            pnum = int(p.get("page_number", 0))
            # Support both "text" and "extracted_text" keys
            text = p.get("text") or p.get("extracted_text") or ""
            if pnum > 0:
                existing_lookup[pnum] = text

    pages_out: list[dict[str, Any]] = []
    total_words = 0
    all_confidences: list[float] = []
    ocr_was_used = False
    pymupdf_was_used = False

    for page_number in range(1, total_pages + 1):
        existing_text = existing_lookup.get(page_number, "")
        existing_wc = len(existing_text.split()) if existing_text else 0

        if existing_wc >= MIN_WORDS_PER_PAGE_THRESHOLD:
            # Sufficient text already — keep it, skip OCR for this page
            pymupdf_was_used = True
            pages_out.append({
                "page_number": page_number,
                "text": existing_text,
                "word_count": existing_wc,
            })
            total_words += existing_wc
            continue

        # Need OCR for this page
        ocr_was_used = True
        image = render_pdf_page_to_image(pdf_path, page_number, dpi=dpi)

        if image is None:
            logger.warning("Could not render page %d to image, skipping OCR.", page_number)
            # Preserve whatever existing text we had (even if sparse)
            pages_out.append({
                "page_number": page_number,
                "text": existing_text,
                "word_count": existing_wc,
            })
            total_words += existing_wc
            continue

        ocr_result = run_tesseract_on_image(image, lang=lang, tesseract_cmd=tess_info["path"])

        if not ocr_result["success"]:
            logger.warning(
                "OCR failed on page %d: %s", page_number, ocr_result.get("error")
            )
            # Fall back to whatever existing text exists
            pages_out.append({
                "page_number": page_number,
                "text": existing_text,
                "word_count": existing_wc,
            })
            total_words += existing_wc
            continue

        ocr_text = ocr_result["text"]
        ocr_wc = ocr_result["word_count"]
        total_words += ocr_wc

        if ocr_result["confidence"] is not None:
            all_confidences.append(ocr_result["confidence"])

        pages_out.append({
            "page_number": page_number,
            "text": ocr_text,
            "word_count": ocr_wc,
        })

    # Determine extraction method label
    if ocr_was_used and pymupdf_was_used:
        extraction_method = "mixed"
    elif ocr_was_used:
        extraction_method = "ocr"
    else:
        extraction_method = "pymupdf"

    avg_confidence: float | None = (
        float(sum(all_confidences) / len(all_confidences))
        if all_confidences
        else None
    )

    has_text = total_words > 0

    return {
        "success": True,
        "total_pages": total_pages,
        "total_words": total_words,
        "has_text": has_text,
        "is_scanned": not has_text,
        "pages": pages_out,
        "metadata": {},
        "extraction_method": extraction_method,
        "ocr_applied": ocr_was_used,
        "ocr_engine": "tesseract" if ocr_was_used else "none",
        "ocr_confidence": avg_confidence,
        "error": None,
    }


# ---------------------------------------------------------------------------
# 5. Decision helper
# ---------------------------------------------------------------------------

def should_use_ocr(
    extraction_result: dict[str, Any],
    min_words_per_page: int = MIN_WORDS_PER_PAGE_THRESHOLD,
) -> bool:
    """
    Decides whether OCR should be applied given a PyMuPDF extraction result.

    OCR is triggered when:
    - The PDF was flagged as scanned, OR
    - Total words == 0, OR
    - Average words per page falls below ``min_words_per_page``

    Args:
        extraction_result:  Result dict from ``extract_text_from_pdf()``.
        min_words_per_page: Minimum words/page threshold.

    Returns:
        True if OCR should be used, False otherwise.
    """
    if extraction_result.get("is_scanned"):
        return True
    if extraction_result.get("total_words", 0) == 0:
        return True
    total_pages = extraction_result.get("total_pages", 1) or 1
    avg_wpp = extraction_result.get("total_words", 0) / total_pages
    return avg_wpp < min_words_per_page
