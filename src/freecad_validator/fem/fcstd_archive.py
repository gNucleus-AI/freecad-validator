"""FCStd archive validation before FreeCAD opens a document."""

import zipfile


def assert_safe_fcstd(path, require_document=False):
    """Reject unsafe archive member paths before opening an FCStd document."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        if require_document:
            raise ValueError("FCStd is not a valid ZIP archive") from exc
        return  # not a zip (unlikely for .FCStd); openDocument will handle it
    if require_document and "Document.xml" not in names:
        raise ValueError("FCStd archive is missing Document.xml")
    for name in names:
        norm = name.replace("\\", "/")
        if norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
            raise ValueError(
                f"[fcstd_adapter] SECURITY: absolute path in FCStd zip member: {name!r}"
            )
        parts = [p for p in norm.split("/") if p not in ("", ".")]
        depth = 0
        for p in parts:
            depth += -1 if p == ".." else 1
            if depth < 0:
                raise ValueError(
                    f"[fcstd_adapter] SECURITY: path traversal in FCStd zip member: {name!r}"
                )
