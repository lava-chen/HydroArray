"""
Figure export helpers.

HydroArray plots use Matplotlib for drawing, but publication figures often
benefit from an SVG-first workflow: text remains editable and precisely
positioned in the vector master, then a PNG is rasterized for final delivery.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import importlib.util

import matplotlib.pyplot as plt


PathLike = Union[str, Path]


@dataclass
class FigureExportConfig:
    """Options for saving figures with an SVG-first workflow."""

    dpi: int = 300
    svg: bool = True
    keep_svg: bool = True
    transparent: bool = False
    bbox_inches: str = "tight"
    pad_inches: float = 0.04


def configure_vector_text(
    *,
    font_family: Optional[Union[str, list[str]]] = None,
    math_fontset: str = "stix",
) -> None:
    """Configure Matplotlib so SVG/PDF exports keep text as text.

    This is useful when figures need final typography polish in Illustrator,
    Inkscape, PowerPoint, or an SVG-aware post-processing workflow.
    """
    rc = {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": math_fontset,
        "axes.unicode_minus": False,
    }
    if font_family is not None:
        rc["font.family"] = font_family
    plt.rcParams.update(rc)


def _svg_path_for(path: Path) -> Path:
    return path if path.suffix.lower() == ".svg" else path.with_suffix(".svg")


def _rasterize_svg(svg_path: Path, png_path: Path, dpi: int) -> bool:
    """Rasterize SVG to PNG with optional dependencies when available."""
    if importlib.util.find_spec("cairosvg") is not None:
        import cairosvg

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            dpi=dpi,
        )
        return True

    return False


def save_figure(
    fig,
    path: PathLike,
    *,
    dpi: int = 300,
    svg: bool = True,
    keep_svg: bool = True,
    transparent: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.04,
) -> Path:
    """Save a figure, optionally using an SVG master before PNG output.

    Args:
        fig: Matplotlib figure.
        path: Output path. If this ends with ``.png`` and ``svg=True``, a
            sibling ``.svg`` master is also written.
        dpi: PNG rasterization resolution.
        svg: Whether to write an SVG master first.
        keep_svg: Keep the SVG file when ``path`` is not itself ``.svg``.
        transparent: Save transparent background.
        bbox_inches: Matplotlib bounding-box mode.
        pad_inches: Padding around the tight bounding box.

    Returns:
        The requested output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    configure_vector_text()

    if out.suffix.lower() == ".svg":
        fig.savefig(
            out,
            format="svg",
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            transparent=transparent,
        )
        return out

    if not svg:
        fig.savefig(
            out,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            transparent=transparent,
        )
        return out

    svg_path = _svg_path_for(out)
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        transparent=transparent,
    )

    rasterized = False
    if out.suffix.lower() == ".png":
        rasterized = _rasterize_svg(svg_path, out, dpi=dpi)

    if not rasterized:
        fig.savefig(
            out,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            transparent=transparent,
        )

    if not keep_svg and svg_path != out:
        try:
            svg_path.unlink()
        except FileNotFoundError:
            pass

    return out


def save_svg_png(
    fig,
    png_path: PathLike,
    *,
    dpi: int = 300,
    keep_svg: bool = True,
    transparent: bool = False,
) -> Path:
    """Convenience wrapper for ``save_figure(..., svg=True)``."""
    return save_figure(
        fig,
        png_path,
        dpi=dpi,
        svg=True,
        keep_svg=keep_svg,
        transparent=transparent,
    )


__all__ = [
    "FigureExportConfig",
    "configure_vector_text",
    "save_figure",
    "save_svg_png",
]
