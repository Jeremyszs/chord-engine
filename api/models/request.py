"""Pydantic request schemas for the chord-engine API."""

from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Query parameters for audio analysis.

    Audio is uploaded as multipart form data (not JSON body).
    This model documents the optional query parameters only.
    """

    smooth_method: Literal["hmm", "median"] = Field(
        default="hmm",
        description='Which post-processing smoothing method to use.',
    )

    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description='Inference device.',
    )

    include_raw_chords: bool = Field(
        default=False,
        description='If True, include the raw per-frame chord array in the response. '
                    'Omit by default to keep response size small.',
    )
