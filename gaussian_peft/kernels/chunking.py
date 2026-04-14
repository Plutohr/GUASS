from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor


def iter_chunks(total: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        yield start, end
        start = end


def chunked_sum(
    values: Tensor,
    chunk_size: int,
    fn,
) -> Tensor:
    if values.ndim == 0:
        raise ValueError("values must have at least one dimension")
    output: Tensor | None = None
    for start, end in iter_chunks(values.shape[0], chunk_size):
        chunk_output = fn(values[start:end], start, end)
        if output is None:
            output = chunk_output.new_empty((values.shape[0], *chunk_output.shape[1:]))
        output[start:end] = chunk_output
    if output is None:
        raise ValueError("values must not be empty")
    return output
