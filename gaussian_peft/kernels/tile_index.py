from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gaussian_peft.kernels.bounds import TileBounds


@dataclass(slots=True)
class TileIndex:
    tile_ptr: Tensor
    gaussian_ids: Tensor
    tile_ids: Tensor
    num_tile_rows: int
    num_tile_cols: int

    @property
    def num_tiles(self) -> int:
        return self.num_tile_rows * self.num_tile_cols


def build_padded_tile_gaussian_index(tile_index: TileIndex) -> tuple[Tensor, Tensor]:
    num_tiles = tile_index.num_tiles
    counts = tile_index.tile_ptr[1:] - tile_index.tile_ptr[:-1]
    if num_tiles == 0:
        empty = torch.empty(0, device=tile_index.tile_ptr.device, dtype=torch.int64)
        return empty.reshape(0, 0), empty

    max_count = int(counts.max().item()) if counts.numel() > 0 else 0
    padded = torch.full(
        (num_tiles, max_count),
        fill_value=-1,
        device=tile_index.tile_ptr.device,
        dtype=torch.int64,
    )
    if max_count == 0 or tile_index.gaussian_ids.numel() == 0:
        return padded, counts

    pair_offsets = torch.arange(
        tile_index.gaussian_ids.shape[0],
        device=tile_index.tile_ptr.device,
        dtype=torch.int64,
    )
    pair_offsets = pair_offsets - tile_index.tile_ptr.index_select(0, tile_index.tile_ids)
    padded[tile_index.tile_ids, pair_offsets] = tile_index.gaussian_ids
    return padded, counts


def build_tile_index(bounds: TileBounds) -> TileIndex:
    device = bounds.tile_r0.device
    num_gaussians = int(bounds.tile_r0.shape[0])
    num_tiles = bounds.num_tile_rows * bounds.num_tile_cols
    total_pairs = int(bounds.tiles_touched.sum().item())
    if total_pairs == 0:
        tile_ptr = torch.zeros(num_tiles + 1, device=device, dtype=torch.int64)
        empty = torch.empty(0, device=device, dtype=torch.int64)
        return TileIndex(
            tile_ptr=tile_ptr,
            gaussian_ids=empty,
            tile_ids=empty,
            num_tile_rows=bounds.num_tile_rows,
            num_tile_cols=bounds.num_tile_cols,
        )

    gaussian_ids = torch.repeat_interleave(
        torch.arange(num_gaussians, device=device, dtype=torch.int64),
        bounds.tiles_touched,
    )
    pair_offsets = torch.arange(total_pairs, device=device, dtype=torch.int64)
    group_starts = torch.repeat_interleave(
        torch.cumsum(bounds.tiles_touched, dim=0) - bounds.tiles_touched,
        bounds.tiles_touched,
    )
    local_offsets = pair_offsets - group_starts
    tile_widths = (bounds.tile_c1 - bounds.tile_c0 + 1).index_select(0, gaussian_ids)
    tile_rows = bounds.tile_r0.index_select(0, gaussian_ids) + torch.div(
        local_offsets,
        tile_widths,
        rounding_mode="floor",
    )
    tile_cols = bounds.tile_c0.index_select(0, gaussian_ids) + torch.remainder(local_offsets, tile_widths)
    tile_ids = tile_rows * bounds.num_tile_cols + tile_cols

    order = torch.argsort(tile_ids)
    tile_ids = tile_ids[order]
    gaussian_ids = gaussian_ids[order]

    counts = torch.bincount(tile_ids, minlength=num_tiles)
    tile_ptr = torch.zeros(num_tiles + 1, device=device, dtype=torch.int64)
    tile_ptr[1:] = torch.cumsum(counts, dim=0)
    return TileIndex(
        tile_ptr=tile_ptr,
        gaussian_ids=gaussian_ids,
        tile_ids=tile_ids,
        num_tile_rows=bounds.num_tile_rows,
        num_tile_cols=bounds.num_tile_cols,
    )
