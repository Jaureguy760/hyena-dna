from typing import List, Optional, cast

from functools import partial
import genvarloader as gvl
import numpy as np
import seqpro as sp
from einops import rearrange, unpack
from genvarloader.util import _set_fixed_length_around_center, read_bedlike
from torch.utils.data import Dataset, DataLoader

from .base import SequenceDataset


class RefAndTracks(Dataset):
    def __init__(
        self,
        fasta: str,
        tracks: str,
        bed: str,
        length: int,
        samples: Optional[List[str]] = None,
        max_jitter: Optional[int] = None,
        rc_prob: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.fasta = gvl.Fasta("haps", fasta, "N")
        self.tracks = gvl.ZarrTracks("track", tracks)
        self.bed = _set_fixed_length_around_center(read_bedlike(bed), length)
        self.length = length
        self.samples = cast(List[str], self.tracks.samples)
        if samples and (missing := set(samples).difference(self.samples)):
            raise ValueError(f"Samples {missing} not found in the dataset")
        elif samples:
            self.samples = list(set(samples).intersection(self.samples))
        self.max_jitter = max_jitter
        self.rc_prob = rc_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.bed.height * len(self.samples)

    def __getitem__(self, idx: int):
        region_idx, sample_idx = np.unravel_index(
            idx, (self.bed.height, len(self.samples))
        )
        sample = self.samples[sample_idx]
        chrom, start, end = self.bed.row(region_idx.item())
        if self.max_jitter is not None:
            start -= self.max_jitter
            end += self.max_jitter
        # (length)
        haps = self.fasta.read(chrom, start, end)
        track = self.tracks.read(chrom, start, end, sample=[sample]).squeeze()
        if self.max_jitter is not None:
            jitter = self.rng.integers(0, 2 * self.max_jitter + 1)
            haps = haps[jitter : jitter + self.length]
            track = track[jitter : jitter + self.length]
        # (l a)
        haps = sp.DNA.ohe(haps)
        # (l)
        track = track[524:1524]
        if self.rc_prob is not None and self.rng.random() < self.rc_prob:
            haps = sp.DNA.reverse_complement(haps, length_axis=-1, ohe_axis=-2)
            track = track[::-1]
        haps = rearrange(haps, "l a -> 1 a l")
        return {"haps": haps.astype(np.float32).copy(), "track": track.copy()}


class HapsAndTracks(Dataset):
    def __init__(
        self,
        fasta: str,
        vcf: str,
        tracks: str,
        bed: str,
        length: int,
        samples: Optional[List[str]] = None,
        max_jitter: Optional[int] = None,
        rc_prob: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        _fasta = gvl.Fasta("haps", fasta, "N")
        _var = gvl.Variants.from_vcf(vcf)
        _tracks = gvl.ZarrTracks("track", tracks)
        self.haps = gvl.Haplotypes(_var, _fasta, _tracks)
        self.bed = _set_fixed_length_around_center(read_bedlike(bed), length)
        self.length = length
        self.samples = cast(List[str], _tracks.samples)
        if samples and (missing := set(samples).difference(self.samples)):
            raise ValueError(f"Samples {missing} not found in the dataset")
        elif samples:
            self.samples = list(set(samples).intersection(self.samples))
        self.max_jitter = max_jitter
        self.rc_prob = rc_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.bed.height * len(self.samples)

    def __getitem__(self, idx: int):
        region_idx, sample_idx = np.unravel_index(
            idx, (self.bed.height, len(self.samples))
        )
        sample = self.samples[sample_idx]
        chrom, start, end = self.bed.row(region_idx.item())
        if self.max_jitter is not None:
            start -= self.max_jitter
            end += self.max_jitter
        # (s=1 p l)
        data = self.haps.read(chrom, start, end, sample=[sample])
        haps = data["haps"]
        # (1 p l)
        track = data["track"]
        if self.max_jitter is not None:
            # (s p l)
            haps, track = sp.jitter(
                haps,
                track,
                max_jitter=self.max_jitter,
                length_axis=-1,
                jitter_axes=(0, 1),
                seed=self.rng.integers(np.iinfo(np.int64).max),
            )
        # (s p l a)
        haps = sp.DNA.ohe(haps)
        # (s p l)
        track = track[..., 524:1524]
        if self.rc_prob is not None and self.rng.random() < self.rc_prob:
            haps = sp.DNA.reverse_complement(haps, length_axis=-1, ohe_axis=-2)
            track = track[..., ::-1]
        haps = rearrange(haps, "1 p l a -> p a l").astype(np.float32)
        # tracks for each haplotype are the same because only SNPs are available (for now)
        # (1 p l) -> (1 l)
        data["track"] = track[0, 0].copy()
        return data


class Seq2Expression(SequenceDataset):
    _name_ = "seq2expression"

    def __init__(
        self,
        reference: str,
        vcf: Optional[str],
        tracks: str,
        bed: str,
        length: int,
        samples: Optional[List[str]] = None,
        max_jitter: Optional[int] = None,
        rc_prob: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        if vcf is not None:
            self.partial_ds = partial(
                HapsAndTracks,
                fasta=reference,
                vcf=vcf,
                tracks=tracks,
                length=length,
                samples=samples,
                max_jitter=max_jitter,
                rc_prob=rc_prob,
                seed=seed,
            )
        else:
            self.partial_ds = partial(
                RefAndTracks,
                fasta=reference,
                tracks=tracks,
                length=length,
                samples=samples,
                max_jitter=max_jitter,
                rc_prob=rc_prob,
                seed=seed,
            )
        self.beds = _set_fixed_length_around_center(
            read_bedlike(bed), length
        ).partition_by("split", as_dict=True, include_key=False)

    def setup(self):
        self.train_ds = self.partial_ds(bed=self.beds["train"])
        self.val_ds = self.partial_ds(bed=self.beds["val"])
        self.test_ds = self.partial_ds(bed=self.beds["test"])

    def train_dataloader(self, **kwargs):
        return DataLoader(self.train_ds, **kwargs)

    def val_dataloader(self, **kwargs):
        return DataLoader(self.val_ds, **kwargs)

    def test_dataloader(self, **kwargs):
        return DataLoader(self.test_ds, **kwargs)
