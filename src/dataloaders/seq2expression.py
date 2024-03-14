from typing import List, Optional, cast, Dict

from attrs import define
from functools import partial
import genvarloader as gvl
import numpy as np
from numpy.typing import NDArray
import seqpro as sp
from einops import rearrange
from genvarloader.util import _set_fixed_length_around_center, read_bedlike
from torch.utils.data import Dataset, DataLoader

from .base import SequenceDataset


def _tokenize(
    seq: NDArray,
    tokenize_table: Dict[bytes, int],
    add_eos: bool,
    eos_id: int,
    dtype=np.int32,
):
    length_axis = seq.ndim - 1
    
    if add_eos:
        shape = seq.shape[:length_axis] + (seq.shape[length_axis] + 1,)
        tokenized = np.empty(shape, dtype=dtype)
        tokenized[..., -1] = eos_id
        _tokenized = tokenized[..., :-1]
    else:
        tokenized = np.empty_like(seq, dtype=dtype)
        _tokenized = tokenized
        
    for nuc, id in tokenize_table.items():
        _tokenized[seq == nuc] = id
    
    return tokenized


@define
class Tokenize:
    name: str
    tokenize_table: Dict[bytes, int]
    add_eos: bool
    eos_id: int

    def __call__(self, batch: Dict[str, NDArray]):
        seq = _tokenize(
            batch[self.name], self.tokenize_table, self.add_eos, self.eos_id
        )
        return seq


NAME = "seq"
TOKENIZE = Tokenize(
    name=NAME,
    tokenize_table={b"A": 7, b"C": 8, b"G": 9, b"T": 10, b"N": 11},
    add_eos=True,
    eos_id=1,
)


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
        self.tokenize = TOKENIZE

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
        # (l)
        track = track[..., self.flank_length : track.shape[-1] - self.flank_length]
        if self.rc_prob is not None and self.rng.random() < self.rc_prob:
            haps = sp.DNA.reverse_complement(haps, length_axis=-1)
            track = track[::-1]
        haps = self.tokenize(rearrange(haps, "l -> 1 l"))
        return haps, track.copy()


class HapsAndTracks(Dataset):
    def __init__(
        self,
        fasta: str,
        vcf: str,
        tracks: str,
        bed: str,
        length: int,
        flank_length: int = 0,
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
        self.flank_length = flank_length
        self.tokenize = TOKENIZE

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
        # (s p l)
        track = track[..., self.flank_length : track.shape[-1] - self.flank_length]
        if self.rc_prob is not None and self.rng.random() < self.rc_prob:
            haps = sp.DNA.reverse_complement(haps, length_axis=-1)
            track = track[..., ::-1]
        haps = self.tokenize(rearrange(haps, "1 p l -> p l"))
        # tracks for each haplotype are the same because only SNPs are available (for now)
        # (1 p l) -> (1 l)
        return haps, track[0, 0].copy()


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
