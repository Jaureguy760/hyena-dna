from typing import Dict, Optional

import genvarloader as gvl
import numpy as np
import polars as pl
import seqpro as sp
from attrs import define
from numpy.typing import NDArray

from .base import SequenceDataset


def _tokenize(
    seq: NDArray[np.bytes_],
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

    def __call__(self, seqs: NDArray[np.bytes_]):
        _seqs = _tokenize(seqs, self.tokenize_table, self.add_eos, self.eos_id)
        return _seqs


NAME = "seq"
TOKENIZE = Tokenize(
    name=NAME,
    tokenize_table={b"A": 7, b"C": 8, b"G": 9, b"T": 10, b"N": 11},
    add_eos=True,
    eos_id=1,
)


@define
class Transform:
    flank_length: int
    rc_prob: Optional[float] = None
    rng: Optional[np.random.Generator] = None
    tokenizer: Tokenize = TOKENIZE

    def __call__(self, seqs: NDArray[np.bytes_], tracks: NDArray[np.float32]):
        batch_size = len(seqs)
        seq_len = seqs.shape[-1]
        if self.rc_prob is not None:
            if self.rng is None:
                self.rng = np.random.default_rng()
            to_rc = self.rng.random(batch_size) < self.rc_prob
            seqs[to_rc] = sp.DNA.reverse_complement(seqs[to_rc], -1)
            tracks[to_rc] = tracks[to_rc, ..., ::-1]

        _seqs = TOKENIZE(seqs)
        tracks = tracks[..., self.flank_length : self.flank_length + seq_len]
        return _seqs, tracks


class Seq2Expression(SequenceDataset):
    _name_ = "seq2expression"

    def __init__(
        self,
        gvl_path: str,
        reference: str,
        bed: str,
        samples: str,
        flank_length: int,
        with_haplotypes: bool = False,
        rc_prob: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        transform = Transform(flank_length, rc_prob, np.random.default_rng(seed))
        sequence_mode = "haplotypes" if with_haplotypes else "reference"
        self.ds = gvl.Dataset.open_with_settings(
            gvl_path, reference, sequence_mode=sequence_mode, transform=transform
        )
        self.beds = gvl.read_bedlike(bed).partition_by(
            "split", as_dict=True, include_key=False
        )
        self.samples = {
            split: df["sample"].to_list()
            for split, df in pl.read_csv(samples)
            .partition_by("split", as_dict=True, include_key=False)
            .items()
        }
        self.setup()

    def setup(self):
        self.train_ds = self.ds.subset_to(self.samples["train"], self.beds["train"])
        self.val_ds = self.ds.subset_to(self.samples["valid"], self.beds["valid"])
        self.test_ds = self.ds.subset_to(self.samples["test"], self.beds["test"])

    def train_dataloader(self, **kwargs):
        return self.train_ds.to_dataloader(**kwargs)

    def val_dataloader(self, **kwargs):
        return self.val_ds.to_dataloader(**kwargs)

    def test_dataloader(self, **kwargs):
        return self.test_ds.to_dataloader(**kwargs)
