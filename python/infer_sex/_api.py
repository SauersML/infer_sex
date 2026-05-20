"""Core implementation of the infer_sex public API.

Mirrors the Rust crate `infer_sex` (see top-level `src/lib.rs` in this
repo) line-for-line on numerics. Differences:

* Pythonic `enum.Enum` + frozen `@dataclass`es replace the Rust enums/structs.
* Streaming `process_variant` is exposed as `add()`; a numpy-vectorised
  `add_batch()` is provided for high-throughput inputs.
* Convenience constructors read directly from VCF and PLINK fileset prefixes.
"""

from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

# Numbers below are taken verbatim from src/lib.rs::AlgorithmConstants::from_build
# Adjusting them here would diverge Python from Rust — don't, unless you also
# update the Rust crate.

_PAR_COORDS = {
    "build37": dict(
        par1_x=(60_001, 2_699_520),
        non_par_x=(2_699_521, 154_931_043),
        par2_x=(154_931_044, 155_260_560),
        par1_y=(10_001, 2_649_520),
        non_par_y=(2_649_521, 59_034_049),
        par2_y=(59_034_050, 59_363_566),
    ),
    "build38": dict(
        par1_x=(10_001, 2_781_479),
        non_par_x=(2_781_480, 155_701_382),
        par2_x=(155_701_383, 156_030_895),
        par1_y=(10_001, 2_781_479),
        non_par_y=(2_781_480, 56_887_902),
        par2_y=(56_887_903, 57_217_415),
    ),
}
_EPSILON = 1e-9
_DEFAULT_SLOPE = 0.3566
_DEFAULT_INTERCEPT = 0.2738


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------


class GenomeBuild(str, Enum):
    """Genome build whose PAR/non-PAR coordinates the inference will use."""

    BUILD37 = "build37"
    BUILD38 = "build38"

    @classmethod
    def parse(cls, value: Union[str, "GenomeBuild"]) -> "GenomeBuild":
        if isinstance(value, cls):
            return value
        norm = str(value).strip().lower().replace("-", "").replace("_", "")
        aliases = {
            "build37": cls.BUILD37,
            "37": cls.BUILD37,
            "hg19": cls.BUILD37,
            "grch37": cls.BUILD37,
            "build38": cls.BUILD38,
            "38": cls.BUILD38,
            "hg38": cls.BUILD38,
            "grch38": cls.BUILD38,
        }
        try:
            return aliases[norm]
        except KeyError:
            raise ValueError(
                f"Unknown genome build {value!r}; expected one of "
                "'build37'/'hg19'/'grch37' or 'build38'/'hg38'/'grch38'."
            ) from None


class Chromosome(str, Enum):
    """Coarse chromosome classes used by the inference algorithm."""

    AUTOSOME = "autosome"
    X = "x"
    Y = "y"


# Numeric encoding for the vectorised batch path. Keep stable: end-users may
# build batches against these codes.
_CHROM_CODE_AUTO = 0
_CHROM_CODE_X = 1
_CHROM_CODE_Y = 2

class InferredSex(str, Enum):
    """Final classification produced by `SexInferenceAccumulator.finish`."""

    MALE = "male"
    FEMALE = "female"
    INDETERMINATE = "indeterminate"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InferenceError(Exception):
    """Base class for all `infer_sex` errors."""


class InvalidPlatformCounts(InferenceError, ValueError):
    """The supplied `PlatformDefinition` is unusable (e.g. zero autosomes)."""


class ObservedExceedsAttempted(InferenceError, ValueError):
    """More variants were observed than the platform definition allows."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlatformDefinition:
    """Platform-level counts used to normalise observed evidence.

    Both numbers refer to the *attempted* loci that will be streamed in.
    Pre-scan your variant manifest (BIM, VCF, etc.) to populate this — the
    library cannot infer them from the variant stream.
    """

    n_attempted_autosomes: int
    n_attempted_y_nonpar: int

    def __post_init__(self) -> None:
        for name in ("n_attempted_autosomes", "n_attempted_y_nonpar"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise InvalidPlatformCounts(
                    f"{name} must be a non-negative integer, got {value!r}"
                )


@dataclass(frozen=True)
class DecisionThresholds:
    """Linear decision boundary for binary Male/Female calls.

    The Rust defaults (slope=0.3566, intercept=0.2738) are the result of a
    fit on real microarray data — change only with empirical evidence.
    """

    slope: float = _DEFAULT_SLOPE
    intercept: float = _DEFAULT_INTERCEPT


@dataclass(frozen=True)
class AlgorithmConstants:
    """PAR/non-PAR ranges for a given build."""

    par1_x: Tuple[int, int]
    non_par_x: Tuple[int, int]
    par2_x: Tuple[int, int]
    par1_y: Tuple[int, int]
    non_par_y: Tuple[int, int]
    par2_y: Tuple[int, int]
    epsilon: float = _EPSILON

    @classmethod
    def from_build(cls, build: Union[str, GenomeBuild]) -> "AlgorithmConstants":
        b = GenomeBuild.parse(build)
        return cls(**_PAR_COORDS[b.value])

    def is_in_x_par(self, pos: int) -> bool:
        return (
            self.par1_x[0] <= pos <= self.par1_x[1]
            or self.par2_x[0] <= pos <= self.par2_x[1]
        )

    def is_in_x_non_par(self, pos: int) -> bool:
        return self.non_par_x[0] <= pos <= self.non_par_x[1]

    def is_in_y_par(self, pos: int) -> bool:
        return (
            self.par1_y[0] <= pos <= self.par1_y[1]
            or self.par2_y[0] <= pos <= self.par2_y[1]
        )

    def is_in_y_non_par(self, pos: int) -> bool:
        return self.non_par_y[0] <= pos <= self.non_par_y[1]


@dataclass(frozen=True)
class VariantInfo:
    """A single observed (i.e. non-missing) variant call."""

    chrom: Chromosome
    pos: int
    is_heterozygous: bool

    def __post_init__(self) -> None:
        if not isinstance(self.chrom, Chromosome):
            object.__setattr__(self, "chrom", _coerce_chromosome(self.chrom))
        if not isinstance(self.pos, (int, np.integer)) or self.pos < 0:
            raise ValueError(f"pos must be a non-negative integer, got {self.pos!r}")
        object.__setattr__(self, "is_heterozygous", bool(self.is_heterozygous))


@dataclass(frozen=True)
class InferenceConfig:
    build: GenomeBuild
    platform: PlatformDefinition
    thresholds: Optional[DecisionThresholds] = None


@dataclass
class EvidenceReport:
    """Detailed counts and ratios produced by the algorithm.

    All `*_count` fields are observed counts. Ratios are `None` when not
    computable (typically because a denominator was zero).
    """

    y_genome_density: Optional[float] = None
    x_autosome_het_ratio: Optional[float] = None
    composite_sex_index: Optional[float] = None

    auto_valid_count: int = 0
    auto_het_count: int = 0
    x_non_par_valid_count: int = 0
    x_non_par_het_count: int = 0
    x_par_valid_count: int = 0
    x_par_het_count: int = 0
    y_non_par_valid_count: int = 0
    y_par_valid_count: int = 0

    def as_dict(self) -> dict:
        """Return a plain dict — handy for JSON / DataFrame round-trips."""
        return {
            "y_genome_density": self.y_genome_density,
            "x_autosome_het_ratio": self.x_autosome_het_ratio,
            "composite_sex_index": self.composite_sex_index,
            "auto_valid_count": self.auto_valid_count,
            "auto_het_count": self.auto_het_count,
            "x_non_par_valid_count": self.x_non_par_valid_count,
            "x_non_par_het_count": self.x_non_par_het_count,
            "x_par_valid_count": self.x_par_valid_count,
            "x_par_het_count": self.x_par_het_count,
            "y_non_par_valid_count": self.y_non_par_valid_count,
            "y_par_valid_count": self.y_par_valid_count,
        }


@dataclass(frozen=True)
class InferenceResult:
    final_call: InferredSex
    report: EvidenceReport

    @property
    def is_male(self) -> bool:
        return self.final_call is InferredSex.MALE

    @property
    def is_female(self) -> bool:
        return self.final_call is InferredSex.FEMALE

    @property
    def is_indeterminate(self) -> bool:
        return self.final_call is InferredSex.INDETERMINATE


# ---------------------------------------------------------------------------
# Chromosome coercion
# ---------------------------------------------------------------------------


def _coerce_chromosome(value) -> Chromosome:
    if isinstance(value, Chromosome):
        return value
    if isinstance(value, (int, np.integer)):
        if value == _CHROM_CODE_AUTO:
            return Chromosome.AUTOSOME
        if value == _CHROM_CODE_X:
            return Chromosome.X
        if value == _CHROM_CODE_Y:
            return Chromosome.Y
        raise ValueError(
            f"chromosome code {value!r} is not one of 0 (autosome), 1 (X), 2 (Y)"
        )
    s = str(value).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    s_upper = s.upper()
    if s_upper == "X":
        return Chromosome.X
    if s_upper == "Y":
        return Chromosome.Y
    if s_upper in {"M", "MT", "MITO", "CHRM"}:
        # Mitochondrial isn't used by the algorithm; map to autosome so it's
        # counted as a "valid" callability sample. Callers who care should
        # filter MT upstream.
        return Chromosome.AUTOSOME
    if s_upper.isdigit():
        n = int(s_upper)
        if 1 <= n <= 22:
            return Chromosome.AUTOSOME
    raise ValueError(f"Cannot interpret {value!r} as a chromosome.")


def _chromosome_codes_from_strings(values: Sequence[str]) -> np.ndarray:
    """Vectorise common chromosome string formats to internal codes."""
    arr = np.asarray(values, dtype=object)
    out = np.full(arr.shape, -1, dtype=np.int8)
    # Normalise: strip leading "chr", uppercase.
    for i, raw in enumerate(arr):
        s = str(raw)
        if s.lower().startswith("chr"):
            s = s[3:]
        u = s.upper()
        if u == "X":
            out[i] = _CHROM_CODE_X
        elif u == "Y":
            out[i] = _CHROM_CODE_Y
        elif u in ("M", "MT", "MITO"):
            out[i] = _CHROM_CODE_AUTO
        elif u.isdigit() and 1 <= int(u) <= 22:
            out[i] = _CHROM_CODE_AUTO
        else:
            # Ignore unknown contigs (e.g. ALT/decoys).
            out[i] = -1
    return out


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class SexInferenceAccumulator:
    """Streaming accumulator — mirrors `SexInferenceAccumulator` in Rust.

    Build one via `SexInferer.accumulator()` (or directly from a config),
    feed variants in via `add()` / `add_batch()` / `add_variants()`, and
    finalise with `finish()`.
    """

    __slots__ = ("_config", "_constants", "_counts")

    def __init__(self, config: InferenceConfig) -> None:
        if not isinstance(config, InferenceConfig):
            raise TypeError(f"config must be an InferenceConfig, got {type(config).__name__}")
        self._config = config
        self._constants = AlgorithmConstants.from_build(config.build)
        # 8-slot int64 vector for SIMD-friendly accumulation:
        # [auto_valid, auto_het, x_non_par_valid, x_non_par_het,
        #  x_par_valid, x_par_het, y_non_par_valid, y_par_valid]
        self._counts = np.zeros(8, dtype=np.int64)

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def add(self, chrom, pos: int, is_heterozygous: bool) -> "SexInferenceAccumulator":
        """Add one observation. Returns self for chaining."""
        chrom_obj = _coerce_chromosome(chrom)
        pos_int = int(pos)
        het = 1 if is_heterozygous else 0
        c = self._constants
        if chrom_obj is Chromosome.AUTOSOME:
            self._counts[0] += 1
            self._counts[1] += het
        elif chrom_obj is Chromosome.X:
            if c.is_in_x_par(pos_int):
                self._counts[4] += 1
                self._counts[5] += het
            elif c.is_in_x_non_par(pos_int):
                self._counts[2] += 1
                self._counts[3] += het
        else:  # Y
            if c.is_in_y_par(pos_int):
                self._counts[7] += 1
            elif c.is_in_y_non_par(pos_int):
                self._counts[6] += 1
        return self

    def add_variants(self, variants: Iterable[VariantInfo]) -> "SexInferenceAccumulator":
        """Add an iterable of `VariantInfo`s. Returns self."""
        for v in variants:
            self.add(v.chrom, v.pos, v.is_heterozygous)
        return self

    def add_batch(
        self,
        chrom,
        pos,
        is_heterozygous,
    ) -> "SexInferenceAccumulator":
        """Add a batch of observations via numpy. Returns self.

        ``chrom`` may be:
          * an int array using codes 0=autosome, 1=X, 2=Y (-1 = ignore), or
          * an array/sequence of strings ('1'..'22', 'X', 'Y', 'chrX', ...).

        ``pos`` and ``is_heterozygous`` must be array-likes of the same
        length. Unknown chromosome labels are silently skipped (consistent
        with the Rust crate, which simply ignores anything that isn't an
        autosome, X, or Y).
        """
        pos = np.asarray(pos, dtype=np.int64)
        het = np.asarray(is_heterozygous, dtype=bool)

        chrom_arr = np.asarray(chrom)
        if chrom_arr.dtype.kind in ("U", "S", "O"):
            codes = _chromosome_codes_from_strings(chrom_arr)
        else:
            codes = np.asarray(chrom, dtype=np.int8)

        if pos.shape != codes.shape or het.shape != codes.shape:
            raise ValueError(
                "chrom, pos, and is_heterozygous must all have the same length "
                f"(got {codes.shape}, {pos.shape}, {het.shape})"
            )

        c = self._constants
        het_i = het.astype(np.int64)

        # Autosomes
        auto_mask = codes == _CHROM_CODE_AUTO
        self._counts[0] += int(auto_mask.sum())
        self._counts[1] += int(het_i[auto_mask].sum())

        # X
        x_mask = codes == _CHROM_CODE_X
        if x_mask.any():
            xp = pos[x_mask]
            xh = het_i[x_mask]
            par_mask = (
                ((xp >= c.par1_x[0]) & (xp <= c.par1_x[1]))
                | ((xp >= c.par2_x[0]) & (xp <= c.par2_x[1]))
            )
            non_par_mask = (xp >= c.non_par_x[0]) & (xp <= c.non_par_x[1]) & ~par_mask
            self._counts[4] += int(par_mask.sum())
            self._counts[5] += int(xh[par_mask].sum())
            self._counts[2] += int(non_par_mask.sum())
            self._counts[3] += int(xh[non_par_mask].sum())

        # Y
        y_mask = codes == _CHROM_CODE_Y
        if y_mask.any():
            yp = pos[y_mask]
            par_mask = (
                ((yp >= c.par1_y[0]) & (yp <= c.par1_y[1]))
                | ((yp >= c.par2_y[0]) & (yp <= c.par2_y[1]))
            )
            non_par_mask = (yp >= c.non_par_y[0]) & (yp <= c.non_par_y[1]) & ~par_mask
            self._counts[7] += int(par_mask.sum())
            self._counts[6] += int(non_par_mask.sum())

        return self

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def snapshot(self) -> EvidenceReport:
        """Return current raw counts (does not run the classification)."""
        return EvidenceReport(
            auto_valid_count=int(self._counts[0]),
            auto_het_count=int(self._counts[1]),
            x_non_par_valid_count=int(self._counts[2]),
            x_non_par_het_count=int(self._counts[3]),
            x_par_valid_count=int(self._counts[4]),
            x_par_het_count=int(self._counts[5]),
            y_non_par_valid_count=int(self._counts[6]),
            y_par_valid_count=int(self._counts[7]),
        )

    def finish(self) -> InferenceResult:
        """Compute the final inference. Does not consume the accumulator."""
        platform = self._config.platform
        c = self._constants
        if platform.n_attempted_autosomes == 0:
            raise InvalidPlatformCounts("n_attempted_autosomes must be > 0")

        report = self.snapshot()

        if report.auto_valid_count > platform.n_attempted_autosomes:
            raise ObservedExceedsAttempted(
                "observed autosomal variants exceed platform definition "
                f"({report.auto_valid_count} > {platform.n_attempted_autosomes})"
            )
        if report.y_non_par_valid_count > platform.n_attempted_y_nonpar:
            raise ObservedExceedsAttempted(
                "observed Y non-PAR variants exceed platform definition "
                f"({report.y_non_par_valid_count} > {platform.n_attempted_y_nonpar})"
            )

        total_sex_observed = (
            report.x_non_par_valid_count
            + report.x_par_valid_count
            + report.y_non_par_valid_count
            + report.y_par_valid_count
        )
        if total_sex_observed == 0:
            return InferenceResult(final_call=InferredSex.INDETERMINATE, report=report)

        y_density: Optional[float]
        if report.auto_valid_count == 0 or platform.n_attempted_y_nonpar == 0:
            y_density = None
        else:
            auto_rate = (report.auto_valid_count + c.epsilon) / platform.n_attempted_autosomes
            y_rate = report.y_non_par_valid_count / platform.n_attempted_y_nonpar
            y_density = y_rate / auto_rate

        x_auto_ratio: Optional[float]
        if report.auto_valid_count == 0 or report.x_non_par_valid_count == 0:
            x_auto_ratio = None
        else:
            auto_het_rate = report.auto_het_count / (report.auto_valid_count + c.epsilon)
            x_het_rate = report.x_non_par_het_count / (report.x_non_par_valid_count + c.epsilon)
            x_auto_ratio = x_het_rate / (auto_het_rate + c.epsilon)

        if y_density is not None and x_auto_ratio is not None:
            composite = y_density / (x_auto_ratio + c.epsilon)
        else:
            composite = None

        report.y_genome_density = y_density
        report.x_autosome_het_ratio = x_auto_ratio
        report.composite_sex_index = composite

        thresholds = self._config.thresholds or DecisionThresholds()
        final_call = _classify(y_density, x_auto_ratio, thresholds)
        return InferenceResult(final_call=final_call, report=report)


def _classify(
    y_density: Optional[float],
    x_auto_ratio: Optional[float],
    thresholds: DecisionThresholds,
) -> InferredSex:
    y = y_density if y_density is not None else 0.0
    x = x_auto_ratio if x_auto_ratio is not None else 0.0
    threshold = thresholds.slope * x + thresholds.intercept
    return InferredSex.MALE if y > threshold else InferredSex.FEMALE


# ---------------------------------------------------------------------------
# High-level façade
# ---------------------------------------------------------------------------


class SexInferer:
    """Convenience façade over `SexInferenceAccumulator`.

    Most users only need this class. Construct once per dataset/platform and
    call one of the `infer_from_*` methods to get an `InferenceResult`.
    """

    def __init__(
        self,
        build: Union[str, GenomeBuild],
        platform: PlatformDefinition,
        thresholds: Optional[DecisionThresholds] = None,
    ) -> None:
        self.config = InferenceConfig(
            build=GenomeBuild.parse(build),
            platform=platform,
            thresholds=thresholds,
        )

    def accumulator(self) -> SexInferenceAccumulator:
        return SexInferenceAccumulator(self.config)

    def infer_from_records(
        self, records: Iterable[Tuple[Union[str, int, Chromosome], int, bool]]
    ) -> InferenceResult:
        """Run inference on an iterable of `(chrom, pos, is_het)` triples."""
        acc = self.accumulator()
        for chrom, pos, is_het in records:
            acc.add(chrom, pos, is_het)
        return acc.finish()

    def infer_from_variants(self, variants: Iterable[VariantInfo]) -> InferenceResult:
        acc = self.accumulator()
        acc.add_variants(variants)
        return acc.finish()

    def infer_from_arrays(
        self, chrom, pos, is_heterozygous
    ) -> InferenceResult:
        """Run inference on parallel arrays (numpy-accelerated)."""
        acc = self.accumulator()
        acc.add_batch(chrom, pos, is_heterozygous)
        return acc.finish()

    def infer_from_vcf(
        self,
        vcf_path: Union[str, os.PathLike],
        *,
        sample: Optional[Union[str, int]] = None,
        chunk_size: int = 50_000,
    ) -> InferenceResult:
        """Infer from a single-sample VCF (.vcf or .vcf.gz).

        If the VCF contains multiple samples, pass ``sample=`` with either
        the sample ID (str) or its 0-based column index (int).
        """
        acc = self.accumulator()
        for codes, positions, hets in _stream_vcf(vcf_path, sample, chunk_size):
            acc.add_batch(codes, positions, hets)
        return acc.finish()

    def infer_from_plink(
        self,
        prefix: Union[str, os.PathLike],
        *,
        sample: Union[str, int] = 0,
    ) -> InferenceResult:
        """Infer from a PLINK 1.9 fileset (``<prefix>.bed/.bim/.fam``).

        ``sample`` is either a sample IID/FID string (matches FAM column 2,
        falls back to column 1) or a 0-based index into the FAM file.
        """
        acc = self.accumulator()
        for codes, positions, hets in _stream_plink(prefix, sample):
            acc.add_batch(codes, positions, hets)
        return acc.finish()


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def infer_from_records(
    records: Iterable[Tuple[Union[str, int, Chromosome], int, bool]],
    *,
    build: Union[str, GenomeBuild],
    platform: PlatformDefinition,
    thresholds: Optional[DecisionThresholds] = None,
) -> InferenceResult:
    return SexInferer(build, platform, thresholds).infer_from_records(records)


def infer_from_vcf(
    vcf_path: Union[str, os.PathLike],
    *,
    build: Union[str, GenomeBuild],
    platform: PlatformDefinition,
    thresholds: Optional[DecisionThresholds] = None,
    sample: Optional[Union[str, int]] = None,
    chunk_size: int = 50_000,
) -> InferenceResult:
    return SexInferer(build, platform, thresholds).infer_from_vcf(
        vcf_path, sample=sample, chunk_size=chunk_size
    )


def infer_from_plink(
    prefix: Union[str, os.PathLike],
    *,
    build: Union[str, GenomeBuild],
    platform: PlatformDefinition,
    thresholds: Optional[DecisionThresholds] = None,
    sample: Union[str, int] = 0,
) -> InferenceResult:
    return SexInferer(build, platform, thresholds).infer_from_plink(prefix, sample=sample)


# ---------------------------------------------------------------------------
# VCF streaming
# ---------------------------------------------------------------------------


def _open_text(path: Union[str, os.PathLike]) -> io.TextIOBase:
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")


def _stream_vcf(
    vcf_path: Union[str, os.PathLike],
    sample: Optional[Union[str, int]],
    chunk_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (codes, positions, is_het) numpy chunks from a VCF.

    Heterozygous = GT alleles are not all equal. Missing GTs ('.') are
    treated as non-observations and dropped.
    """
    chrom_buf: list = []
    pos_buf: list = []
    het_buf: list = []
    sample_idx: Optional[int] = None

    with _open_text(vcf_path) as f:
        for raw in f:
            if not raw:
                continue
            if raw.startswith("##"):
                continue
            if raw.startswith("#CHROM"):
                header = raw.rstrip("\n").split("\t")
                # Format columns 0..8 are fixed; samples start at index 9.
                samples = header[9:]
                if not samples:
                    raise ValueError(
                        f"{vcf_path}: VCF has no sample columns; cannot infer genotypes."
                    )
                if sample is None:
                    sample_idx = 9
                    if len(samples) > 1:
                        # Default to first sample but tell the user.
                        # (Doesn't raise; surface via a clear behaviour.)
                        pass
                elif isinstance(sample, int):
                    if not (0 <= sample < len(samples)):
                        raise IndexError(
                            f"sample index {sample} out of range (file has {len(samples)} samples)"
                        )
                    sample_idx = 9 + sample
                else:
                    try:
                        sample_idx = 9 + samples.index(sample)
                    except ValueError:
                        raise KeyError(
                            f"sample {sample!r} not found in VCF (available: {samples})"
                        ) from None
                continue
            if sample_idx is None:
                raise ValueError(f"{vcf_path}: VCF missing #CHROM header line.")
            fields = raw.rstrip("\n").split("\t")
            if len(fields) <= sample_idx:
                continue
            chrom_buf.append(fields[0])
            try:
                pos_buf.append(int(fields[1]))
            except ValueError:
                # malformed line; skip
                chrom_buf.pop()
                continue
            fmt = fields[8].split(":") if len(fields) > 8 else ["GT"]
            try:
                gt_field_idx = fmt.index("GT")
            except ValueError:
                chrom_buf.pop()
                pos_buf.pop()
                continue
            gt_raw = fields[sample_idx].split(":")[gt_field_idx]
            het = _parse_gt_is_het(gt_raw)
            if het is None:
                chrom_buf.pop()
                pos_buf.pop()
                continue
            het_buf.append(het)
            if len(chrom_buf) >= chunk_size:
                yield (
                    np.asarray(chrom_buf, dtype=object),
                    np.asarray(pos_buf, dtype=np.int64),
                    np.asarray(het_buf, dtype=bool),
                )
                chrom_buf.clear()
                pos_buf.clear()
                het_buf.clear()
    if chrom_buf:
        yield (
            np.asarray(chrom_buf, dtype=object),
            np.asarray(pos_buf, dtype=np.int64),
            np.asarray(het_buf, dtype=bool),
        )


def _parse_gt_is_het(gt: str) -> Optional[bool]:
    """Parse a VCF GT field. Returns None for missing, else bool het."""
    if not gt or gt in (".", "./.", ".|."):
        return None
    # Split on either phased or unphased separators.
    parts = gt.replace("|", "/").split("/")
    alleles = []
    for p in parts:
        if p == ".":
            return None
        try:
            alleles.append(int(p))
        except ValueError:
            return None
    if not alleles:
        return None
    return any(a != alleles[0] for a in alleles[1:])


# ---------------------------------------------------------------------------
# PLINK streaming
# ---------------------------------------------------------------------------


def _stream_plink(
    prefix: Union[str, os.PathLike], sample: Union[str, int]
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    prefix = Path(prefix)
    bed = prefix.with_suffix(".bed") if prefix.suffix == "" else Path(str(prefix) + ".bed")
    bim = prefix.with_suffix(".bim") if prefix.suffix == "" else Path(str(prefix) + ".bim")
    fam = prefix.with_suffix(".fam") if prefix.suffix == "" else Path(str(prefix) + ".fam")
    # If the user passed an already-suffixed path, the construction above
    # produces silly names; recover by stripping known suffixes.
    if not bed.exists() and prefix.suffix in {".bed", ".bim", ".fam"}:
        base = prefix.with_suffix("")
        bed, bim, fam = base.with_suffix(".bed"), base.with_suffix(".bim"), base.with_suffix(".fam")
    for p in (bed, bim, fam):
        if not p.exists():
            raise FileNotFoundError(p)

    # Resolve sample index from FAM.
    with open(fam, "r") as f:
        fam_rows = [line.rstrip("\n").split() for line in f if line.strip()]
    if isinstance(sample, int):
        if not (0 <= sample < len(fam_rows)):
            raise IndexError(f"sample index {sample} out of range; FAM has {len(fam_rows)} rows")
        sample_idx = sample
    else:
        sample_idx = None
        for i, row in enumerate(fam_rows):
            if len(row) >= 2 and (row[1] == sample or row[0] == sample):
                sample_idx = i
                break
        if sample_idx is None:
            raise KeyError(f"sample {sample!r} not found in {fam}")

    n_samples = len(fam_rows)
    bytes_per_variant = (n_samples + 3) // 4
    byte_in_variant = sample_idx // 4
    bit_in_byte = (sample_idx % 4) * 2

    # Parse BIM positions.
    bim_chrom: list = []
    bim_pos: list = []
    with open(bim, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split()
            if len(parts) < 4:
                continue
            bim_chrom.append(parts[0])
            bim_pos.append(int(parts[3]))
    n_variants = len(bim_chrom)
    if n_variants == 0:
        return

    with open(bed, "rb") as f:
        magic = f.read(3)
        if magic[:2] != b"\x6c\x1b":
            raise ValueError(f"{bed}: not a PLINK .bed file (bad magic)")
        if magic[2] != 0x01:
            raise ValueError(
                f"{bed}: only variant-major .bed files are supported (magic byte {magic[2]:#x})"
            )

        # Read only the byte of interest per variant.
        chunk = 50_000
        codes_chunk = np.empty(chunk, dtype=object)
        pos_chunk = np.empty(chunk, dtype=np.int64)
        het_chunk = np.empty(chunk, dtype=bool)
        idx = 0
        for v in range(n_variants):
            f.seek(3 + v * bytes_per_variant + byte_in_variant)
            b = f.read(1)
            if not b:
                break
            gt = (b[0] >> bit_in_byte) & 0b11
            # PLINK encoding: 00 hom A1, 01 missing, 10 het, 11 hom A2.
            if gt == 0b01:
                continue
            is_het = gt == 0b10
            codes_chunk[idx] = bim_chrom[v]
            pos_chunk[idx] = bim_pos[v]
            het_chunk[idx] = is_het
            idx += 1
            if idx == chunk:
                yield codes_chunk[:idx].copy(), pos_chunk[:idx].copy(), het_chunk[:idx].copy()
                idx = 0
        if idx:
            yield codes_chunk[:idx].copy(), pos_chunk[:idx].copy(), het_chunk[:idx].copy()
