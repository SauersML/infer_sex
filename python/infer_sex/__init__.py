"""Infer genetic sex from variant data.

Pythonic port of the upstream Rust `infer_sex` crate. The algorithm and all
constants (PAR/non-PAR coordinates per build, decision thresholds, epsilon)
are kept byte-identical to the Rust reference so calls match across
languages.

Quick start
-----------

>>> from infer_sex import SexInferer, GenomeBuild, PlatformDefinition
>>> inferer = SexInferer(
...     build=GenomeBuild.BUILD38,
...     platform=PlatformDefinition(n_attempted_autosomes=2000, n_attempted_y_nonpar=1000),
... )
>>> result = inferer.infer_from_records(records)  # records yields (chrom, pos, is_het)
>>> result.final_call
<InferredSex.MALE: 'male'>
>>> result.report.composite_sex_index
2.41...

For streaming/online use, drop down to the accumulator:

>>> acc = inferer.accumulator()
>>> for chrom, pos, is_het in records:
...     acc.add(chrom, pos, is_het)
>>> result = acc.finish()

For high-throughput bulk processing, pass numpy arrays:

>>> acc.add_batch(chrom_codes, positions, is_het)  # vectorised, no Python loop
"""

from ._api import (
    SexInferer,
    SexInferenceAccumulator,
    InferenceConfig,
    InferenceResult,
    EvidenceReport,
    PlatformDefinition,
    DecisionThresholds,
    AlgorithmConstants,
    GenomeBuild,
    Chromosome,
    InferredSex,
    VariantInfo,
    InferenceError,
    InvalidPlatformCounts,
    ObservedExceedsAttempted,
    infer_from_records,
    infer_from_vcf,
    infer_from_plink,
)

__all__ = [
    "SexInferer",
    "SexInferenceAccumulator",
    "InferenceConfig",
    "InferenceResult",
    "EvidenceReport",
    "PlatformDefinition",
    "DecisionThresholds",
    "AlgorithmConstants",
    "GenomeBuild",
    "Chromosome",
    "InferredSex",
    "VariantInfo",
    "InferenceError",
    "InvalidPlatformCounts",
    "ObservedExceedsAttempted",
    "infer_from_records",
    "infer_from_vcf",
    "infer_from_plink",
]

__version__ = "0.1.0"
