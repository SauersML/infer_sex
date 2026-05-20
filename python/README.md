# infer_sex (Python)

Pythonic port of [`SauersML/infer_sex`](https://github.com/SauersML/infer_sex).

The algorithm and constants are kept byte-identical to the Rust reference
crate so calls match across languages.

## Install

```bash
pip install infer_sex
# or, from this directory:
pip install -e .
```

## Quick start

```python
from infer_sex import SexInferer, PlatformDefinition

inferer = SexInferer(
    build="hg38",
    platform=PlatformDefinition(
        n_attempted_autosomes=2_000,
        n_attempted_y_nonpar=1_000,
    ),
)

# Plain VCF / VCF.gz
result = inferer.infer_from_vcf("sample.vcf.gz")
print(result.final_call)            # InferredSex.MALE / .FEMALE / .INDETERMINATE
print(result.report.as_dict())

# Already streaming variants from somewhere else?
result = inferer.infer_from_records(
    (("1", pos, is_het) for pos, is_het in my_stream)
)

# High-throughput numpy batches
result = inferer.infer_from_arrays(chrom_codes, positions, is_het)
```

PAR/non-PAR coordinates, decision thresholds, and rounding behaviour all
match the upstream Rust crate exactly.

## Inputs

* VCF (.vcf / .vcf.gz) — single sample by default; pass `sample=...` to
  pick by ID or 0-based index.
* PLINK 1.9 fileset (variant-major `.bed/.bim/.fam`) — pass the prefix.
* Any iterable of `(chrom, pos, is_heterozygous)` triples.
* Parallel numpy arrays, when you want to avoid the per-variant Python overhead.

Missing genotypes (`./.` in VCF, `0b01` in PLINK) are dropped from the
denominators, matching the Rust convention.

## Errors

* `InvalidPlatformCounts` — the `PlatformDefinition` is unusable.
* `ObservedExceedsAttempted` — more observations than the platform claims.

Both subclass `InferenceError`.
