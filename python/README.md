# infer_sex (Python)

Pythonic port of [`SauersML/infer_sex`](https://github.com/SauersML/infer_sex).
Counts heterozygous calls per chromosome region and applies a linear
decision boundary on `(x_autosome_het_ratio, y_genome_density)` to call
genetic sex. The algorithm + every numeric constant are kept byte-
identical to the Rust crate.

```python
from infer_sex import SexInferer, platform_from_bim

inferer = SexInferer(
    build="hg38",
    platform=platform_from_bim("/data/cohort.bim", build="hg38"),
)

# Pick your input — they're all type-checked and return the same shape:
result = inferer.infer_from_vcf("/data/sample.vcf.gz")
# or  inferer.infer_from_plink("/data/cohort")
# or  inferer.infer_from_records([("X", 100_000_000, True), ...])
# or  inferer.infer_from_arrays(chrom_codes, positions, is_het)

print(result.final_call)   # InferredSex.MALE / .FEMALE / .INDETERMINATE
print(result.report.composite_sex_index)
```

## Install

```bash
pip install infer_sex
```

Pure Python + numpy. No Rust toolchain required.

## Platform definitions

The algorithm normalises observed counts by the *attempted* counts on
the platform — pass them in via `PlatformDefinition`. Two helpers
compute them for you in one call:

```python
from infer_sex import platform_from_bim, platform_from_vcf

platform = platform_from_bim("/data/cohort.bim", build="hg38")
platform = platform_from_vcf("/data/cohort.vcf.gz", build="hg38")
```

These walk the file once, counting autosomal rows and Y-non-PAR rows.
Everything else (X, Y-PAR, MT, alt contigs) is ignored — exactly the
locus set the inference algorithm uses for normalisation.

If you already know the counts (e.g. from a manifest), construct
`PlatformDefinition` directly:

```python
from infer_sex import PlatformDefinition

platform = PlatformDefinition(
    n_attempted_autosomes=2_000,
    n_attempted_y_nonpar=1_000,
)
```

## Shortcuts: pass what you already know

`infer_sex` never touches the network. Skip build detection by passing
`build=` directly. Use custom decision thresholds (e.g. one fit on your
own labelled data) via `DecisionThresholds`:

```python
from infer_sex import DecisionThresholds

inferer = SexInferer(
    build="hg38",
    platform=PlatformDefinition(...),
    thresholds=DecisionThresholds(slope=0.30, intercept=0.25),
)
```

## Inputs

* `infer_from_vcf(path)` — `.vcf` / `.vcf.gz`. Multi-sample files
  default to the first column with a `UserWarning`; pass `sample=` to
  pick by ID (str) or 0-based index (int).
* `infer_from_plink(prefix)` — variant-major `.bed/.bim/.fam`. Pass
  `sample=` to pick a specific row of the FAM (string IID/FID or 0-based
  index). Reads via `np.memmap`; biobank-scale `.bed`s are fine.
* `infer_from_records(iterable)` — accepts `(chrom, pos, is_het)`
  triples. Useful when reading from a custom source.
* `infer_from_arrays(chrom, pos, is_het)` — parallel numpy arrays;
  ~10× faster than `infer_from_records` for the same data.

Missing genotypes (`./.` in VCF, `0b01` in PLINK) are dropped — they
don't count toward the denominator, matching the Rust crate.

## Streaming API

`SexInferenceAccumulator` is the streaming primitive:

```python
acc = inferer.accumulator()
for chrom, pos, is_het in my_stream:
    acc.add(chrom, pos, is_het)
# bulk path:
acc.add_batch(chrom_array, pos_array, is_het_array)
# any time:
print(acc.snapshot())       # raw counts, no classification
result = acc.finish()        # full inference; does not consume the accumulator
```

## Results

```python
result.final_call           # InferredSex enum
result.is_male / .is_female / .is_indeterminate
result.report.y_genome_density          # Optional[float]
result.report.x_autosome_het_ratio      # Optional[float]
result.report.composite_sex_index       # Optional[float]
result.report.auto_valid_count          # int (and many more counts)
result.report.as_dict()                  # plain dict, JSON-ready
```

## Errors

* `InvalidPlatformCounts` — the `PlatformDefinition` is unusable
  (e.g. zero autosomes).
* `ObservedExceedsAttempted` — more observations than the platform
  claims (your platform definition doesn't match the input stream).

Both subclass `InferenceError`.

## Cross-language guarantee

Every PAR/non-PAR coordinate, the `1e-9` epsilon, the default
`DecisionThresholds(slope=0.3566, intercept=0.2738)`, and the
classification formula are kept byte-identical to `src/lib.rs` in this
same repo. Calls match across languages — feed the same variant stream
to the Rust accumulator and the Python `SexInferer` and you get the
same `InferredSex`.
