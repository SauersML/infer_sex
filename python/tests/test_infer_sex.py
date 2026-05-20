"""Tests for the Python port of `infer_sex`.

The numeric cases below mirror the Rust integration tests in
`src/lib.rs` so that any divergence would be caught by either suite.
"""

from __future__ import annotations

import gzip
import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from infer_sex import (
    AlgorithmConstants,
    Chromosome,
    DecisionThresholds,
    EvidenceReport,
    GenomeBuild,
    InferenceConfig,
    InferredSex,
    InvalidPlatformCounts,
    ObservedExceedsAttempted,
    PlatformDefinition,
    SexInferenceAccumulator,
    SexInferer,
    VariantInfo,
    infer_from_records,
    infer_from_plink,
    infer_from_vcf,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def cfg38(n_attempted_autosomes=2_000, n_attempted_y_nonpar=1_000, thresholds=None):
    return InferenceConfig(
        build=GenomeBuild.BUILD38,
        platform=PlatformDefinition(
            n_attempted_autosomes=n_attempted_autosomes,
            n_attempted_y_nonpar=n_attempted_y_nonpar,
        ),
        thresholds=thresholds or DecisionThresholds(),
    )


# Reusable position generators — placed once per chromosome category.
B38 = AlgorithmConstants.from_build(GenomeBuild.BUILD38)
B37 = AlgorithmConstants.from_build(GenomeBuild.BUILD37)
X_NON_PAR_POS = B38.non_par_x[0] + 100
X_PAR_POS = B38.par1_x[0] + 100
Y_NON_PAR_POS = B38.non_par_y[0] + 100
Y_PAR_POS = B38.par1_y[0] + 100


# ---------------------------------------------------------------------------
# Build/parse helpers
# ---------------------------------------------------------------------------


def test_genome_build_parses_aliases():
    for alias in ["hg19", "GRCh37", "build37", "BUILD37", "37", "Hg19"]:
        assert GenomeBuild.parse(alias) is GenomeBuild.BUILD37
    for alias in ["hg38", "GRCh38", "build38", "Build_38", "38", "grch-38"]:
        assert GenomeBuild.parse(alias) is GenomeBuild.BUILD38


def test_genome_build_rejects_unknown():
    with pytest.raises(ValueError):
        GenomeBuild.parse("hg42")


def test_algorithm_constants_classify_par_correctly():
    c = B38
    assert c.is_in_x_par(c.par1_x[0]) and c.is_in_x_par(c.par1_x[1])
    assert not c.is_in_x_par(c.non_par_x[0])
    assert c.is_in_x_non_par(c.non_par_x[0]) and c.is_in_x_non_par(c.non_par_x[1])
    assert c.is_in_y_par(c.par2_y[1])
    assert c.is_in_y_non_par(c.non_par_y[0])


# ---------------------------------------------------------------------------
# Accumulator semantics — match Rust integration tests
# ---------------------------------------------------------------------------


def test_female_sample_calls_female():
    acc = SexInferenceAccumulator(cfg38())
    # Autosomes: solid callability, moderate heterozygosity.
    for i in range(1_500):
        acc.add(Chromosome.AUTOSOME, 1_000_000 + i, i % 4 == 0)  # ~25% het
    # X non-PAR with high het rate.
    for i in range(500):
        acc.add(Chromosome.X, X_NON_PAR_POS + i, True)
    # No Y variants → low Y density.
    result = acc.finish()
    assert result.final_call is InferredSex.FEMALE
    assert result.report.x_autosome_het_ratio is not None
    assert result.report.y_genome_density == 0.0


def test_male_sample_calls_male():
    acc = SexInferenceAccumulator(cfg38())
    for i in range(1_500):
        acc.add(Chromosome.AUTOSOME, 1_000_000 + i, i % 4 == 0)
    # Few X non-PAR variants, all homozygous.
    for i in range(20):
        acc.add(Chromosome.X, X_NON_PAR_POS + i, False)
    # Y non-PAR signal.
    for i in range(700):
        acc.add(Chromosome.Y, Y_NON_PAR_POS + i, False)
    result = acc.finish()
    assert result.final_call is InferredSex.MALE
    assert result.report.y_genome_density and result.report.y_genome_density > 0


def test_par_variants_do_not_influence_non_par_metrics():
    acc = SexInferenceAccumulator(cfg38())
    for i in range(1_500):
        acc.add(Chromosome.AUTOSOME, 1_000_000 + i, i % 4 == 0)
    for i in range(500):
        acc.add(Chromosome.X, X_PAR_POS + i, True)  # PAR
    for i in range(500):
        acc.add(Chromosome.Y, Y_PAR_POS + i, False)  # PAR
    result = acc.finish()
    # No non-PAR observations → ratios should reflect that.
    assert result.report.x_non_par_valid_count == 0
    assert result.report.y_non_par_valid_count == 0
    assert result.report.x_autosome_het_ratio is None
    assert result.report.y_genome_density == 0.0


def test_indeterminate_when_only_autosomes_seen():
    acc = SexInferenceAccumulator(cfg38())
    for i in range(500):
        acc.add(Chromosome.AUTOSOME, 1_000_000 + i, False)
    result = acc.finish()
    assert result.final_call is InferredSex.INDETERMINATE


def test_metrics_are_invariant_to_uniform_scaling():
    """If autosomes and Y non-PAR scale identically, y_genome_density is stable."""

    def run(scale: int):
        cfg = cfg38(
            n_attempted_autosomes=1000 * scale,
            n_attempted_y_nonpar=500 * scale,
        )
        acc = SexInferenceAccumulator(cfg)
        for i in range(800 * scale):
            acc.add(Chromosome.AUTOSOME, 1_000_000 + i, False)
        for i in range(350 * scale):
            acc.add(Chromosome.Y, Y_NON_PAR_POS + i, False)
        return acc.finish().report.y_genome_density

    base = run(1)
    for s in (2, 5, 10):
        scaled = run(s)
        assert math.isclose(base, scaled, rel_tol=1e-9), (base, scaled, s)


def test_invalid_platform_counts_raises():
    cfg = InferenceConfig(
        build=GenomeBuild.BUILD38,
        platform=PlatformDefinition(n_attempted_autosomes=0, n_attempted_y_nonpar=10),
    )
    acc = SexInferenceAccumulator(cfg)
    acc.add(Chromosome.X, X_NON_PAR_POS, True)
    with pytest.raises(InvalidPlatformCounts):
        acc.finish()


def test_observed_exceeds_attempted_raises():
    acc = SexInferenceAccumulator(cfg38(n_attempted_autosomes=5))
    for i in range(10):
        acc.add(Chromosome.AUTOSOME, 1_000_000 + i, False)
    with pytest.raises(ObservedExceedsAttempted):
        acc.finish()


def test_platform_definition_rejects_negative_or_non_int():
    with pytest.raises(InvalidPlatformCounts):
        PlatformDefinition(n_attempted_autosomes=-1, n_attempted_y_nonpar=10)


# ---------------------------------------------------------------------------
# Batch (numpy) path
# ---------------------------------------------------------------------------


def test_add_batch_matches_streaming_path():
    rng = np.random.default_rng(0)
    n = 5_000
    codes = rng.choice([0, 1, 2], size=n, p=[0.85, 0.1, 0.05]).astype(np.int8)
    pos = rng.integers(low=1_000_000, high=100_000_000, size=n, dtype=np.int64)
    het = rng.random(n) < 0.25

    chrom_objs = [Chromosome.AUTOSOME, Chromosome.X, Chromosome.Y]

    cfg = cfg38()
    a_stream = SexInferenceAccumulator(cfg)
    a_batch = SexInferenceAccumulator(cfg)
    for i in range(n):
        a_stream.add(chrom_objs[int(codes[i])], int(pos[i]), bool(het[i]))

    a_batch.add_batch(codes, pos, het)
    assert a_stream.snapshot() == a_batch.snapshot()


def test_add_batch_accepts_string_chromosomes():
    cfg = cfg38()
    acc = SexInferenceAccumulator(cfg)
    chrom = np.array(["1", "X", "chrY", "MT", "M", "GL000001.1", "22"])
    pos = np.array([1_000_000, X_NON_PAR_POS, Y_NON_PAR_POS, 100, 100, 1, 5_000], dtype=np.int64)
    het = np.array([False, True, False, False, False, False, True])
    acc.add_batch(chrom, pos, het)
    snap = acc.snapshot()
    # MT and 22 → autosome; GL000001.1 → ignored.
    assert snap.auto_valid_count == 4
    assert snap.x_non_par_valid_count == 1
    assert snap.y_non_par_valid_count == 1


def test_add_batch_rejects_mismatched_lengths():
    acc = SexInferenceAccumulator(cfg38())
    with pytest.raises(ValueError):
        acc.add_batch([0, 0], [1, 2, 3], [False, True, False])


def test_add_batch_accepts_chromosome_enum_objects():
    """Regression: object-dtype arrays of Chromosome members used to be
    silently dropped because str(Chromosome.X) == 'Chromosome.X', which
    doesn't match any handled pattern."""
    acc = SexInferenceAccumulator(cfg38())
    chroms = np.array(
        [Chromosome.AUTOSOME, Chromosome.X, Chromosome.Y, Chromosome.AUTOSOME],
        dtype=object,
    )
    pos = np.array([1_000_000, X_NON_PAR_POS, Y_NON_PAR_POS, 2_000_000], dtype=np.int64)
    het = np.array([False, True, False, True])
    acc.add_batch(chroms, pos, het)
    snap = acc.snapshot()
    assert snap.auto_valid_count == 2
    assert snap.x_non_par_valid_count == 1
    assert snap.x_non_par_het_count == 1
    assert snap.y_non_par_valid_count == 1


def test_add_batch_accepts_chromosome_enum_values():
    """Lowercase 'autosome'/'x'/'y' strings (Chromosome enum values) work too."""
    acc = SexInferenceAccumulator(cfg38())
    acc.add_batch(
        np.array(["autosome", "x", "y"], dtype=object),
        np.array([1_000_000, X_NON_PAR_POS, Y_NON_PAR_POS], dtype=np.int64),
        np.array([False, True, False]),
    )
    snap = acc.snapshot()
    assert snap.auto_valid_count == 1
    assert snap.x_non_par_valid_count == 1
    assert snap.y_non_par_valid_count == 1


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------


def test_sexinferer_infer_from_records():
    records = [(Chromosome.AUTOSOME, 1_000_000 + i, i % 4 == 0) for i in range(1_500)]
    records += [(Chromosome.X, X_NON_PAR_POS + i, True) for i in range(500)]
    result = SexInferer(
        build="hg38",
        platform=PlatformDefinition(n_attempted_autosomes=2_000, n_attempted_y_nonpar=1_000),
    ).infer_from_records(records)
    assert result.is_female


def test_module_level_infer_from_records():
    records = [("Y", Y_NON_PAR_POS + i, False) for i in range(500)]
    records += [("1", 1_000_000 + i, False) for i in range(1_500)]
    result = infer_from_records(
        records,
        build="GRCh38",
        platform=PlatformDefinition(n_attempted_autosomes=2_000, n_attempted_y_nonpar=1_000),
    )
    # Heavy Y signal but no X → male.
    assert result.is_male


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _make_vcf(path: Path, lines: list, gzipped: bool = False):
    body = textwrap.dedent(
        """\
        ##fileformat=VCFv4.5
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1
        """
    ) + "\n".join(lines) + "\n"
    if gzipped:
        with gzip.open(path, "wt") as f:
            f.write(body)
    else:
        path.write_text(body)


def test_infer_from_vcf_plain(tmp_path):
    vcf = tmp_path / "sample.vcf"
    lines = []
    for i in range(1_500):
        lines.append(f"1\t{1_000_000 + i}\t.\tA\tG\t.\t.\t.\tGT\t0/0")
    for i in range(500):
        lines.append(f"X\t{X_NON_PAR_POS + i}\t.\tA\tG\t.\t.\t.\tGT\t0/1")
    _make_vcf(vcf, lines)
    result = infer_from_vcf(
        vcf,
        build="hg38",
        platform=PlatformDefinition(n_attempted_autosomes=2_000, n_attempted_y_nonpar=1_000),
    )
    assert result.is_female
    assert result.report.x_non_par_het_count == 500


def test_infer_from_vcf_gzip_and_phased_and_missing(tmp_path):
    vcf = tmp_path / "sample.vcf.gz"
    lines = []
    for i in range(1_500):
        lines.append(f"chr1\t{1_000_000 + i}\t.\tA\tG\t.\t.\t.\tGT\t0|0")
    for i in range(20):
        lines.append(f"chrX\t{X_NON_PAR_POS + i}\t.\tA\tG\t.\t.\t.\tGT\t0|0")
    for i in range(700):
        lines.append(f"chrY\t{Y_NON_PAR_POS + i}\t.\tA\tG\t.\t.\t.\tGT\t0")
    # Missing genotypes must be skipped.
    for i in range(100):
        lines.append(f"chr2\t{2_000_000 + i}\t.\tA\tG\t.\t.\t.\tGT\t./.")
    _make_vcf(vcf, lines, gzipped=True)
    result = infer_from_vcf(
        vcf,
        build="hg38",
        platform=PlatformDefinition(n_attempted_autosomes=2_000, n_attempted_y_nonpar=1_000),
    )
    assert result.is_male
    # Missing GTs were dropped.
    assert result.report.auto_valid_count == 1_500


def test_infer_from_vcf_warns_on_multi_sample_default(tmp_path):
    """Regression: previously the default-to-first-sample fallback was
    completely silent. It now emits a UserWarning naming the picked
    sample so an incorrect call is debuggable."""
    vcf = tmp_path / "multi.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.5\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE_A\tSAMPLE_B\n"
        f"X\t{X_NON_PAR_POS}\t.\tA\tG\t.\t.\t.\tGT\t0/1\t0/0\n"
        f"1\t100\t.\tA\tG\t.\t.\t.\tGT\t0/0\t0/0\n"
    )
    with pytest.warns(UserWarning, match="SAMPLE_A"):
        infer_from_vcf(
            vcf,
            build="hg38",
            platform=PlatformDefinition(n_attempted_autosomes=10, n_attempted_y_nonpar=10),
        )


def test_infer_from_vcf_no_warning_when_sample_explicit(tmp_path):
    vcf = tmp_path / "multi.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.5\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
        f"1\t100\t.\tA\tG\t.\t.\t.\tGT\t0/0\t0/0\n"
    )
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")  # any UserWarning becomes a test failure
        infer_from_vcf(
            vcf,
            build="hg38",
            platform=PlatformDefinition(n_attempted_autosomes=10, n_attempted_y_nonpar=10),
            sample="B",
        )


def test_infer_from_vcf_selects_sample_by_id(tmp_path):
    vcf = tmp_path / "multi.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.5\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
        f"1\t1\t.\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\n"
        f"X\t{X_NON_PAR_POS}\t.\tA\tG\t.\t.\t.\tGT\t0/0\t1/1\n"
    )
    a = infer_from_vcf(
        vcf,
        build="hg38",
        platform=PlatformDefinition(n_attempted_autosomes=10, n_attempted_y_nonpar=10),
        sample="A",
    )
    b = infer_from_vcf(
        vcf,
        build="hg38",
        platform=PlatformDefinition(n_attempted_autosomes=10, n_attempted_y_nonpar=10),
        sample="B",
    )
    assert a.report.auto_het_count == 0
    assert b.report.auto_het_count == 1


def test_infer_from_vcf_missing_sample_raises(tmp_path):
    vcf = tmp_path / "bad.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.5\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\n"
        "1\t1\t.\tA\tG\t.\t.\t.\tGT\t0/0\n"
    )
    with pytest.raises(KeyError):
        infer_from_vcf(
            vcf,
            build="hg38",
            platform=PlatformDefinition(n_attempted_autosomes=10, n_attempted_y_nonpar=10),
            sample="NONESUCH",
        )


def _write_plink(prefix: Path, variants: list, samples: list):
    """variants: list of (chrom, pos, gt_for_sample_0, gt_for_sample_1, ...)
    where gt is one of 0 (hom A1), 1 (missing), 2 (het), 3 (hom A2).
    """
    bim_lines = []
    for i, v in enumerate(variants):
        bim_lines.append(f"{v[0]}\trs{i}\t0\t{v[1]}\tA\tG")
    (prefix.parent / (prefix.name + ".bim")).write_text("\n".join(bim_lines) + "\n")
    fam_lines = [f"FAM\t{s}\t0\t0\t0\t-9" for s in samples]
    (prefix.parent / (prefix.name + ".fam")).write_text("\n".join(fam_lines) + "\n")

    n = len(samples)
    bpv = (n + 3) // 4
    bed = bytearray()
    bed.extend(b"\x6c\x1b\x01")
    for v in variants:
        gts = v[2:]
        buf = bytearray(bpv)
        for sample_idx, gt in enumerate(gts):
            byte_i = sample_idx // 4
            bit_i = (sample_idx % 4) * 2
            buf[byte_i] |= (gt & 0b11) << bit_i
        bed.extend(buf)
    (prefix.parent / (prefix.name + ".bed")).write_bytes(bytes(bed))


def test_infer_from_plink(tmp_path):
    prefix = tmp_path / "ds"
    variants = []
    for i in range(1_500):
        variants.append(("1", 1_000_000 + i, 0, 2))  # sample0 hom A1, sample1 het
    for i in range(500):
        variants.append(("X", X_NON_PAR_POS + i, 0, 2))
    _write_plink(prefix, variants, samples=["A", "B"])

    res_a = infer_from_plink(
        prefix,
        build="hg38",
        platform=PlatformDefinition(n_attempted_autosomes=2_000, n_attempted_y_nonpar=1_000),
        sample="A",
    )
    res_b = infer_from_plink(
        prefix,
        build="hg38",
        platform=PlatformDefinition(n_attempted_autosomes=2_000, n_attempted_y_nonpar=1_000),
        sample="B",
    )
    assert res_a.report.x_non_par_het_count == 0
    assert res_b.report.x_non_par_het_count == 500


def test_infer_from_plink_rejects_truncated_bed(tmp_path):
    """Regression: previously the per-variant seek/read could silently
    return an empty byte and continue; we now precheck file size."""
    prefix = tmp_path / "trunc"
    (prefix.parent / "trunc.bim").write_text(
        "1\trs1\t0\t1\tA\tG\n1\trs2\t0\t2\tA\tG\n1\trs3\t0\t3\tA\tG\n"
    )
    (prefix.parent / "trunc.fam").write_text("F\tA\t0\t0\t0\t-9\nF\tB\t0\t0\t0\t-9\n")
    # Magic + one variant only (need three).
    (prefix.parent / "trunc.bed").write_bytes(b"\x6c\x1b\x01\x00")
    with pytest.raises(ValueError, match="truncated"):
        infer_from_plink(
            prefix,
            build="hg38",
            platform=PlatformDefinition(n_attempted_autosomes=10, n_attempted_y_nonpar=10),
        )


def test_infer_from_plink_rejects_sample_major(tmp_path):
    prefix = tmp_path / "bad"
    (prefix.parent / "bad.bim").write_text("1\trs1\t0\t1\tA\tG\n")
    (prefix.parent / "bad.fam").write_text("F\tA\t0\t0\t0\t-9\n")
    (prefix.parent / "bad.bed").write_bytes(b"\x6c\x1b\x00\x00")  # sample-major
    with pytest.raises(ValueError):
        infer_from_plink(
            prefix,
            build="hg38",
            platform=PlatformDefinition(n_attempted_autosomes=10, n_attempted_y_nonpar=10),
        )


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def test_evidence_report_as_dict_round_trip():
    rep = EvidenceReport(auto_valid_count=10, auto_het_count=2)
    d = rep.as_dict()
    assert d["auto_valid_count"] == 10
    assert d["auto_het_count"] == 2
    assert d["composite_sex_index"] is None


def test_variantinfo_validates_position():
    with pytest.raises(ValueError):
        VariantInfo(chrom=Chromosome.X, pos=-1, is_heterozygous=True)


def test_inference_result_helpers():
    rep = EvidenceReport()
    for sex, attrs in [
        (InferredSex.MALE, ("is_male",)),
        (InferredSex.FEMALE, ("is_female",)),
        (InferredSex.INDETERMINATE, ("is_indeterminate",)),
    ]:
        from infer_sex import InferenceResult

        r = InferenceResult(final_call=sex, report=rep)
        for attr in attrs:
            assert getattr(r, attr) is True
