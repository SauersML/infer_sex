//! A high-performance, zero-dependency Rust library for inferring genetic sex
//! from summarized variant data.
//!
//! The algorithm consumes an iterator of [`VariantInfo`] structs in a single
//! pass, counting valid and heterozygous observations across autosomes and sex
//! chromosomes. Metrics are normalized by platform-level "attempted" locus
//! counts that the caller must provide via [`PlatformDefinition`], making the
//! library resilient to platform density and sample quality differences.
//!
//! # Example
//!
//! ```
//! use infer_sex::{
//!     Chromosome, DecisionThresholds, GenomeBuild, InferenceConfig, InferenceResult,
//!     InferredSex, PlatformDefinition, SexInferenceAccumulator, VariantInfo,
//! };
//!
//! let config = InferenceConfig {
//!     build: GenomeBuild::Build38,
//!     platform: PlatformDefinition {
//!         n_attempted_autosomes: 2_000,
//!         n_attempted_y_nonpar: 1_000,
//!     },
//!     thresholds: Some(DecisionThresholds::default()),
//! };
//!
//! let mut acc = SexInferenceAccumulator::new(config);
//! let variants = vec![
//!     // Autosomal signal for normalization.
//!     VariantInfo { chrom: Chromosome::Autosome, pos: 1_000_000, is_heterozygous: true },
//!     VariantInfo { chrom: Chromosome::Autosome, pos: 2_000_000, is_heterozygous: false },
//!     // X non-PAR heterozygosity (diploid X implies female).
//!     VariantInfo { chrom: Chromosome::X, pos: 10_000_000, is_heterozygous: true },
//!     VariantInfo { chrom: Chromosome::X, pos: 20_000_000, is_heterozygous: true },
//! ];
//!
//! for v in &variants {
//!     acc.process_variant(v);
//! }
//!
//! let result: InferenceResult = acc.finish().expect("valid platform counts");
//! assert_eq!(result.final_call, InferredSex::Female);
//! println!("Report: {:?}", result.report);
//! ```
//!
//! The library returns `InferredSex::Male`, `InferredSex::Female`, or
//! `InferredSex::Indeterminate` when no sex-chromosome evidence is observed. If
//! you do not supply [`DecisionThresholds`], a built-in default heuristic is
//! used to derive the call while still exposing the underlying metrics for
//! custom downstream logic.
//!
//! ## Platform definitions (`n_attempted_*`)
//!
//! The attempted locus counts must match the exact loci that will be streamed into
//! [`process_variant`]. A common pattern is to pre-scan a BIM (or similar) file:
//!
//! ```no_run
//! use infer_sex::PlatformDefinition;
//!
//! struct BimRow { chrom: String, pos: u64 }
//!
//! fn derive_platform_from_bim(rows: impl Iterator<Item = BimRow>) -> PlatformDefinition {
//!     let mut auto = 0u64;
//!     let mut y_nonpar = 0u64;
//!     fn is_in_y_par(_pos: u64) -> bool { unimplemented!("project-specific PAR check") }
//!     for row in rows {
//!         match row.chrom.as_str() {
//!             "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12"
//!             | "13" | "14" | "15" | "16" | "17" | "18" | "19" | "20" | "21" | "22" => {
//!                 auto += 1;
//!             }
//!             "Y" => {
//!                 if !is_in_y_par(row.pos) {
//!                     y_nonpar += 1;
//!                 }
//!             }
//!             _ => {}
//!         }
//!     }
//!     PlatformDefinition {
//!         n_attempted_autosomes: auto,
//!         n_attempted_y_nonpar: y_nonpar,
//!     }
//! }
//! ```
//!
//! The variant stream passed to [`SexInferenceAccumulator`] must be derived from the
//! same locus set; down-sampling autosomes for speed requires that
//! `n_attempted_autosomes` reflect the down-sampled set.

//=============================================================================
//  Public API Surface
//=============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenomeBuild {
    Build37,
    Build38,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlgorithmConstants {
    pub par1_x: (u64, u64),
    pub non_par_x: (u64, u64),
    pub par2_x: (u64, u64),
    pub par1_y: (u64, u64),
    pub non_par_y: (u64, u64),
    pub par2_y: (u64, u64),
    pub epsilon: f64,
}

impl AlgorithmConstants {
    pub fn from_build(build: GenomeBuild) -> Self {
        match build {
            GenomeBuild::Build37 => Self {
                par1_x: (60_001, 2_699_520),
                non_par_x: (2_699_521, 154_931_043),
                par2_x: (154_931_044, 155_260_560),
                par1_y: (10_001, 2_649_520),
                non_par_y: (2_649_521, 59_034_049),
                par2_y: (59_034_050, 59_363_566),
                epsilon: 1e-9,
            },
            GenomeBuild::Build38 => Self {
                par1_x: (10_001, 2_781_479),
                non_par_x: (2_781_480, 155_701_382),
                par2_x: (155_701_383, 156_030_895),
                par1_y: (10_001, 2_781_479),
                non_par_y: (2_781_480, 56_887_902),
                par2_y: (56_887_903, 57_217_415),
                epsilon: 1e-9,
            },
        }
    }

    pub fn is_in_x_par(&self, pos: u64) -> bool {
        (pos >= self.par1_x.0 && pos <= self.par1_x.1)
            || (pos >= self.par2_x.0 && pos <= self.par2_x.1)
    }

    pub fn is_in_x_non_par(&self, pos: u64) -> bool {
        pos >= self.non_par_x.0 && pos <= self.non_par_x.1
    }

    pub fn is_in_y_par(&self, pos: u64) -> bool {
        (pos >= self.par1_y.0 && pos <= self.par1_y.1)
            || (pos >= self.par2_y.0 && pos <= self.par2_y.1)
    }

    pub fn is_in_y_non_par(&self, pos: u64) -> bool {
        pos >= self.non_par_y.0 && pos <= self.non_par_y.1
    }
}

impl GenomeBuild {
    pub fn algorithm_constants(&self) -> AlgorithmConstants {
        AlgorithmConstants::from_build(*self)
    }

    pub fn is_in_x_par(&self, pos: u64) -> bool {
        self.algorithm_constants().is_in_x_par(pos)
    }

    pub fn is_in_x_non_par(&self, pos: u64) -> bool {
        self.algorithm_constants().is_in_x_non_par(pos)
    }

    pub fn is_in_y_par(&self, pos: u64) -> bool {
        self.algorithm_constants().is_in_y_par(pos)
    }

    pub fn is_in_y_non_par(&self, pos: u64) -> bool {
        self.algorithm_constants().is_in_y_non_par(pos)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformDefinition {
    /// Total attempted autosomal loci for this dataset/platform.
    pub n_attempted_autosomes: u64,
    /// Total attempted Y non-PAR loci for this dataset/platform.
    pub n_attempted_y_nonpar: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionThresholds {
    /// Slope of the linear decision boundary used to differentiate samples.
    pub slope: f64,
    /// Intercept of the linear decision boundary used to differentiate samples.
    pub intercept: f64,
}

impl Default for DecisionThresholds {
    fn default() -> Self {
        Self {
            slope: 0.3566,
            intercept: 0.2738,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InferenceConfig {
    pub build: GenomeBuild,
    pub platform: PlatformDefinition,
    /// Optional threshold set used to derive a binary call when sex evidence is
    /// observed. If omitted, default heuristics are applied to force a
    /// Male/Female decision.
    pub thresholds: Option<DecisionThresholds>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chromosome {
    Autosome,
    X,
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariantInfo {
    /// Chromosome the variant belongs to.
    pub chrom: Chromosome,
    /// 1-based position of the variant.
    pub pos: u64,
    /// Whether the genotype is heterozygous. Missing calls must not be emitted.
    pub is_heterozygous: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferredSex {
    Male,
    Female,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvidenceReport {
    /// Y genome density metric.
    pub y_genome_density: Option<f64>,
    /// X-to-autosome heterozygosity ratio.
    pub x_autosome_het_ratio: Option<f64>,
    /// Composite index combining Y density and X normalization.
    pub composite_sex_index: Option<f64>,

    // Raw counts for QC/debugging.
    pub auto_valid_count: u64,
    pub auto_het_count: u64,
    pub x_non_par_valid_count: u64,
    pub x_non_par_het_count: u64,
    pub x_par_valid_count: u64,
    pub x_par_het_count: u64,
    pub y_non_par_valid_count: u64,
    pub y_par_valid_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceResult {
    pub final_call: InferredSex,
    pub report: EvidenceReport,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceError {
    /// Platform definition provided zero attempted loci for required regions.
    InvalidPlatformCounts(&'static str),
    /// Observed counts exceeded platform attempted counts, indicating mismatched inputs.
    ObservedExceedsAttempted(&'static str),
}

#[derive(Debug)]
pub struct SexInferenceAccumulator {
    config: InferenceConfig,
    constants: AlgorithmConstants,
    counters: internal::EvidenceCounters,
}

//=============================================================================
//  Internal Implementation Details
//=============================================================================

mod internal {
    #[derive(Default, Debug)]
    pub(crate) struct EvidenceCounters {
        pub(crate) auto_valid_count: u64,
        pub(crate) auto_het_count: u64,
        pub(crate) x_non_par_valid_count: u64,
        pub(crate) x_non_par_het_count: u64,
        pub(crate) x_par_valid_count: u64,
        pub(crate) x_par_het_count: u64,
        pub(crate) y_non_par_valid_count: u64,
        pub(crate) y_par_valid_count: u64,
    }
}

//=============================================================================
//  Public Struct Implementations
//=============================================================================

impl SexInferenceAccumulator {
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            constants: AlgorithmConstants::from_build(config.build),
            config,
            counters: internal::EvidenceCounters::default(),
        }
    }

    pub fn process_variant(&mut self, variant: &VariantInfo) {
        match variant.chrom {
            Chromosome::Autosome => {
                self.counters.auto_valid_count += 1;
                if variant.is_heterozygous {
                    self.counters.auto_het_count += 1;
                }
            }
            Chromosome::X => {
                if self.constants.is_in_x_par(variant.pos) {
                    self.counters.x_par_valid_count += 1;
                    if variant.is_heterozygous {
                        self.counters.x_par_het_count += 1;
                    }
                } else if self.constants.is_in_x_non_par(variant.pos) {
                    self.counters.x_non_par_valid_count += 1;
                    if variant.is_heterozygous {
                        self.counters.x_non_par_het_count += 1;
                    }
                }
            }
            Chromosome::Y => {
                if self.constants.is_in_y_par(variant.pos) {
                    self.counters.y_par_valid_count += 1;
                } else if self.constants.is_in_y_non_par(variant.pos) {
                    self.counters.y_non_par_valid_count += 1;
                }
            }
        }
    }

    pub fn finish(self) -> Result<InferenceResult, InferenceError> {
        if self.config.platform.n_attempted_autosomes == 0 {
            return Err(InferenceError::InvalidPlatformCounts(
                "n_attempted_autosomes must be > 0",
            ));
        }
        if self.counters.auto_valid_count > self.config.platform.n_attempted_autosomes {
            return Err(InferenceError::ObservedExceedsAttempted(
                "observed autosomal variants exceed platform definition",
            ));
        }
        if self.counters.y_non_par_valid_count > self.config.platform.n_attempted_y_nonpar {
            return Err(InferenceError::ObservedExceedsAttempted(
                "observed Y non-PAR variants exceed platform definition",
            ));
        }

        let mut report = EvidenceReport {
            auto_valid_count: self.counters.auto_valid_count,
            auto_het_count: self.counters.auto_het_count,
            x_non_par_valid_count: self.counters.x_non_par_valid_count,
            x_non_par_het_count: self.counters.x_non_par_het_count,
            x_par_valid_count: self.counters.x_par_valid_count,
            x_par_het_count: self.counters.x_par_het_count,
            y_non_par_valid_count: self.counters.y_non_par_valid_count,
            y_par_valid_count: self.counters.y_par_valid_count,
            ..EvidenceReport::default()
        };

        let total_sex_observed = self.counters.x_non_par_valid_count
            + self.counters.x_par_valid_count
            + self.counters.y_non_par_valid_count
            + self.counters.y_par_valid_count;

        if total_sex_observed == 0 {
            return Ok(InferenceResult {
                final_call: InferredSex::Indeterminate,
                report,
            });
        }

        let y_density = if self.counters.auto_valid_count == 0
            || self.config.platform.n_attempted_y_nonpar == 0
        {
            None
        } else {
            let auto_rate = (self.counters.auto_valid_count as f64 + self.constants.epsilon)
                / self.config.platform.n_attempted_autosomes as f64;
            let y_rate = self.counters.y_non_par_valid_count as f64
                / self.config.platform.n_attempted_y_nonpar as f64;
            Some(y_rate / auto_rate)
        };

        let x_auto_ratio =
            if self.counters.auto_valid_count == 0 || self.counters.x_non_par_valid_count == 0 {
                None
            } else {
                let auto_het_rate = self.counters.auto_het_count as f64
                    / (self.counters.auto_valid_count as f64 + self.constants.epsilon);
                let x_het_rate = self.counters.x_non_par_het_count as f64
                    / (self.counters.x_non_par_valid_count as f64 + self.constants.epsilon);
                Some(x_het_rate / (auto_het_rate + self.constants.epsilon))
            };

        let composite = match (y_density, x_auto_ratio) {
            (Some(y), Some(x)) => Some(y / (x + self.constants.epsilon)),
            _ => None,
        };

        report.y_genome_density = y_density;
        report.x_autosome_het_ratio = x_auto_ratio;
        report.composite_sex_index = composite;

        let thresholds = self.config.thresholds.unwrap_or_default();
        let final_call = classify_sex(y_density, x_auto_ratio, thresholds);

        Ok(InferenceResult { final_call, report })
    }
}

fn classify_sex(
    y_density: Option<f64>,
    x_auto_ratio: Option<f64>,
    thresholds: DecisionThresholds,
) -> InferredSex {
    let y = y_density.unwrap_or(0.0);
    let x = x_auto_ratio.unwrap_or(0.0);
    let calculated_threshold = (thresholds.slope * x) + thresholds.intercept;

    if y > calculated_threshold {
        InferredSex::Male
    } else {
        InferredSex::Female
    }
}

//=============================================================================
//  Tests
//=============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    fn config38() -> InferenceConfig {
        InferenceConfig {
            build: GenomeBuild::Build38,
            platform: PlatformDefinition {
                n_attempted_autosomes: 2_000,
                n_attempted_y_nonpar: 1_000,
            },
            thresholds: Some(DecisionThresholds::default()),
        }
    }

    fn gen_variant(chrom: Chromosome, pos: u64, is_het: bool) -> VariantInfo {
        VariantInfo {
            chrom,
            pos,
            is_heterozygous: is_het,
        }
    }

    #[test]
    fn female_sample_shows_high_x_het_ratio_and_low_y_density() {
        let mut acc = SexInferenceAccumulator::new(config38());

        // Autosomes: solid callability and moderate heterozygosity.
        for i in 0..1_500 {
            acc.process_variant(&gen_variant(
                Chromosome::Autosome,
                1_000_000 + i,
                i % 2 == 0,
            ));
        }

        // X non-PAR: heterozygous sites consistent with XX.
        for i in 0..400 {
            acc.process_variant(&gen_variant(Chromosome::X, 10_000_000 + i, true));
        }

        // Y: minimal non-PAR signal.
        for i in 0..2 {
            acc.process_variant(&gen_variant(Chromosome::Y, 5_000_000 + i, false));
        }

        let result = acc.finish();
        let report = result.as_ref().unwrap().report.clone();
        assert_eq!(result.unwrap().final_call, InferredSex::Female);
        assert!(report.y_genome_density.unwrap() < 0.1);
        assert!(report.x_autosome_het_ratio.unwrap() > 0.5);
        assert!(report.composite_sex_index.unwrap() < 0.5);
    }

    #[test]
    fn male_sample_shows_high_y_density_and_low_x_het_ratio() {
        let mut acc = SexInferenceAccumulator::new(config38());
        let constants = AlgorithmConstants::from_build(GenomeBuild::Build38);

        // Autosomes: valid and heterozygous calls for normalization.
        for i in 0..1_000 {
            acc.process_variant(&gen_variant(
                Chromosome::Autosome,
                1_000_000 + i,
                i % 3 == 0,
            ));
        }

        // X non-PAR: haploid-like (no hets).
        for i in 0..400 {
            acc.process_variant(&gen_variant(
                Chromosome::X,
                constants.non_par_x.0 + 10 + i,
                false,
            ));
        }

        // Y non-PAR: strong presence.
        for i in 0..400 {
            acc.process_variant(&gen_variant(
                Chromosome::Y,
                constants.non_par_y.0 + 10 + i,
                false,
            ));
        }

        let result = acc.finish();
        let report = result.as_ref().unwrap().report.clone();
        assert_eq!(result.unwrap().final_call, InferredSex::Male);
        assert!(report.y_genome_density.unwrap() > 0.5);
        assert!(report.x_autosome_het_ratio.unwrap() < 0.2);
        assert!(report.composite_sex_index.unwrap() > 1.0);
    }

    #[test]
    fn par_variants_do_not_influence_non_par_metrics() {
        let mut acc = SexInferenceAccumulator::new(config38());
        let constants = AlgorithmConstants::from_build(GenomeBuild::Build38);

        // Autosome baseline.
        for i in 0..100 {
            acc.process_variant(&gen_variant(Chromosome::Autosome, 1_000_000 + i, false));
        }

        // X PAR variants: should not affect X non-PAR counts.
        acc.process_variant(&gen_variant(Chromosome::X, constants.par1_x.0 + 10, true));
        acc.process_variant(&gen_variant(Chromosome::X, constants.par2_x.0 + 10, true));

        // Y PAR variants: should not affect Y non-PAR counts.
        acc.process_variant(&gen_variant(Chromosome::Y, constants.par1_y.0 + 10, false));
        acc.process_variant(&gen_variant(Chromosome::Y, constants.par2_y.0 + 10, false));

        let report = acc.finish().unwrap().report;
        assert_eq!(report.x_non_par_valid_count, 0);
        assert_eq!(report.x_non_par_het_count, 0);
        assert_eq!(report.y_non_par_valid_count, 0);
    }

    #[test]
    fn metrics_are_platform_invariant_when_counts_scale() {
        let constants = AlgorithmConstants::from_build(GenomeBuild::Build38);

        let config_small = InferenceConfig {
            build: GenomeBuild::Build38,
            platform: PlatformDefinition {
                n_attempted_autosomes: 2_000,
                n_attempted_y_nonpar: 500,
            },
            thresholds: None,
        };

        let config_large = InferenceConfig {
            build: GenomeBuild::Build38,
            platform: PlatformDefinition {
                n_attempted_autosomes: 4_000,
                n_attempted_y_nonpar: 1_000,
            },
            thresholds: None,
        };

        let mut acc_small = SexInferenceAccumulator::new(config_small);
        let mut acc_large = SexInferenceAccumulator::new(config_large);

        // Small platform counts.
        for i in 0..1_000 {
            acc_small.process_variant(&gen_variant(
                Chromosome::Autosome,
                1_000_000 + i,
                i % 5 == 0,
            ));
        }
        for i in 0..400 {
            acc_small.process_variant(&gen_variant(
                Chromosome::X,
                constants.non_par_x.0 + 10 + i,
                i % 10 == 0,
            ));
        }
        for i in 0..200 {
            acc_small.process_variant(&gen_variant(
                Chromosome::Y,
                constants.non_par_y.0 + 10 + i,
                false,
            ));
        }

        // Large platform counts with doubled observations.
        for i in 0..2_000 {
            acc_large.process_variant(&gen_variant(
                Chromosome::Autosome,
                2_000_000 + i,
                i % 5 == 0,
            ));
        }
        for i in 0..800 {
            acc_large.process_variant(&gen_variant(
                Chromosome::X,
                constants.non_par_x.0 + 1000 + i,
                i % 10 == 0,
            ));
        }
        for i in 0..400 {
            acc_large.process_variant(&gen_variant(
                Chromosome::Y,
                constants.non_par_y.0 + 1000 + i,
                false,
            ));
        }

        let small_report = acc_small.finish().unwrap().report;
        let large_report = acc_large.finish().unwrap().report;

        let small_y = small_report.y_genome_density.unwrap();
        let large_y = large_report.y_genome_density.unwrap();
        assert!((small_y - large_y).abs() < 1e-9);

        let small_x = small_report.x_autosome_het_ratio.unwrap();
        let large_x = large_report.x_autosome_het_ratio.unwrap();
        assert!((small_x - large_x).abs() < 1e-9);
    }

    #[test]
    fn allows_x_only_platforms_and_derives_call_from_x_ratio() {
        let constants = AlgorithmConstants::from_build(GenomeBuild::Build38);
        let config = InferenceConfig {
            build: GenomeBuild::Build38,
            platform: PlatformDefinition {
                n_attempted_autosomes: 1_000,
                n_attempted_y_nonpar: 0,
            },
            thresholds: Some(DecisionThresholds::default()),
        };

        let mut acc = SexInferenceAccumulator::new(config);

        // Autosomes: balanced heterozygosity to normalize against.
        for i in 0..500 {
            acc.process_variant(&gen_variant(
                Chromosome::Autosome,
                1_000_000 + i,
                i % 2 == 0,
            ));
        }

        // X non-PAR: strong heterozygosity indicative of XX.
        for i in 0..200 {
            acc.process_variant(&gen_variant(
                Chromosome::X,
                constants.non_par_x.0 + 10 + i,
                true,
            ));
        }

        let result = acc.finish().unwrap();
        let report = result.report;

        assert!(report.y_genome_density.is_none());
        assert!(report.x_autosome_het_ratio.unwrap() > 0.5);
        assert_eq!(result.final_call, InferredSex::Female);
    }

    #[test]
    fn autosome_only_samples_return_indeterminate() {
        let mut acc = SexInferenceAccumulator::new(config38());

        for i in 0..1_000 {
            acc.process_variant(&gen_variant(
                Chromosome::Autosome,
                1_000_000 + i,
                i % 2 == 0,
            ));
        }

        let result = acc.finish().unwrap();
        assert_eq!(result.final_call, InferredSex::Indeterminate);
        assert!(result.report.y_genome_density.is_none());
        assert!(result.report.x_autosome_het_ratio.is_none());
        assert!(result.report.composite_sex_index.is_none());
    }

    #[test]
    fn errors_when_platform_counts_are_zero_or_mismatched() {
        let bad_platform = PlatformDefinition {
            n_attempted_autosomes: 0,
            n_attempted_y_nonpar: 10,
        };
        let config = InferenceConfig {
            build: GenomeBuild::Build38,
            platform: bad_platform,
            thresholds: None,
        };
        let acc = SexInferenceAccumulator::new(config);
        assert!(matches!(
            acc.finish(),
            Err(InferenceError::InvalidPlatformCounts(_))
        ));

        let mut acc = SexInferenceAccumulator::new(config38());
        // Feed more autosomal variants than allowed.
        for i in 0..2_100 {
            acc.process_variant(&gen_variant(Chromosome::Autosome, 1_000_000 + i, false));
        }
        let err = acc.finish().unwrap_err();
        assert!(matches!(err, InferenceError::ObservedExceedsAttempted(_)));
    }

    fn for_each_dtc_row<F>(path: &str, mut f: F) -> std::io::Result<()>
    where
        F: FnMut(Chromosome, u64, &str),
    {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = trimmed.split('\t').collect();
            if fields.len() < 4 {
                continue;
            }

            let chrom = match fields[1].trim() {
                "X" => Chromosome::X,
                "Y" => Chromosome::Y,
                "MT" => continue,
                "23" => Chromosome::X,
                "24" => Chromosome::Y,
                "25" => continue,
                other => match other.parse::<u8>() {
                    Ok(n) if (1..=22).contains(&n) => Chromosome::Autosome,
                    _ => continue,
                },
            };

            let pos = match fields[2].trim().parse::<u64>() {
                Ok(pos) => pos,
                Err(_) => continue,
            };

            f(chrom, pos, fields[3].trim());
        }

        Ok(())
    }

    fn is_missing_genotype(genotype: &str) -> bool {
        let g = genotype.trim();
        g.is_empty() || g.contains('-') || g == "0" || g == "00" || g.eq_ignore_ascii_case("NA")
    }

    fn is_heterozygous(genotype: &str) -> bool {
        let g = genotype.as_bytes();
        g.len() == 2 && g[0] != g[1]
    }

    #[test]
    fn dtc_sample_infers_male() {
        let path = format!(
            "{}/data/genome_Christopher_Smith_v5_Full_20230926164611_test.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        let constants = AlgorithmConstants::from_build(GenomeBuild::Build37);

        let mut attempted_autosomes = 0_u64;
        let mut attempted_y_nonpar = 0_u64;
        for_each_dtc_row(&path, |chrom, pos, _geno| {
            match chrom {
                Chromosome::Autosome => attempted_autosomes += 1,
                Chromosome::Y => {
                    if constants.is_in_y_non_par(pos) {
                        attempted_y_nonpar += 1;
                    }
                }
                Chromosome::X => {}
            }
        })
        .expect("failed to scan DTC data for platform counts");

        assert!(attempted_autosomes > 0);
        assert!(attempted_y_nonpar > 0);

        let config = InferenceConfig {
            build: GenomeBuild::Build37,
            platform: PlatformDefinition {
                n_attempted_autosomes: attempted_autosomes,
                n_attempted_y_nonpar: attempted_y_nonpar,
            },
            thresholds: Some(DecisionThresholds::default()),
        };
        let mut acc = SexInferenceAccumulator::new(config);

        for_each_dtc_row(&path, |chrom, pos, geno| {
            if is_missing_genotype(geno) {
                return;
            }
            let variant = VariantInfo {
                chrom,
                pos,
                is_heterozygous: is_heterozygous(geno),
            };
            acc.process_variant(&variant);
        })
        .expect("failed to parse DTC data for inference");

        let result = acc.finish().unwrap();
        println!("inferred sex: {:?}", result.final_call);
        println!("report: {:#?}", result.report);
        assert_eq!(result.final_call, InferredSex::Male);
    }
}
