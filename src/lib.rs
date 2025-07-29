//! A high-performance, zero-dependency Rust library for inferring genetic sex
//! from summarized variant data.
//!
//! This library uses a robust, opinionated, multi-pronged algorithm based on
//! sex chromosome biology. It operates on an iterator of `VariantInfo` structs
//! in a single pass, ensuring minimal memory usage and high throughput.
//!
//! The core of the library is the [`SexInferenceAccumulator`], a state machine
//! that you feed variant data into.
//!
//! # Example
//!
//! ```
//! use infer_sex::{
//!     SexInferenceAccumulator, InferenceConfig, GenomeBuild,
//!     VariantInfo, Chromosome, InferenceResult, InferredSex
//! };
//!
//! // 1. Create a configuration.
//! let config = InferenceConfig { build: GenomeBuild::Build37 };
//!
//! // 2. Initialize the accumulator state machine.
//! let mut accumulator = SexInferenceAccumulator::new(config);
//!
//! // 3. In your application, parse your VCF and create VariantInfo structs.
//! //    This example uses a small, hardcoded set of variants characteristic of XX.
//! let variants: Vec<VariantInfo> = {
//!     let mut v = Vec::new();
//!     // Simulate 1250 X variants for the heterozygosity check to pass the minimum threshold.
//!     // 1000/1250 = 80% heterozygous -> 0.8 ratio --> XX
//!     for i in 0..1000 {
//!         v.push(VariantInfo { chrom: Chromosome::X, pos: 30_000_000 + i, is_heterozygous: true });
//!     }
//!     for i in 0..250 {
//!         v.push(VariantInfo { chrom: Chromosome::X, pos: 50_000_000 + i, is_heterozygous: false });
//!     }
//!     // Simulate 50 Y variants, all in PAR, to pass the Y-presence check.
//!     // XX sample might have some reads mis-mapped to Y-PAR.
//!     for i in 0..50 {
//!         v.push(VariantInfo { chrom: Chromosome::Y, pos: 1_000_000 + i * 1_000, is_heterozygous: false });
//!     }
//!     v
//! };
//!
//! // 4. Process all variants in a single pass.
//! for variant in &variants {
//!     accumulator.process_variant(variant);
//! }
//!
//! // 5. Finalize the analysis to get the result. This consumes the accumulator.
//! let result: InferenceResult = accumulator.finish();
//!
//! // 6. Use the clear, structured result.
//! assert_eq!(result.final_call, InferredSex::Female);
//! println!("Final Call: {:?}", result.final_call);
//! println!("Evidence Report: {:#?}", result.report);
//! ```

//=============================================================================
//  Public API Surface
//=============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenomeBuild {
    Build37,
    Build38,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceConfig {
    pub build: GenomeBuild,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chromosome {
    X,
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariantInfo {
    pub chrom: Chromosome,
    pub pos: u64,
    pub is_heterozygous: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferredSex {
    Male,
    Female,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvidenceReport {
    pub x_heterozygosity_check: Option<(f64, InferredSex)>,
    pub y_presence_check: Option<(u64, u64, InferredSex)>,
    pub sry_presence_check: Option<InferredSex>,
    pub par_non_par_het_check: Option<(f64, InferredSex)>,
    pub final_male_votes: u8,
    pub final_female_votes: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceResult {
    pub final_call: InferredSex,
    pub report: EvidenceReport,
}

#[derive(Debug)]
pub struct SexInferenceAccumulator {
    constants: internal::constants::AlgorithmConstants,
    counters: internal::EvidenceCounters,
}

//=============================================================================
//  Internal Implementation Details
//=============================================================================

mod internal {
    // This module is private to the crate by default.
    pub(crate) mod constants {
        use crate::GenomeBuild;

        // FIX: Add `Debug` derive and use `pub(crate)` for visibility within the crate.
        #[derive(Debug)]
        pub(crate) struct AlgorithmConstants {
            pub(crate) par1_x: (u64, u64),
            pub(crate) non_par_x: (u64, u64),
            pub(crate) par2_x: (u64, u64),
            pub(crate) par1_y: (u64, u64),
            pub(crate) non_par_y: (u64, u64),
            pub(crate) par2_y: (u64, u64),
            pub(crate) sry_region: (u64, u64),
            pub(crate) x_het_threshold: f64,
            pub(crate) par_non_par_ratio_threshold: f64,
            pub(crate) min_x_variants_for_het_check: u64,
            pub(crate) min_y_variants_for_presence_check: u64,
        }

        impl AlgorithmConstants {
            // FIX: Use `pub(crate)` for the constructor.
            pub(crate) fn from_build(build: GenomeBuild) -> Self {
                match build {
                    GenomeBuild::Build37 => Self {
                        par1_x: (60_001, 2_699_520),
                        non_par_x: (2_699_521, 154_931_043),
                        par2_x: (154_931_044, 155_260_560),
                        par1_y: (10_001, 2_649_520),
                        non_par_y: (2_649_521, 59_034_049),
                        par2_y: (59_034_050, 59_363_566),
                        sry_region: (2_786_855, 2_787_682),
                        x_het_threshold: 0.1,
                        par_non_par_ratio_threshold: 2.0,
                        min_x_variants_for_het_check: 1000,
                        min_y_variants_for_presence_check: 50,
                    },
                    GenomeBuild::Build38 => Self {
                        par1_x: (10_001, 2_781_479),
                        non_par_x: (2_781_480, 155_701_382),
                        par2_x: (155_701_383, 156_030_895),
                        par1_y: (10_001, 2_781_479),
                        non_par_y: (2_781_480, 56_887_902),
                        par2_y: (56_887_903, 57_217_415),
                        sry_region: (2_654_896, 2_655_723),
                        x_het_threshold: 0.1,
                        par_non_par_ratio_threshold: 2.0,
                        min_x_variants_for_het_check: 1000,
                        min_y_variants_for_presence_check: 50,
                    },
                }
            }
            
            // FIX: Use `pub(crate)` for all helper methods.
            pub(crate) fn is_in_x_par(&self, pos: u64) -> bool {
                (pos >= self.par1_x.0 && pos <= self.par1_x.1) || (pos >= self.par2_x.0 && pos <= self.par2_x.1)
            }
            pub(crate) fn is_in_x_non_par(&self, pos: u64) -> bool {
                pos >= self.non_par_x.0 && pos <= self.non_par_x.1
            }
            pub(crate) fn is_in_y_par(&self, pos: u64) -> bool {
                (pos >= self.par1_y.0 && pos <= self.par1_y.1) || (pos >= self.par2_y.0 && pos <= self.par2_y.1)
            }
            pub(crate) fn is_in_y_non_par(&self, pos: u64) -> bool {
                pos >= self.non_par_y.0 && pos <= self.non_par_y.1
            }
            pub(crate) fn is_in_sry(&self, pos: u64) -> bool {
                pos >= self.sry_region.0 && pos <= self.sry_region.1
            }
        }
    }

    #[derive(Default, Debug)]
    pub(crate) struct EvidenceCounters {
        pub(crate) x_total_variants: u64,
        pub(crate) x_het_variants: u64,
        pub(crate) y_par_variants: u64,
        pub(crate) y_non_par_variants: u64,
        pub(crate) y_sry_variants: u64,
        pub(crate) x_par_total: u64,
        pub(crate) x_par_het: u64,
        pub(crate) x_non_par_total: u64,
        pub(crate) x_non_par_het: u64,
    }
}

//=============================================================================
//  Public Struct Implementations
//=============================================================================

impl SexInferenceAccumulator {
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            constants: internal::constants::AlgorithmConstants::from_build(config.build),
            counters: internal::EvidenceCounters::default(),
        }
    }

    pub fn process_variant(&mut self, variant: &VariantInfo) {
        match variant.chrom {
            Chromosome::X => {
                self.counters.x_total_variants += 1;
                if variant.is_heterozygous {
                    self.counters.x_het_variants += 1;
                }

                if self.constants.is_in_x_par(variant.pos) {
                    self.counters.x_par_total += 1;
                    if variant.is_heterozygous {
                        self.counters.x_par_het += 1;
                    }
                } else if self.constants.is_in_x_non_par(variant.pos) {
                    self.counters.x_non_par_total += 1;
                    if variant.is_heterozygous {
                        self.counters.x_non_par_het += 1;
                    }
                }
            }
            Chromosome::Y => {
                if self.constants.is_in_y_par(variant.pos) {
                    self.counters.y_par_variants += 1;
                } else if self.constants.is_in_y_non_par(variant.pos) {
                    self.counters.y_non_par_variants += 1;
                }
                
                if self.constants.is_in_sry(variant.pos) {
                    self.counters.y_sry_variants += 1;
                }
            }
        }
    }

    pub fn finish(self) -> InferenceResult {
        let mut report = EvidenceReport::default();
        let mut male_votes = 0;
        let mut female_votes = 0;

        // Check 1: X Heterozygosity
        if self.counters.x_total_variants >= self.constants.min_x_variants_for_het_check {
            let ratio = self.counters.x_het_variants as f64 / self.counters.x_total_variants as f64;
            let vote = if ratio > self.constants.x_het_threshold {
                female_votes += 1;
                InferredSex::Female
            } else {
                male_votes += 1;
                InferredSex::Male
            };
            report.x_heterozygosity_check = Some((ratio, vote));
        }

        // Check 2: Y Presence
        let total_y_variants = self.counters.y_par_variants + self.counters.y_non_par_variants;
        if total_y_variants >= self.constants.min_y_variants_for_presence_check {
            let vote = if self.counters.y_non_par_variants > self.counters.y_par_variants {
                male_votes += 1;
                InferredSex::Male
            } else {
                female_votes += 1;
                InferredSex::Female
            };
            report.y_presence_check = Some((self.counters.y_non_par_variants, self.counters.y_par_variants, vote));
        }

        // Check 3: SRY Presence
        if self.counters.y_sry_variants > 0 {
            male_votes += 1;
            report.sry_presence_check = Some(InferredSex::Male);
        }

        // Check 4: PAR vs. Non-PAR Heterozygosity
        if self.counters.x_non_par_total > 0 {
            let non_par_het_rate = self.counters.x_non_par_het as f64 / self.counters.x_non_par_total as f64;
            if non_par_het_rate == 0.0 {
                male_votes += 1;
                report.par_non_par_het_check = Some((-1.0, InferredSex::Male));
            } else if self.counters.x_par_total > 0 {
                let par_het_rate = self.counters.x_par_het as f64 / self.counters.x_par_total as f64;
                let ratio = par_het_rate / non_par_het_rate;
                let vote = if ratio >= self.constants.par_non_par_ratio_threshold {
                    male_votes += 1;
                    InferredSex::Male
                } else {
                    female_votes += 1;
                    InferredSex::Female
                };
                report.par_non_par_het_check = Some((ratio, vote));
            }
        }
        
        let final_call = if female_votes > male_votes {
            InferredSex::Female
        } else {
            InferredSex::Male
        };
        
        report.final_male_votes = male_votes;
        report.final_female_votes = female_votes;

        InferenceResult { final_call, report }
    }
}

//=============================================================================
//  Tests
//=============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_variants(count: u64, chrom: Chromosome, pos_start: u64, is_het: bool) -> Vec<VariantInfo> {
        (0..count).map(|i| VariantInfo {
            chrom,
            pos: pos_start + i * 1000,
            is_heterozygous: is_het,
        }).collect()
    }

    #[test]
    fn test_clear_female_case() {
        let config = InferenceConfig { build: GenomeBuild::Build37 };
        let mut accumulator = SexInferenceAccumulator::new(config);

        let mut variants = generate_variants(1000, Chromosome::X, 30_000_000, true);
        variants.extend(generate_variants(250, Chromosome::X, 50_000_000, false));
        variants.extend(generate_variants(50, Chromosome::Y, 1_000_000, false));
        
        for variant in variants {
            accumulator.process_variant(&variant);
        }

        let result = accumulator.finish();
        assert_eq!(result.final_call, InferredSex::Female);
        assert_eq!(result.report.final_female_votes, 2);
        assert_eq!(result.report.final_male_votes, 0);
    }

    #[test]
    fn test_clear_male_case() {
        let config = InferenceConfig { build: GenomeBuild::Build37 };
        let constants = internal::constants::AlgorithmConstants::from_build(config.build);
        let mut accumulator = SexInferenceAccumulator::new(config);
    
        // This data is specifically crafted to trigger all four male-voting checks.
        let mut variants = Vec::new();
    
        // CHECK 1 DATA: X Heterozygosity (Expect Male Vote)
        // Create 1000 total X variants with a low heterozygosity rate (50/1000 = 0.05).
        // This is below the 0.1 threshold, voting Male.
        // Place them in the non-PAR region to also contribute to Check 4.
        variants.extend(generate_variants(50, Chromosome::X, constants.non_par_x.0 + 1, true));
        variants.extend(generate_variants(950, Chromosome::X, constants.non_par_x.0 + 1_000_000, false));
    
        // CHECK 2 & 3 DATA: Y Presence & SRY Presence (Expect 2 Male Votes)
        // Create >50 Y variants, with all of them in the non-PAR region.
        // This makes non-PAR count > PAR count, voting Male.
        // Also, place one variant specifically within the SRY region, voting Male again.
        variants.extend(generate_variants(100, Chromosome::Y, constants.non_par_y.0 + 1, false));
        variants.push(VariantInfo { chrom: Chromosome::Y, pos: constants.sry_region.0 + 1, is_heterozygous: false });
    
        // CHECK 4 DATA: PAR vs. Non-PAR Het Ratio (Expect Male Vote)
        // Add heterozygous variants to the X-PAR. This ensures `x_par_total > 0`.
        // The PAR het rate will be 1.0 (10/10). The non-PAR het rate is 0.05 (from above).
        // The ratio (1.0 / 0.05 = 20.0) is >= 2.0, voting Male.
        variants.extend(generate_variants(10, Chromosome::X, constants.par1_x.0 + 1, true));
        
        for variant in variants {
            accumulator.process_variant(&variant);
        }
    
        let result = accumulator.finish();
    
        // The final call must be Male.
        assert_eq!(result.final_call, InferredSex::Male);
    
        // With data for all four checks, we now expect 4 male votes and 0 female votes.
        assert_eq!(result.report.final_male_votes, 4, "Expected 4 male votes from 4 successful checks");
        assert_eq!(result.report.final_female_votes, 0);
    }

    #[test]
    fn test_male_par_vs_non_par_zero_het_case() {
        let config = InferenceConfig { build: GenomeBuild::Build37 };
        let constants = internal::constants::AlgorithmConstants::from_build(config.build);
        let mut accumulator = SexInferenceAccumulator::new(config);
        
        let variants = vec![
            VariantInfo { chrom: Chromosome::X, pos: constants.par1_x.0 + 1, is_heterozygous: true },
            VariantInfo { chrom: Chromosome::X, pos: constants.non_par_x.0 + 1, is_heterozygous: false },
        ];
        
        for variant in variants {
            accumulator.process_variant(&variant);
        }

        let result = accumulator.finish();
        let (ratio, vote) = result.report.par_non_par_het_check.unwrap();
        assert_eq!(vote, InferredSex::Male);
        assert_eq!(ratio, -1.0);
    }

    #[test]
    fn test_tie_break_defaults_to_male() {
        let config = InferenceConfig { build: GenomeBuild::Build37 };
        let mut accumulator = SexInferenceAccumulator::new(config);
        
        let mut variants = generate_variants(50, Chromosome::Y, 1_000_000, false);
        variants.push(VariantInfo {
            chrom: Chromosome::Y,
            pos: 2_787_000,
            is_heterozygous: false,
        });

        for variant in variants {
            accumulator.process_variant(&variant);
        }

        let result = accumulator.finish();
        assert_eq!(result.report.final_female_votes, 1);
        assert_eq!(result.report.final_male_votes, 1);
        assert_eq!(result.final_call, InferredSex::Male);
    }

    #[test]
    fn test_zero_variants_case() {
        let config = InferenceConfig { build: GenomeBuild::Build37 };
        let accumulator = SexInferenceAccumulator::new(config);
        let result = accumulator.finish();

        assert_eq!(result.final_call, InferredSex::Male);
        assert!(result.report.x_heterozygosity_check.is_none());
        assert!(result.report.y_presence_check.is_none());
        assert!(result.report.sry_presence_check.is_none());
        assert!(result.report.par_non_par_het_check.is_none());
    }

    #[test]
    fn test_insufficient_variants_case() {
        let config = InferenceConfig { build: GenomeBuild::Build37 };
        let mut accumulator = SexInferenceAccumulator::new(config);
        
        let variants = generate_variants(999, Chromosome::X, 10_000_000, true);
        for variant in variants {
            accumulator.process_variant(&variant);
        }
        let result = accumulator.finish();
        
        assert!(result.report.x_heterozygosity_check.is_none());
    }
}
