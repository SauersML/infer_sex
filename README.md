# infer_sex
A high-performance, zero-dependency Rust library for classifying samples based on sex chromosome characteristics from summarized variant data.

## About

This library uses a multi-pronged algorithm based on chromosome biology. It operates on an iterator of `VariantInfo` structs in a single pass, ensuring minimal memory usage and high throughput.

The core of the library is the `SexInferenceAccumulator`, a state machine that you feed variant data into to receive a classification and a detailed evidence report.

## Highlights

*   **Performant:** Zero-dependency and processes variants in a single stream.
*   **Memory Efficient:** State machine design uses constant memory.
*   **Transparent:** The final call is accompanied by a detailed `EvidenceReport` showing the result of each internal check.
*   **Simple API:** A straightforward, builder-like pattern for processing data.
*   **Robust:** Supports both GRCh37/hg19 and GRCh38/hg38 genome builds.

## Usage

The primary workflow involves creating a `SexInferenceAccumulator`, processing `VariantInfo` structs from your data source (e.g., a VCF file), and finalizing the analysis to get a result.

1.  Create a configuration for your genome build.
2.  Initialize the accumulator state machine.
    ```rust
    let mut accumulator = SexInferenceAccumulator::new(config);
    ```
3.  In your application, create `VariantInfo` structs from your data.
4.  Process all variants in a single pass.
5.  Finalize the analysis to get the result. This consumes the accumulator.
6.  Use the structured result.

## Algorithm

The final classification is determined by a voting system based on four biological checks.

1.  **X Chromosome Heterozygosity:** Compares the ratio of heterozygous to total variants on chromosome X against a threshold. A high ratio suggests one outcome, a low ratio suggests the other.
2.  **Y Chromosome Presence:** Evaluates the ratio of variants in the non-pseudoautosomal region (non-PAR) to the pseudoautosomal region (PAR) of chromosome Y. A high proportion of non-PAR variants provides strong evidence for one classification.
3.  **SRY Presence:** Checks for any variants within the SRY. The presence of such variants is a strong indicator for a specific outcome.
4.  **PAR vs. Non-PAR X Heterozygosity:** Compares the heterozygosity rate within the X-PAR to the rate in the X-non-PAR. A significantly higher rate in the PAR is a key indicator. Zero heterozygosity in the non-PAR region is also a special case providing strong evidence.

This algorithm has been validated on real-world microarray data.

## API Overview

*   `InferenceConfig`: Specifies the genome build (`Build37` or `Build38`).
*   `VariantInfo`: The input struct representing a single variant's chromosome, position, and heterozygosity.
*   `SexInferenceAccumulator`: The main state machine that consumes `VariantInfo` structs.
*   `InferenceResult`: The final output, containing the `final_call` (`InferredSex`) and the `report`.
*   `EvidenceReport`: A breakdown of the results and vote from each of the internal checks.
