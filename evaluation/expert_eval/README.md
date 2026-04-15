# Expert Rating Study

This directory contains the materials for a multi-topic expert rating study that compares MAESTRO-generated research reports against published human literature reviews on three domain topics. The goal is to measure report quality along five rubric dimensions and to quantify the effect of configuration choices (research model, web search on/off) on perceived quality.

## Contents

- `rubric.md` - the five-dimension scoring rubric used by the raters
- `ai_reports/` - 24 MAESTRO-generated research reports that were rated

## Design

- **Topics (3):**
  1. *Cognitive biases in workplace decision-making* - baseline: Ohms, B. (2025), "A Systematic Literature Review of Cognitive Biases in Workplace Decision-Making," *International Journal of Business Administration*, 16(3). doi:10.5430/ijba.v16n3p26
  2. *AI integration in supply chain management* - baseline: Samuels, A. (2025), "Examining the integration of artificial intelligence in supply chain management from Industry 4.0 to 6.0: a systematic literature review," *Frontiers in Artificial Intelligence*. doi:10.3389/frai.2024.1477044
  3. *AI in hospital management* - baseline: Bhagat, S.V. & Kanyal, D. (2024), "Navigating the Future: The Transformative Impact of Artificial Intelligence on Hospital Management - A Comprehensive Review," *Cureus*, 16(2). PMC10955674
- **Models (4 generated, 3 analyzed):** Gemini 2.5 Flash, GPT-OSS-120B, Qwen3.5-27B. A fourth model (GLM) is included in `ai_reports/` for completeness but was excluded from the final quality analysis.
- **Conditions (2):** RAG-only (local corpus) and RAG + web search.
- **Raters (3):** three independent raters scored each (topic, model, condition) report alongside an anonymized version of the corresponding published baseline review. Reports were shuffled within each topic and labeled only with neutral identifiers, so raters did not know which report came from which source.
- **Rubric (5 dimensions, 1-5 Likert):** Correctness, Coverage, Coherence, Usefulness, Citation Quality. See `rubric.md` for operational definitions.

## `ai_reports/` naming

Files are named `topic<N>_<model>_<condition>.md` where:

- `<N>` in {1, 2, 3} - the three topics above
- `<model>` in {gemini, oss120, qwen3.5, glm}
  - `gemini` = Gemini 2.5 Flash
  - `oss120` = GPT-OSS-120B
  - `qwen3.5` = Qwen3.5-27B (run with thinking mode disabled)
  - `glm` = included for completeness; excluded from the final quality analysis
- `<condition>` in {rag-only, rag-web}

This gives 3 topics x 4 models x 2 conditions = 24 files.

## Results summary

### Inter-rater reliability (ICC)

| Dimension        | ICC(2,k) | F     | p       |
|------------------|----------|-------|---------|
| Correctness      | 0.741    | 4.46  | <0.001  |
| Coverage         | 0.838    | 7.22  | <0.001  |
| Coherence        | 0.822    | 5.99  | <0.001  |
| Usefulness       | 0.906    | 11.52 | <0.001  |
| Citation Quality | 0.888    | 11.40 | <0.001  |
| Total (sum of 5) | 0.897    | 13.38 | <0.001  |

ICC(2,k) is the two-way random effects average-measures model, appropriate for generalizing to other potential raters. All dimensions show statistically significant agreement (p < 0.001). Using established thresholds, values indicate good reliability for Correctness (0.741) and excellent reliability for every other dimension.

### Mean ratings by model and condition

Averaged across 3 topics and 3 raters. Columns: Corr. = Correctness, Cov. = Coverage, Coh. = Coherence, Use. = Usefulness, Cit.Q. = Citation Quality; Total is the sum across the five dimensions (maximum 25).

| Source                         | Corr. | Cov. | Coh. | Use. | Cit.Q. | Total |
|--------------------------------|-------|------|------|------|--------|-------|
| Published reviews (anonymized) | 2.11  | 2.89 | 3.22 | 2.44 | 1.78   | 12.44 |
| Gemini 2.5 Flash (RAG only)    | 2.56  | 3.67 | 3.33 | 2.67 | 1.44   | 13.67 |
| Gemini 2.5 Flash (RAG + web)   | 2.56  | 4.11 | 4.22 | 3.11 | 1.56   | 15.56 |
| GPT-OSS-120B (RAG only)        | 2.56  | 3.78 | 3.11 | 3.22 | 1.67   | 14.33 |
| GPT-OSS-120B (RAG + web)       | 2.33  | 4.22 | 3.89 | 3.67 | 2.00   | 16.11 |
| Qwen3.5-27B (RAG only)         | 2.33  | 4.56 | 4.11 | 3.78 | 2.44   | 17.22 |
| Qwen3.5-27B (RAG + web)        | 3.11  | 4.11 | 4.22 | 4.00 | 3.22   | 18.67 |

All MAESTRO-generated reports received higher total scores than the anonymized published baselines across every model and condition combination. The best configuration (Qwen3.5-27B with web search) scored 18.67 / 25 compared to 12.44 for the published baselines.

### Effect of web search

Averaged across 3 models and 3 topics.

| Dimension        | RAG Only | RAG + Web | Difference |
|------------------|----------|-----------|------------|
| Correctness      | 2.59     | 2.67      | +0.07      |
| Coverage         | 4.00     | 4.15      | +0.15      |
| Coherence        | 3.52     | 4.11      | +0.59      |
| Usefulness       | 3.22     | 3.59      | +0.37      |
| Citation Quality | 1.85     | 2.26      | +0.41      |
| **Total**        | **15.19**| **16.78** | **+1.59**  |

Enabling web search improved total scores by an average of 1.59 points, with the largest gains in Coherence (+0.59) and Citation Quality (+0.41).

## Data availability

The MAESTRO-generated reports in `ai_reports/` are released under the repository's AGPL-3.0 license. The anonymized published baseline reviews used for comparison are not redistributed: their full text remains the copyright of the respective original authors. The three baselines can be obtained from the DOIs listed under "Topics" above.
