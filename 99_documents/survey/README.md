# Refinement practice in NTC model papers (Table 2, Fig. 1, and Table B.1 of the article)

`surveyed_papers.csv` lists the 68 peer-reviewed NTC model papers (2020-2025) of the survey in Section 2.2, with the
level assigned to each and the verbatim evidence from the paper (or its released code) on which the level rests.

| Level | Name | Meaning | Papers |
|---|---|---|---|
| 0 | no validation | no validation described | 10 |
| 1 | format-only | operations inside samples (header removal, address masking, truncation) or removal for format reasons only | 39 |
| 2 | conventional | removal of a few conventional task-irrelevant types without a definition of noise | 18 |
| 3 | systematic | systematic sample-level refinement under a defined notion of noise | 1 |

Papers per year (citation year): 2020: 11, 2021: 11, 2022: 10, 2023: 13, 2024: 12, 2025: 11.

Columns: `year` (citation year), `title`, `venue`, `level`, `feature_ops` (operations inside samples), `sample_ops`
(removal of samples), `evidence` (verbatim wording), `source` (where the text was read), `verified_from`.

Inclusion: peer-reviewed venue, published 2020-2025, proposes or substantially evaluates an ML or DL model for traffic
classification or intrusion detection on packet or flow data, and reports performance on at least one dataset; arXiv-only
papers, surveys, dataset papers, and benchmark-methodology studies are excluded. The full text or the released code was
read for every paper.

`excluded_papers.csv` lists one paper that was coded but removed from the survey because the publisher has issued a
notice of removal for it.
