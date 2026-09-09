# ICASSP 2027 submission checklist

## Verified against the official CFP and Paper Kit

- Full-paper deadline: **16 September 2026**.
- Regular paper maximum: **5 pages total**.
- Pages 1--4 may contain technical content, figures, tables, and references.
- Page 5, if used, may contain only references, funding acknowledgements, and a Compliance with Ethical Standards statement.
- Two-column layout, minimum 9 pt type, no page numbers.
- Submission is **not double-blind**; the PDF must contain the author list.
- Official LaTeX files: `spconf.sty` and `IEEEbib.bst` from the 2027 Paper Kit.
- Paper topics relevant to this work include Machine Learning and Generative AI, Signal Processing Theory & Methods, Signal Processing for Robotics, and Signal Processing for Memory and Storage.

## Files

- Main source: `POLICY_CLASS_Q_MEMORY_ICASSP2027.tex`
- Main figure source image: `policy_class_q_memory_main_figure_clear.png`
- Official style: `spconf.sty`
- Bibliography style: `IEEEbib.bst`
- Compiled PDF: `icassp2027_build/POLICY_CLASS_Q_MEMORY_ICASSP2027.pdf` (5 pages; page 5 references only)
- Abstract text for the web form: `ICASSP2027_ABSTRACT.txt`

## Required author action before upload

1. Use exactly the same author names and order in the online submission form; ICASSP warns that mismatches can withdraw a paper.
2. Mark **Haowen Chen (陈浩文)** as the corresponding author and use `hwchen@hnu.edu.cn` in the submission form.
3. Add ORCID identifiers for all listed authors if requested by the submission system.
4. Select the most accurate technical committee, likely **Machine Learning for Signal Processing** or **Signal Processing Theory and Methods**; confirm the final taxonomy when the submission form opens.
5. Verify every reference against the final source/DOI record and add a stable public repository URL for code and data.
6. Run the PDF eXpress / IEEE PDF compliance check required by the submission portal, then upload the checked PDF and ASCII abstract.

## Current technical status

- The manuscript compiles with the official style and `pdflatex`.
- Current output is 4 pages on US Letter, within the 5-page maximum, with technical content and references distributed across all four pages.
- With the main figure included, the current output is 5 pages: pages 1--4 contain technical content and page 5 contains references only, as permitted by the Paper Kit.
- The reference list now contains 18 entries. Classical POMDP/state-abstraction entries were added and cited in the Related Work section; the existing recent memory/compression entries were retained as contextual support.
- Visual inspection found no clipped text, table overlap, or page numbers.
- The manuscript intentionally excludes open-ended agent benchmarks and limits claims to policy-relative Q-memory evidence.
