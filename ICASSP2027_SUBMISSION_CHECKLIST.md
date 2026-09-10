# ICASSP 2027 submission checklist

Verified on 10 September 2026 against the official [Call for Papers](https://2027.ieeeicassp.org/call-for-papers/), [submission portal](https://cmsworkshops.com/ICASSP2027/papers.php), [Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php), and [topic list](https://cmsworkshops.com/ICASSP2027/papers/paper_topics.php).

## Conference requirements

- Full-paper deadline: **16 September 2026** on the conference date line, shown by the portal as **20:00 on 17 September 2026 in Beijing**.
- Maximum length: four pages of technical content plus one optional fifth page containing only references, funding acknowledgements, and a Compliance with Ethical Standards statement.
- Format: two columns on US Letter or A4; minimum 9 pt type throughout, including figure labels and captions; no page numbers.
- PDF: unencrypted, first page first, all fonts embedded and subset, and no larger than 5 MB.
- Review is not double-blind. The PDF must contain the complete author list.
- The title, abstract, author names, author order, and affiliations in the PDF must exactly match the online form.
- Every listed author **must** provide a valid ORCID URL. A missing ORCID or PDF/form author mismatch causes automatic withdrawal.
- The web abstract should be approximately 100--150 words and match the PDF abstract exactly.
- The upload filename should be the first author's surname: `shen.pdf`.
- Authors must personally comply with the ICASSP 2027 LLM policy: no manuscript section may be entirely LLM-produced, and all generated text, visual content, and code must be thoroughly checked and owned by the authors.

## Prepared files

- Upload PDF: `shen.pdf`
- Main LaTeX source: `POLICY_CLASS_Q_MEMORY_ICASSP2027.tex`
- Web-form abstract: `ICASSP2027_ABSTRACT.txt` (147 words)
- Editable main figure: `policy_class_q_memory_main_figure.drawio`
- Manuscript figure export: `policy_class_q_memory_main_figure.pdf`
- Official 2027 style files: `spconf.sty` and `IEEEbib.bst`
- Submission source archive: `ICASSP2027_submission_source_v9.zip`

## Recommended portal entries

- Title: `Decision-Preserving Discrete Memory via Policy-Class Q-Distortion`
- Primary topic: **2.2 Reinforcement learning [ML-REI]**
- Secondary topic: **2.3.7 Representation learning [ML-DLR-REPR]**
- Corresponding author: **Haowen Chen**, `hwchen@hnu.edu.cn`
- Yujie Shen ORCID: `https://orcid.org/0009-0009-0447-9913`
- Haowen Chen ORCID: `https://orcid.org/0000-0002-4777-7525`
- Funding acknowledgment: included in the manuscript with grant/project identifiers `62472165`, `JYB2025XDXM602`, and `YLS-2026-ZY01002`.

## Automated checks

- [x] The final build has exactly five pages.
- [x] Page 5 contains only the funding acknowledgment and references, with no technical continuation.
- [x] The PDF is below 5 MB and is not encrypted.
- [x] Every font is embedded and subset; no Type 3 fonts are present.
- [x] No page numbers, clipping, overlapping tables, overfull boxes, or undefined citations are present.
- [x] The ASCII abstract exactly matches the rendered abstract.
- [x] Citation keys are defined, cited, and ordered by first appearance.

These boxes are updated only after rebuilding and inspecting the final artifact.

## Author checks

Complete every item in `AUTHOR_INPUT_REQUIRED.md` before submission. In particular, the authors must resolve the mismatch between the manuscript's centered-Q/Huber method description and the classification losses used by the headline experiment scripts. The current formatting checks cannot establish scientific validity.
