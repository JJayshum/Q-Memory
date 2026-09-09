# Reference audit for ICASSP 2027 manuscript

## Automated consistency checks

- Bibliography entries: 18
- Distinct citation keys used in the manuscript: 18
- Citation keys missing from the bibliography: 0
- Bibliography entries never cited: 0
- LaTeX compilation: successful; no undefined citation/reference warnings

## Metadata review

The eight added foundational references were normalized against their canonical titles, author lists, publication venues, and years:

1. Kaelbling, Littman & Cassandra (1998), *Artificial Intelligence*, ``Planning and acting in partially observable stochastic domains.''
2. Puterman (1994), Wiley monograph, *Markov Decision Processes: Discrete Stochastic Dynamic Programming*.
3. Li, Walsh & Littman (2006), ISAIM proceedings, ``Towards a unified theory of state abstraction for MDPs.''
4. Ferns, Panangaden & Precup (2004), UAI, ``Metrics for finite Markov decision processes.''
5. Abel, Hershkowitz & Littman (2016), ICML, ``Near-optimal behavior via approximate state abstraction.''
6. Hausknecht & Stone (2015), arXiv:1507.06527, ``Deep recurrent Q-learning for partially observable MDPs.''
7. Kapturowski et al. (2019), ICLR, ``Recurrent experience replay in distributed reinforcement learning.''
8. Laskin, Srinivas & Abbeel (2020), ICML, ``CURL: Contrastive unsupervised representations for reinforcement learning.''

The original contextual references are retained because they are used to position memory/prompt-compression work. Their metadata is internally consistent with the supplied manuscript records, but the following recent entries should be checked against the final publisher or proceedings record before upload: Wang et al. (2025), Xu et al. (2025), Ouyang et al. (2026), Zhou et al. (2026), Wu et al. (2025), and Zhu et al. (2025). No DOI is claimed for these entries, so the manuscript does not contain a DOI-to-wrong-paper mismatch.

## Submission action

Before submission, confirm the six recent entries above using the official proceedings/publisher pages or the authors' final BibTeX. If any venue or publication status differs, update only the corresponding `\\bibitem` and recompile; the citation-key audit should remain unchanged.
