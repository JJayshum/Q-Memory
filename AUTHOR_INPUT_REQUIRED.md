# Author input required before ICASSP 2027 submission

The formatting and packaging work can be completed automatically. The items below require author identity, scientific judgment, or an explicit author attestation and therefore cannot be supplied by an automated system.

## 1. Mandatory author metadata

Please provide:

- Yujie Shen's ORCID URL in the form `https://orcid.org/0000-0000-0000-0000`.
- Haowen Chen's ORCID URL in the same form.
- Confirmation of the exact author names, order, affiliations, email addresses, and full postal mailing address for each author.
- Confirmation that Haowen Chen is the corresponding/contact author. The manuscript currently uses `hwchen@hnu.edu.cn`.

The PDF and submission form must match exactly. ICASSP 2027 states that missing ORCIDs or mismatched author lists cause automatic withdrawal.

## 2. Funding and ethics declarations

Please provide either the exact funding acknowledgement required by the sponsors or the statement `No funding acknowledgement is required`.

Please also confirm whether a `Compliance with Ethical Standards` statement is required under the IEEE Signal Processing Society policies. The present experiments appear synthetic and do not describe human or animal subjects, but this is an author/legal determination and cannot be inferred safely.

## 3. Scientific method decision

The current manuscript describes a decoder trained by Huber regression on centered action advantages. The headline scripts instead optimize classification objectives:

- `attention_memory_pilot.py`: cross-entropy on an action class.
- `ambiguous_memory_pilot.py`: cross-entropy on an action class.
- `multistate_memory_benchmark.py`: cross-entropy on an action class.
- `minigrid_qmemory_benchmark.py`: cross-entropy on memory-state labels.

Please choose one scientifically defensible route:

1. **Run the stated method.** Implement and rerun true centered-Q/advantage regression, then regenerate every affected table and claim.
2. **Reframe the paper.** Describe the implemented method as supervised policy-class memory/classification and remove unsupported Huber/Q-regression claims.

This decision changes the paper's central contribution. It must be made and verified by the authors. If route 1 is chosen, provide current GPU SSH access; one 24 GB NVIDIA GPU should be sufficient for these pilots, while the existing MiniGrid reruns can largely run without an LLM.

## 4. Frozen-Qwen control

`qwen_learned_multi.py` currently prompts the model with `The correct action is to open the {learned} door`. This reveals the decision semantically instead of presenting a neutral learned code. Either rerun with an opaque code and an independently defined decoder/interface, or remove/reframe the frozen-Qwen retention claim. A rerun requires current GPU SSH access and model-download availability.

## 5. Main-figure legibility

The supplied PNG is retained as requested, but some text inside it falls below the Paper Kit's 9 pt minimum at its current manuscript scale. Please provide the editable figure source, or explicitly approve a faithful redraw with larger labels. Scaling the existing raster wider may consume technical-page space and still leaves limited control over individual label sizes.

## 6. Required author verification under the LLM policy

Both authors must read and substantively verify the entire manuscript, all citations, all tables/figures, and the experiment code. ICASSP permits LLM assistance for language editing, code development, visualization, research, and ideation, but prohibits entirely LLM-produced sections and direct use of unverified generated text or code. The authors must rewrite any section they did not originate sufficiently to take full responsibility and ownership.

## 7. Final portal actions

- Enter the exact prepared title and 147-word abstract.
- Use primary topic `2.2 Reinforcement learning [ML-REI]` and secondary topic `2.3.7 Representation learning [ML-DLR-REPR]`, unless the authors deliberately select a different technical committee.
- Upload `shen.pdf` and complete the portal's document inspection.
- Confirm one author can present in person in Toronto if accepted and that the required non-student author registration can be completed by the conference deadline.
