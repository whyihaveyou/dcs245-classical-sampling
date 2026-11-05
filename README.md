# Classical sampling of the QAT #245 circuit

Author: Zhongpu Qiu. Affiliations: Shanghai Artificial Intelligence Laboratory; Beijing Normal University.

Data and code for our classical sampling of `ibm_boston_dcs_nq64_depth73_checks12` ([QAT #245](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/245); S. Martiel et al., arXiv:2607.25941): 64 qubits, depth 73, 314 T gates, ring.

| Quantity | Value |
|---|---:|
| Samples | 1,389 |
| Fidelity lower bound (95%, 64 preselected audits) | 0.908 |
| GPUs | 8 × NVIDIA H200 on one node, one GPU per worker |
| Worker start to last sample written (slowest worker) | 864 s |
| Generation only / including audits (slowest worker) | 852 s / 950 s |
| Quantum runtime (#245) | 870 s |

## Method in brief

Each T gate is replaced by a gadget qubit, which turns the circuit into a 378-qubit stabilizer state; amplitudes are contracted vertex by vertex over the corresponding graph state (linear rank-width contraction). For each sample we draw a uniform 44-bit prefix, compute all 2^20 amplitudes of the remaining 20 bits in one contraction, and draw those bits exactly. With D = 2^44 and p_y the true prefix probability, the sampled state has fidelity F = (E_y sqrt(D p_y))^2 with the ideal state. For 64 sample indices fixed before the run, an independent complex128 contraction recomputes the block; the overlaps c_y satisfy sqrt(F) = E c_y, and a one-sided Chernoff bound gives F ≥ 0.908 at 95% confidence.

## Layout

- `inputs/`: circuit (SHA-256 `7a0b8aa6…73e5`, identical to the #245 attachment), injected graph, contraction plans, provenance.
- `source/`: contraction and sampling code (`dcs_rankwidth/`, `kernel_opt/`, `sampling/`, `production/`, `review/`). The cluster file-transfer module listed in `manifest.json` is not included; it plays no role in the computation.
- `manifest.json`, `protocol.json`: frozen hashes, seeds, sample count and audit indices, written before the run.
- `run_r1/worker<k>/`: per-worker records (random-number states, prefixes, CDF intervals, bitstrings, vector hashes), audit results, hardware and timing. The 16 MB conditional vectors behind the audits are not included; their hashes are in the records and they can be regenerated from the frozen inputs.
- `run_r1/analysis/`: independent verification of all records and audits.

## Attribution

The contraction code starts from Kim Renaud's [dcs-rankwidth](https://github.com/kim-renaud/dcs-rankwidth) (Apache-2.0; see `source/LICENSE` and `source/NOTICE`), used for #249. The uniform-prefix strategy follows H. Manabe, H. Gu and F. Pan, arXiv:2608.13110, Sec. 4.2.
