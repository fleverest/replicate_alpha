# Code to reproduce (Stark 2022)

Replicate 'ALPHA: Audit that Learns from Previously Hand-audited Ballots' by Philip Stark.

Currently only reproduces Table 2.

* `generate_datasets.py` produces precisely the Bernoulli sequences used in the paper.
* `params.py` contains all parameters for the experiment.
* `compute_ss.py` computes the ALPHA martingales for each sequence.
