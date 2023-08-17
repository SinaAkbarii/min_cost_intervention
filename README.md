# min_cost_intervention
Python implementation of the algorithms described in the paper [''Minimum Cost Intervention Design for Causal Effect Identification''](https://proceedings.mlr.press/v162/akbari22a.html), which was awarded ['outstanding paper runner-up'](https://icml.cc/virtual/2022/oral/17380) at ICML2022, getting shortlisted among the top 15 best papers of the conference.

Main algorithms, including Algorithm 2 in the paper (both exact and approximation), along with the described minimum-vertex-cut-based heuristic algorithms are included in the algorithms.

utils includes certain necessary base functions, and admg contains the main class for defining ancestral directed mixed graph (ADMG) instances. This class also provides a brute-force algorithm for sanity check.

min_cost.py illustrates an example of calling different algorithms provided in this work.

To cite this work:
Akbari, Sina, Jalal Etesami, and Negar Kiyavash. "Minimum cost intervention design for causal effect identification." International Conference on Machine Learning. PMLR, 2022.
