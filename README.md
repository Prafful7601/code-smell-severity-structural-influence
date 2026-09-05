# Code Smell Severity and Structural Influence in Extensive Open-Source Systems

Empirical analysis of code smell severity, persistence, and structural
complexity across 522 open-source repositories, built on the MLCQ
"Industry-Relevant Code Smell Data Set" (Madeyski & Lewowski, 2020).

Unlike a narrower Apache/Eclipse-only analysis, this project uses the
**full MLCQ export across every GitHub organisation it covers** —
9,517 (sample, smell) instances — and adds two measurements the raw
dataset does not provide on its own: how long a smell survives in a
project's git history before it's removed, and its structural complexity
(via the [CK](https://github.com/mauricioaniche/ck) tool) at the moment
it was reviewed.

## Authors

- Neelam Rawat
- Prafful Gupta ([prafful.25161161@kiet.edu](mailto:prafful.25161161@kiet.edu))
- Prashant Kumar Singh ([prashantkumar532004@gmail.com](mailto:prashantkumar532004@gmail.com))
- Mohd. Aatir
- Shweta Singh ([shweta.vidudi272@gmail.com](mailto:shweta.vidudi272@gmail.com))

Department of Computer Applications, Krishna Institute of Engineering &
Technology (KIET), Ghaziabad, Delhi-NCR, Uttar Pradesh, India.

## Hypotheses

- **H1**: industry-relevant projects carry a larger share of major and
  critical smells than other projects.
- **H2**: higher-severity smells persist across more revisions before
  removal.
- **H3**: co-occurring Blob + Data Class instances have higher structural
  complexity than isolated instances.
- **O4**: can a model trained on structural metrics and reviewer context
  predict severity with useful reliability?

## What's in this repository

```
scripts/           Full pipeline, in order:
                    01_build_instances.py   consensus severity per instance
                    02_clone_repos.py       blobless partial clone of every repo
                    03_persistence.py       git-history persistence walk (H2)
                    04_complexity.py        CK structural metrics (H3)
                    05_stats_tests.py       H1-H3 statistical tests
                    06_ml_classifier.py     O4: severity prediction models
                    07_feature_selection.py three-stage feature selection
                    08_paper_figures.py     regenerates the paper's data figures
                    09_edit_paper.py        applies real results into the manuscript
                    lib/                    shared helpers (git remap detection,
                                             the persistence proxy's entity finder)
data/               instances.csv, persistence.csv, complexity.csv
outputs/            figures/, and every stats/ML result as JSON
failures.csv        every instance that couldn't be processed, with a reason
METHOD.md           full methodology: operationalizations, bugs found and
                    fixed, exact attrition counts at every stage
RESULTS.md          every statistic reported in the paper, with exact values
requirements.txt    pinned dependency versions
```

**Not included**: the raw MLCQ export (`Test Smell Dataset.xlsx`) and the
`cache/`/`tools/` scratch directories the pipeline builds while running
(cloned repos, CK build) — both are large and fully reproducible from the
steps below.

## Data

This repo does not redistribute the raw dataset. Download it and place it
in the repo root before running the pipeline from scratch:

- Dataset (Zenodo): **https://doi.org/10.5281/zenodo.3590101**
- Paper: Lech Madeyski and Tomasz Lewowski. 2020. *MLCQ: Industry-Relevant
  Code Smell Data Set*. In **Proceedings of the 24th International
  Conference on Evaluation and Assessment in Software Engineering (EASE
  2020)**, April 15-17, 2020, Trondheim, Norway. ACM, New York, NY, USA,
  6 pages. **https://doi.org/10.1145/3383219.3383264**

## Reproduce

```bash
pip install -r requirements.txt
python scripts/01_build_instances.py
python scripts/02_clone_repos.py --all
python scripts/03_persistence.py --all
python scripts/04_complexity.py --all
python scripts/05_stats_tests.py
python scripts/06_ml_classifier.py
python scripts/07_feature_selection.py
python scripts/08_paper_figures.py
```

Random seed is fixed at `SEED = 42` throughout. Full methodology,
operationalizations, and every bug found and fixed along the way are
documented in [`METHOD.md`](METHOD.md); every reported statistic is in
[`RESULTS.md`](RESULTS.md).

## Known limitations

See `METHOD.md` §4 and `RESULTS.md` for the full picture — in particular:
persistence and complexity data are only available for 89.6% and 85.5%
of instances respectively (repos that moved/disappeared, CK crashes on a
handful of very large codebases, and other documented attrition causes);
H1 is not supported in this dataset; the ML classifier's absolute
performance is modest (macro-F1 0.53 across four severity grades) and
should be read as a triage signal, not a substitute for review.
