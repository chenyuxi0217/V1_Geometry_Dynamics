# Population Dimensionality and Timescale Organization in Mouse V1

Analysis and visualization code accompanying **“Population dimensionality and timescale organization in mouse V1 under grating and natural movie stimulation”** by Yuxi Chen.

## Overview

This project uses the [Allen Brain Observatory Neuropixels Visual Coding dataset](https://observatory.brain-map.org/visualcoding/) to characterize the relationship between effective population dimensionality and modal decay timescales in mouse primary visual cortex (VISp). The primary analysis compares unaveraged responses to drifting gratings and `natural_movie_one` in the same 26 recording sessions.

Movie responses exhibited higher effective dimensionality and longer mean modal decay timescales than grating responses. Across sessions, dimensionality and mean timescale showed no consistent association, although movie responses showed a tentative negative association at 20-ms resolution. Exploratory analyses also suggested a relationship with the relative organization of slower and faster timescales, including a tendency toward relative spectrum compression at higher movie dimensionality. These unaveraged-response associations did not survive multiple-testing correction. The study provides a phenomenological characterization, rather than a test establishing mechanistic independence of geometry and dynamics.

## Repository layout

Use the following script names and directory structure. Paths are resolved relative to each script, so the Python files must remain in `scripts/`.

```text
V1_Geometry_Dynamics/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── scripts/
│   ├── main.py
│   ├── plot_figures.py
│   ├── filter_sessions.py
│   ├── grating_orientation_tf_counts.py
│   ├── get_movie_real_contents.py
│   └── ask_for_url.py
├── results/
│   ├── *.csv
│   └── figures/
└── allen_ecephys_cache/
    └── manifest.json
```

`results/` contains the processed analysis tables, and `results/figures/` receives the generated figures. `allen_ecephys_cache/` stores data and metadata retrieved through AllenSDK. The analysis and plotting scripts create their output directories; the SDK manages its cache.

Only `main.py` and `plot_figures.py` are required for the analysis-to-figure workflow. The other four scripts are optional inspection or download utilities. They do not need to be run before `main.py`.

## Environment and installation

The analyses reported in the manuscript were performed with **Python 3.10 on Windows 11**. No GPU is required.

Use any Python environment or IDE of your choice. The package versions used for the analysis are listed in `requirements.txt`. For example, the dependencies can be installed with:

```bash
pip install -r requirements.txt
```

The analysis scripts access the Allen Brain Observatory Neuropixels dataset through AllenSDK. Downloaded data are cached under allen_ecephys_cache/ and are not included in this repository.

## Reproduce figures from processed results

Place the complete processed CSV set from one analysis run in `results/`, then run:

```bash
python scripts/plot_figures.py
```

The plotting script reads 18 of the 22 analysis outputs listed below. It does not fit neural models or recompute the multiple-testing families. It uses stored inferential results and calculates the across-session summaries needed for display.

**The CSV files alone are not sufficient for an uncached first run.** Figure 1 uses actual frames from the Allen natural-movie template. AllenSDK and access to that cached template, or network access to retrieve it, are required. Plotting from processed tables does not require reloading the session-level spike recordings.

The script generates four main figures and ten Extended Data figures, each as **PNG and LZW-compressed TIFF at 300 dpi**, under `results/figures/`. Files with the same names are overwritten. Use processed tables from a single completed run.

## Run the full analysis

From the repository root:

```bash
python scripts/main.py
python scripts/plot_figures.py
```

`main.py` uses the fixed session lists near the beginning of the file. It obtains missing session data through AllenSDK, reuses cached downloads, performs the primary and supplementary analyses, and writes the processed tables after completing the statistical calculations and consistency checks.

All three branches run as part of the same program: primary unaveraged responses, identity-aligned repeat-averaged responses and their common-support comparisons, and neuron-count sensitivity. The scripts do not expose command-line switches for running individual branches.

Raw session files can be large. Disk usage and runtime depend on the selected sessions, existing cache, download speed, hardware; no fixed storage or runtime estimate is specified here. The optional session-screening utility can inspect more sessions than the fixed analysis cohort and therefore may download additional data.

## Analysis configuration

The supplied `main.py` implements the following study design:

| Item | Setting |
|---|---|
| Primary cohort | 26 sessions in `SESSION_IDS_40` |
| Primary population | 40 eligible VISp neurons, sampled without replacement |
| Unit screening | Validity and out-of-brain filtering; amplitude cutoff maximum 0.1, presence ratio minimum 0.95, ISI-violation maximum 0.5; finite mean firing rate at least 0.5 Hz |
| Gratings | Eight drift directions, 0–315 degrees in 45-degree steps, at 4 Hz |
| Movie | `natural_movie_one`; eight segment identities selected from 17 candidates |
| Window | 1.7 s; onset exclusion of 0.3 s as specified in the manuscript |
| Bin widths | 20, 50, and 100 ms: 85, 34, and 17 bins per window |
| Within-session sampling | 50 matched subset draws; selections shared across bin widths |
| Predictive evaluation | Five group-split seeds: 10, 42, 100, 2024, 999 |
| Spectral summaries | The 20 slowest valid stable modes |
| Correlation intervals | 10,000 session-bootstrap draws; pointwise 95% percentile intervals |
| Neuron-count sensitivity | Nested populations of 40, 50, and 60 neurons in the same 15-session cohort |

The 19-session `SESSION_IDS_50` list records the intermediate eligibility threshold. The neuron-count comparison uses the 15 sessions in `SESSION_IDS_60` **at all three population sizes**, not a different cohort for each size.

Neurons, stimulus identities, and repetitions are selected without replacement within a subset draw. Movie repetitions are shared across the selected segment identities. Final inclusion also requires valid analysis windows and complete movie sequences. `filter_sessions.py` screens stimulus availability and eligible-unit counts only; it does not duplicate this final window/repetition screening.

Effective dimensionality is the participation ratio of neuron-wise standardized activity. The dynamical model is an affine, unregularized ordinary least-squares autoregression. Current and next-bin states use the same standardization, and transition pairs never cross window boundaries. Modal decay times are calculated as `tau_ms = -bin_ms / ln(abs(eigenvalue))` for `1e-6 < abs(eigenvalue) < 1`.

Spectral shape is computed after averaging accepted spectra rank by rank within a session. The log-spectrum slope across ranks 1–20 is the main spectral-shape summary. Regional means summarize the slow tail (ranks 1–3), slower half (ranks 1–10), and faster half (ranks 11–20). Supplementary analyses include a tail log-ratio comparing ranks 1–3 with ranks 11–20 and a half-spectrum log-ratio comparing ranks 1–10 with ranks 11–20. “Faster half” refers to the retained spectrum, not the fastest modes of the full fitted system.

Repeat-averaged responses average repetitions **within each identity**, never across different identities. Their predictive evaluation divides repetitions into separate training and test groups before averaging. Direct unaveraged/repeat-averaged comparisons use their jointly valid subset draws and the same finite session set for each correlation.

### Statistical interpretation

The recording session, not an individual subset draw, is the unit of cross-session inference. Within-session standard deviations describe sampling variation; session-bootstrap intervals quantify uncertainty in cross-session correlations.

BH and BY adjustments are computed separately from the same unadjusted probabilities within each family:

| Family | Scope |
|---|---|
| F1 | Primary unaveraged condition effects |
| F2 | Primary unaveraged associations, including scalar, rank-resolved, cumulative-mode, spectral-shape, and prediction associations |
| F3 | Repeat-averaged condition effects |
| F4 | Repeat-averaged and distinct common-support associations |
| F5 | Neuron-count condition effects, pooled across tested sizes and bin widths |
| F6 | Neuron-count associations, pooled across sizes, bin widths, and relationship types |

Equivalent records are counted once only when their definitions, input values, session sets, and relevant sampling support agree. Applicable unchanged common-support references reuse the original F2 adjustment. The actual numbers of distinct and estimable tests are exported in `multiple_testing_family_summary.csv`.

`p_raw` and the original test-specific probability columns are unadjusted. `q_bh_family` and `q_by_family` contain the family-specific adjusted values. Figure asterisks use BH-adjusted values. Correlation intervals remain pointwise and are not simultaneous or multiplicity-adjusted bands.

## Output tables

A completed analysis writes **22 CSV files**. The final column indicates whether the current plotting script reads the file.

| File in `results/` | Contents | Plot input |
|---|---|:---:|
| `per_session.csv` | Primary session-level estimates and diagnostics | Yes |
| `subsampling_raw_data.csv` | Individual within-session subset draws | Yes |
| `acf_data.csv` | Session-level autocorrelation curves | Yes |
| `tau_spectrum_data.csv` | Retained session-level timescale spectra | Yes |
| `session_metadata.csv` | Eligibility, matching, exclusions, recorded animal labels, and runtime metadata | No |
| `summary_stats.csv` | Primary condition effects and selected summary associations | Yes |
| `pca_summary.csv` | Across-session cumulative PCA summaries | Yes |
| `tau_summary_dim_correlations.csv` | Scalar-timescale associations | Yes |
| `tau_mode_by_mode_dim_correlations.csv` | Associations at each retained rank | Yes |
| `tau_topk_dim_correlations.csv` | Associations for cumulative K-mode means | Yes |
| `tau_spectral_shape_per_session.csv` | Session-level spectral-region and shape summaries | Yes |
| `tau_spectral_shape_dim_correlations.csv` | Spectral-region and shape associations | Yes |
| `psth_per_session.csv` | Repeat-averaged estimates and diagnostics | Yes |
| `psth_summary_stats.csv` | Repeat-averaged condition effects | Yes |
| `psth_dim_correlations.csv` | Repeat-averaged associations | No |
| `raw_psth_common_per_session.csv` | Common-support estimates for both representations | Yes |
| `raw_psth_dim_correlations.csv` | Matched unaveraged/repeat-averaged association comparisons | Yes |
| `psth_qc_summary.csv` | Repeat-average model coverage and session counts | Yes |
| `neuron_count_sensitivity_per_session.csv` | Population-size estimates and common-support diagnostics | No |
| `neuron_count_sensitivity_summary_stats.csv` | Population-size condition effects | Yes |
| `neuron_count_sensitivity_dim_correlations.csv` | Population-size associations | Yes |
| `multiple_testing_family_summary.csv` | Six-family test inventory and discovery counts | No |

No Excel workbook is required.

## Figure files

Each stem below is saved with both `.png` and `.tiff` extensions.

| Manuscript figure | Output stem | Content |
|---|---|---|
| Figure 1 | `Figure1_Geometry` | Stimulus examples, PCA, and effective dimensionality |
| Figure 2 | `Figure2_Dynamics` | Autocorrelation and modal timescales |
| Figure 3 | `Figure3_Geometry_Dynamics_Forest` | Cross-session correlation summaries |
| Figure 4 | `Figure4_Spectral_Hierarchy` | Movie rank profile and log-spectrum slope |
| Extended Data Figure 1-1 | `Figure1-1` | PCA at 20 and 50 ms |
| Extended Data Figure 1-2 | `Figure1-2` | Within-session sampling variation |
| Extended Data Figure 1-3 | `Figure1-3` | Neuron-count sensitivity |
| Extended Data Figure 1-4 | `Figure1-4` | Fano factor and SNR |
| Extended Data Figure 2-1 | `Figure2-1` | Unaveraged prediction and stable-mode coverage |
| Extended Data Figure 4-1 | `Figure4-1` | Grating and stimulus-change rank profiles |
| Extended Data Figure 4-2 | `Figure4-2` | Cumulative K-mode associations |
| Extended Data Figure 4-3 | `Figure4-3` | Repeat-average and common-support comparisons |
| Extended Data Figure 4-4 | `Figure4-4` | Repeat-average prediction, conditioning, and coverage |
| Extended Data Figure 4-5 | `Figure4-5` | Tail and half-spectrum log-ratio correlations and scatter plots |
## Optional helper scripts

### Inspect candidate session lists

```bash
python scripts/filter_sessions.py
```

The script scans available sessions for both stimulus types and counts eligible VISp units above the firing-rate threshold. It prints candidate session lists for the thresholds in `HARD_FLOOR = [40, 50, 60]`, together with their minimum eligible-unit counts. It does not edit `main.py` or export analysis results. A candidate passing this screen must still pass the more detailed checks in `main.py`.

### Inspect grating direction and temporal-frequency availability

```bash
python scripts/grating_orientation_tf_counts.py
```

For the listed sessions, this script counts grating presentations by the SDK's `orientation` and `temporal_frequency` fields. It prints, for each temporal frequency, the number of sessions containing all eight expected directions and summaries of the within-session minimum repetition count across those directions.

These are counts in the retrieved stimulus tables, **before the main script's 1.7-s window and relevant-invalid-interval screening**. They need not equal the final usable repetition counts. The script prints its summary; it does not save a CSV or change the chosen 4-Hz analysis setting.

### View an example movie frame

```bash
python scripts/get_movie_real_contents.py
```

The script retrieves `natural_movie_one`, prints the template shape and data type, and displays the zero-based frame selected by `FRAME_IDX` (55 by default). It is a single-frame viewer, not a movie-segmentation or export step. It requires a graphical display for its interactive window. The plotting script independently selects frames 200, 213, and 226 for Figure 1.

### Print session download links

```bash
python scripts/ask_for_url.py
```

The script queries Allen's metadata API for the session NWB file associated with each entry in `TARGET_SESSIONS` and prints a download link or diagnostic message. It does not download the NWB files, retrieve movie templates, or register files in the SDK cache.

For manual downloads, preserve the cache layout expected by the SDK. With the supplied manifest location, the standard session-file path is `allen_ecephys_cache/session_<session_id>/session_<session_id>.nwb`. Additional cached metadata may still be required.

The normal SDK-managed download is the default workflow.

See the [AllenSDK cache documentation](https://allensdk.readthedocs.io/en/latest/allensdk.brain_observatory.ecephys.ecephys_project_cache.html) for dataset-access details.

## Citation and license

The associated manuscript is under review. For the underlying dataset, cite:

Siegle JH et al. (2021). *Survey of spiking in the mouse visual system reveals functional hierarchy.* Nature 592:86–92. DOI: 10.1038/s41586-020-03171-x.

Repository code is distributed under the MIT License; see `LICENSE`. Allen Institute datasets and stimulus templates are obtained separately; consult the provider's documentation for their access and reuse conditions.
