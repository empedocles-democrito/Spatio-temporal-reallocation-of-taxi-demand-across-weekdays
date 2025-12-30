[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18099753.svg)](https://doi.org/10.5281/zenodo.18099753)
# Spatio-temporal-reallocation-of-taxi-demand-across-weekdays
Data and code for spatio-temporal analysis of taxi demand across weekdays
# Data and code for spatio-temporal analysis of NYC taxi demand
**Zenodo DOI:** https://doi.org/10.5281/zenodo.18099753


## Data Description

This repository contains the data and Python scripts used to analyze
spatio-temporal variation in New York City taxi demand across weekdays.
The materials support replication of temporal divergence tests,
spatial hotspot identification, SIMPER-style decompositions,
and revenue-based analyses.

The repository is organized into two main folders: `data/` and `Python_script/`.

---

## Repository Structure

### 📁 data/

This folder contains the processed datasets used as inputs and key derived outputs.

- **`Taxi_manhattan_correct_datetime_numbers_with_day.csv`**  
  Main input dataset. Cleaned taxi trip records for Manhattan including
  pickup date, pickup time, weekday label, and numerical formatting.

- **`stdbscan_best_config.json`**  
  Configuration file for the calibrated ST-DBSCAN procedure
  (spatial radius and minimum samples).

- **`Taxi_clusters_STDBSCAN_fullweek.csv`**  
  Output of the global ST-DBSCAN clustering.
  Defines the canonical spatio-temporal hotspots used in all spatial analyses.

- **`daily_vectors_t30.csv`**  
  Normalized intra-day demand vectors (30-minute resolution) for each calendar day.

- **`permanova_global_t30.csv`**  
  Global PERMANOVA results for temporal profiles across weekdays.

- **`permanova_pairwise_t30.csv`**  
  Pairwise PERMANOVA comparisons between weekdays.

- **`bootstrap_by_year_t30.csv`**  
  Bootstrap validation of temporal PERMANOVA results using year blocks.

---

### 📁 Python_script/

This folder contains the Python scripts used to generate the results reported
in the associated article.

#### Core analytical scripts

- **`weekday_temporal_permanova_pipeline.py`**  
  Constructs normalized intra-day temporal profiles and performs
  Jensen–Shannon distance calculations and PERMANOVA tests across weekdays.

- **`spatial_weekday_permanova_pipeline.py`**  
  Performs spatial PERMANOVA and a SIMPER-style decomposition to identify
  hotspots driving weekday–weekend spatial divergence.

- **`stdbscan_final_calibrated.py`**  
  Identifies canonical spatio-temporal hotspots using a calibrated
  ST-DBSCAN algorithm applied globally.

- **`revenue_analysis_from_stdbscan.py`**  
  Analyzes fare and revenue concentration across spatio-temporal hotspots,
  allowing comparison between spatial persistence and economic dominance.

#### Supporting scripts

- **`simper_spatial.py`**  
  Computes spatial SIMPER-style contributions by hotspot.

- **`simper_temporal.py`**  
  Computes temporal SIMPER-style contributions across intra-day intervals.

---

## Experimental Design, Materials and Methods

1. **Temporal profiling**  
   Intra-day taxi demand is aggregated into fixed 30-minute intervals and
   normalized to probability distributions for each calendar day.

2. **Temporal divergence analysis**  
   Jensen–Shannon distance and PERMANOVA are used to assess systematic
   differences in temporal demand profiles across weekdays.

3. **Canonical hotspot identification**  
   A calibrated ST-DBSCAN algorithm is applied to the full dataset to identify
   spatially and temporally coherent demand hotspots that are held fixed
   across all subsequent analyses.

4. **Spatial contribution analysis**  
   A SIMPER-style decomposition quantifies the contribution of each hotspot
   to weekday–weekend spatial divergence.

5. **Economic relevance assessment**  
   Hotspot persistence is contrasted with revenue concentration to identify
   economically dominant activity regions.

---

## Usage Notes

All scripts require standard Python scientific libraries
(`numpy`, `pandas`, `scipy`, `scikit-learn`).

Example execution:

```bash
python Python_script/weekday_temporal_permanova_pipeline.py \
  --input data/Taxi_manhattan_correct_datetime_numbers_with_day.csv \
  --t 30
