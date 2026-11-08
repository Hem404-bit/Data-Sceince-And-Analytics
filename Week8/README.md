# Week 8 Capstone: Real World Business Analysis

The project uses all three provided datasets, analyzed independently: customer churn (500 rows), sales (100 rows), and house prices (300 rows). Only churn reaches the brief's stated 500-row minimum.

## Structure
- `data/`: original copies plus cleaned versions of each CSV. `raw_data.csv` is the churn source in the requested template.
- `notebooks/1_data_cleaning.ipynb`: quality checks and cleaning for all three datasets.
- `notebooks/2_eda.ipynb`: cross-domain EDA and visualizations.
- `notebooks/3_analysis.ipynb`: churn statistical checks and house-price regression.
- `reports/`: executive summary and technical report in PDF, with TXT source copies.
- `presentations/business_presentation.pptx`: 12-slide editable deck.
- `outputs/`: 11 visualizations and analysis results.

## Run in VS Code
Install dependencies with `python -m pip install -r requirements.txt`, then run notebooks in order. `build_capstone.py` regenerates the cleaned CSVs, charts, reports, and analysis results. `make_presentation.py` regenerates the PowerPoint.

These are exploratory analyses. Datasets are unrelated and are not merged. Currencies and sampling frames are unknown; associations are not causal.
