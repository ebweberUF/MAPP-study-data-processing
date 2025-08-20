# MAPP Study Report Pipeline

## Overview
Multi-modal Assessment of Pain Prediction (MAPP) study data processing and reporting pipeline for observational research on pre and post-surgery outcomes.

## Final Report Pipeline

### Required Files
1. **Raw Data**: `data/MAPP_DATA_2025-08-20_0816.csv`
2. **Data Dictionary**: `metadata/MAPP_DataDictionary_2025-08-14.csv`

### Processing Scripts (Run in Order)
1. **`scripts/deidentified_labeled_separator.py`**
   - Separates events into individual CSV files
   - Combines EMA data by timepoints
   - Removes HIPAA identifiers while preserving subject_id
   - Filters variables based on data dictionary metadata
   - Creates labeled datasets with proper variable names

2. **`scripts/improved_visual_reports.py`**
   - Generates professional visualizations
   - Creates executive summary dashboard
   - Produces PROMIS outcomes charts
   - Generates pain trajectory analyses
   - Creates baseline demographics summaries

3. **`scripts/create_improved_html_report.py`**
   - Combines all visualizations into professional HTML report
   - Embeds images as base64 for portability
   - Creates stakeholder-ready presentation format

### Output Structure
```
data/deidentified_labeled_data_2025-08-20/
├── events/                     # Individual event CSV files
├── ema_combined/              # Combined EMA data by timepoint
├── saliva_data/               # Separated saliva variables
├── metadata/                  # Processing summaries and labels
└── improved_visual_reports/   # Final visualizations and HTML report
```

### Key Features
- HIPAA-compliant de-identification
- Metadata-based variable filtering
- Professional publication-quality visualizations
- Longitudinal pain trajectory analysis
- Comprehensive demographic summaries
- Portable HTML reports for stakeholders

### Usage

## Repository Notes

**⚠️ Data Privacy**: This repository contains only the processing scripts and metadata. All sensitive data files are excluded via `.gitignore` to maintain HIPAA compliance and data security.

### To use this pipeline:
1. Clone this repository
2. Add your raw data files to the `data/` folder (not tracked in git)
3. Run the scripts in the specified order
4. Generated outputs will be created in `data/deidentified_labeled_data_*/`

### Requirements
- Python 3.x
- Required packages: pandas, numpy, matplotlib, seaborn, plotly
- REDCap data export files
- Data dictionary metadata

````