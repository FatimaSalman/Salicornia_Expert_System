[![DOI](https://zenodo.org/badge/1288978332.svg)](https://doi.org/10.5281/zenodo.23126238)

# Salicornia Expert System

A hybrid AI decision-support system for *Salicornia europaea* under salinity
stress. A Random Forest layer predicts fresh biomass and cell wall stiffness
(0-1000 mM NaCl), and a rule-based inference engine explains each prediction
using a published Pearson correlation matrix of cell wall traits as its
knowledge base.

**Live app:** https://salicorniaexpertsystem-fms.streamlit.app/

## Repository structure
- `app.py` - the deployed Streamlit application
- `analysis/` - scripts reproducing all metrics reported in the manuscript
  (LOO-CV, SHAP/LIME benchmark); see `analysis/requirements.txt`
- `generate_paper_figures.py` - manuscript figure generation

## Reproducing the analysis
    cd analysis
    pip install -r requirements.txt
    python shap_lime_benchmark.py

## License
MIT (see LICENSE). Data source: Cardenas Perez et al. 2026, Scientific
Reports 16, 964 (Table S1/S2), with 600/800 mM extensions from the data
provider.
