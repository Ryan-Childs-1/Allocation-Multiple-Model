# Allocation Split Expert v3.9 Feature-Pruned AK + Site 802 — Flat Streamlit App

This is a flat-folder Streamlit package for the v3.9 Feature-Pruned AK + Site 802 model branch.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Included model path

- Separate Allocate and Review two-stage models
- Site 802 specialist model
- AK specialist model for sites 248, 159, 212, 145, and 121
- v3.9 feature-pruned AK Review logic
- No residual correction model

## Expected input columns

Class Name, Line Name, Site, MIL, FLM, Cost, L30, D30, D60, LW, TTM, Supply, Dc Avail, Rank, Proj. Demand, Alloc. Rec., Flag.

## Output

The app fills `Final Alloc.` and returns a downloadable CSV plus an audit CSV.
