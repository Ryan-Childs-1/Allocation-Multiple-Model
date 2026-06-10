# Allocation Split Expert v3.2 Enhanced No-Residual

This version keeps the original v3 split model family and adds a no-residual demand/supply feature layer. It removes the v3.1 residual correction model entirely.

## Important behavior
- Original v3 Allocate/Review classifier-regressor predictions remain the base.
- Additional L30, D30, D60, LW, TTM, Supply, Proj. Demand, Alloc. Rec., FLM, and Dc Avail features are used in a deterministic adjustment layer.
- Original worksheet features are intentionally prioritized over derived-only signals.
- Review rows are mostly preserved because original v3 Review performance was already strong.
- Output artifacts are consolidated under `compact_artifacts/`.

## Test decision
On the holdout extracted sample test, v3.2 reduced false positives and total unit bias, but original v3 still had better row-level MAE and exact-match rate.
