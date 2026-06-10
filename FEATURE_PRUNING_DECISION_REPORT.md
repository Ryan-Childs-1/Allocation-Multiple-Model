# v3.9 Feature-Pruning Decision Report

## Summary

The v3.8 AK specialist improved overall and AK-store MAE, but its AK Review path added a few low-unit false positives. The feature audit showed that the broad v3.7/v3.8 feature stack had useful signals, but the following feature families were either redundant, noisy, or risky for the deployed specialist layer:

- base-model-output-derived features in the primary feature set,
- duplicated demand summary features,
- unstable DC ratio features,
- sparse-demand flags used as positive rescue signals,
- low-unit AK Review rescue behavior when sheet-need support was weak.

## Features/rules pruned or de-emphasized

The next trainer keeps these features out of the primary whitelist or marks them as secondary-only:

```text
base_pred_units
base_flms
base_confidence
base_to_rec
base_x_rank
post_base_to_velocity
post_base_to_sheet_need
max_demand_signal
median_demand_signal
std_demand_signal
proj_to_dc
rec_to_dc
only_ttm_supports_demand
only_d60_supports_demand
only_proj_supports_demand
recent_sales_weak_but_proj_high
proj_high_but_no_recent_sales
rec_high_but_no_recent_sales
d60_high_l30_low_flag
ttm_high_recent_low_flag
site_is_high_error_candidate
site_is_high_overalloc_candidate
site_is_high_underalloc_candidate
site_requires_conservative_mode
site_requires_rescue_mode
```

## Features retained as high-value

The trainer and app keep the original sheet fields and the strongest engineered groups:

```text
Class Name, Line Name, Site, Rank, Flag
MIL, FLM, Cost, L30, D30, D60, LW, TTM, Supply, Dc Avail, Proj. Demand, Alloc. Rec.
weighted_velocity
sheet_need
mean_demand_signal
demand_consensus
demand_above_supply_share
high_conf_demand_share
shortage_sheet_flms
raw Dc Avail buckets
Proj. Demand / Alloc. Rec. agreement and gap features
one/two/three FLM outcome features
class-line competition ranks
class-line DC pressure
Site 802 specialist features
AK specialist features
```

## Deployed pruning action

The immediate app-level pruning is intentionally narrow: it removes only weak low-unit AK Review rescues where `shortage_sheet_flms < 2.5`. This avoids overfitting the smoke test while preserving the high-impact AK Allocate and higher-shortage AK Review rescues.
