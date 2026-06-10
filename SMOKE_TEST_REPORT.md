# v3.9 Feature Deep Dive and Feature-Importance Review

This file replaces the old smoke-test report. The Streamlit app now focuses this tab on feature behavior, model inputs, feature pruning, and how the v3.9 Feature-Pruned AK + Site 802 model uses the approved worksheet columns.

## Approved worksheet inputs

The model only uses these worksheet fields as raw inputs:

- Class Name
- Line Name
- Site
- MIL
- FLM
- Cost
- L30
- D30
- D60
- LW
- TTM
- Supply
- Dc Avail
- Rank
- Proj. Demand
- Alloc. Rec.
- Flag

Everything else is engineered from those fields. No outside data is required.

## Model paths

### Allocate path

Allocate rows use a two-stage structure:

1. A classifier decides whether the row should receive allocation.
2. A regressor estimates the number of FLMs to allocate.
3. The postprocessor applies integer rounding, FLM logic, and DC availability caps.

The most important Allocate features are usually the direct worksheet recommendation and demand/supply features:

- Alloc. Rec.
- Proj. Demand
- Supply
- Dc Avail
- FLM
- MIL
- L30 / D30 / D60 / LW / TTM
- shortage_sheet_flms
- demand agreement signals
- rec/proj agreement features
- one-FLM and two-FLM outcome features

### Review path

Review rows behave more like a prioritization problem. The Review model ranks rows by need and allocates until the available DC pool is exhausted. This makes competition-aware features especially important:

- class_line_need_percentile
- need_rank_in_class_line
- shortage_flm_rank_in_class_line
- class_line_dc_pressure_score
- review_competition_score
- demand_signal_share_above_supply
- high_conf_demand_share
- rank_score

### Site 802 specialist

The Site 802 specialist is used only for Site 802. It exists because Site 802 historically behaved differently from the rest of the workbook. The most important Site 802 feature patterns are:

- high recommendation but weak velocity
- projected-demand and recommendation disagreement
- single-FLM oversupply risk
- strong demand support
- cut / rescue candidate flags

### AK specialist

The AK specialist is used for sites 248, 159, 212, 145, and 121. These sites are handled separately because their allocation behavior differs from normal lower-48 stores. Useful AK features include:

- AK site flag
- shortage_sheet_flms
- rec_trust_score
- class_line_dc_pressure_score
- demand_signal_share_above_supply
- high_conf_demand_share
- single-FLM and below-FLM remainder logic

## Feature families

### 1. Original worksheet fields

These are intentionally preserved as first-class features because they represent the strongest business context in the sheet. The app strongly favors:

- Alloc. Rec. as the starting recommendation
- Proj. Demand as the demand target
- Supply as the current inventory position
- Dc Avail as the hard cap
- FLM as the pack-size constraint
- Rank as a store priority signal

### 2. Demand agreement features

Demand agreement features measure whether multiple demand signals say the store is under-supplied. These help reduce allocations where only one noisy column is high.

Examples:

- demand_signal_count_above_supply
- demand_signal_share_above_supply
- demand_signal_count_above_supply_plus_1flm
- demand_signal_share_above_supply_plus_1flm
- high_conf_demand_share

These are valuable because they distinguish between a well-supported allocation and a weak one-signal allocation.

### 3. Demand volatility and disagreement features

These features measure whether recent demand, medium-term demand, and long-term demand disagree.

Examples:

- demand_signal_range
- demand_signal_cv
- recent_vs_long_term_disagreement
- short_term_vs_medium_term_disagreement
- demand_disagreement_risk

High disagreement is not always bad, but it warns the model that demand may be stale, seasonal, or noisy.

### 4. Projection and recommendation trust features

These features compare Proj. Demand and Alloc. Rec. directly. They help the model decide whether to follow, cut, or rescue the recommendation.

Examples:

- rec_proj_abs_gap
- rec_proj_gap_flms
- rec_proj_agreement_flag
- rec_supported_by_recent_sales
- proj_supported_by_recent_sales
- rec_trust_score
- rec_cut_candidate
- rec_follow_candidate
- rec_add_candidate

These are especially useful on Allocate rows.

### 5. Pack-size and one-more-FLM features

The model needs to know whether one FLM is small, reasonable, or excessive. A single FLM may be trivial for one item and too aggressive for another.

Examples:

- flm_to_l30
- flm_to_d30
- flm_to_proj
- flm_to_weighted_velocity
- supply_after_1flm
- supply_after_2flm
- supply_after_3flm
- oversupply_after_1flm
- oversupply_after_2flm
- single_flm_oversupply_risk

These features directly help with exact-match rate because many errors are off by one FLM.

### 6. DC availability bucket features

The model uses both FLM-normalized DC availability and raw DC availability buckets. The raw buckets match the structure you requested:

- 0 / empty
- 1–25
- 26–50
- 51–100
- 101–300
- 301–600
- 601–1000
- 1001–2000
- 2000+

These features help the model behave differently when DC is tight versus abundant.

### 7. Class-line competition features

Allocation is competitive within a class-line group. These features help the model compare a row to its peers.

Examples:

- need_rank_in_class_line
- need_percentile_in_class_line_v37
- rec_rank_in_class_line
- velocity_rank_in_class_line
- shortage_flm_rank_in_class_line
- class_line_rows_competing_for_dc
- class_line_dc_pressure_score

These are especially important for Review rows.

### 8. Site-relative features

Site features help the model determine whether a store is unusually under-supplied compared with other stores.

Examples:

- site_need_percentile
- site_velocity_percentile
- site_rec_percentile
- site_supply_pressure_percentile
- site_share_of_class_line_need

These features support store-level prioritization without requiring external store data.

### 9. Cost-aware risk features

Cost should not dominate allocation, but it can help identify expensive mistakes.

Examples:

- allocation_cost_if_1flm
- allocation_cost_if_rec
- shortage_cost_proxy
- overalloc_cost_proxy
- cost_adjusted_oversupply_risk

These features help distinguish low-risk pack decisions from expensive over-allocation risk.

### 10. Sparse-demand guardrails

Sparse-demand features prevent weak allocations where demand is not supported by recent activity.

Examples:

- zero_l30_flag
- zero_d30_flag
- zero_d60_flag
- zero_lw_flag
- zero_ttm_flag
- zero_recent_sales_flag
- sparse_demand_count
- sparse_guardrail_score

These are especially useful for reducing false positives.

## Features removed or de-emphasized

The v3.9 pruning pass removed features that were likely redundant or harmful:

- old-model-output features such as base_pred_units, base_flms, and base_confidence
- unstable ratios such as proj_to_dc and rec_to_dc
- duplicate summary features when better demand-agreement features already existed
- weak sparse-demand single-signal flags that created noisy rescues
- high-error site trigger flags that were too broad and could overfit

## How to audit results

The Audit page compares predictions to an existing Final Alloc. column. Use it when the uploaded file already has manual or historical final allocations.

The most important audit metrics are:

- MAE Units: average absolute error in units
- Exact Rate: percent of rows where prediction exactly matches Final Alloc.
- Within 1 FLM: percent of rows within one pack size
- False Positives: predicted allocation where actual was blank or zero
- False Negatives: predicted blank or zero where actual had allocation
- Unit Delta: total predicted units minus actual units
- Over-DC Violations: predictions above available DC inventory

For model debugging, look at the segment breakdowns:

- Allocate rows
- Review rows
- Site 802 rows
- AK store rows
- Site 802 specialist applied
- AK specialist applied
- v3.9 pruning applied

Those segments show whether errors are coming from the base model or one of the specialist paths.
