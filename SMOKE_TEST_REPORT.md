# v3.9 Feature-Pruned AK Smoke Test Report

Compared **v3.9 Feature-Pruned AK + Site 802** against **v3.8 AK Specialist** on the provided allocation workbooks. v3.9 prunes weak AK Review rescue features/rules while preserving the successful AK Allocate rescue path and the Site 802 specialist.

Files tested: 4

## Pruning parameters

```json
{
  "version": "v3.9_feature_pruned_ak",
  "ak_allocate_enabled": true,
  "ak_review_enabled": true,
  "allocate_min_confidence": null,
  "allocate_min_rec_flms": null,
  "allocate_min_high_conf_demand_share": null,
  "allocate_min_shortage_flms": null,
  "review_min_confidence": null,
  "review_min_rec_flms": null,
  "review_min_high_conf_demand_share": null,
  "review_min_shortage_flms": 2.5,
  "review_max_rec_trust": 1.0,
  "review_max_class_line_dc_pressure": 1.0,
  "review_remove_sparse_guardrail": false,
  "review_low_unit_rescue_max_units": 2.0,
  "notes": "Prunes weak low-unit AK Review rescues with less than 2.5 sheet-need FLMs while preserving larger, high-impact AK Review rescues and all successful AK Allocate rescues."
}
```

## Weighted row-level summary

| model               | segment     |   rows |   mae_weighted |   exact_weighted |   false_pos |   false_neg |   pred_units |   actual_units |   unit_delta |
|:--------------------|:------------|-------:|---------------:|-----------------:|------------:|------------:|-------------:|---------------:|-------------:|
| v3_8_ak_specialist  | All         |   7623 |       0.404434 |         0.892693 |         210 |         299 |        19826 |          19615 |          211 |
| v3_8_ak_specialist  | Allocate    |   3388 |       0.550177 |         0.866883 |         136 |         102 |        15720 |          15270 |          450 |
| v3_8_ak_specialist  | Review      |   4235 |       0.287839 |         0.913341 |          74 |         197 |         4106 |           4345 |         -239 |
| v3_8_ak_specialist  | AK Stores   |    455 |       0.773626 |         0.821978 |          24 |           8 |         2029 |           1887 |          142 |
| v3_8_ak_specialist  | AK Allocate |    220 |       1.06364  |         0.772727 |          16 |           0 |         1626 |           1518 |          108 |
| v3_8_ak_specialist  | AK Review   |    235 |       0.502128 |         0.868085 |           8 |           8 |          403 |            369 |           34 |
| v3_8_ak_specialist  | Non-AK      |   7168 |       0.380999 |         0.897182 |         186 |         291 |        17797 |          17728 |           69 |
| v3_8_ak_specialist  | Site 802    |    203 |       0.586207 |         0.812808 |           4 |          26 |          161 |            236 |          -75 |
| v3_9_feature_pruned | All         |   7623 |       0.404172 |         0.892956 |         208 |         299 |        19824 |          19615 |          209 |
| v3_9_feature_pruned | Allocate    |   3388 |       0.550177 |         0.866883 |         136 |         102 |        15720 |          15270 |          450 |
| v3_9_feature_pruned | Review      |   4235 |       0.287367 |         0.913813 |          72 |         197 |         4104 |           4345 |         -241 |
| v3_9_feature_pruned | AK Stores   |    455 |       0.769231 |         0.826374 |          22 |           8 |         2027 |           1887 |          140 |
| v3_9_feature_pruned | AK Allocate |    220 |       1.06364  |         0.772727 |          16 |           0 |         1626 |           1518 |          108 |
| v3_9_feature_pruned | AK Review   |    235 |       0.493617 |         0.876596 |           6 |           8 |          401 |            369 |           32 |
| v3_9_feature_pruned | Non-AK      |   7168 |       0.380999 |         0.897182 |         186 |         291 |        17797 |          17728 |           69 |
| v3_9_feature_pruned | Site 802    |    203 |       0.586207 |         0.812808 |           4 |          26 |          161 |            236 |          -75 |


## MAE delta, v3.9 minus v3.8

| segment     |   v3_8_ak_specialist |   v3_9_feature_pruned |   mae_delta_v39_minus_v38 |
|:------------|---------------------:|----------------------:|--------------------------:|
| AK Allocate |             1.06364  |              1.06364  |               0           |
| AK Review   |             0.502128 |              0.493617 |              -0.00851064  |
| AK Stores   |             0.773626 |              0.769231 |              -0.0043956   |
| All         |             0.404434 |              0.404172 |              -0.000262364 |
| Allocate    |             0.550177 |              0.550177 |               0           |
| Non-AK      |             0.380999 |              0.380999 |               0           |
| Review      |             0.287839 |              0.287367 |              -0.000472255 |
| Site 802    |             0.586207 |              0.586207 |               0           |


## File-average summary

| model               | segment           |   rows |   mae_file_avg |   exact_file_avg |   false_pos |   false_neg |   pred_units |   actual_units |   unit_delta |
|:--------------------|:------------------|-------:|---------------:|-----------------:|------------:|------------:|-------------:|---------------:|-------------:|
| v3_8_ak_specialist  | All               |   7623 |       0.44194  |         0.893479 |         210 |         299 |        19826 |          19615 |          211 |
| v3_8_ak_specialist  | Allocate          |   3388 |       0.573568 |         0.859061 |         136 |         102 |        15720 |          15270 |          450 |
| v3_8_ak_specialist  | Non-802           |   7420 |       0.442591 |         0.895032 |         206 |         273 |        19665 |          19379 |          286 |
| v3_8_ak_specialist  | Review            |   4235 |       0.301631 |         0.919113 |          74 |         197 |         4106 |           4345 |         -239 |
| v3_8_ak_specialist  | Site 802          |    203 |       0.573791 |         0.807898 |           4 |          26 |          161 |            236 |          -75 |
| v3_8_ak_specialist  | Site 802 Allocate |    138 |       0.761962 |         0.730702 |           2 |          21 |          142 |            203 |          -61 |
| v3_8_ak_specialist  | Site 802 Review   |     65 |       0.273077 |         0.924038 |           2 |           5 |           19 |             33 |          -14 |
| v3_9_feature_pruned | All               |   7623 |       0.441422 |         0.893997 |         208 |         299 |        19824 |          19615 |          209 |
| v3_9_feature_pruned | Allocate          |   3388 |       0.573568 |         0.859061 |         136 |         102 |        15720 |          15270 |          450 |
| v3_9_feature_pruned | Non-802           |   7420 |       0.442049 |         0.895573 |         204 |         273 |        19663 |          19379 |          284 |
| v3_9_feature_pruned | Review            |   4235 |       0.300418 |         0.920327 |          72 |         197 |         4104 |           4345 |         -241 |
| v3_9_feature_pruned | Site 802          |    203 |       0.573791 |         0.807898 |           4 |          26 |          161 |            236 |          -75 |
| v3_9_feature_pruned | Site 802 Allocate |    138 |       0.761962 |         0.730702 |           2 |          21 |          142 |            203 |          -61 |
| v3_9_feature_pruned | Site 802 Review   |     65 |       0.273077 |         0.924038 |           2 |           5 |           19 |             33 |          -14 |