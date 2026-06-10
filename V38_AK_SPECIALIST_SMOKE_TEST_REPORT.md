# v3.8 AK Store Specialist Smoke Test Report

Compared **v3.8 AK Store Specialist** against **v3.7 Competition-Aware** on the provided allocation workbooks. AK stores are Sites **248, 159, 212, 145, and 121**. v3.8 keeps the v3.7 feature stack and adds separate AK Allocate and AK Review specialists similar to the Site 802 specialist.

Files tested: 4

## AK specialist configuration

```json
{
  "ak_sites": [
    "248",
    "159",
    "212",
    "145",
    "121"
  ],
  "model_type": "rule_rescue_specialists",
  "description": "Separate AK Allocate and AK Review specialists. They rescue likely false negatives from the v3.7 base path by allocating one FLM/remainder when original worksheet features support demand.",
  "segments": {
    "allocate": {
      "enabled": true,
      "rows": 220,
      "rescue_conf_min": 0.5,
      "rescue_rec_flms_min": 0.0,
      "rescue_high_conf_demand_share_min": 0.0,
      "rescue_shortage_flms_min": 0.0
    },
    "review": {
      "enabled": true,
      "rows": 235,
      "rescue_conf_min": 0.3,
      "rescue_rec_flms_min": 1.5,
      "rescue_high_conf_demand_share_min": 0.1,
      "rescue_shortage_flms_min": 0.0
    }
  }
}
```

## Weighted row-level summary

| model              | segment     |   rows |   mae_weighted |   exact_weighted |   false_pos |   false_neg |   pred_units |   actual_units |   unit_delta |
|:-------------------|:------------|-------:|---------------:|-----------------:|------------:|------------:|-------------:|---------------:|-------------:|
| v3_7_competition   | All         |   7623 |       0.411387 |         0.8923   |         207 |         305 |        19757 |          19615 |          142 |
| v3_7_competition   | Allocate    |   3388 |       0.560803 |         0.865998 |         136 |         105 |        15684 |          15270 |          414 |
| v3_7_competition   | Review      |   4235 |       0.291854 |         0.913341 |          71 |         200 |         4073 |           4345 |         -272 |
| v3_7_competition   | AK Stores   |    455 |       0.89011  |         0.815385 |          21 |          14 |         1960 |           1887 |           73 |
| v3_7_competition   | AK Allocate |    220 |       1.22727  |         0.759091 |          16 |           3 |         1590 |           1518 |           72 |
| v3_7_competition   | AK Review   |    235 |       0.574468 |         0.868085 |           5 |          11 |          370 |            369 |            1 |
| v3_7_competition   | Non-AK      |   7168 |       0.380999 |         0.897182 |         186 |         291 |        17797 |          17728 |           69 |
| v3_7_competition   | Site 802    |    203 |       0.586207 |         0.812808 |           4 |          26 |          161 |            236 |          -75 |
| v3_8_ak_specialist | All         |   7623 |       0.404434 |         0.892693 |         210 |         299 |        19826 |          19615 |          211 |
| v3_8_ak_specialist | Allocate    |   3388 |       0.550177 |         0.866883 |         136 |         102 |        15720 |          15270 |          450 |
| v3_8_ak_specialist | Review      |   4235 |       0.287839 |         0.913341 |          74 |         197 |         4106 |           4345 |         -239 |
| v3_8_ak_specialist | AK Stores   |    455 |       0.773626 |         0.821978 |          24 |           8 |         2029 |           1887 |          142 |
| v3_8_ak_specialist | AK Allocate |    220 |       1.06364  |         0.772727 |          16 |           0 |         1626 |           1518 |          108 |
| v3_8_ak_specialist | AK Review   |    235 |       0.502128 |         0.868085 |           8 |           8 |          403 |            369 |           34 |
| v3_8_ak_specialist | Non-AK      |   7168 |       0.380999 |         0.897182 |         186 |         291 |        17797 |          17728 |           69 |
| v3_8_ak_specialist | Site 802    |    203 |       0.586207 |         0.812808 |           4 |          26 |          161 |            236 |          -75 |


## MAE delta, v3.8 minus v3.7

| segment     |   v3_7_competition |   v3_8_ak_specialist |   mae_delta_v38_minus_v37 |
|:------------|-------------------:|---------------------:|--------------------------:|
| AK Allocate |           1.22727  |             1.06364  |               -0.163636   |
| AK Review   |           0.574468 |             0.502128 |               -0.0723404  |
| AK Stores   |           0.89011  |             0.773626 |               -0.116484   |
| All         |           0.411387 |             0.404434 |               -0.00695264 |
| Allocate    |           0.560803 |             0.550177 |               -0.0106257  |
| Non-AK      |           0.380999 |             0.380999 |                0          |
| Review      |           0.291854 |             0.287839 |               -0.00401417 |
| Site 802    |           0.586207 |             0.586207 |                0          |


## File-average summary

| model              | segment           |   rows |   mae_file_avg |   exact_file_avg |   false_pos |   false_neg |   pred_units |   actual_units |   unit_delta |
|:-------------------|:------------------|-------:|---------------:|-----------------:|------------:|------------:|-------------:|---------------:|-------------:|
| v3_7_competition   | All               |   7623 |       0.455667 |         0.892731 |         207 |         305 |        19757 |          19615 |          142 |
| v3_7_competition   | Allocate          |   3388 |       0.589814 |         0.857707 |         136 |         105 |        15684 |          15270 |          414 |
| v3_7_competition   | Non-802           |   7420 |       0.456969 |         0.894247 |         203 |         279 |        19596 |          19379 |          217 |
| v3_7_competition   | Review            |   4235 |       0.312378 |         0.919118 |          71 |         200 |         4073 |           4345 |         -272 |
| v3_7_competition   | Site 802          |    203 |       0.573791 |         0.807898 |           4 |          26 |          161 |            236 |          -75 |
| v3_7_competition   | Site 802 Allocate |    138 |       0.761962 |         0.730702 |           2 |          21 |          142 |            203 |          -61 |
| v3_7_competition   | Site 802 Review   |     65 |       0.273077 |         0.924038 |           2 |           5 |           19 |             33 |          -14 |
| v3_8_ak_specialist | All               |   7623 |       0.44194  |         0.893479 |         210 |         299 |        19826 |          19615 |          211 |
| v3_8_ak_specialist | Allocate          |   3388 |       0.573568 |         0.859061 |         136 |         102 |        15720 |          15270 |          450 |
| v3_8_ak_specialist | Non-802           |   7420 |       0.442591 |         0.895032 |         206 |         273 |        19665 |          19379 |          286 |
| v3_8_ak_specialist | Review            |   4235 |       0.301631 |         0.919113 |          74 |         197 |         4106 |           4345 |         -239 |
| v3_8_ak_specialist | Site 802          |    203 |       0.573791 |         0.807898 |           4 |          26 |          161 |            236 |          -75 |
| v3_8_ak_specialist | Site 802 Allocate |    138 |       0.761962 |         0.730702 |           2 |          21 |          142 |            203 |          -61 |
| v3_8_ak_specialist | Site 802 Review   |     65 |       0.273077 |         0.924038 |           2 |           5 |           19 |             33 |          -14 |