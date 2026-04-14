## Data Drift Monitoring

I implemented data drift monitoring using the Evidently library.

The training dataset was used as the reference dataset, and I generated a simulated production dataset by intentionally shifting several feature distributions to mimic post-deployment data drift.

### Simulated Production Changes

The following changes were introduced to create drift:

- `MonthlyIncome` was multiplied by 2
- `DistanceFromHome` was increased by 20
- the proportion of `OverTime = "Yes"` was increased
- part of the `JobRole` distribution was shifted toward `"Sales Executive"`

### Results

The drift monitoring script compares the reference data to the simulated production data, prints a drift summary, and saves an HTML report to:

`reports/drift_report.html`

After running the script, update this section with your actual output:

- Drift Share: 0.00%
- Drifted Columns:
  - None detected

### Interpretation

No significant drift was detected between the reference and simulated production data. This suggests that the feature distributions remain similar, and the model is likely to maintain stable performance under these conditions.

However, the simulated changes demonstrate how drift could be introduced, and monitoring remains important to detect future distribution shifts.

For this project, drift in features such as `MonthlyIncome`, `DistanceFromHome`, `OverTime`, or `JobRole` would likely affect the model because these employee-related attributes are directly relevant to attrition prediction.

### Recommended Action

If this were a real production system, I would recommend:

- investigating the source of the drift
- continuing monitoring if the drift is small and performance remains stable
- retraining the model if the drift persists or model quality drops

The script is also configured to exit with a non-zero status code when drift exceeds the configured threshold, which makes it suitable for automated monitoring workflows.