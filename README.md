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

The detected drift suggests that the incoming production data no longer follows the same distribution as the training data. This matters because the model was trained on the reference distribution, so substantial changes in numeric or categorical feature patterns can reduce model performance and reliability.

For this project, drift in features such as `MonthlyIncome`, `DistanceFromHome`, `OverTime`, or `JobRole` would likely affect the model because these employee-related attributes are directly relevant to attrition prediction.

### Recommended Action

If this were a real production system, I would recommend:

- investigating the source of the drift
- continuing monitoring if the drift is small and performance remains stable
- retraining the model if the drift persists or model quality drops

The script is also configured to exit with a non-zero status code when drift exceeds the configured threshold, which makes it suitable for automated monitoring workflows.