# Project Charter

## Business Question
- How can we predict minutes_late for the Long Island Rail Road (LIRR) using weather conditions so commuters can adjust travel plans?
- What weather factors (e.g., precipitation, temperature, visibility) most strongly influence delays?
- At what time granularity (daily vs. line‑level aggregations) do weather features best explain delays?

## Stakeholder
- Primary: LIRR customers/commuters deciding whether to travel and which mode to use.
- Secondary: LIRR operations/planning teams who can use weather‑delay insights for staffing, maintenance, or service advisories.
- Tertiary: Course instructors/graders evaluating data engineering and modeling rigor.

## Success Metrics
### Technical
- Predict minutes_late with strong regression performance (low MAE/RMSE on a held‑out test set).
- Clean, reproducible data pipeline with documented steps for acquisition, cleaning, and feature engineering.
- Successful join between delay records and weather station data using time (day, month, year) with minimal missingness.
- Model interpretability: clear feature importance or partial dependence that aligns with domain intuition.

### Business
- Provide actionable delay estimates so commuters can choose whether to travel or select an alternative mode.
- Summarize weather patterns that meaningfully impact LIRR delays for operational awareness.
- Deliver insights in a format suitable for non‑technical stakeholders (short brief + visuals).

## Data Sources
- Primary: MTA LIRR delay records from data.ny.gov (CSV; historical delays since 2010).
- Enrichment: NOAA/NCDC Long Island weather station data (CSV).

## Integration Strategy
- Join key: time (day, month, year).
- Granularity: delay data by line; aggregate to match weather reporting granularity where needed.

## Risks & Mitigations
- Risk: High‑cardinality categorical features (e.g., weather_type, train_line) can inflate dimensionality.
  - Mitigation: consolidate categories, target encoding, or regularization.
- Risk: Missing or inconsistent timestamps between datasets.
  - Mitigation: date normalization, imputation rules, and sensitivity checks.

## Defined Roles
- Data Engineer / UI: Taylor Shipley — data ingestion, cleaning, joins, and any lightweight UI/visualization.
- Modeling Lead: Skyler Turner — model selection, training, evaluation, and error analysis.
- Project Lead / Git‑Project Manager: Mylee Anderson — coordination, repo hygiene, and milestone tracking.
- Reviewer / QA: Chase Powers — code review, reproducibility checks, and documentation QA.
- Research Lead: Mylee Anderson — literature scan, feature rationale, and interpretation write‑up.
