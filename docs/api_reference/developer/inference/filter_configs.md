# Filter Configurations

The single `Filter()` handler is directed to the appropriate filtering algorithm via the provided `FilterConfig`.

Several shared fields on `BaseFilterConfig` now control predicted-observation recordings (`record_predicted_observations_mean`, `record_predicted_observations_cov`, and `record_predicted_observations_ensemble`). These backend summaries are distinct from the scoring path documented on the companion [Scoring](scoring.md) page: scoring can run without recording them into the trace.

::: dynestyx.inference.filter_configs
    options:
      filters: []
