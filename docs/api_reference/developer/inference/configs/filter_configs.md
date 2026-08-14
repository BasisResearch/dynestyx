# Filter Configurations

The single `Filter()` handler is directed to the appropriate filtering algorithm via the provided `FilterConfig`.

`include_predicted_observations` controls result-level collection of supported backend predictive-observation outputs. The `record_predicted_observations_mean`, `record_predicted_observations_cov`, and `record_predicted_observations_ensemble` fields separately control NumPyro recording. Collection and recording default to enabled; unavailable fields are omitted without making Filter fail.

::: dynestyx.inference.configs.filter
    options:
      filters: []
