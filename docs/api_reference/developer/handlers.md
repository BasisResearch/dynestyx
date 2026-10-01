# Handlers

The private `dynestyx.handlers._dynestyx_stack_kind()` operation reports built-in
interpretation kinds from innermost to outermost. Its default is `[]`; each
interpretation prepends its `_DynestyxStackKind` member to `fwd()`. For example,
`with Filter(), Discretizer():` reports `[DISCRETIZER, FILTER]`. Repeated kinds
are preserved, and unrelated effects do not contribute entries. This is an
internal operation, not part of the package's public API.

::: dynestyx.handlers
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - condition
        - sample
        - plate

::: dynestyx.api
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - log_prob
        - simulate
