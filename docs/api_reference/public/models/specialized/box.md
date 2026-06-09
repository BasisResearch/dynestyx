# Box

::: dynestyx.models.wsp.Box
    options:
      show_root_heading: false
      show_root_toc_entry: false
      filters:
        - "!^__init__$"

!!! note "Domain for WSP"
    `Box` is an axis-aligned hyper-rectangle $\prod_d [a_d, b_d]$ that specifies the
    compact domain a [`WSP`](wsp.md)-wrapped SDE must remain inside. It supplies the
    geometric quantities WSP needs: the box center (its Chebyshev center), the
    per-dimension weight $w(x)$, the inward center pull, and the input-clip safeguard.
