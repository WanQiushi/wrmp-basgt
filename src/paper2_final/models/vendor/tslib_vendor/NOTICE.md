# Vendored Time-Series-Library subset

The files in this directory are derived from the official
[`thuml/Time-Series-Library`](https://github.com/thuml/Time-Series-Library)
repository at commit `4e938a1767106324dd753b2a44832bf870a0252e`
(upstream commit date: 2026-04-18).

Included model implementations:

- `DLinear.py`
- `iTransformer.py`
- `TimeXer.py`

The supporting layer files are copied from the same revision. Imports were
made package-relative. Unused optional imports for Reformer, Informer masks,
and two-stage attention were removed from `SelfAttention_Family.py`; none of
the three included models uses those optional classes. Model mathematics and
forward paths used by these baselines are otherwise retained.

The upstream project is distributed under the MIT License. The complete
license text is retained in `LICENSE`.
