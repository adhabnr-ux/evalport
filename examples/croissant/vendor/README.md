# Vendored Croissant Tasks files

`croissant-tasks.ttl` and `croissant-tasks-shapes.ttl` are copied unmodified
from [mlcommons/croissant](https://github.com/mlcommons/croissant) `tasks/` at
commit `401f6fff81db26a49c0d1704f02bffc4e4fa8fe2` (2026-07-15), licensed under
the Apache License 2.0 (see the upstream `LICENSE.md`).

They are vendored only so `../test_croissant_tasks.py` can validate the
exporter's output offline. `test_croissant_tasks.py` works around one known
upstream shape bug ("Bug A" in mlcommons/croissant#1025 and #1027, re-reported
with the other test-suite failures in mlcommons/croissant#1054) on an in-memory
copy; these files stay byte-identical to upstream.
