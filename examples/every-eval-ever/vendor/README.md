# Vendored Every Eval Ever files

`eval.schema.json` is copied unmodified from
[evaleval/every_eval_ever](https://github.com/evaleval/every_eval_ever) at
`every_eval_ever/schemas/eval.schema.json`, schema `version` `0.3.0`, at commit
`1eb9d39aed34505e15db637153de72318bd946d4` (2026-09-28, the tip of `main` when this
was vendored; the file itself was last changed in commit
`09496fc02c6aa64ed696a95109be08bbd2164020`, 2026-09-09). Source URL:
https://raw.githubusercontent.com/evaleval/every_eval_ever/1eb9d39aed34505e15db637153de72318bd946d4/every_eval_ever/schemas/eval.schema.json

SHA-256 of the file: `c9c6195aec8a9dfa0b2aba4924ac1aa8a184c9d8ca97cee778cb531088e39b48`.
The commit SHAs came from the GitHub commits listing; the file bytes were fetched
from `raw.githubusercontent.com` at that commit SHA. They are also byte-identical to
`every_eval_ever/schemas/eval.schema.json` inside the `every-eval-ever==0.3.0` wheel
from PyPI (compared with `cmp` when this was vendored).

`LICENSE-every_eval_ever` is upstream's `LICENSE` at the same commit: the MIT License,
Copyright (c) 2026 EvalEval.

The file is vendored only so `../test_evalport_to_eee.py` can validate the converter's
output offline. The instance-level schema (`instance_level_eval.schema.json`) is not
vendored because the converter does not write instance-level records.
