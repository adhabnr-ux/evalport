# autogen-openeval-adapter

Convert AutoGen (https://github.com/microsoft/autogen) agent evaluation tasks
and results to and from EvalPort (https://github.com/adhabnr-ux/evalport),
the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

As of August 2026, microsoft/autogen is in maintenance mode: it no longer
accepts new-feature pull requests, only bug fixes, security patches, and
documentation improvements. That means a native EvalPort adapter can't land
inside AutoGen itself going forward - see the discussion on PR #8009
(https://github.com/microsoft/autogen/pull/8009) and issue #8005
(https://github.com/microsoft/autogen/issues/8005).

This package fills that gap without needing to modify AutoGen: it works
against AutoGen's public eval task/result shapes (objects or dicts) from the
outside, so you get EvalPort import/export today.

## Install

```
pip install autogen-openeval-adapter
```

## Usage

```
from autogen_openeval_adapter import to_openeval, from_openeval

suite = to_openeval(my_autogen_eval_result)

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
f = open("my_suite.json", "w")
json.dump(suite, f, indent=2)
f.close()

tasks = from_openeval(suite)
```

By default, to_openeval() generates an exact_match grader (ignore_case=True).
For agent evals where exact string matching is too strict, pass
grader_type="llm_judge" instead:

```
suite = to_openeval(my_autogen_eval_result, grader_type="llm_judge")
```

expected_tools on a task (when present) is carried through as EvalPort's
expected_tools field, so tool-call verification round-trips correctly.

## Credit

The to_openeval() / from_openeval() mapping mirrors the design originally
proposed by the EvalPort spec author in microsoft/autogen#8005, and was
shaped by review of a draft implementation contributed by DresdenGman
(https://github.com/DresdenGman) in microsoft/autogen#8009.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 - see LICENSE.
