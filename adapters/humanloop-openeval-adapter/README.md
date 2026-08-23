# humanloop-openeval-adapter

Convert Humanloop AI datasets and evaluation runs to/from the EvalPort Open Standard format.

## Installation

```bash
pip install humanloop-openeval-adapter
```

Or install with humanloop extras:
```bash
pip install "humanloop-openeval-adapter[humanloop]"
```

## Usage

### Converting Datasets (to_openeval / from_openeval)

```python
from humanloop_openeval_adapter import to_openeval, from_openeval
from humanloop.types import DatapointResponse

# List of Humanloop datapoints
datapoints = [
    DatapointResponse(
        id="dp_1",
        inputs={"question": "What is 2+2?"},
        target={"target": "4"}
    ),
    DatapointResponse(
        id="dp_2",
        messages=[
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side."}
        ]
    )
]

# Convert to EvalPort suite
suite = to_openeval(datapoints, suite_id="my_humanloop_dataset")

# Reconstruct back to Parea-compatible dicts
reconstructed = from_openeval(suite)
```

### Converting Evaluation Runs (result_to_openeval)

```python
from humanloop_openeval_adapter import result_to_openeval
from humanloop.types import EvaluationResponse, EvaluatorLogResponse

# Convert logs and evaluator definition to EvalPort ResultSet
result_set = result_to_openeval(
    evaluation=evaluation_response,
    logs=logs_list
)
```
