import re

with open("src/literalai_openeval_adapter/__init__.py", "r") as f:
    code = f.read()

# Fix 1: empty dict fallback
code = code.replace(
    "        if not value:\n            raise ValueError(\"cannot flatten an empty dict\")",
    "        if not value:\n            return \"{}\""
)

# Fix 2: map_grader_type explicit None check
code = code.replace(
    "    if not isinstance(literalai_type, str):",
    "    if literalai_type is None:\n        raise ValueError(\"Literal AI score type is missing (None)\")\n    if not isinstance(literalai_type, str):"
)
code = code.replace(
    "    if not isinstance(openeval_type, str):",
    "    if openeval_type is None:\n        raise ValueError(\"EvalPort grader type is missing (None)\")\n    if not isinstance(openeval_type, str):"
)

# Fix 3: clamp_score NaN check
clamp_orig = """    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise TypeError(f"score must be numeric, got {type(raw_value)!r}")

    clamped = float(raw_value)"""

clamp_new = """    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise TypeError(f"score must be numeric, got {type(raw_value)!r}")
    import math
    if math.isnan(raw_value):
        raise ValueError("score cannot be NaN")

    clamped = float(raw_value)"""
code = code.replace(clamp_orig, clamp_new)

with open("src/literalai_openeval_adapter/__init__.py", "w") as f:
    f.write(code)


with open("tests/test_adapter.py", "r") as f:
    test_code = f.read()

# Update empty dict test
test_code = test_code.replace(
    "    def test_empty_dict_raises(self):\n        with pytest.raises(ValueError):\n            flatten_dict_field({})",
    "    def test_empty_dict_serializes_to_empty_json(self):\n        assert flatten_dict_field({}) == \"{}\""
)

# Add NaN test
nan_test = """
    def test_nan_score_raises(self):
        import math
        with pytest.raises(ValueError, match="NaN"):
            clamp_score(float('nan'))
"""
test_code = test_code.replace("    def test_non_numeric_score_raises(self):", nan_test + "\n    def test_non_numeric_score_raises(self):")

with open("tests/test_adapter.py", "w") as f:
    f.write(test_code)

print("Patch applied.")
