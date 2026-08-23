with open("src/literalai_openeval_adapter/__init__.py", "r") as f:
    code = f.read()

# Revert empty dict fallback to ValueError
code = code.replace(
    "        if not value:\n            return \"{}\"",
    "        if not value:\n            raise ValueError(\"cannot flatten an empty dict\")"
)

with open("src/literalai_openeval_adapter/__init__.py", "w") as f:
    f.write(code)


with open("tests/test_adapter.py", "r") as f:
    test_code = f.read()

# Revert empty dict test
test_code = test_code.replace(
    "    def test_empty_dict_serializes_to_empty_json(self):\n        assert flatten_dict_field({}) == \"{}\"",
    "    def test_empty_dict_raises(self):\n        with pytest.raises(ValueError):\n            flatten_dict_field({})"
)

with open("tests/test_adapter.py", "w") as f:
    f.write(test_code)

print("Revert applied.")
