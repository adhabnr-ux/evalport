with open("README.md", "r") as f:
    text = f.read()

# Replace the text about empty dicts to match what the user requested
old_text = "If the dict contains **no string-valued fields at all**, it serializes the whole\ndict to JSON. An arbitrary Literal AI row is still valid data that deserves a\n(documented) string representation, rather than a hard failure."

new_text = "If the dict contains **no string-valued fields at all**, it falls back to JSON-serializing the whole dict (instead of raising) — an empty dict still raises `ValueError`, since there's nothing to represent."

text = text.replace(old_text, new_text)

with open("README.md", "w") as f:
    f.write(text)
print("README updated")
