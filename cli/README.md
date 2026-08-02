# @evalport/cli

Command-line tool for EvalPort — validate, convert, and init eval suites.

## Install

```bash
npm install -g @evalport/cli
```

## Commands

### validate

Validate an EvalPort document against its schema.

```bash
openeval validate my-suite.json
openeval validate my-suite.json --type=resultset
```

### convert

Convert between evaluation formats.

```bash
openeval convert promptfoo openeval config.json output.json
```

Supported conversions:
- `promptfoo` → `openeval`

### init

Create a starter eval suite.

```bash
openeval init my-eval-suite
# Creates my-eval-suite.json
```

### summary

Print summary statistics of a result set.

```bash
openeval summary results.json
```

## License

Apache 2.0
