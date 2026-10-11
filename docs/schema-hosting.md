# Schema hosting: `evalport.org` does not resolve

Status as of 2026-09-27. This page explains why the schema URLs in the spec are currently dead, what the `Pages` workflow in this repo does about it, and the one decision the repo owner has to make.

## The situation

Every EvalPort schema identifies itself with an `https://evalport.org/...` URL, and every example document points its `$schema` at one. The domain does not resolve:

| Check (2026-09-27) | Result |
|---|---|
| `getent hosts evalport.org` | no output, exit code 2 (not found) |
| `dig evalport.org A` | `status: NXDOMAIN`; the authority section is the `.org` zone's SOA (`a0.org.afilias-nst.info.`), i.e. the `.org` registry has no delegation for the name |
| `dig www.evalport.org A` | `status: NXDOMAIN` |
| `curl -sI https://evalport.org/schema/suite.json` | no HTTP response from the site: curl exit 56, `CONNECT tunnel failed, response 502` from the egress proxy these checks ran behind, which could not reach the host |
| RDAP, `.org` registry (`https://rdap.publicinterestregistry.org/rdap/domain/evalport.org`) | HTTP 404, `"title": "Object not found"` |
| `whois evalport.org` (port 43) | timed out from the environment these checks ran in; no answer either way. The RDAP row above is the same registry's data over HTTPS. |

The RDAP 404 means the `.org` registry (Public Interest Registry) has no registration record for `evalport.org`. That strongly suggests the name is unregistered. It does not prove the name can be bought, because registries can reserve names or price them as premium. A registrar's availability search is the authoritative check.

For context: the project's old name, `openeval.org`, does resolve. It has been registered since 2009-02-08 (registrar GoDaddy.com, LLC; expires 2027-02-08, per the same RDAP service; the registrant is not published). Nothing in this repo indicates the project controls it, and the spec never used it. `open-eval.org` returns NXDOMAIN.

### What is broken because of this

- **Editor validation.** VS Code and other JSON-Schema-aware editors fetch whatever `$schema` points to. Today that fetch fails, so users get no completion and no validation.
- **Cross-file `$ref`s.** `suite.json` references `https://evalport.org/schema/grader.json` and `https://evalport.org/schema/testcase.json`, and `testcase.json` references `grader.json`, all by absolute URL. A tool that loads `suite.json` from anywhere, even a working mirror, still has to fetch `grader.json` from `evalport.org`. So a mirror at a different host does **not** fully fix validation for `suite.json` and `testcase.json`. Only serving the files at the `evalport.org` URLs does, or changing the `$ref`s (Option B).
- **SchemaStore.** A SchemaStore catalog entry points editors at a schema URL. Registering URLs that do not resolve would ship the same breakage to every SchemaStore consumer.
- **Not broken:** the reference SDKs, their tests, and CI. `sdk/python/tests/test_schema_consistency.py`, `sdk/typescript/tests/schema-consistency.test.ts`, and the `validate-schemas` job in `.github/workflows/ci.yml` load all four files from disk and register them by `$id`, so `$ref` resolution never touches the network.

## URL inventory

These are all the `evalport.org` URLs in the repository (commit `3d64214`), found with `grep -rn "evalport.org"`. There are 51 matching lines.

**(a) Schema identity: `$id` and `$ref`.** There are 14 occurrences, 7 in `spec/schemas/*.json` and the same 7 in the byte-identical mirror `schema/*.json`:
- Each of the four schemas has `"$id": "https://evalport.org/schema/<name>.json"`.
- `suite.json` has 2 `$ref`s and `testcase.json` has 1, all to other schemas by absolute URL.
- CI and both SDK test suites build an offline registry keyed by these `$id` values. They read the `$id` from the files, so they are not hard-coded, but changing an `$id` is still a spec-level change (Option B).

**(b) `$schema` values in documents.** There are 28 occurrences:
- `SPEC.md` and `spec/SPEC.md`, 8 each: the example `TestCase`, `Grader`, `EvalSuite`, and `ResultSet`, plus the examples further down.
- `examples/*.json`: 6. `spec/examples/*.json`: 3.
- `cli/src/index.ts`: 1 (the `evalport init` template).
- `docs/getting-started/README.md`: 1.
- `adapters/longtracer-openeval-adapter` output: 1.

**(c) Prose and other URLs.** There are 9 occurrences:
- `SPEC.md` and `spec/SPEC.md` "Schema Evolution" (2 each): `.../schema/testcase.json` (latest) and `.../schema/v1.0.0/testcase.json` (pinned).
- `SPEC.md` and `spec/SPEC.md` "Extension Mechanism" (1 each): `https://evalport.org/extensions`, described as an extensions registry. No such registry content exists in this repo, so the Pages site does not publish one.
- SDK test comments (2): `https://evalport.org/schema/*.json`, a glob in a comment.
- `adapters/agenta-openeval-adapter` (1): `https://evalport.org/test-case/{raw_id}`, used only as a UUIDv5 namespace string and never fetched. It needs no hosting.

Distinct schema URLs: `/schema/{suite,testcase,grader,resultset}.json` and `/schema/v1.0.0/testcase.json`.

## What this PR adds

- **`scripts/build_pages_site.py`** builds a static site whose paths come from each schema's `$id`:
  ```
  index.html                      <- docs/landing-page.html
  schema/grader.json              <- spec/schemas/grader.json (byte-for-byte)
  schema/resultset.json
  schema/suite.json
  schema/testcase.json
  schema/v1.0.0/grader.json       <- same bytes (see "Pinned paths")
  schema/v1.0.0/resultset.json
  schema/v1.0.0/suite.json
  schema/v1.0.0/testcase.json
  ```
  The script then scans the repo for every `https://evalport.org/schema/<file>.json` URL and fails the build if one does not map to a built file. Run it locally with `python scripts/build_pages_site.py --out /tmp/site`.
- **`.github/workflows/pages.yml`** runs that script and deploys the result with `actions/configure-pages`, `actions/upload-pages-artifact`, and `actions/deploy-pages`. It needs the `pages: write` and `id-token: write` permissions. It runs on pushes to `main` that touch `spec/schemas/**`, `docs/landing-page.html`, the build script, or the workflow itself, and it can also be started manually (`workflow_dispatch`).
- **`docs/landing-page.html`**: its GitHub links pointed at `github.com/openeval/openeval`, which returns 404. They now point at this repo. Nothing else on the page changed.

No `$id`, `$ref`, or `$schema` value is changed by this PR.

Once Pages is enabled, the files are served at `https://adhabnr-ux.github.io/evalport/schema/suite.json` and so on. With a custom domain on a project site, GitHub serves the site at the domain root, so the same build answers `https://evalport.org/schema/suite.json`.

### Why there is no `CNAME` file

GitHub's docs: *"If you are publishing from a custom GitHub Actions workflow, no CNAME file is created, and any existing CNAME file is ignored and is not required."* With this workflow, the custom domain is configured in **Settings → Pages → Custom domain**, not in a file. Adding a `CNAME` file would have no effect.

### Pinned paths

`SPEC.md` documents `https://evalport.org/schema/v1.0.0/testcase.json` as the pinned URL form. The spec is still at `1.0.0-rc.x`, so for now the `v1.0.0/` copies carry the current schema bytes and change whenever `spec/schemas/` changes. They are not frozen yet. When 1.0.0 final is released, freeze them by changing `build_pages_site.py` to take `v1.0.0/` from the release tag (`git show v1.0.0:spec/schemas/<name>.json`) instead of the working tree. Note that the repo's existing `v1.x` tags (`v1.1.0`, `v1.3.0`, `v1.3.1`) are SDK release tags, not spec versions. Pinned copies keep the unversioned `$id` inside them. That is harmless for validation, but a strict implementation of pinning would give each pinned copy a versioned `$id`, which is itself an Option-B-style spec change.

## Owner actions

### Step 1 (either option): enable Pages

1. **Settings → Pages → Build and deployment → Source: "GitHub Actions".** Until this is done, `actions/configure-pages` fails and the workflow cannot deploy. That includes the first run the merge itself triggers.
2. Run **Actions → Pages → Run workflow** once, or push any change under `spec/schemas/`.
3. Check that `https://adhabnr-ux.github.io/evalport/schema/suite.json` returns 200.

### Option A (recommended if the domain can be registered): point `evalport.org` at Pages

With Option A, every existing `$id`, `$ref`, and `$schema` URL starts working, with **no spec change** and nothing for consumers to update.

1. Register `evalport.org` at any registrar. The registry currently has no record for it (see above), but confirm availability and price at the registrar.
2. Optionally, verify the domain for the `adhabnr-ux` account (your **account** Settings → Pages → Add a domain, not the repo's settings). GitHub recommends this to prevent takeover of the domain by other Pages sites.
3. In the repo: **Settings → Pages → Custom domain**: `evalport.org` → Save.
4. At the DNS provider, create the records GitHub documents for an apex domain:
   - `A` records for `evalport.org`: `185.199.108.153`, `185.199.109.153`, `185.199.110.153`, `185.199.111.153`
   - Optionally, `AAAA` records: `2606:50c0:8000::153`, `2606:50c0:8001::153`, `2606:50c0:8002::153`, `2606:50c0:8003::153`
   - `CNAME` for `www.evalport.org` → `adhabnr-ux.github.io` (the account's Pages host, without the repository name)
   - Do **not** use wildcard DNS records (`*.evalport.org`). GitHub warns they create an immediate domain-takeover risk.

   These values come from GitHub's "Managing a custom domain for your GitHub Pages site" doc (docs.github.com), checked on 2026-09-27. Re-check them there when you make the change.
5. After the DNS check passes and the certificate is issued, tick **Enforce HTTPS**.
6. Verify with `dig +short evalport.org` (it should list the four IPs) and `curl -sI https://evalport.org/schema/suite.json` (it should return 200).
7. Then the schemas can be submitted to SchemaStore under their `https://evalport.org/schema/*.json` URLs.

Keep the domain renewed. If it lapses, every URL breaks again, and a re-registration by someone else would let them serve arbitrary "schemas" at the spec's identifiers.

### Option B (if the domain is not available or not wanted): move the URLs via the RFC process

Change `$id`, `$ref`, and `$schema` to the Pages URL, for example `https://adhabnr-ux.github.io/evalport/schema/suite.json`. This is a normative spec change, so it goes through the RFC process in `.github/CONTRIBUTING.md`: a `[Spec Change]` Discussion, a two-week comment period, and spec-lead sign-off because it breaks backward compatibility. It should not be done in a docs/CI PR.

**What breaks:**

- **Consumers who keyed a schema registry on the old `$id`s.** That is the pattern this repo's own tests use and recommend (`Registry().with_resources([(schema["$id"], ...)])`, `ajv.addSchema(schema, schema.$id)`). Code that reads the `$id` from the file keeps working. Code that hard-codes `"https://evalport.org/schema/grader.json"` for lookups, `getSchema(...)`, or pre-seeded caches stops finding the schema.
- **Mixed versions.** A consumer that updates `suite.json` but keeps a cached or vendored old `grader.json` (or the reverse) gets `$ref`s that point at an `$id` it does not have. Resolution then fails, or it silently tries the network.
- **Bundled or dereferenced schemas** that other tools generated from the old files still embed the old URLs until they are regenerated.
- **Pinned URLs** (`/schema/v1.0.0/...`) move too. Anyone who pinned the old form has to switch.
- **The identifier becomes tied to a GitHub account and repo name.** The `github.io` host and path come from the account and repo names, so a future rename or transfer of `adhabnr-ux/evalport` changes the identifiers again. That would mean a second breaking change, which Option A avoids for good.

**What does not break:**

- Suite and result documents stay valid. In `suite.json` and `resultset.json`, `$schema` is just `{"type": "string"}`, so documents with the old `$schema` string still validate against the new schemas. (`testcase.json` and `grader.json` do not accept `$schema` at all. See "Related finding" below.)
- Semantic validation is unchanged. No field, type, or constraint changes.
- Editors that were already failing on the old URL gain nothing and lose nothing until documents are updated.

**Files an Option B PR must touch** (from the inventory above):
- The 14 `$id`/`$ref` occurrences, kept byte-identical between `spec/schemas/` and `schema/`, which CI enforces.
- The 28 `$schema` occurrences.
- The "Schema Evolution" prose, with `SPEC.md` and `spec/SPEC.md` kept byte-identical.
- The CLI `init` template and the `longtracer` adapter output.
- A changelog row in the spec.

`scripts/build_pages_site.py` currently requires `$id`s under `evalport.org`. Under Option B, update `SCHEMA_HOST` and `URL_RE` in the same PR, and note that for a project site the path inside the site stays `/schema/<name>.json`.

## Related finding (not changed here)

`testcase.json` and `grader.json` set `"additionalProperties": false` and do not declare a `$schema` property. As a result, the spec's own standalone `TestCase` example (`SPEC.md` §1) and `Grader` example (§2), both of which carry `"$schema": ...`, fail validation against those schemas: `Additional properties are not allowed ('$schema' was unexpected)`. This was checked with `jsonschema`'s `Draft202012Validator` and the four schemas registered by `$id`. `suite.json` and `resultset.json` declare `"$schema": {"type": "string"}` and accept their examples. Once the URLs resolve, an editor will show this error on any standalone test case or grader file that uses `$schema`. The fix, adding `"$schema": {"type": "string"}` to both schemas or dropping `$schema` from those two examples, is a schema change and belongs in its own PR.

## Stopgap that needs no decision

Until one of the options lands, users can point their editor at a working copy of a single schema without touching their documents, for example with a VS Code `json.schemas` mapping to `https://adhabnr-ux.github.io/evalport/schema/grader.json` once Pages is on. Today, both `https://raw.githubusercontent.com/adhabnr-ux/evalport/main/spec/schemas/suite.json` (`text/plain`) and `https://cdn.jsdelivr.net/gh/adhabnr-ux/evalport@main/spec/schemas/suite.json` (`application/json`) return 200. For `suite.json` and `testcase.json`, the absolute `$ref`s to `evalport.org` still fail from any mirror, as explained above.
