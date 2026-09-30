import fs from "node:fs";
import path from "node:path";
import { test, expect } from "vitest";
import { validateDocument } from "../src/validate";
import type { DocumentType } from "../src/types";

interface ConformanceFixture {
  description: string;
  type: DocumentType;
  // PROPOSED (Discussion #108): absent/"strict" = default strict validation,
  // "allow_unknown" = lenient consumption (unknown properties accepted).
  mode?: "strict" | "allow_unknown";
  expect: {
    valid: boolean;
    error_paths?: string[];
  };
  document: unknown;
}

const fixturesDir = path.join(__dirname, "../../../spec/conformance/fixtures");
const fixtureNames = fs.readdirSync(fixturesDir)
  .filter((name) => name.endsWith(".json"))
  .sort();

for (const name of fixtureNames) {
  test(`conformance: ${name}`, () => {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(fixturesDir, name), "utf8")
    ) as ConformanceFixture;

    const mode = fixture.mode ?? "strict";
    expect(["strict", "allow_unknown"]).toContain(mode);
    const result = validateDocument(fixture.document, fixture.type, { allowUnknown: mode === "allow_unknown" });
    expect(result.valid).toBe(fixture.expect.valid);

    if (fixture.expect.error_paths) {
      const actualPaths = result.errors.map((error) => error.path);
      for (const expectedPath of fixture.expect.error_paths) {
        expect(actualPaths).toContain(expectedPath);
      }
    }
  });
}
