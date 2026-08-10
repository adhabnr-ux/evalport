import { describe, expect, test } from "vitest";
import { JsonPathSyntaxError, parseJsonPath, queryJsonPath, queryJsonPathFirst } from "../../../src/run/graders/jsonPath";

describe("parseJsonPath", () => {
  test("requires a leading $", () => {
    expect(() => parseJsonPath("foo.bar")).toThrow(JsonPathSyntaxError);
  });

  test("parses dotted keys", () => {
    expect(parseJsonPath("$.a.b")).toEqual([{ kind: "key", key: "a" }, { kind: "key", key: "b" }]);
  });

  test("parses bracket index and quoted key", () => {
    expect(parseJsonPath("$.items[0]")).toEqual([{ kind: "key", key: "items" }, { kind: "index", index: 0 }]);
    expect(parseJsonPath('$["a"]')).toEqual([{ kind: "key", key: "a" }]);
  });

  test("parses wildcards in both forms", () => {
    expect(parseJsonPath("$.*")).toEqual([{ kind: "wildcard" }]);
    expect(parseJsonPath("$[*]")).toEqual([{ kind: "wildcard" }]);
  });

  test("rejects empty key segments and unclosed brackets", () => {
    expect(() => parseJsonPath("$.")).toThrow(JsonPathSyntaxError);
    expect(() => parseJsonPath("$.a[0")).toThrow(JsonPathSyntaxError);
  });
});

describe("queryJsonPath / queryJsonPathFirst", () => {
  const doc = { a: { b: [{ id: 1, name: "x" }, { id: 2, name: "y" }] }, top: "hi" };

  test("simple key traversal", () => {
    expect(queryJsonPathFirst(doc, "$.top")).toEqual({ found: true, value: "hi" });
  });

  test("array index, including negative", () => {
    expect(queryJsonPathFirst(doc, "$.a.b[0].id")).toEqual({ found: true, value: 1 });
    expect(queryJsonPathFirst(doc, "$.a.b[-1].id")).toEqual({ found: true, value: 2 });
  });

  test("wildcard fans out over array elements", () => {
    const names = queryJsonPath(doc, "$.a.b[*].name");
    expect(names).toEqual(["x", "y"]);
  });

  test("wildcard over object values", () => {
    const values = queryJsonPath({ x: 1, y: 2 }, "$.*");
    expect(values.sort()).toEqual([1, 2]);
  });

  test("missing path returns found:false", () => {
    expect(queryJsonPathFirst(doc, "$.nope")).toEqual({ found: false, value: undefined });
    expect(queryJsonPathFirst(doc, "$.a.b[99].id")).toEqual({ found: false, value: undefined });
  });

  test("indexing into a non-array yields nothing, not a throw", () => {
    expect(queryJsonPathFirst(doc, "$.top[0]")).toEqual({ found: false, value: undefined });
  });
});
