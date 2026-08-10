import { describe, expect, test } from "vitest";
import { validateAgainstJsonSchema } from "../../../src/run/graders/jsonSchema";

describe("validateAgainstJsonSchema", () => {
  test("type: string passes/fails", () => {
    expect(validateAgainstJsonSchema("hi", { type: "string" }).valid).toBe(true);
    const r = validateAgainstJsonSchema(5, { type: "string" });
    expect(r.valid).toBe(false);
    expect(r.errors[0].message).toMatch(/expected type string/);
  });

  test("type array (union types)", () => {
    expect(validateAgainstJsonSchema(5, { type: ["string", "number"] }).valid).toBe(true);
    expect(validateAgainstJsonSchema(true, { type: ["string", "number"] }).valid).toBe(false);
  });

  test("integer vs number distinction", () => {
    expect(validateAgainstJsonSchema(5, { type: "integer" }).valid).toBe(true);
    expect(validateAgainstJsonSchema(5.5, { type: "integer" }).valid).toBe(false);
    expect(validateAgainstJsonSchema(5.5, { type: "number" }).valid).toBe(true);
  });

  test("const", () => {
    expect(validateAgainstJsonSchema("x", { const: "x" }).valid).toBe(true);
    expect(validateAgainstJsonSchema("y", { const: "x" }).valid).toBe(false);
  });

  test("enum", () => {
    expect(validateAgainstJsonSchema("b", { enum: ["a", "b", "c"] }).valid).toBe(true);
    expect(validateAgainstJsonSchema("z", { enum: ["a", "b", "c"] }).valid).toBe(false);
  });

  test("string length and pattern bounds", () => {
    expect(validateAgainstJsonSchema("ab", { minLength: 3 }).valid).toBe(false);
    expect(validateAgainstJsonSchema("abcd", { maxLength: 3 }).valid).toBe(false);
    expect(validateAgainstJsonSchema("abc", { pattern: "^[a-c]+$" }).valid).toBe(true);
    expect(validateAgainstJsonSchema("abcX", { pattern: "^[a-c]+$" }).valid).toBe(false);
  });

  test("number bounds", () => {
    expect(validateAgainstJsonSchema(5, { minimum: 6 }).valid).toBe(false);
    expect(validateAgainstJsonSchema(5, { maximum: 4 }).valid).toBe(false);
    expect(validateAgainstJsonSchema(5, { exclusiveMinimum: 5 }).valid).toBe(false);
    expect(validateAgainstJsonSchema(5, { exclusiveMaximum: 5 }).valid).toBe(false);
    expect(validateAgainstJsonSchema(9, { multipleOf: 3 }).valid).toBe(true);
    expect(validateAgainstJsonSchema(10, { multipleOf: 3 }).valid).toBe(false);
  });

  test("array constraints: minItems/maxItems/uniqueItems/items", () => {
    expect(validateAgainstJsonSchema([1], { minItems: 2 }).valid).toBe(false);
    expect(validateAgainstJsonSchema([1, 2, 3], { maxItems: 2 }).valid).toBe(false);
    expect(validateAgainstJsonSchema([1, 1], { uniqueItems: true }).valid).toBe(false);
    expect(validateAgainstJsonSchema([1, 2], { uniqueItems: true }).valid).toBe(true);
    expect(validateAgainstJsonSchema(["a", 1], { items: { type: "string" } }).valid).toBe(false);
    expect(validateAgainstJsonSchema([1, "a"], { items: [{ type: "number" }, { type: "string" }] }).valid).toBe(true);
  });

  test("object: required, properties, additionalProperties", () => {
    const schema = { type: "object", required: ["a"], properties: { a: { type: "string" } }, additionalProperties: false };
    expect(validateAgainstJsonSchema({ a: "x" }, schema).valid).toBe(true);
    expect(validateAgainstJsonSchema({}, schema).valid).toBe(false);
    expect(validateAgainstJsonSchema({ a: "x", b: 1 }, schema).valid).toBe(false);
    expect(validateAgainstJsonSchema({ a: 1 }, schema).valid).toBe(false);
  });

  test("additionalProperties as a schema validates extras", () => {
    const schema = { type: "object", properties: { a: { type: "string" } }, additionalProperties: { type: "number" } };
    expect(validateAgainstJsonSchema({ a: "x", b: 1 }, schema).valid).toBe(true);
    expect(validateAgainstJsonSchema({ a: "x", b: "not a number" }, schema).valid).toBe(false);
  });

  test("allOf / anyOf / oneOf / not", () => {
    expect(validateAgainstJsonSchema(5, { allOf: [{ minimum: 1 }, { maximum: 10 }] }).valid).toBe(true);
    expect(validateAgainstJsonSchema(5, { allOf: [{ minimum: 1 }, { maximum: 3 }] }).valid).toBe(false);
    expect(validateAgainstJsonSchema("x", { anyOf: [{ type: "number" }, { type: "string" }] }).valid).toBe(true);
    expect(validateAgainstJsonSchema(true, { anyOf: [{ type: "number" }, { type: "string" }] }).valid).toBe(false);
    expect(validateAgainstJsonSchema(5, { oneOf: [{ minimum: 0 }, { maximum: 10 }] }).valid).toBe(false); // matches both -> not exactly 1
    expect(validateAgainstJsonSchema(5, { oneOf: [{ minimum: 10 }, { maximum: 10 }] }).valid).toBe(true);
    expect(validateAgainstJsonSchema("x", { not: { type: "number" } }).valid).toBe(true);
    expect(validateAgainstJsonSchema(5, { not: { type: "number" } }).valid).toBe(false);
  });

  test("boolean schemas", () => {
    expect(validateAgainstJsonSchema("anything", true).valid).toBe(true);
    expect(validateAgainstJsonSchema("anything", false).valid).toBe(false);
  });

  test("nested object/array error paths are reported", () => {
    const schema = { type: "object", properties: { items: { type: "array", items: { type: "object", required: ["id"], properties: { id: { type: "string" } } } } } };
    const r = validateAgainstJsonSchema({ items: [{ id: "ok" }, {}] }, schema);
    expect(r.valid).toBe(false);
    expect(r.errors.some((e) => e.path === "$.items[1].id")).toBe(true);
  });
});
