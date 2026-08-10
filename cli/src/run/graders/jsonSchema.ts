// A minimal, dependency-free JSON Schema (draft-07-ish) validator.
//
// Why not just use ajv? The Phase 2 spec explicitly calls for Tier 1 graders
// (exact_match, contains, regex, json_schema, json_path) to ship with no
// external dependencies, so `evalport run` stays installable and auditable
// without pulling in a schema-validation library and its transitive tree.
// This covers the subset of JSON Schema that shows up in real eval suites —
// type, enum, const, required, properties/additionalProperties, items,
// string/number bounds, pattern, and the boolean combinators — not the
// entire spec (no $ref resolution, no format validation, no $defs).

export interface JsonSchemaValidationError {
  path: string;
  message: string;
}

export interface JsonSchemaValidationResult {
  valid: boolean;
  errors: JsonSchemaValidationError[];
}

type Schema = Record<string, unknown> | boolean;

function typeOf(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  if (typeof v === "number") return Number.isInteger(v) ? "integer" : "number";
  return typeof v; // "string" | "number" | "boolean" | "object" | "undefined"
}

function matchesType(v: unknown, type: string): boolean {
  const actual = typeOf(v);
  if (type === "number") return actual === "number" || actual === "integer";
  if (type === "integer") return actual === "integer";
  return actual === type;
}

function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => deepEqual(x, b[i]));
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ak = Object.keys(a as object), bk = Object.keys(b as object);
    return ak.length === bk.length && ak.every((k) => deepEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
  }
  return false;
}

function validateNode(value: unknown, schema: Schema, path: string, errors: JsonSchemaValidationError[]): void {
  if (typeof schema === "boolean") {
    if (schema === false) errors.push({ path, message: "schema is `false`; nothing is valid here" });
    return;
  }
  if (!schema || typeof schema !== "object") return;

  const type = schema.type;
  if (typeof type === "string") {
    if (!matchesType(value, type)) errors.push({ path, message: `expected type ${type}, got ${typeOf(value)}` });
  } else if (Array.isArray(type)) {
    if (!type.some((t) => typeof t === "string" && matchesType(value, t))) {
      errors.push({ path, message: `expected one of [${type.join(", ")}], got ${typeOf(value)}` });
    }
  }

  if ("const" in schema && !deepEqual(value, schema.const)) {
    errors.push({ path, message: `value does not equal const ${JSON.stringify(schema.const)}` });
  }
  if (Array.isArray(schema.enum) && !schema.enum.some((e) => deepEqual(value, e))) {
    errors.push({ path, message: `value not in enum ${JSON.stringify(schema.enum)}` });
  }

  if (typeof value === "string") {
    if (typeof schema.minLength === "number" && value.length < schema.minLength) errors.push({ path, message: `string shorter than minLength ${schema.minLength}` });
    if (typeof schema.maxLength === "number" && value.length > schema.maxLength) errors.push({ path, message: `string longer than maxLength ${schema.maxLength}` });
    if (typeof schema.pattern === "string" && !new RegExp(schema.pattern).test(value)) errors.push({ path, message: `string does not match pattern ${schema.pattern}` });
  }

  if (typeof value === "number") {
    if (typeof schema.minimum === "number" && value < schema.minimum) errors.push({ path, message: `${value} < minimum ${schema.minimum}` });
    if (typeof schema.maximum === "number" && value > schema.maximum) errors.push({ path, message: `${value} > maximum ${schema.maximum}` });
    if (typeof schema.exclusiveMinimum === "number" && value <= schema.exclusiveMinimum) errors.push({ path, message: `${value} <= exclusiveMinimum ${schema.exclusiveMinimum}` });
    if (typeof schema.exclusiveMaximum === "number" && value >= schema.exclusiveMaximum) errors.push({ path, message: `${value} >= exclusiveMaximum ${schema.exclusiveMaximum}` });
    if (typeof schema.multipleOf === "number" && schema.multipleOf > 0) {
      const q = value / schema.multipleOf;
      if (Math.abs(q - Math.round(q)) > 1e-9) errors.push({ path, message: `${value} is not a multiple of ${schema.multipleOf}` });
    }
  }

  if (Array.isArray(value)) {
    if (typeof schema.minItems === "number" && value.length < schema.minItems) errors.push({ path, message: `array shorter than minItems ${schema.minItems}` });
    if (typeof schema.maxItems === "number" && value.length > schema.maxItems) errors.push({ path, message: `array longer than maxItems ${schema.maxItems}` });
    if (schema.uniqueItems === true) {
      const seen: unknown[] = [];
      for (const item of value) {
        if (seen.some((s) => deepEqual(s, item))) { errors.push({ path, message: "array items must be unique" }); break; }
        seen.push(item);
      }
    }
    if (schema.items !== undefined) {
      if (Array.isArray(schema.items)) {
        value.forEach((item, i) => { if (schema.items && Array.isArray(schema.items) && schema.items[i] !== undefined) validateNode(item, schema.items[i] as Schema, `${path}[${i}]`, errors); });
      } else {
        value.forEach((item, i) => validateNode(item, schema.items as Schema, `${path}[${i}]`, errors));
      }
    }
  }

  if (typeOf(value) === "object" && value !== null) {
    const obj = value as Record<string, unknown>;
    const required = Array.isArray(schema.required) ? (schema.required as string[]) : [];
    for (const key of required) {
      if (!(key in obj)) errors.push({ path: `${path}.${key}`, message: "required property missing" });
    }
    const properties = (schema.properties && typeof schema.properties === "object" ? schema.properties : {}) as Record<string, Schema>;
    for (const [key, propSchema] of Object.entries(properties)) {
      if (key in obj) validateNode(obj[key], propSchema, `${path}.${key}`, errors);
    }
    if (schema.additionalProperties === false) {
      const known = new Set(Object.keys(properties));
      for (const key of Object.keys(obj)) {
        if (!known.has(key)) errors.push({ path: `${path}.${key}`, message: "additional property not allowed" });
      }
    } else if (schema.additionalProperties && typeof schema.additionalProperties === "object") {
      const known = new Set(Object.keys(properties));
      for (const key of Object.keys(obj)) {
        if (!known.has(key)) validateNode(obj[key], schema.additionalProperties as Schema, `${path}.${key}`, errors);
      }
    }
  }

  if (Array.isArray(schema.allOf)) {
    for (const sub of schema.allOf) validateNode(value, sub as Schema, path, errors);
  }
  if (Array.isArray(schema.anyOf)) {
    const anyOk = schema.anyOf.some((sub) => collectErrors(value, sub as Schema, path).length === 0);
    if (!anyOk) errors.push({ path, message: "value does not match any schema in anyOf" });
  }
  if (Array.isArray(schema.oneOf)) {
    const matchCount = schema.oneOf.filter((sub) => collectErrors(value, sub as Schema, path).length === 0).length;
    if (matchCount !== 1) errors.push({ path, message: `value matched ${matchCount} of oneOf schemas, expected exactly 1` });
  }
  if (schema.not !== undefined) {
    if (collectErrors(value, schema.not as Schema, path).length === 0) errors.push({ path, message: "value must not match the 'not' schema" });
  }
}

function collectErrors(value: unknown, schema: Schema, path: string): JsonSchemaValidationError[] {
  const errors: JsonSchemaValidationError[] = [];
  validateNode(value, schema, path, errors);
  return errors;
}

export function validateAgainstJsonSchema(value: unknown, schema: Schema): JsonSchemaValidationResult {
  const errors = collectErrors(value, schema, "$");
  return { valid: errors.length === 0, errors };
}
