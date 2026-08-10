// A minimal, dependency-free JSONPath evaluator covering the subset that
// shows up in real eval-suite json_path graders: `$`, `.field`, `.nested.field`,
// `[0]` / `[-1]` array indexing, and `[*]` / `.*` wildcards. Not a full
// JSONPath implementation (no filter expressions, no slices, no recursive
// descent `..`) — kept dependency-free per the Tier 1 grader requirement.

type Segment = { kind: "key"; key: string } | { kind: "index"; index: number } | { kind: "wildcard" };

export class JsonPathSyntaxError extends Error {}

export function parseJsonPath(path: string): Segment[] {
  let p = path.trim();
  if (!p.startsWith("$")) throw new JsonPathSyntaxError(`JSONPath must start with "$": ${path}`);
  p = p.slice(1);
  const segments: Segment[] = [];
  let i = 0;
  while (i < p.length) {
    const ch = p[i];
    if (ch === ".") {
      i++;
      if (p[i] === "*") { segments.push({ kind: "wildcard" }); i++; continue; }
      let key = "";
      while (i < p.length && p[i] !== "." && p[i] !== "[") { key += p[i]; i++; }
      if (!key) throw new JsonPathSyntaxError(`Empty key segment in path: ${path}`);
      segments.push({ kind: "key", key });
    } else if (ch === "[") {
      const close = p.indexOf("]", i);
      if (close === -1) throw new JsonPathSyntaxError(`Unclosed "[" in path: ${path}`);
      const inner = p.slice(i + 1, close).trim();
      if (inner === "*") {
        segments.push({ kind: "wildcard" });
      } else {
        const stripped = inner.replace(/^['"]|['"]$/g, "");
        const idx = Number(stripped);
        if (Number.isInteger(idx)) segments.push({ kind: "index", index: idx });
        else segments.push({ kind: "key", key: stripped });
      }
      i = close + 1;
    } else {
      throw new JsonPathSyntaxError(`Unexpected character "${ch}" at position ${i} in path: ${path}`);
    }
  }
  return segments;
}

function step(values: unknown[], segment: Segment): unknown[] {
  const out: unknown[] = [];
  for (const v of values) {
    if (segment.kind === "key") {
      if (v && typeof v === "object" && !Array.isArray(v) && segment.key in (v as Record<string, unknown>)) {
        out.push((v as Record<string, unknown>)[segment.key]);
      }
    } else if (segment.kind === "index") {
      if (Array.isArray(v)) {
        const idx = segment.index < 0 ? v.length + segment.index : segment.index;
        if (idx >= 0 && idx < v.length) out.push(v[idx]);
      }
    } else {
      // wildcard
      if (Array.isArray(v)) out.push(...v);
      else if (v && typeof v === "object") out.push(...Object.values(v as Record<string, unknown>));
    }
  }
  return out;
}

/** Returns every value the path matches (0, 1, or many — wildcards can fan out). */
export function queryJsonPath(root: unknown, path: string): unknown[] {
  const segments = parseJsonPath(path);
  let current: unknown[] = [root];
  for (const seg of segments) current = step(current, seg);
  return current;
}

/** Convenience for the common single-value case: the json_path grader
 * compares one extracted value against `expected`. Returns undefined if the
 * path matched nothing, and the first match if it matched one or more. */
export function queryJsonPathFirst(root: unknown, path: string): { found: boolean; value: unknown } {
  const matches = queryJsonPath(root, path);
  return matches.length > 0 ? { found: true, value: matches[0] } : { found: false, value: undefined };
}
