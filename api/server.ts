import { createServer } from "http";
import { validateSuite } from "../sdk/typescript/src/validate";
import { createResultSet } from "../sdk/typescript/src/convert";

const PORT = 9876;
const suites = new Map<string, any>();
const results = new Map<string, any>();

function readBody(req: any): Promise<string> {
  return new Promise(resolve => { let d = ""; req.on("data", c => d += c); req.on("end", () => resolve(d)); });
}

function send(res: any, code: number, data: unknown) {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

createServer(async (req: any, res: any) => {
  const url = req.url || "/";
  const method = req.method || "GET";

  if (url === "/suites" && method === "POST") {
    const body = JSON.parse(await readBody(req));
    const v = validateSuite(body);
    if (!v.valid) return send(res, 400, { error: "Invalid suite", details: v.errors });
    suites.set(body.id, body);
    return send(res, 201, { id: body.id, message: "Suite stored" });
  }

  if (url.startsWith("/suites/") && method === "GET") {
    const id = url.split("/")[2];
    const suite = suites.get(id);
    if (!suite) return send(res, 404, { error: "Suite not found" });
    return send(res, 200, suite);
  }

  if (url === "/suites" && method === "GET") {
    return send(res, 200, Array.from(suites.keys()));
  }

  if (url === "/results" && method === "POST") {
    const body = JSON.parse(await readBody(req));
    const { suite_id, results: rs, run_id } = body;
    const suite = suites.get(suite_id);
    if (!suite) return send(res, 404, { error: "Suite not found" });
    const resultSet = createResultSet(suite, rs, run_id || `run_${Date.now()}`);
    results.set(resultSet.run_id, resultSet);
    return send(res, 201, resultSet);
  }

  if (url.startsWith("/results/") && method === "GET") {
    const id = url.split("/")[2];
    const rs = results.get(id);
    if (!rs) return send(res, 404, { error: "Result set not found" });
    return send(res, 200, rs);
  }

  if (url === "/health") {
    return send(res, 200, { status: "ok", suites: suites.size, results: results.size });
  }

  send(res, 404, { error: "Not found" });
}).listen(PORT, () => console.log(`OpenEval API running on http://localhost:${PORT}`));
