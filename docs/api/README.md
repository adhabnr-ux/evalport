# EvalPort REST API Reference

The EvalPort reference API server provides endpoints for storing and retrieving eval suites and result sets.

## Running the Server

```bash
npx tsx api/server.ts
# EvalPort API running on http://localhost:9876
```

## Endpoints

### POST /suites

Store an eval suite (validated on intake).

```bash
curl -X POST http://localhost:9876/suites \
  -H "Content-Type: application/json" \
  -d @examples/basic-suite.json
```

**Response**: `201 Created`
```json
{"id": "suite_qa_basic", "message": "Suite stored"}
```

If invalid: `400 Bad Request`
```json
{"error": "Invalid suite", "details": [{"path": "$.version", "message": "...", "code": "INVALID_VERSION"}]}
```

### GET /suites

List all stored suite IDs.

```bash
curl http://localhost:9876/suites
```

### GET /suites/:id

Retrieve a stored suite.

```bash
curl http://localhost:9876/suites/suite_qa_basic
```

### POST /results

Submit results for a suite run. Creates a ResultSet with computed summary.

```bash
curl -X POST http://localhost:9876/results \
  -H "Content-Type: application/json" \
  -d '{"suite_id": "suite_qa_basic", "run_id": "run_001", "results": [...]}'
```

### GET /results/:id

Retrieve a stored result set.

### GET /health

Health check.
```json
{"status": "ok", "suites": 3, "results": 5}
```

## Architecture

The reference server uses in-memory storage. For production use, replace the `Map` with a database (PostgreSQL, MongoDB, etc.).
