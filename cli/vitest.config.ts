import { fileURLToPath } from "url";
import { defineConfig } from "vitest/config";

// In this repo the CLI is tested against the sibling SDK source, not whatever
// evalport-sdk version npm installed into node_modules. tsconfig.json's
// `paths` does the same for `npm run typecheck`. The published CLI resolves
// evalport-sdk from its own dependencies at runtime.
export default defineConfig({
  resolve: {
    alias: {
      "evalport-sdk": fileURLToPath(new URL("../sdk/typescript/src/index.ts", import.meta.url)),
    },
  },
});
