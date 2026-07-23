import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const componentSource = fs.readFileSync(
  path.join(root, "components", "browser_storage", "index.html"),
  "utf8",
);
const script = componentSource.match(/<script>([\s\S]*?)<\/script>/)[1];

function createComponentHarness(initialStorage = {}) {
  const storage = new Map(Object.entries(initialStorage));
  const messages = [];
  let renderHandler;
  const context = {
    window: {
      localStorage: {
        getItem: (key) => storage.get(key) ?? null,
        setItem: (key, value) => storage.set(key, value),
        removeItem: (key) => storage.delete(key),
      },
      parent: { postMessage: (message) => messages.push(message) },
      addEventListener: (event, handler) => {
        if (event === "message") renderHandler = handler;
      },
    },
  };
  vm.runInNewContext(script, context);
  return {
    messages,
    storage,
    render: (args) => renderHandler({ data: { type: "streamlit:render", args } }),
  };
}

test("browser storage reloads a value saved after the initial empty load", () => {
  const harness = createComponentHarness();
  const storageKey = "kb_quality_llm_config_v1";
  const savedConfig = { api_key: "secret", base_url: "https://example.test/v1", model: "model-a" };

  harness.render({ action: "load", storage_key: storageKey, payload: { session_id: "session-a" } });
  harness.render({ action: "save", storage_key: storageKey, payload: savedConfig });
  harness.render({ action: "load", storage_key: storageKey, payload: { session_id: "session-a" } });
  harness.render({ action: "load", storage_key: storageKey, payload: { session_id: "session-b" } });

  assert.deepEqual(JSON.parse(harness.storage.get(storageKey)), savedConfig);
  assert.ok(
    harness.messages.some((message) => message.value?.api_key === savedConfig.api_key),
    "a later load should return the saved browser configuration",
  );
});
