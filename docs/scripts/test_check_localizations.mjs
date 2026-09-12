#!/usr/bin/env node

import assert from "node:assert/strict";
import { parseFrontmatter, validateTranslation } from "./check_localizations.mjs";

const source = "---\ntitle: Example\n---\n\nRun this.\n\n```bash\nsglang serve --model example\n```\n";
const valid = "---\ntitle: 예제\ntranslation_of: cookbook/example.mdx\ntranslation_source_commit: 358c163250ad3b1f62939b01ce1314a0a31a0365\n---\n\n다음을 실행하세요.\n\n```bash\nsglang serve --model example\n```\n";

assert.deepEqual(validateTranslation({
  path: "ko/example",
  content: valid,
  sourceContent: source,
  metadata: parseFrontmatter(valid),
}), []);

for (const [name, content, expected] of [
  ["English-only prose", valid.replace("다음을 실행하세요.", "Run this."), "contain Hangul"],
  ["mutated command", valid.replace("--model example", "--model changed"), "code blocks must match"],
  ["abbreviated SHA", valid.replace("358c163250ad3b1f62939b01ce1314a0a31a0365", "358c163"), "40-character"],
]) {
  const errors = validateTranslation({
    path: name,
    content,
    sourceContent: source,
    metadata: parseFrontmatter(content),
  });
  assert(errors.some((error) => error.includes(expected)), `${name} was not rejected: ${errors}`);
}

console.log("Localization checker adversarial tests passed.");
