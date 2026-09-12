#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join, normalize, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const DOCS_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export function parseFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n/);
  if (!match) return {};
  return Object.fromEntries(match[1].split("\n").flatMap((line) => {
    const field = line.match(/^([a-zA-Z0-9_]+):\s*(.*?)\s*$/);
    if (!field) return [];
    return [[field[1], field[2].replace(/^(["'])(.*)\1$/, "$2")]];
  }));
}

export function fencedBlocks(content) {
  return [...content.matchAll(/^(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1\s*$/gm)]
    .map((match) => match[0]);
}

export function validateTranslation({ path, content, sourceContent, metadata }) {
  const errors = [];
  if (!/[\uac00-\ud7a3]/u.test(content.replace(/^---[\s\S]*?---/u, ""))) {
    errors.push("translated prose must contain Hangul");
  }
  if (!metadata.translation_of?.endsWith(".mdx")) {
    errors.push("translation_of must name an MDX source page");
  }
  if (!/^[0-9a-f]{40}$/.test(metadata.translation_source_commit || "")) {
    errors.push("translation_source_commit must be a full 40-character commit SHA");
  }
  if (sourceContent !== undefined) {
    const sourceBlocks = fencedBlocks(sourceContent);
    const translatedBlocks = fencedBlocks(content);
    if (JSON.stringify(sourceBlocks) !== JSON.stringify(translatedBlocks)) {
      errors.push("fenced code blocks must match the source exactly and in order");
    }
  }
  return errors.map((message) => `${path}: ${message}`);
}

function collectPages(value, withinPages = false) {
  if (typeof value === "string") return withinPages ? [value] : [];
  if (Array.isArray(value)) return value.flatMap((child) => collectPages(child, withinPages));
  if (!value || typeof value !== "object") return [];
  return Object.entries(value).flatMap(([key, child]) => collectPages(child, key === "pages"));
}

function safeSourcePath(source) {
  const path = normalize(source || "");
  if (!path || path.startsWith(`..${sep}`) || path === "..") return null;
  const absolute = resolve(DOCS_ROOT, path);
  return relative(DOCS_ROOT, absolute).startsWith("..") ? null : absolute;
}

export function checkLocalizations() {
  const failures = [];
  const config = JSON.parse(readFileSync(join(DOCS_ROOT, "docs.json"), "utf8"));
  const languages = config.navigation?.languages || [];
  const english = languages.find((entry) => entry.language === "en");
  const korean = languages.find((entry) => entry.language === "ko");

  if (!english?.default) failures.push("docs.json: English must remain the default language");
  if (!korean) failures.push("docs.json: Korean navigation is missing");

  const englishPages = new Set(collectPages(english?.tabs || []));
  const koreanPages = collectPages(korean?.tabs || []);
  if (!koreanPages.length) failures.push("docs.json: Korean navigation has no pages");

  for (const page of koreanPages) {
    if (!page.startsWith("ko/") || page.includes("..")) {
      failures.push(`docs.json: unsafe or non-Korean page in Korean navigation: ${page}`);
      continue;
    }
    const translatedPath = join(DOCS_ROOT, `${page}.mdx`);
    let content;
    try {
      content = readFileSync(translatedPath, "utf8");
    } catch {
      failures.push(`${page}: navigation target does not exist`);
      continue;
    }
    const metadata = parseFrontmatter(content);
    const sourcePath = safeSourcePath(metadata.translation_of);
    if (!sourcePath) {
      failures.push(`${page}: translation_of escapes the docs root or is missing`);
      continue;
    }
    if (!englishPages.has(metadata.translation_of.replace(/\.mdx$/, ""))) {
      failures.push(`${page}: translation_of is not present in English navigation`);
    }
    let sourceContent;
    try {
      sourceContent = readFileSync(sourcePath, "utf8");
    } catch {
      failures.push(`${page}: source page does not exist: ${metadata.translation_of}`);
      continue;
    }
    failures.push(...validateTranslation({ path: page, content, sourceContent, metadata }));

    if (/^[0-9a-f]{40}$/.test(metadata.translation_source_commit || "")) {
      try {
        const changed = execFileSync("git", ["diff", "--name-only",
          `${metadata.translation_source_commit}..HEAD`, "--", metadata.translation_of],
          { cwd: DOCS_ROOT, encoding: "utf8" }).trim();
        if (changed) {
          failures.push(`${page}: source changed after translation_source_commit; review and resync it`);
        }
      } catch (error) {
        failures.push(`${page}: cannot verify translation_source_commit: ${error.message}`);
      }
    }
  }

  if (failures.length) {
    console.error(failures.join("\n"));
    process.exit(1);
  }
  console.log(`Checked ${koreanPages.length} Korean localized pages.`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  checkLocalizations();
}
