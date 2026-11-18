# Ascend LLM Wiki — Schema & Conventions

## Overview

This is a persistent, LLM-maintained knowledge base for **Ascend AI Infrastructure (昇腾 AI Infra)**. It covers the full software stack from PyTorch down to NPU hardware, with a focus on training optimization, FP8 quantization, HCCL communication, and operational practices.

## Directory Structure

```
ascend-llm-wiki/
├── CLAUDE.md              ← This file — schema and conventions
├── index.md               ← Content catalog (LLM maintains)
├── log.md                 ← Append-only change log (LLM maintains)
├── raw/                   ← Immutable source documents
│   └── sources/           ←   Web articles, papers, notes (read-only)
├── wiki/                  ← LLM-maintained knowledge base
│   ├── architecture/      ←   System architecture pages
│   ├── concepts/          ←   Core concept and theory pages
│   ├── components/        ←   Component-specific deep dives
│   ├── troubleshooting/   ←   Problem diagnosis and resolution
│   ├── operations/        ←   Operational guides and best practices
│   └── references/        ←   External reference links and cheatsheets
└── assets/                ← Images, diagrams, attachments
```

## Three-Layer Architecture

1. **Raw sources** (`raw/sources/`) — immutable. Once written, never modified. Source of truth.
2. **The wiki** (`wiki/`) — LLM-maintained. Create, update, cross-reference freely.
3. **The schema** (this file) — conventions and workflows. Co-evolve with the wiki.

## Page Conventions

### Frontmatter
Every wiki page MUST start with YAML frontmatter:
```yaml
---
title: Page Title
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [source-file-1, source-file-2]
---
```

### Cross-references
Use `[[wikilinks]]` for internal cross-references. Examples:
- `[[architecture/ascend-software-stack]]`
- `[[concepts/fp8-training]]`
### Categories
- **architecture/** — high-level system design, software stack layers
- **concepts/** — theoretical concepts, algorithms, quantization methods
- **components/** — specific components: HCCL, CANN, TorchNPU, etc.
- **troubleshooting/** — root cause analysis, error codes, workarounds
- **operations/** — how-to guides, environment setup, best practices
- **references/** — external links, API references, version tables

## Workflows

### Ingest
1. Place source document in `raw/sources/`
2. Read and extract key information
3. Create/update relevant wiki pages
4. Update `index.md` with new/updated entries
5. Append entry to `log.md`

### Query
1. Read `index.md` to find relevant pages
2. Read the identified pages
3. Synthesize answer with citations to wiki pages
4. Optionally file the answer as a new wiki page

### Lint
Periodically check for:
- Orphan pages (no inbound links)
- Stale claims (newer sources supersede old ones)
- Missing cross-references
- Important concepts mentioned but lacking their own page
- Contradictions between pages

## Style Guidelines

- Use Chinese for page titles and content (user's primary language)
- Keep technical terms in their original English form (e.g., FP8, HCCL, CANN, AscendCL)
- Include version information when referencing specific components
- Cite sources with `[source: filename.md]` notation
- Use code blocks for commands, configs, and error messages
- Use tables for comparisons, version mappings, and parameter lists