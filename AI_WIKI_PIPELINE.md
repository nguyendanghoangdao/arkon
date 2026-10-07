# Arkon — AI Wiki Pipeline

> How Arkon turns raw source documents into a structured, interlinked enterprise knowledge wiki using LLMs.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Source Ingestion](#2-source-ingestion)
3. [MRP Pipeline (Primary)](#3-mrp-pipeline-primary)
   - [Phase 0 — Triage](#phase-0--triage)
   - [Phase 1 — MAP](#phase-1--map)
   - [Phase 2 — REDUCE](#phase-2--reduce)
   - [Human Review Gate](#human-review-gate)
   - [Phase 3 — REFINE](#phase-3--refine)
   - [Phase 4 — VERIFY](#phase-4--verify)
   - [Phase 5 — COMMIT](#phase-5--commit)
4. [Wiki Agent (Alternative Path)](#4-wiki-agent-alternative-path)
5. [Wiki Compiler (Legacy Single-Shot)](#5-wiki-compiler-legacy-single-shot)
6. [AI Provider System](#6-ai-provider-system)
7. [Data Model](#7-data-model)
8. [API Endpoints](#8-api-endpoints)
9. [Feature Summary](#9-feature-summary)

---

## 1. Architecture Overview

Arkon is an enterprise knowledge management system that **compiles** source documents into a permanent, interlinked wiki. It is not a summarizer — it preserves specific numbers, regulations, procedures, names, and edge cases in a queryable, permanent structure that compounds over time as more sources arrive.

### Technology Stack

| Layer          | Technology                                      |
|----------------|------------------------------------------------|
| Backend        | FastAPI (async Python)                          |
| Database       | PostgreSQL + pgvector                           |
| Task Queue     | Redis + arq (async background workers)          |
| File Storage   | MinIO (S3-compatible)                           |
| AI Providers   | Anthropic, OpenAI, Google (pluggable via registry) |
| Migrations     | Alembic                                         |

### High-Level Flow

```
Source Upload (file/URL)
    │
    ▼
Text Extraction + Image Extraction + Outline Building
    │
    ▼
Token Gate (optional human approval for large docs)
    │
    ▼
┌─────────────────────────────────────────┐
│  MRP Pipeline (primary)                 │
│  Phase 0: Triage                        │
│  Phase 1: MAP (parallel chunk extraction)│
│  Phase 2: REDUCE (dedup + plan)         │
│    ⏸ Human review of Compilation Plan   │
│  Phase 3: REFINE (parallel page writing)│
│  Phase 4: VERIFY (coverage + conflicts) │
│  Phase 5: COMMIT (atomic DB write)      │
└─────────────────────────────────────────┘
    │
    ▼
Wiki Pages (markdown, versioned, embedded, interlinked)
```

### Key Files Overview

| File | Role |
|------|------|
| `app/ai/mrp/pipeline.py` | MRP orchestrator — two entry points for phases 0-2 and 3-5 |
| `app/ai/mrp/mapper.py` | Phase 0 (Triage) + Phase 1 (MAP) |
| `app/ai/mrp/reducer.py` | Phase 2 (REDUCE) — dedup, reconciliation, planning |
| `app/ai/mrp/writer.py` | Phase 3 (REFINE) — evidence assembly + page writing |
| `app/ai/mrp/verifier.py` | Phase 4 (VERIFY) — coverage, conflicts, lifecycle |
| `app/ai/mrp/merger.py` | LLM-based content merging for multi-source pages |
| `app/ai/wiki_agent.py` | Tool-calling wiki agent (alternative path) |
| `app/ai/wiki_agent_tools.py` | Agent tool schemas and handler implementations |
| `app/ai/wiki_compiler.py` | Legacy single-shot compiler |
| `app/ai/wiki_analyzer.py` | Pre-analysis for the wiki agent |
| `app/ai/registry.py` | Runtime AI provider resolution from DB config |
| `app/ai/providers/base.py` | Abstract base classes for all AI providers |
| `app/ai/llm_catalog.py` | Whitelisted LLM models with metadata |
| `app/ai/embedding_catalog.py` | Whitelisted embedding models with dimensions |
| `app/ai/vision_catalog.py` | Whitelisted vision models for image captioning |
| `app/ai/agent_protocol.py` | Provider-agnostic types for agent tool-calling loops |
| `app/worker.py` | Background task definitions (arq workers) |
| `app/routers/sources.py` | REST API for source upload, plan review, approval |
| `app/database/models.py` | All SQLAlchemy ORM models |

---

## 2. Source Ingestion

Source ingestion handles the pipeline from upload through text/image extraction to the point where the MRP pipeline takes over.

### 2.1 File Upload

**API Endpoint:** `POST /sources/upload`
**File:** `app/routers/sources.py` → `upload_source()` (line 319)

1. Accepts multipart file upload with metadata (title, knowledge_type_id, department_ids, scope)
2. Creates a `Source` record in DB with `status="pending"`
3. Uploads the file to MinIO at `sources/{source_id}/original/{file_name}`
4. Enqueues `ingest_file_task` via arq

**API Endpoint:** `POST /sources/url`
**File:** `app/routers/sources.py` → `add_url_source()` (line 412)

Same flow for URL sources, enqueues `ingest_url_task`.

### 2.2 File Ingestion Task

**File:** `app/worker.py` → `ingest_file_task()` (line 91)

```
Step 1: Download from MinIO                      → 10%
Step 2: Extract text per page (with OCR fallback) → 25%
Step 3: Extract images + inline markers            → 40%
Step 4: Build document outline + assemble full_text→ 50%
Step 5: Token count                                → 55%
Step 6: Gate or auto-proceed to MRP                → 55%+
```

**Details:**

- **Text extraction** uses `_extract_text_from_file()` from `app/services/kb_service.py`. Supports PDF, DOCX, DOC, plain text. Falls back to a `VisionProvider` for OCR on image-only PDF pages.
- **Image extraction** uses `extract_images()` from `app/services/image_service.py`. Extracted images are persisted as `SourceImage` rows with MinIO keys. Image markers `![caption](image://<uuid>)` are inlined into per-page text.
- **Outline building** uses `build_outline()` from `app/services/source_outline.py` — produces a heading-based TOC tree stored as `source.outline_json`.
- **Full text assembly** via `assemble_full_text()` — concatenates all page texts, records character offsets per page in `source.page_offsets`.
- **Token gate:** If `extracted_token_count > auto_approve_extraction_threshold_tokens`, the source enters `status="awaiting_approval"` for human review. Otherwise proceeds automatically.
- **Image captioning** is offloaded to a separate task `caption_images_task` (line 1173) so ingestion isn't blocked by image count. Uses the configured `VisionProvider`.

### 2.3 URL Ingestion Task

**File:** `app/worker.py` → `ingest_url_task()` (line 252)

Same flow without the MinIO download step. Uses `_extract_text_from_url()` instead.

### 2.4 Verbatim Mode

When `source.preserve_verbatim = True`, the MRP pipeline is skipped entirely. The raw `full_text` is chunked and embedded as-is into `source_chunk_embeddings_<dim>` tables, making it searchable alongside wiki pages but never rewritten. Designed for high-fidelity documents like decrees and official gazettes.

**File:** `app/worker.py` → `finalize_verbatim_source()` (line 68)

---

## 3. MRP Pipeline (Primary)

The MRP (Map-Reduce-Plan) pipeline is the primary compilation path. It is deterministic, multi-phase, and supports human review of compilation plans before writing wiki pages.

### Two Entry Points

**File:** `app/ai/mrp/pipeline.py`

| Entry Point | Phases | Worker Task | Line |
|---|---|---|---|
| `run_mrp_pipeline()` | 0 → 1 → 2 | `ingest_map_reduce_task` (worker.py:723) | 287 |
| `run_refine_pipeline()` | 3 → 4 → 5 | `ingest_refine_task` (worker.py:827) | 404 |

The split allows a human review gate between Phase 2 (plan generation) and Phase 3 (page writing).

---

### Phase 0 — Triage

**File:** `app/ai/mrp/mapper.py` → `classify_strategy()` (line 58)

Classifies the document by character length into one of three strategies:

| Strategy | Document Length | Chunk Count | Target Pages |
|---|---|---|---|
| `single_pass` | < 30,000 chars | 1–2 chunks | 3–30 |
| `standard` | 30,000–200,000 chars | ~10 chunks | 8–80 |
| `hierarchical` | > 200,000 chars | Many chunks | 15–200 |

The strategy controls chunking granularity in Phase 1 and target page count in Phase 2.

---

### Phase 1 — MAP

**File:** `app/ai/mrp/mapper.py` → `run_map_phase()` (line 340)

**Goal:** Split the document into chunks, extract structured knowledge from each chunk in parallel.

#### 1a. Chunking

**Function:** `build_chunks()` (line 83)

Splits `full_text` into `DocumentChunk` objects (~20K chars each) using the document's heading-based outline:

- Groups level-1 and level-2 headings until accumulated chars exceed `CHUNK_TARGET_CHARS = 20,000`
- Each chunk gets an overlap prefix from the previous chunk (`OVERLAP_CHARS = 1,000`) for context continuity
- Falls back to `_sliding_window_chunks()` (line 179) when no outline exists

**Dataclass:** `DocumentChunk` (line 42)
```python
@dataclass
class DocumentChunk:
    index: int
    start_char: int        # absolute offset in full_text
    end_char: int          # absolute offset in full_text
    section_path: str      # e.g. "Chapter 2 > Section 2.1"
    text: str              # chunk body (may include overlap prefix)
    overlap_prefix_len: int = 0
```

#### 1b. Parallel Extraction

**Function:** `extract_chunk()` (line 321)

Each chunk is sent to the LLM with a structured extraction prompt (`EXTRACTION_SYSTEM` + `EXTRACTION_PROMPT_TEMPLATE`). The LLM returns JSON with:

```json
{
  "entities": [...],
  "concepts": [
    {
      "term": "string",
      "definition_excerpt": "string",
      "local_offset": 0
    }
  ],
  "claims": [
    {
      "statement": "string",
      "subject": "string",
      "local_offset": 0,
      "evidence_length": 200,
      "confidence": "explicit"
    }
  ]
}
```

`local_offset` values are converted to absolute offsets via `_convert_offsets()` (line 302).

**Concurrency:** Up to `MAX_MAP_CONCURRENCY = 6` parallel LLM calls via `asyncio.Semaphore`.

**Persistence:** Each chunk result is immediately persisted to a `SourceChunkExtract` row (status `"done"`) enabling crash resume. Failed chunks are retried once sequentially.

---

### Phase 2 — REDUCE

**File:** `app/ai/mrp/reducer.py` → `run_reduce_phase()` (line 549)

**Goal:** Deduplicate extracted knowledge, reconcile with existing wiki pages, produce a Compilation Plan.

#### Step 2.1 — Collect Raw Items

**Function:** `collect_raw_items()` (line 62)

Flattens entities, concepts, and claims from all `SourceChunkExtract` rows. Entities are merged into concepts (the "general pages" pivot).

#### Step 2.2 — Exact Deduplication

**Function:** `exact_dedup_concepts()` (line 104)

Groups concepts by normalized name (lowercased, punctuation stripped). Keeps the highest-mention-count variant and longest definition excerpt.

#### Step 2.3 — Embedding-Based Deduplication

**Function:** `embedding_dedup_concepts()` (line 150)

Embeds all concept names in a single batch API call. Pairs with cosine similarity:

| Similarity | Action |
|---|---|
| ≥ 0.90 (`MERGE_THRESHOLD`) | Auto-merge |
| 0.75–0.90 (`AMBIGUOUS_LOW`) | Flag as ambiguous |
| < 0.75 | Keep separate |

#### Step 2.4 — LLM Disambiguation

**Function:** `resolve_ambiguous_concepts()` (line 215)

Sends all ambiguous pairs to the LLM in a single batch call. The LLM returns a boolean array indicating same-or-different for each pair. Merged results applied via `_apply_merges_concepts()` (line 272).

#### Step 2.5 — KB Reconciliation

**Function:** `reconcile_with_kb()` (line 298)

For each canonical concept, searches existing wiki pages via semantic similarity:

| Similarity | Action | Meaning |
|---|---|---|
| ≥ 0.85 (`KB_UPDATE_THRESHOLD`) | `UPDATE` | Matches existing page |
| 0.60–0.85 (`KB_MAYBE_THRESHOLD`) | `MAYBE` | Needs LLM confirmation |
| < 0.60 | `CREATE` | New page needed |

`MAYBE` items are resolved via `_resolve_maybe_items()` (line 374) — another single-batch LLM call.

Reconciliation searches across **all scopes** the source belongs to (via `_resolve_wiki_scopes()` in pipeline.py:30) to prevent duplicate pages across scopes.

#### Step 2.7 — Planning Call

**Function:** `run_planning_call()` (line 477)

One LLM call produces the **Compilation Plan** JSON — a list of pages to CREATE or UPDATE, with slugs, titles, page types, entity assignments, and priorities.

The plan is persisted to `SourceCompilationPlan` (line 247 in models.py) with `status="pending_review"`.

```
PLANNING_PROMPT_TEMPLATE includes:
  - Source title and knowledge type context
  - All canonical concepts with mention counts and KB reconciliation decisions
  - KB reconciliation summary (which pages to update)
  - Optional human reviewer feedback (for plan regeneration)
  - Target page count based on strategy
```

---

### Human Review Gate

After Phase 2, the pipeline pauses for human review. The source enters `status="plan_ready"`.

| API Endpoint | File / Function | Effect |
|---|---|---|
| `GET /sources/{id}/plan` | `sources.py` → `get_compilation_plan()` (line 653) | View the plan |
| `POST /sources/{id}/plan/approve` | `sources.py` → `approve_compilation_plan()` (line 731) | Approve → enqueue Phase 3 |
| `POST /sources/{id}/plan/regenerate` | `sources.py` → `regenerate_compilation_plan()` (line 786) | Re-run planning with feedback |
| `POST /sources/{id}/plan/reject` | `sources.py` → `reject_compilation_plan()` (line 835) | Reject → source enters error state |

**Auto-approve:** If `settings.mrp_auto_approve_plan = True`, the plan is auto-approved and Phase 3 is enqueued immediately via `_auto_trigger_refine()` (pipeline.py:371).

**Plan regeneration** runs as a background task `regenerate_plan_task` (worker.py:907) that re-runs KB reconciliation + the planning call with the reviewer's feedback note injected into the prompt.

---

### Phase 3 — REFINE

**File:** `app/ai/mrp/writer.py` → `run_refine_phase()` (line 1256)

**Goal:** Write each planned wiki page using dedicated writers with pre-assembled evidence.

#### Evidence Assembly

**Function:** `assemble_evidence()` (line 115)

For each page in the plan, collects all claims whose `subject` matches any of the page's `entity_names` (whole-word/whole-phrase matching, case-insensitive). Each claim includes a source excerpt from the full document.

#### Source Context Building

**Function:** `_build_source_context()` (line 269)

Builds the source context the writer sees:

- **Budget calculation:** `_get_source_context_budget()` (line 245) — uses ~85% of the LLM's context window (from `config.spec.context_window_tokens`). Falls back to 120K chars when no spec is attached.
- **Short documents** (fits in budget): Include the full text.
- **Long documents:** Scores sections by evidence density (`_score_sections()`, line 386), includes the intro, then greedily adds highest-scored sections until budget is filled. Sections are kept in original document order.

#### Writer Strategy Decision

**Function:** `_decide_writer_strategy()` (line 595)

| Condition | Strategy |
|---|---|
| Source context + evidence + existing content fits in 70% of budget | `single` — one LLM call |
| Otherwise | `multipass` — multiple write/extend/polish passes |

Additionally, pages with many evidence items (≥ 8) or large existing content (≥ 3,000 chars) trigger the **complex writer** (mini agent loop).

#### Simple Writer

**Function:** `_write_page_simple()` (line 725)

Single `llm.generate()` call. The prompt includes:
- Page slug, title, type, action (CREATE/UPDATE)
- Available wiki page slugs for wikilinks
- Evidence checklist
- Source context (full text or smart-extracted sections)
- Existing page content (for UPDATEs)
- Relevant image markers

Returns `(content_md, summary, citations_meta)`.

#### Complex Writer

**Function:** `_write_page_complex()` (line 851)

Mini agent loop (up to `WRITER_AGENT_MAX_STEPS = 10` turns) with 3 tools:

| Tool | Description |
|---|---|
| `read_kb_page` | Read full content of an existing wiki page |
| `read_source_excerpt` | Read more context from the source by char offset |
| `finish` | Submit completed content_md + summary |

Used when pages need to integrate many evidence items or merge with substantial existing content.

#### Multi-Pass Writer

**Function:** `_write_page_multipass()` (line 1161)

For very large pages that exceed single-pass budgets:

1. **Create pass** (`_writer_pass_create()`, line 1072) — write initial content from highest-priority sections
2. **Extend pass(es)** (`_writer_pass_extend()`, line 1111) — add content from remaining section batches
3. **Polish pass** (`_writer_pass_polish()`, line 1137) — final coherence and formatting pass

Sections are batched via `build_writer_batches()` (line 551) respecting document order.

#### Concurrency

All writers run in parallel with `MAX_WRITER_CONCURRENCY = 4` via `asyncio.Semaphore`.

#### Output

Each writer produces a `PageWriteResult` (line 53):

```python
@dataclass
class PageWriteResult:
    slug: str
    title: str
    page_type: str
    action: str          # CREATE | UPDATE
    content_md: str
    summary: str
    citations: list[dict]
    entity_names: list[str]
    related_kb_pages: list[str]
```

---

### Phase 4 — VERIFY

**File:** `app/ai/mrp/verifier.py` → `run_verify_phase()` (line 209)

**Goal:** Quality checks on generated pages. All checks are **non-blocking** — they log warnings and annotate content but never fail the pipeline.

#### 4.1 Coverage Check

**Function:** `check_coverage()` (line 33)

Identifies concepts mentioned ≥ 3 times in chunk extracts but not covered by any generated page. Logged as warnings.

#### 4.2 Conflict Check

**Function:** `check_conflicts()` (line 73)

For each new/updated page:
1. Embeds the page content
2. Finds existing wiki pages with cosine similarity ≥ `CONFLICT_SIM_THRESHOLD = 0.80`
3. Asks the LLM: "Do these two texts contain contradictory factual statements?"
4. If a contradiction is detected, prepends an Obsidian-style callout to the page:

```markdown
> [!contradiction] **Mâu thuẫn tri thức phát hiện bởi AI:**
> <description of the contradiction>
```

#### 4.3 Lifecycle Status Assessment

**Function:** `assess_page_status()` (line 160)

Programmatically assigns a lifecycle status based on content length and wikilink count:

| Status | Criteria |
|---|---|
| `seed` | < 600 chars OR < 1 wikilink |
| `developing` | < 2,000 chars OR < 3 wikilinks |
| `mature` | < 4,000 chars OR < 5 wikilinks |
| `evergreen` | ≥ 4,000 chars AND ≥ 5 wikilinks |

Source index pages are always `evergreen`.

---

### Phase 5 — COMMIT

**File:** `app/ai/mrp/pipeline.py` → `run_commit_phase()` (line 66)

**Goal:** Atomically write all pages to the database and generate embeddings.

#### Scope Resolution

**Function:** `_resolve_wiki_scopes()` (line 30)

Determines which (scope_type, scope_id) tuples to commit pages into:
- Project scope takes priority
- Department assignments create one scope per department
- Falls back to global

#### Page Writing

For each scope × each page result:

1. **Advisory lock:** `pg_advisory_xact_lock` per (slug, scope) prevents race conditions from concurrent pipelines.

2. **CREATE:** Calls `wiki_service.apply_create()`. If the slug already exists (concurrent creation), falls through to UPDATE.

3. **UPDATE with merge:** When a new source touches an existing page from a *different* source, invokes `merge_page_content()` from `app/ai/mrp/merger.py` (line 75):
   - LLM merges existing + incoming content preserving ALL facts from both
   - **Sanity check:** If merged output is < 70% of the longest input (`BODY_SHRINK_THRESHOLD = 0.7`), the merge is rejected (LLM likely stripped content) and falls back to new content
   - Fast paths: skip merge if existing content is empty or identical to new content

4. **Embedding:** Generates a vector via the active `EmbeddingProvider` and upserts into the appropriate `wiki_page_embeddings_<dim>` table.

5. **Index regeneration:** `wiki_service.regenerate_index()` rebuilds the wiki index page.

6. **Source finalization:** Sets `source.status = "ready"`, `source.progress = 100`.

---

## 4. Wiki Agent (Alternative Path)

The wiki agent is a **tool-calling agent loop** that replaces the deterministic MRP pipeline with an iterative, exploratory approach. Each tool call gets the LLM's full token budget, producing denser output per page.

**File:** `app/ai/wiki_agent.py` → `compile_source_with_agent()` (line 259)

### Pre-Analysis

**File:** `app/ai/wiki_analyzer.py` → `analyze_source()` (line 88)

A single cheap LLM call (temperature 0.1, max first 30K chars) produces an advisory map:

```json
{
  "document_type": "regulation|sop|report|technical_spec|other",
  "primary_language": "vi|en|...",
  "key_themes": ["..."],
  "named_entities": [{"name": "...", "type": "...", "significance": "..."}],
  "key_concepts": [{"name": "...", "suggested_slug": "...", "description": "..."}],
  "existing_pages_to_update": [{"slug": "...", "reason": "..."}],
  "new_pages_to_create": [{"suggested_slug": "...", "page_type": "...", "title": "..."}]
}
```

Formatted via `format_analysis_section()` (line 133) and injected into the agent's initial message.

### Agent Loop

- **Max turns:** `MAX_STEPS = 50`
- **LLM call timeout:** `LLM_CALL_TIMEOUT = 180` seconds per call
- **Initial excerpt:** First 30,000 chars of source text

### Agent Tools

**File:** `app/ai/wiki_agent_tools.py`

**Tool schemas:** `TOOL_SCHEMAS` (line 39) — OpenAI function-calling format

**Tool handlers:** `build_tool_handlers()` (line 279) — returns name→async_callable mapping

| Tool | Description |
|------|-------------|
| `read_wiki_index` | List all existing wiki pages (slug, type, summary) |
| `read_wiki_page` | Read full markdown content of a specific page |
| `search_wiki` | Semantic search over existing pages (top-K by embedding similarity) |
| `read_source_excerpt` | Read a portion of the source document by character offset (max 15K chars) |
| `create_page` | Create a new wiki page with full content |
| `update_page` | Replace content of an existing page |
| `append_log` | Write to the wiki activity log |
| `finish` | Signal completion with a summary report |

### Agent State

**Class:** `AgentState` (line 225)

Tracks pages created/updated, read pages, tool call count, and completion signal. The `summary()` method produces the final result dict.

### Agent Protocol

**File:** `app/ai/agent_protocol.py`

Provider-agnostic types for multi-turn tool calling:

| Type | Description | Line |
|------|-------------|------|
| `ToolCall` | Single tool invocation (id, name, arguments) | 18 |
| `AssistantTurn` | LLM response with optional tool_calls | 25 |
| `assistant_message_from_turn()` | Build neutral message dict | 37 |
| `tool_results_message()` | Build tool results message | 47 |

Converter functions for each provider: `neutral_to_anthropic_messages()`, `neutral_to_openai_messages()`, `neutral_to_gemini_contents()`.

---

## 5. Wiki Compiler (Legacy Single-Shot)

The wiki compiler is the original, single-LLM-call approach. Predates the MRP pipeline.

**File:** `app/ai/wiki_compiler.py` → `compile_source_into_wiki()` (line 290)

### Flow

1. **Build context:**
   - Wiki index: up to 200 existing pages (slug + summary) via `_render_wiki_index()` (line 555)
   - Top-8 semantically-relevant existing pages via `_render_relevant_pages()` (line 580)

2. **Single LLM call:** Sends full source text (capped at `MAX_DOCUMENT_CHARS = 200,000`), wiki context, and detailed instructions. Temperature 0.2.

3. **Parse operations:** `_parse_operations()` (line 617) extracts JSON operations from LLM output:
   ```json
   [
     {"op": "create", "slug": "...", "title": "...", "page_type": "...", "content_md": "...", "summary": "..."},
     {"op": "update", "slug": "...", "new_content_md": "...", "summary": "..."},
     {"op": "log",    "entry": "..."}
   ]
   ```

4. **Sanitize image markers:** `_sanitize_image_markers()` (line 494) strips any hallucinated `image://<uuid>` references not in the source's actual image set.

5. **Apply operations:** Each operation calls `wiki_service.apply_create()` or `wiki_service.apply_update()` with advisory locks and savepoints for race condition handling.

6. **Re-embed:** `_reembed_pages()` (line 652) generates new embeddings for all touched pages.

---

## 6. AI Provider System

The AI layer is fully provider-agnostic. The rest of the codebase never imports a specific SDK.

### Abstract Base Classes

**File:** `app/ai/providers/base.py`

| Class | Methods | Line |
|-------|---------|------|
| `EmbeddingProvider` | `embed()`, `embed_batch()`, `test_connection()` | 60 |
| `LLMProvider` | `generate()`, `generate_with_tools()`, `test_connection()` | 87 |
| `VisionProvider` | `analyze_image()`, `test_connection()` | 125 |

All providers receive a `ProviderConfig` dataclass (line 36) loaded from the database at runtime.

### Provider Registry

**File:** `app/ai/registry.py` → `ProviderRegistry` (line 80)

Runtime resolver that reads active model configuration from the DB (`AppConfig` table) and instantiates the correct provider class.

| Method | Returns | Line |
|--------|---------|------|
| `get_embedding(task, spec_id)` | `EmbeddingProvider` | 93 |
| `get_llm()` | `LLMProvider` | 128 |
| `get_vision()` | `VisionProvider` or `None` | 150 |
| `test_all()` | `dict[str, (bool, str)]` | 178 |
| `get_active_embedding_spec_id()` | `Optional[str]` | 114 |
| `get_active_llm_spec_id()` | `Optional[str]` | 134 |

The registry attaches the catalog spec to `config.spec` so callers can read context window, cost, and capability metadata without hard-coding model IDs.

### Concrete Providers

| File | Provider |
|------|----------|
| `app/ai/providers/anthropic_provider.py` | Anthropic Claude |
| `app/ai/providers/openai_provider.py` | OpenAI GPT |
| `app/ai/providers/google.py` | Google Gemini |

### Model Catalogs

Whitelisted models — admins pick from these in the settings UI. No free-form model IDs.

#### LLM Catalog

**File:** `app/ai/llm_catalog.py` → `LLM_CATALOG` dict

| Spec ID | Label | Context | Tools | Input Cost |
|---------|-------|---------|-------|------------|
| `anthropic/claude-opus-4-7` | Claude Opus 4.7 | 1M | ✓ | $15.00/1M |
| `anthropic/claude-sonnet-4-6` | Claude Sonnet 4.6 | 1M | ✓ | $3.00/1M |
| `google/gemini-3.1-pro` | Gemini 3.1 Pro | 1M | ✓ | $1.25/1M |
| `google/gemini-3.5-flash` | Gemini 3.5 Flash | 1M | ✓ | $0.50/1M |
| `google/gemini-3.1-flash-lite` | Gemini 3.1 Flash-Lite | 1M | ✓ | $0.25/1M |
| `openai/gpt-5.4` | GPT-5.4 | 1M | ✓ | $2.50/1M |
| `openai/gpt-5.2` | GPT-5.2 | 256K | ✓ | $1.50/1M |
| `openai/gpt-4.1-mini` | GPT-4.1 Mini | 1M | ✓ | $0.40/1M |
| `openai/gpt-4o` | GPT-4o | 128K | ✓ | $2.50/1M |
| `openai/gpt-4o-mini` | GPT-4o Mini | 128K | ✓ | $0.15/1M |

Key metadata: `context_window_tokens` (drives writer source context budget), `supports_tools` (gates agent mode), `cost_per_1m_input_tokens` / `cost_per_1m_output_tokens`.

#### Embedding Catalog

**File:** `app/ai/embedding_catalog.py` → `EMBEDDING_CATALOG` dict

| Spec ID | Dimensions | Max Tokens | Cost |
|---------|-----------|------------|------|
| `google/gemini-embedding-001` | 3072 | 2,048 | $0.15/1M |
| `google/gemini-embedding-2` | 3072 | 8,192 | $0.15/1M |
| `openai/text-embedding-3-small` | 1536 | 8,191 | $0.02/1M |
| `openai/text-embedding-3-large` | 3072 | 8,191 | $0.13/1M |

Each `dimension` value must have a matching `wiki_page_embeddings_<dim>` table. Currently supported: 768, 1024, 1536, 3072.

#### Vision Catalog

**File:** `app/ai/vision_catalog.py` → `VISION_CATALOG` dict

| Spec ID | Label | Input Cost |
|---------|-------|------------|
| `google/gemini-3.5-flash` | Gemini 3.5 Flash | N/A |
| `google/gemini-3.1-flash-lite` | Gemini 3.1 Flash-Lite | $0.25/1M |
| `google/gemini-2.5-flash` | Gemini 2.5 Flash | $0.075/1M |
| `openai/gpt-4o` | GPT-4o | $2.50/1M |
| `openai/gpt-4o-mini` | GPT-4o Mini | $0.15/1M |

---

## 7. Data Model

**File:** `app/database/models.py`

### Source Models

| Model | Table | Description | Line |
|-------|-------|-------------|------|
| `Source` | `sources` | Source document with full_text, outline, pipeline state | 76 |
| `SourceDepartment` | `source_departments` | M2M: Source ↔ Department | 161 |
| `SourceImage` | `source_images` | Extracted images stored in MinIO | 181 |
| `SourceChunkExtract` | `source_chunk_extracts` | Phase 1 MAP output per chunk | 217 |
| `SourceCompilationPlan` | `source_compilation_plans` | Phase 2 REDUCE output: the compilation plan | 247 |

#### Source Status Flow

```
pending → processing → awaiting_approval (optional)
                     → plan_ready → processing → ready
                     → error (at any point)
```

#### Source Pipeline Phase

```
map → reduce → plan_review → refine → verify → commit
```

### Wiki Models

| Model | Table | Description | Line |
|-------|-------|-------------|------|
| `WikiPage` | `wiki_pages` | Core wiki page with markdown content, scope, version | 281 |
| `WikiLink` | `wiki_links` | Directed edges from `[[slug]]` patterns in content | 350 |
| `WikiBranch` | `wiki_branches` | Named contribution branch grouping drafts | 374 |
| `WikiPageDraft` | `wiki_page_drafts` | Proposed edits from contributors | 412 |
| `WikiPageRevision` | `wiki_page_revisions` | Immutable content snapshots per version | 481 |
| `WikiDraftRound` | `wiki_draft_rounds` | Review round snapshots for back-and-forth | 517 |

#### WikiPage Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `slug` | `String(300)` | URL-safe identifier, unique per scope |
| `title` | `String(500)` | Human-readable title |
| `content_md` | `Text` | Markdown content with `[[wikilinks]]` and `image://` markers |
| `summary` | `Text` | Short summary for index and search |
| `status` | `String(20)` | `seed` \| `developing` \| `mature` \| `evergreen` |
| `scope_type` | `String(20)` | `global` \| `project` |
| `scope_id` | `UUID` | Project/workspace ID (null for global) |
| `knowledge_type_slugs` | `ARRAY(String)` | Tags for categorization |
| `source_ids` | `ARRAY(UUID)` | Which sources contributed to this page |
| `version` | `Integer` | Incremented on each content change |
| `page_type` | hybrid property | Derived from slug: `index`, `log`, `hot`, `source`, or `concept` |

#### WikiPageDraft Key Fields

| Field | Description |
|-------|-------------|
| `draft_kind` | `edit` (modify existing) or `create` (propose new page) |
| `status` | `pending` \| `needs_revision` \| `withdrawn` \| `approved` \| `rejected` |
| `ai_check_status` | `pending` \| `running` \| `passed` \| `warned` \| `failed` |
| `base_version` | Version of target page when authored (detects mid-air collisions) |
| `revision_round` | Increments each time author resubmits after `needs_revision` |
| `source` | Origin: `web_ui` \| `mcp_claude_desktop` \| `mcp_claude_code` \| `api_direct` |

### Embedding Models

| Model | Table | Dimension | Line |
|-------|-------|-----------|------|
| `WikiPageEmbedding768` | `wiki_page_embeddings_768` | 768 | 1010 |
| `WikiPageEmbedding1024` | `wiki_page_embeddings_1024` | 1024 | 1015 |
| `WikiPageEmbedding1536` | `wiki_page_embeddings_1536` | 1536 | 1020 |
| `WikiPageEmbedding3072` | `wiki_page_embeddings_3072` | 3072 | 1025 |
| `SourceChunkEmbedding768–3072` | `source_chunk_embeddings_<dim>` | various | 1080–1095 |

Per-dimension tables using pgvector, supporting multiple embedding models with different output dimensions simultaneously.

---

## 8. API Endpoints

**File:** `app/routers/sources.py`

### Source Management

| Method | Path | Function | Line | Description |
|--------|------|----------|------|-------------|
| `POST` | `/sources/upload` | `upload_source()` | 319 | File upload → ingestion |
| `POST` | `/sources/url` | `add_url_source()` | 412 | URL → ingestion |
| `GET` | `/sources` | `list_sources()` | 170 | List all sources with filters |
| `GET` | `/sources/{id}` | `get_source()` | 261 | Source detail |
| `PATCH` | `/sources/{id}` | `update_source()` | 459 | Update metadata |
| `DELETE` | `/sources/{id}` | `delete_source()` | 881 | Delete source + cleanup |
| `POST` | `/sources/{id}/retry` | `retry_source()` | 563 | Retry failed source |

### Pipeline Control

| Method | Path | Function | Line | Description |
|--------|------|----------|------|-------------|
| `GET` | `/sources/{id}/progress` | `get_source_progress()` | 299 | Poll pipeline progress |
| `POST` | `/sources/{id}/approve-extraction` | `approve_extraction()` | 685 | Approve token-gated source |
| `GET` | `/sources/{id}/plan` | `get_compilation_plan()` | 653 | View compilation plan |
| `POST` | `/sources/{id}/plan/approve` | `approve_compilation_plan()` | 731 | Approve plan → enqueue REFINE |
| `POST` | `/sources/{id}/plan/regenerate` | `regenerate_compilation_plan()` | 786 | Re-run planning with feedback |
| `POST` | `/sources/{id}/plan/reject` | `reject_compilation_plan()` | 835 | Reject plan |

### Background Workers

**File:** `app/worker.py`

| Task | Function | Line | Trigger |
|------|----------|------|---------|
| `ingest_file_task` | `ingest_file_task()` | 91 | Source file upload |
| `ingest_url_task` | `ingest_url_task()` | 252 | Source URL addition |
| `ingest_map_reduce_task` | `ingest_map_reduce_task()` | 723 | After text extraction |
| `ingest_refine_task` | `ingest_refine_task()` | 827 | After plan approval |
| `regenerate_plan_task` | `regenerate_plan_task()` | 907 | Plan regeneration request |
| `caption_images_task` | `caption_images_task()` | 1173 | After image extraction |
| `ai_pre_review_draft_task` | `ai_pre_review_draft_task()` | 1293 | Draft AI pre-review |
| `reembed_all_pages_task` | `reembed_all_pages_task()` | 556 | Embedding model change |
| `sweep_stuck_processing_cron` | `sweep_stuck_processing_cron()` | 1043 | Periodic recovery |
| `daily_stats_rollup_cron` | `daily_stats_rollup_cron()` | 1157 | Daily stats aggregation |

---

## 9. Feature Summary

### Core Pipeline Features

| Feature | Description |
|---------|-------------|
| **MRP Pipeline** | 6-phase deterministic pipeline: Triage → MAP → REDUCE → REFINE → VERIFY → COMMIT |
| **Parallel extraction** | Up to 6 concurrent LLM calls for chunk extraction (Phase 1) |
| **Parallel writing** | Up to 4 concurrent page writers (Phase 3) |
| **Three writer strategies** | Simple (1 call), Complex (agent loop, 10 steps), Multi-pass (create/extend/polish) |
| **Dynamic context budgeting** | Writer source context budget scales with the model's context window from catalog metadata |
| **Human-reviewable plans** | Compilation plans pause for approval; reviewers can approve, reject, or regenerate with feedback |
| **Auto-approve mode** | Plans can be auto-approved via `mrp_auto_approve_plan` setting |

### Knowledge Quality

| Feature | Description |
|---------|-------------|
| **Compilation, not summarization** | Preserves specific numbers, regulations, procedures, names — never condenses |
| **Multi-source merging** | LLM-based content merge when new sources touch existing pages. Sanity check rejects if merged content is too short |
| **Contradiction detection** | Phase 4 finds factual conflicts between new and existing pages, annotates with callout blocks |
| **Coverage checking** | Phase 4 flags frequently-mentioned concepts not covered by any generated page |
| **Lifecycle assessment** | Automated page status (seed → developing → mature → evergreen) based on content depth and link density |
| **Wikilinks** | `[[slug]]` cross-linking between pages; `WikiLink` edges tracked in DB |
| **Image preservation** | Extracted images are captioned, stored in MinIO, and referenced via `image://<uuid>` markers in page content |

### Reliability & Safety

| Feature | Description |
|---------|-------------|
| **Crash resumability** | Each pipeline phase persists output to DB. Crashed pipelines resume from last completed phase |
| **Token gating** | Large documents require human approval before spending AI tokens |
| **Advisory locks** | `pg_advisory_xact_lock` per (slug, scope) prevents race conditions between concurrent pipelines |
| **Savepoints** | `session.begin_nested()` + `IntegrityError` handling for concurrent slug creation |
| **Merge sanity guard** | `BODY_SHRINK_THRESHOLD = 0.7` rejects LLM merges that are suspiciously shorter than inputs |
| **Image marker sanitization** | Hallucinated `image://<uuid>` references not in the source's actual images are stripped |
| **Auto-recovery cron** | `sweep_stuck_processing_cron` flips stuck sources back to error (max `max_auto_recover_attempts`) |

### Scope & Multi-Tenancy

| Feature | Description |
|---------|-------------|
| **Scope isolation** | Wiki pages are scoped: `global`, `project`, or `department`. All operations filter by scope |
| **Multi-scope commit** | A single source can produce pages in multiple scopes (e.g., one per assigned department) |
| **Knowledge types** | Sources and pages are categorized by `KnowledgeType` slugs |
| **Department assignments** | Sources can be assigned to multiple departments via M2M table |

### Contribution Workflow

| Feature | Description |
|---------|-------------|
| **Drafts** | Contributors propose edits via `WikiPageDraft` (both `edit` and `create` kinds) |
| **AI pre-review** | Drafts are automatically checked by AI before human review |
| **Revision rounds** | Reviewers can send drafts back for revisions; each round is snapshotted |
| **Branches** | Multiple drafts can be grouped into named `WikiBranch` for batch review |
| **Base version tracking** | Detects mid-air collisions when the target page changed while a draft was being authored |
| **Page revisions** | Every content change creates an immutable `WikiPageRevision` snapshot |

### AI Provider Flexibility

| Feature | Description |
|---------|-------------|
| **Provider-agnostic architecture** | Abstract base classes; no SDK imports in business logic |
| **Model catalogs** | Whitelisted models with metadata (context window, cost, capabilities) — no free-form IDs |
| **Runtime provider switching** | Admins change models via settings UI; takes effect on next request |
| **Multi-provider support** | Anthropic, OpenAI, Google for LLM; Google + OpenAI for embeddings and vision |
| **Per-dimension embedding tables** | Support multiple embedding models with different output dimensions simultaneously |
| **Connection testing** | `registry.test_all()` validates all configured providers in one call |
| **Verbatim mode** | Skip LLM pipeline entirely; index raw document chunks for semantic search only |
