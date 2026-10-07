"""
Wiki Service — CRUD, semantic search, and wikilink graph for WikiPage.

The wiki is the LLM-compiled knowledge layer. It replaces chunk-based RAG.
Each page is markdown that may contain `[[slug]]` wikilinks; after every
upsert, refresh_links() re-parses the content and rewrites the wiki_links
edge table so 1-2 hop graph queries (backlinks, neighborhood) stay fast in
PostgreSQL — no separate graph DB needed.

Scope support: every page belongs to a scope (global or workspace). Query
functions accept scope_type/scope_id to isolate results. Default is global.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from sqlalchemy import and_, delete, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import WikiLink, WikiPage, WikiPageDraft, WikiPageRevision

# Reserved page slugs — these are regular WikiPage rows but treated specially.
INDEX_SLUG = "_index"
LOG_SLUG = "_log"
HOT_SLUG = "_hot"

# Recognized page types — used for filtering and prompt hints to the compiler.
PAGE_TYPES = {"entity", "concept", "source", "topic", "index", "log", "hot"}

# `[[slug]]` or `[[slug|display text]]` — captures the slug only.
_WIKILINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]*)?]]")


# ---------------------------------------------------------------------------
# Scope filter helper
# ---------------------------------------------------------------------------

def _scope_filter(scope_type: str = "global", scope_id: Optional[uuid.UUID] = None):
    """Return SQLAlchemy WHERE clauses for exact scope filtering."""
    if scope_id:
        return and_(WikiPage.scope_type == scope_type, WikiPage.scope_id == scope_id)
    return and_(WikiPage.scope_type == scope_type, WikiPage.scope_id.is_(None))


def _scope_filter_with_dept(department_ids: Optional[list[uuid.UUID]] = None):
    """OR-filter: global pages + department pages visible to the given dept members.

    DEPRECATED for MCP read paths — does NOT include project-scoped pages,
    which made wiki pages of workspaces invisible to their own members. Use
    `_scope_filter_for_identity` instead.
    """
    if department_ids:
        return or_(
            and_(WikiPage.scope_type == "global", WikiPage.scope_id.is_(None)),
            and_(WikiPage.scope_type == "department", WikiPage.scope_id.in_(department_ids)),
        )
    return _scope_filter("global")


def _scope_filter_for_identity(
    department_ids: Optional[list[uuid.UUID]] = None,
    project_ids: Optional[list[uuid.UUID]] = None,
):
    """OR-filter for the MCP read path: every wiki page the user can see.

    Includes:
      - All global pages.
      - Department pages of every department the user belongs to.
      - Project pages of every workspace the user is a member of.

    Without the project branch, members of a workspace cannot find their own
    workspace's wiki pages via search — they fall through to raw-source
    drill-down and assume the page doesn't exist.
    """
    clauses = [and_(WikiPage.scope_type == "global", WikiPage.scope_id.is_(None))]
    if department_ids:
        clauses.append(
            and_(WikiPage.scope_type == "department", WikiPage.scope_id.in_(department_ids))
        )
    if project_ids:
        clauses.append(
            and_(WikiPage.scope_type == "project", WikiPage.scope_id.in_(project_ids))
        )
    return or_(*clauses)


def _inverse_scope_filter_for_identity(
    department_ids: Optional[list[uuid.UUID]] = None,
    project_ids: Optional[list[uuid.UUID]] = None,
):
    """Pages OUTSIDE the user's accessible scope — used by out-of-scope hints.

    Excludes global pages (everyone sees those) so the inverse is just:
      - Department pages of departments the user does NOT belong to.
      - Project pages of workspaces the user is NOT a member of.
    """
    project_clause = (
        and_(WikiPage.scope_type == "project", WikiPage.scope_id.notin_(project_ids))
        if project_ids
        else WikiPage.scope_type == "project"
    )
    dept_clause = (
        and_(WikiPage.scope_type == "department", WikiPage.scope_id.notin_(department_ids))
        if department_ids
        else WikiPage.scope_type == "department"
    )
    return or_(dept_clause, project_clause)


# ---------------------------------------------------------------------------
# Wikilink parsing & graph maintenance
# ---------------------------------------------------------------------------

def extract_wikilinks(content_md: str) -> list[str]:
    """Return the list of slugs referenced by `[[slug]]` patterns, deduped."""
    if not content_md:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _WIKILINK_RE.finditer(content_md):
        slug = match.group(1).strip()
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out


async def refresh_links(
    session: AsyncSession,
    from_page_id: uuid.UUID,
    from_slug: str,
    content_md: str,
) -> None:
    """
    Replace all outgoing edges from the page identified by `from_page_id` with
    wikilinks parsed from its current `content_md`. Self-links (matching the
    page's own slug) are dropped to keep the graph sane.
    """
    await session.execute(
        delete(WikiLink).where(WikiLink.from_page_id == from_page_id)
    )
    targets = [s for s in extract_wikilinks(content_md) if s != from_slug]
    if not targets:
        return
    await session.execute(
        pg_insert(WikiLink)
        .values([{"from_page_id": from_page_id, "to_slug": t} for t in targets])
        .on_conflict_do_nothing()
    )


async def get_backlinks(
    session: AsyncSession,
    slug: str,
    scope_type: Optional[str] = None,
    scope_id: Optional[uuid.UUID] = None,
) -> list[str]:
    """Slugs of pages that link to `slug`.

    If scope filters are given, only return slugs of origin pages in the same
    scope OR in global scope (global referrers are visible from any scope).
    """
    stmt = (
        select(WikiPage.slug)
        .join(WikiLink, WikiLink.from_page_id == WikiPage.id)
        .where(WikiLink.to_slug == slug)
    )
    if scope_type is not None:
        stmt = stmt.where(
            or_(
                and_(WikiPage.scope_type == scope_type, WikiPage.scope_id == scope_id) if scope_id is not None
                else and_(WikiPage.scope_type == scope_type, WikiPage.scope_id.is_(None)),
                and_(WikiPage.scope_type == "global", WikiPage.scope_id.is_(None)),
            )
        )
    result = await session.execute(stmt.distinct())
    return [row[0] for row in result.all()]


async def get_outlinks(
    session: AsyncSession,
    slug: str,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
) -> list[str]:
    """Slugs that the page (`slug`, scope) links to."""
    page = await get_page_by_slug(session, slug, scope_type=scope_type, scope_id=scope_id)
    if page is None:
        return []
    result = await session.execute(
        select(WikiLink.to_slug).where(WikiLink.from_page_id == page.id)
    )
    return [row[0] for row in result.all()]


async def get_neighborhood(
    session: AsyncSession,
    slug: str,
    depth: int = 1,
) -> dict:
    """
    Return nodes (slug, title, page_type) and edges within `depth` hops of `slug`.
    Uses an undirected recursive CTE — useful for Obsidian-style graph view.
    """
    depth = max(1, min(depth, 3))  # cap at 3 hops to keep queries cheap
    # Recursive CTE walking both directions over (origin_slug, target_slug)
    # tuples derived from wiki_links joined with wiki_pages on from_page_id.
    # WITH RECURSIVE is required because `walk` self-references inside its
    # own definition. Without the RECURSIVE keyword Postgres treats `walk` as
    # not-yet-defined when it parses the second arm of the UNION, raising
    # `relation "walk" does not exist`. The `edges` non-recursive CTE is
    # allowed in the same WITH clause as long as RECURSIVE is set once.
    cte_sql = text(
        """
        WITH RECURSIVE edges AS (
            SELECT wp.slug AS from_slug, wl.to_slug AS to_slug
            FROM wiki_links wl
            JOIN wiki_pages wp ON wp.id = wl.from_page_id
        ),
        walk(slug, dist) AS (
            SELECT CAST(:start AS varchar), 0
          UNION
            SELECT
              CASE WHEN e.from_slug = w.slug THEN e.to_slug ELSE e.from_slug END,
              w.dist + 1
            FROM walk w
            JOIN edges e
              ON e.from_slug = w.slug OR e.to_slug = w.slug
            WHERE w.dist < :depth
        )
        SELECT DISTINCT slug FROM walk
        """
    )
    rows = await session.execute(cte_sql, {"start": slug, "depth": depth})
    slugs = [r[0] for r in rows.all()]
    if not slugs:
        return {"nodes": [], "edges": []}

    pages_result = await session.execute(
        select(WikiPage.slug, WikiPage.title, WikiPage.page_type)
        .where(WikiPage.slug.in_(slugs))
    )
    nodes = [
        {"slug": r.slug, "title": r.title, "page_type": r.page_type}
        for r in pages_result.all()
    ]
    edges_result = await session.execute(
        select(WikiPage.slug.label("from_slug"), WikiLink.to_slug)
        .join(WikiLink, WikiLink.from_page_id == WikiPage.id)
        .where(and_(WikiPage.slug.in_(slugs), WikiLink.to_slug.in_(slugs)))
    )
    edges = [{"from": r.from_slug, "to": r.to_slug} for r in edges_result.all()]
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Page CRUD
# ---------------------------------------------------------------------------

async def get_page_by_slug(
    session: AsyncSession,
    slug: str,
    allowed_kt_slugs: Optional[list[str]] = None,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
) -> Optional[WikiPage]:
    """
    Fetch a page by slug within a specific scope. If `allowed_kt_slugs` is
    given (RBAC), only return the page when it overlaps the allowed set or is
    a reserved slug.
    """
    stmt = select(WikiPage).where(
        WikiPage.slug == slug,
        _scope_filter(scope_type, scope_id),
    )
    result = await session.execute(stmt)
    page = result.scalars().first()
    if page is None:
        return None
    if allowed_kt_slugs is None or slug in (INDEX_SLUG, LOG_SLUG, HOT_SLUG):
        return page
    if not page.knowledge_type_slugs:
        return page
    if any(s in allowed_kt_slugs for s in page.knowledge_type_slugs):
        return page
    return None


async def get_page_by_slug_any_scope(
    session: AsyncSession,
    slug: str,
) -> Optional[WikiPage]:
    """
    Fetch a page by slug across ALL scopes (no scope filtering).
    Used as a fallback when no explicit scope is specified, e.g. global graph view
    clicking on a workspace-scoped wiki page.
    """
    stmt = select(WikiPage).where(WikiPage.slug == slug).limit(1)
    result = await session.execute(stmt)
    return result.scalars().first()


async def list_pages(
    session: AsyncSession,
    page_type: Optional[str] = None,
    knowledge_type_slug: Optional[str] = None,
    allowed_kt_slugs: Optional[list[str]] = None,
    query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
    department_ids: Optional[list[uuid.UUID]] = None,
    project_ids: Optional[list[uuid.UUID]] = None,
    all_scopes: bool = False,
) -> list[WikiPage]:
    """List pages with filtering within a scope.

    Scope behaviour:
      - `all_scopes=True`: no scope filter at all (admin bypass).
      - `department_ids` (and optionally `project_ids`) given: union of global
        + every department the user belongs to + every workspace the user is
        a member of.
      - Otherwise: exact `scope_type`/`scope_id` (pipeline write path).
    """
    if all_scopes:
        scope_clause = None
    elif department_ids or project_ids:
        scope_clause = _scope_filter_for_identity(department_ids, project_ids)
    else:
        scope_clause = _scope_filter(scope_type, scope_id)
    stmt = (
        select(WikiPage)
        .where(WikiPage.slug.notin_([INDEX_SLUG, LOG_SLUG, HOT_SLUG]))
        .order_by(WikiPage.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if scope_clause is not None:
        stmt = stmt.where(scope_clause)
    if page_type:
        stmt = stmt.where(WikiPage.page_type == page_type)
    if knowledge_type_slug:
        stmt = stmt.where(WikiPage.knowledge_type_slugs.any(knowledge_type_slug))  # type: ignore[arg-type]
    if allowed_kt_slugs:
        stmt = stmt.where(
            or_(
                WikiPage.knowledge_type_slugs.overlap(allowed_kt_slugs),
                func.cardinality(WikiPage.knowledge_type_slugs) == 0,
            )
        )
    if query:
        stmt = stmt.where(
            or_(
                WikiPage.title.ilike(f"%{query}%"),
                WikiPage.slug.ilike(f"%{query}%"),
                WikiPage.content_md.ilike(f"%{query}%"),
            )
        )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def search_pages_semantic(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 10,
    allowed_kt_slugs: Optional[list[str]] = None,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
    spec_id: Optional[str] = None,
    department_ids: Optional[list[uuid.UUID]] = None,
    project_ids: Optional[list[uuid.UUID]] = None,
    inverse_scope: bool = False,
    all_scopes: bool = False,
) -> list[tuple[WikiPage, float]]:
    """
    Cosine-similarity search over wiki page embeddings within a scope.

    Embeddings live in per-dimension tables (`wiki_page_embeddings_<dim>`).
    The active embedding model spec determines which table to query and which
    `model_spec_id` rows to filter to. Pass `spec_id` explicitly to override —
    only used by tests and internal tooling.

    Scope behaviour:
      - If `department_ids` or `project_ids` is given: returns pages from
        global + user's departments + user's workspaces (MCP read path).
      - If `inverse_scope=True`: returns pages OUTSIDE that scope (other
        departments, workspaces the user isn't a member of). Used to surface
        "you don't have access" hints.
      - Otherwise uses exact scope_type/scope_id matching (pipeline write path).

    Returns (page, similarity) pairs sorted by similarity descending. Returns
    an empty list if no active embedding model is configured.
    """
    from app.ai.embedding_catalog import get_spec
    from app.ai.registry import ProviderRegistry
    from app.database.models import get_embedding_model_for_dim

    if spec_id is None:
        registry = ProviderRegistry(session)
        spec_id = await registry.get_active_embedding_spec_id()
    if not spec_id:
        return []

    spec = get_spec(spec_id)
    Emb = get_embedding_model_for_dim(spec.dimension)

    if all_scopes and not inverse_scope:
        scope_clause = None
    elif inverse_scope:
        scope_clause = _inverse_scope_filter_for_identity(department_ids, project_ids)
    elif department_ids or project_ids:
        scope_clause = _scope_filter_for_identity(department_ids, project_ids)
    else:
        scope_clause = _scope_filter(scope_type, scope_id)

    where_clauses = [
        Emb.model_spec_id == spec.id,
        WikiPage.slug.notin_([INDEX_SLUG, LOG_SLUG, HOT_SLUG]),
    ]
    if scope_clause is not None:
        where_clauses.append(scope_clause)

    stmt = (
        select(
            WikiPage,
            (1 - Emb.embedding.cosine_distance(query_embedding)).label("similarity"),
        )
        .join(Emb, Emb.page_id == WikiPage.id)
        .where(and_(*where_clauses))
        .order_by(Emb.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    if allowed_kt_slugs:
        stmt = stmt.where(
            or_(
                WikiPage.knowledge_type_slugs.overlap(allowed_kt_slugs),
                func.cardinality(WikiPage.knowledge_type_slugs) == 0,
            )
        )
    result = await session.execute(stmt)
    return [(row[0], float(row[1])) for row in result.all()]


async def search_source_chunks_semantic(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 10,
    allowed_source_ids: Optional[set[str]] = None,
    spec_id: Optional[str] = None,
):
    """
    Cosine-similarity search over verbatim source chunk embeddings.

    Mirrors `search_pages_semantic` but over `source_chunk_embeddings_<dim>`
    (raw, never-rewritten slices of preserve_verbatim sources). Lets high-fidelity
    docs (decrees, gazettes) be retrieved in the same semantic pool as wiki pages.

    RBAC: pass `allowed_source_ids` (the set returned by the MCP layer's
    `_get_allowed_source_ids`). None means open access; an empty set means no
    access (returns nothing).

    Returns (Source, chunk_row, similarity) tuples sorted by similarity desc.
    Returns [] if no active embedding model is configured.
    """
    import uuid as _uuid

    from app.ai.embedding_catalog import get_spec
    from app.ai.registry import ProviderRegistry
    from app.database.models import (
        Source,
        get_source_chunk_embedding_model_for_dim,
    )

    if allowed_source_ids is not None and len(allowed_source_ids) == 0:
        return []

    if spec_id is None:
        registry = ProviderRegistry(session)
        spec_id = await registry.get_active_embedding_spec_id()
    if not spec_id:
        return []

    spec = get_spec(spec_id)
    Emb = get_source_chunk_embedding_model_for_dim(spec.dimension)

    where_clauses = [
        Emb.model_spec_id == spec.id,
        Source.preserve_verbatim.is_(True),
        Source.status == "ready",
    ]
    if allowed_source_ids is not None:
        where_clauses.append(
            Source.id.in_([_uuid.UUID(s) for s in allowed_source_ids])
        )

    stmt = (
        select(
            Source,
            Emb,
            (1 - Emb.embedding.cosine_distance(query_embedding)).label("similarity"),
        )
        .join(Source, Source.id == Emb.source_id)
        .where(and_(*where_clauses))
        .order_by(Emb.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await session.execute(stmt)
    return [(row[0], row[1], float(row[2])) for row in result.all()]


# ---------------------------------------------------------------------------
# Compiler ops application
# ---------------------------------------------------------------------------

async def apply_create(
    session: AsyncSession,
    slug: str,
    title: str,
    page_type: str,
    content_md: str,
    summary: str,
    knowledge_type_slugs: list[str],
    source_ids: list[uuid.UUID],
    embedding: Optional[list[float]] = None,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
    status: str = "seed",
) -> WikiPage:
    """Insert a new page in the given scope. Conflicts raise — caller should use update."""
    page = WikiPage(
        slug=slug,
        title=title,
        status=status,
        content_md=content_md,
        summary=summary,
        knowledge_type_slugs=list(knowledge_type_slugs or []),
        source_ids=list(source_ids or []),
        # embedding intentionally omitted: stored in wiki_page_embeddings_<dim>
        scope_type=scope_type,
        scope_id=scope_id,
        version=1,
    )
    _ = embedding  # backward-compat parameter, ignored
    session.add(page)
    await session.flush()
    await refresh_links(session, page.id, slug, content_md)
    session.add(WikiPageRevision(
        page_id=page.id, version=page.version,
        content_md=content_md, change_type="agent_compile",
    ))
    return page


async def apply_update(
    session: AsyncSession,
    slug: str,
    new_content_md: str,
    summary: Optional[str] = None,
    title: Optional[str] = None,
    add_knowledge_type_slug: Optional[str] = None,
    add_source_id: Optional[uuid.UUID] = None,
    embedding: Optional[list[float]] = None,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
    status: Optional[str] = None,
) -> Optional[WikiPage]:
    """
    Update an existing page atomically within the given scope:
      - Replace content_md with new_content_md.
      - Optionally update title/summary.
      - Union add_knowledge_type_slug into knowledge_type_slugs.
      - Append add_source_id to source_ids if not present.
      - Optionally update lifecycle status.
      - Bump version, refresh updated_at, refresh embedding if supplied.
    Returns None if the page does not exist.
    """
    page = await get_page_by_slug(session, slug, scope_type=scope_type, scope_id=scope_id)
    if page is None:
        return None

    page.content_md = new_content_md
    if title is not None:
        page.title = title
    if summary is not None:
        page.summary = summary
    if status is not None:
        page.status = status
    if add_knowledge_type_slug and add_knowledge_type_slug not in (page.knowledge_type_slugs or []):
        page.knowledge_type_slugs = [*(page.knowledge_type_slugs or []), add_knowledge_type_slug]
    if add_source_id and add_source_id not in (page.source_ids or []):
        page.source_ids = [*(source_id for source_id in (page.source_ids or [])), add_source_id]
    # Embeddings are no longer stored on WikiPage; the compiler calls
    # _reembed_pages after this returns, which writes into the active
    # wiki_page_embeddings_<dim> table. The `embedding` parameter is accepted
    # only for backward compatibility and ignored here.
    _ = embedding
    page.version = (page.version or 1) + 1
    await session.flush()
    await refresh_links(session, page.id, slug, new_content_md)
    session.add(WikiPageRevision(
        page_id=page.id, version=page.version,
        content_md=new_content_md, change_type="agent_compile",
    ))
    return page


async def upsert_page(
    session: AsyncSession,
    slug: str,
    title: str,
    page_type: str,
    content_md: str,
    summary: str,
    knowledge_type_slugs: list[str],
    source_ids: list[uuid.UUID],
    embedding: Optional[list[float]] = None,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
    status: Optional[str] = None,
) -> WikiPage:
    """Create-or-update by slug within a scope."""
    # Acquire a transaction-level advisory lock based on the hash of the slug
    # to serialize concurrent upserts for the exact same page.
    lock_query = select(func.pg_advisory_xact_lock(func.hashtext(slug)))
    await session.execute(lock_query)

    existing = await get_page_by_slug(session, slug, scope_type=scope_type, scope_id=scope_id)
    if existing is None:
        return await apply_create(
            session, slug, title, page_type, content_md, summary,
            knowledge_type_slugs, source_ids, embedding,
            scope_type=scope_type, scope_id=scope_id,
            status=status or "seed",
        )
    return await apply_update(
        session,
        slug=slug,
        new_content_md=content_md,
        summary=summary,
        title=title,
        add_knowledge_type_slug=knowledge_type_slugs[0] if knowledge_type_slugs else None,
        add_source_id=source_ids[0] if source_ids else None,
        embedding=embedding,
        scope_type=scope_type, scope_id=scope_id,
        status=status,
    ) or existing


# ---------------------------------------------------------------------------
# Reserved pages: _index and _log
# ---------------------------------------------------------------------------

async def regenerate_index(
    session: AsyncSession,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
) -> WikiPage:
    """
    Rebuild the `_index` page within the given scope.
    Grouped by page_type, alphabetical within group. Excludes reserved slugs.
    """
    stmt = (
        select(WikiPage.slug, WikiPage.title, WikiPage.page_type, WikiPage.summary)
        .where(
            WikiPage.slug.notin_([INDEX_SLUG, LOG_SLUG, HOT_SLUG]),
            _scope_filter(scope_type, scope_id),
        )
        .order_by(WikiPage.page_type, WikiPage.title)
    )
    rows = (await session.execute(stmt)).all()

    by_type: dict[str, list[tuple[str, str, str]]] = {}
    for r in rows:
        by_type.setdefault(r.page_type, []).append((r.slug, r.title, r.summary or ""))

    lines = ["# Wiki Index", ""]
    if not by_type:
        lines.append("_(empty — no pages yet)_")
    else:
        for ptype in sorted(by_type.keys()):
            lines.append(f"## {ptype.capitalize()}")
            lines.append("")
            for slug, title, summary in by_type[ptype]:
                summary_part = f" — {summary}" if summary else ""
                lines.append(f"- [[{slug}|{title}]]{summary_part}")
            lines.append("")

    new_md = "\n".join(lines).rstrip() + "\n"
    page = await get_page_by_slug(session, INDEX_SLUG, scope_type=scope_type, scope_id=scope_id)
    if page is None:
        page = WikiPage(
            slug=INDEX_SLUG,
            title="Wiki Index",
            # page_type="index",
            content_md=new_md,
            summary="Catalog of all wiki pages",
            knowledge_type_slugs=[],
            source_ids=[],
            scope_type=scope_type,
            scope_id=scope_id,
        )
        session.add(page)
    else:
        page.content_md = new_md
        page.version = (page.version or 1) + 1
    await session.flush()
    return page


async def append_log(
    session: AsyncSession,
    entry: str,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
) -> WikiPage:
    """
    Append a timestamped line to the `_log` page within the given scope.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    line = f"## [{ts}] {entry.strip()}"
    page = await get_page_by_slug(session, LOG_SLUG, scope_type=scope_type, scope_id=scope_id)
    if page is None:
        page = WikiPage(
            slug=LOG_SLUG,
            title="Wiki Log",
            # page_type="log",
            content_md=f"# Wiki Log\n\n{line}\n",
            summary="Chronological activity log",
            knowledge_type_slugs=[],
            source_ids=[],
            scope_type=scope_type,
            scope_id=scope_id,
        )
        session.add(page)
    else:
        existing = page.content_md or "# Wiki Log\n"
        if "_(empty" in existing:
            existing = "# Wiki Log\n"
        page.content_md = existing.rstrip() + f"\n\n{line}\n"
        page.version = (page.version or 1) + 1
    await session.flush()
    return page


# ---------------------------------------------------------------------------
# Page deletion — cascade cleanup
# ---------------------------------------------------------------------------

async def delete_page_cascade(
    session: AsyncSession,
    page: WikiPage,
) -> None:
    """
    Delete a wiki page and cascade-cleanup all references:
    1. Delete all outgoing links from this page
    2. Delete all incoming links pointing to this page
    3. Remove [[slug]] and [[slug|text]] wikilinks from pages that reference this one
    4. Delete the page itself

    Caller passes the already-resolved page so we never accidentally fall back
    to a different scope's copy of the same slug.
    """
    slug = page.slug
    del_scope_type = page.scope_type or "global"
    del_scope_id = page.scope_id

    # 1+2: Remove edges. Outgoing edges from this page cascade via FK. Incoming
    # edges (to this slug) are removed only from referrers in the same scope
    # OR from global referrers (which logically point at the deleted page if
    # it is global) — leave edges in other scopes intact because they target
    # *that* scope's same-slug page, not the one we're deleting.
    if del_scope_type == "global":
        # Deleting a global page invalidates ALL [[slug]] references because
        # those links resolve to global by default. Clear all incoming edges.
        await session.execute(
            delete(WikiLink).where(WikiLink.to_slug == slug)
        )
    else:
        same_scope_pages = select(WikiPage.id).where(
            WikiPage.scope_type == del_scope_type,
            WikiPage.scope_id == del_scope_id,
        )
        await session.execute(
            delete(WikiLink).where(
                WikiLink.to_slug == slug,
                WikiLink.from_page_id.in_(same_scope_pages),
            )
        )

    # 3: Find pages that reference this slug in their content and clean up.
    # Scope the scrub the same way: only rewrite same-scope pages (and globals
    # when deleting a global page).
    ref_stmt = select(WikiPage).where(
        WikiPage.content_md.contains(f"[[{slug}]]")
        | WikiPage.content_md.contains(f"[[{slug}|")
    )
    if del_scope_type != "global":
        ref_stmt = ref_stmt.where(
            WikiPage.scope_type == del_scope_type,
            WikiPage.scope_id == del_scope_id,
        )
    referring_pages = (await session.execute(ref_stmt)).scalars().all()

    for ref_page in referring_pages:
        if ref_page.id == page.id:
            continue
        cleaned = ref_page.content_md or ""
        # Replace [[slug|display]] with just display text
        cleaned = re.sub(
            rf"\[\[{re.escape(slug)}\|([^\]]+)]]",
            r"\1",
            cleaned,
        )
        # Replace [[slug]] with slug text
        cleaned = cleaned.replace(f"[[{slug}]]", slug.split("/")[-1])
        ref_page.content_md = cleaned

    # 4: Delete the page itself
    await session.delete(page)

    await session.flush()
    logger.info(f"delete_page_cascade({slug}): deleted page + cleaned {len(referring_pages)} references")


# ---------------------------------------------------------------------------
# Source removal — used when deleting a source
# ---------------------------------------------------------------------------

async def detach_source_from_wiki(
    session: AsyncSession,
    source_id: uuid.UUID,
) -> int:
    """
    Remove `source_id` from every WikiPage.source_ids.
    - Pages that have other contributing sources: keep, just remove this source_id.
    - Pages whose only source was this one: delete immediately.

    Returns the number of pages deleted.
    """
    stmt = select(WikiPage).where(WikiPage.source_ids.any(source_id))  # type: ignore[arg-type]
    pages = list((await session.execute(stmt)).scalars().all())
    deleted_count = 0
    for page in pages:
        remaining = [sid for sid in (page.source_ids or []) if sid != source_id]
        if not remaining:
            await session.delete(page)
            deleted_count += 1
        else:
            page.source_ids = remaining
    await session.flush()
    if deleted_count:
        logger.info(f"detach_source_from_wiki({source_id}): deleted {deleted_count} single-source pages")
    return deleted_count


# ---------------------------------------------------------------------------
# Draft workflow
# ---------------------------------------------------------------------------

class DraftConflictError(Exception):
    """Raised when a draft's base_version is older than the current page version."""
    def __init__(self, current_version: int, base_version: int):
        self.current_version = current_version
        self.base_version = base_version
        super().__init__(
            f"Draft is based on v{base_version} but the page has advanced to "
            f"v{current_version}. Re-base the draft against the latest content."
        )


async def create_draft(
    session: AsyncSession,
    page_id: Optional[uuid.UUID],
    author_id: uuid.UUID,
    content_md: str,
    note: Optional[str] = None,
    source: str = "web_ui",
    source_metadata: Optional[dict] = None,
    base_version: Optional[int] = None,
    draft_kind: str = "edit",
    suggested_metadata: Optional[dict] = None,
) -> WikiPageDraft:
    """Create a pending draft for editor review.

    For draft_kind='edit', page_id is required. For 'create', page_id stays
    None and suggested_metadata holds the contributor's proposed slug/title/
    page_type/knowledge_type_slugs/scope. The reviewer can override the
    metadata at approve time before the page is materialised.
    """
    draft = WikiPageDraft(
        page_id=page_id,
        author_id=author_id,
        content_md=content_md,
        note=note,
        status="pending",
        source=source,
        source_metadata=source_metadata,
        base_version=base_version,
        draft_kind=draft_kind,
        suggested_metadata=suggested_metadata,
    )
    session.add(draft)
    await session.flush()
    return draft


class CreateDraftSlugConflict(Exception):
    """Raised when approving a create-draft whose slug already exists in scope."""
    def __init__(self, slug: str, scope_type: str, scope_id: Optional[uuid.UUID]):
        self.slug = slug
        self.scope_type = scope_type
        self.scope_id = scope_id
        scope_label = scope_type if scope_id is None else f"{scope_type}:{scope_id}"
        super().__init__(
            f"Slug '{slug}' already exists in {scope_label}. "
            "Override final_slug, or have the contributor edit the existing page instead."
        )


async def approve_draft(
    session: AsyncSession,
    draft: WikiPageDraft,
    reviewer_id: uuid.UUID,
    reviewer_note: Optional[str] = None,
    edited_content_md: Optional[str] = None,
    allow_conflict: bool = False,
    metadata_overrides: Optional[dict] = None,
) -> WikiPage:
    """
    Approve a pending draft. Writes the final content to wiki_pages.content_md,
    creates a revision, and marks the draft approved.
    If edited_content_md is provided, that is used instead of the original draft content.

    For draft_kind='create' the page is materialised from
    `draft.suggested_metadata` (or the reviewer-supplied `metadata_overrides`)
    using `apply_create`. The reviewer may override slug / title / page_type /
    knowledge_type_slugs before commit.

    Raises DraftConflictError when an edit draft was authored against an older
    page version than the current one, unless `allow_conflict=True` or
    `edited_content_md` is supplied. Raises CreateDraftSlugConflict when a
    create draft's chosen slug already exists in the target scope.
    """
    final_content = edited_content_md.strip() if edited_content_md else draft.content_md

    # Serialise concurrent approves on the same page. Without this, two
    # reviewers clicking Approve on different pending drafts of the same
    # page within the same second can both read page.version=N, both set
    # N+1, and both INSERT a WikiPageRevision(version=N+1) — leaving a
    # duplicate revision row and a non-deterministic last-writer-wins for
    # the page content. Lock by slug (when known) so we don't block the
    # entire page table.
    target_slug: Optional[str] = None
    existing_page: Optional[WikiPage] = None
    if draft.draft_kind == "create":
        target_slug = (draft.suggested_metadata or {}).get("slug")
    else:
        existing_page = await session.get(WikiPage, draft.page_id) if draft.page_id else None
        target_slug = existing_page.slug if existing_page else None
    if target_slug:
        await session.execute(
            select(func.pg_advisory_xact_lock(func.hashtext(target_slug)))
        )
        # The page row was loaded BEFORE the lock; another reviewer may have
        # bumped its version while we waited. Refresh from DB so version /
        # content_md reflect the committed state inside the critical section.
        if existing_page is not None:
            await session.refresh(existing_page)

    if draft.draft_kind == "create":
        meta = dict(draft.suggested_metadata or {})
        overrides = metadata_overrides or {}
        slug = (overrides.get("final_slug") or meta.get("slug") or "").strip()
        title = (overrides.get("final_title") or meta.get("title") or "").strip()
        page_type = overrides.get("final_page_type") or meta.get("page_type") or "concept"
        kt_slugs = (
            overrides.get("final_knowledge_type_slugs")
            if overrides.get("final_knowledge_type_slugs") is not None
            else meta.get("knowledge_type_slugs") or []
        )
        scope_type = meta.get("scope_type") or "global"
        scope_id_raw = meta.get("scope_id")
        try:
            scope_id = uuid.UUID(scope_id_raw) if isinstance(scope_id_raw, str) else scope_id_raw
        except (ValueError, TypeError):
            scope_id = None
        if scope_id is not None and not isinstance(scope_id, uuid.UUID):
            # Hand-crafted metadata with a non-string non-UUID (e.g. int)
            # shouldn't propagate downstream. Treat as missing scope.
            scope_id = None

        if not slug or slug in (INDEX_SLUG, LOG_SLUG, HOT_SLUG):
            raise ValueError(f"Invalid slug for new page: '{slug}'")
        if not title:
            raise ValueError("Title is required to materialise a new page")

        existing = await get_page_by_slug(session, slug, scope_type=scope_type, scope_id=scope_id)
        if existing is not None:
            raise CreateDraftSlugConflict(slug, scope_type, scope_id)

        page = await apply_create(
            session,
            slug=slug, title=title, page_type=page_type,
            content_md=final_content, summary="",
            knowledge_type_slugs=list(kt_slugs), source_ids=[],
            scope_type=scope_type, scope_id=scope_id,
        )
        # Tag the create-approval revision with reviewer context.
        session.add(WikiPageRevision(
            page_id=page.id,
            version=page.version,
            content_md=final_content,
            change_type="draft_approved_create",
            draft_id=draft.id,
            changed_by_id=reviewer_id,
            change_note=reviewer_note,
        ))
        # Backfill draft.page_id so subsequent UI reads can join cleanly.
        draft.page_id = page.id
    else:
        page = await session.get(WikiPage, draft.page_id) if draft.page_id else None
        if page is None:
            raise ValueError(f"Wiki page {draft.page_id} not found")

        if (
            not allow_conflict
            and edited_content_md is None
            and draft.base_version is not None
            and page.version is not None
            and draft.base_version < page.version
        ):
            raise DraftConflictError(page.version, draft.base_version)

        page.content_md = final_content
        page.version = (page.version or 1) + 1
        await session.flush()
        await refresh_links(session, page.id, page.slug, final_content)

        session.add(WikiPageRevision(
            page_id=page.id,
            version=page.version,
            content_md=final_content,
            change_type="draft_approved",
            draft_id=draft.id,
            changed_by_id=reviewer_id,
            change_note=reviewer_note,
        ))

    draft.status = "approved"
    draft.reviewed_by_id = reviewer_id
    draft.reviewed_at = datetime.now(timezone.utc)
    draft.reviewer_note = reviewer_note
    await session.flush()
    return page


async def reject_draft(
    session: AsyncSession,
    draft: WikiPageDraft,
    reviewer_id: uuid.UUID,
    reviewer_note: str,
) -> WikiPageDraft:
    """Reject a pending draft with a required reason."""
    draft.status = "rejected"
    draft.reviewed_by_id = reviewer_id
    draft.reviewed_at = datetime.now(timezone.utc)
    draft.reviewer_note = reviewer_note
    await session.flush()
    return draft


async def direct_edit_page(
    session: AsyncSession,
    page: WikiPage,
    editor_id: uuid.UUID,
    content_md: str,
    change_note: Optional[str] = None,
) -> WikiPage:
    """
    Sync write by an editor/admin — no review step.
    Creates a revision immediately.
    """
    page.content_md = content_md
    page.version = (page.version or 1) + 1
    await session.flush()
    await refresh_links(session, page.id, page.slug, content_md)

    session.add(WikiPageRevision(
        page_id=page.id,
        version=page.version,
        content_md=content_md,
        change_type="editor_edit",
        changed_by_id=editor_id,
        change_note=change_note,
    ))
    await session.flush()
    return page


async def rollback_to_revision(
    session: AsyncSession,
    page: WikiPage,
    target_version: int,
    actor_id: uuid.UUID,
) -> WikiPage:
    """
    Restore a page to a previous revision snapshot.
    Creates a new revision recording the rollback.
    """
    revision = (await session.execute(
        select(WikiPageRevision).where(
            WikiPageRevision.page_id == page.id,
            WikiPageRevision.version == target_version,
        )
    )).scalar_one_or_none()
    if revision is None:
        raise ValueError(f"Revision v{target_version} not found for page {page.slug}")

    page.content_md = revision.content_md
    page.version = (page.version or 1) + 1
    await session.flush()
    await refresh_links(session, page.id, page.slug, revision.content_md)

    session.add(WikiPageRevision(
        page_id=page.id,
        version=page.version,
        content_md=revision.content_md,
        change_type="rollback",
        changed_by_id=actor_id,
        change_note=f"rollback to v{target_version}",
    ))
    await session.flush()
    return page


async def regenerate_hot_cache(
    session: AsyncSession,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
) -> WikiPage:
    """
    Generate or rebuild the dynamic hot cache page (`_hot`) for the given scope.
    It summarizes:
      - The 10 most recently updated pages.
      - A list of active 'seed' pages that need more knowledge.
      - Any detected and unresolved contradictions (pages containing `[!contradiction]`).
    It calls the active LLM to compile this into a dense, high-value briefing.
    """
    from app.ai.registry import ProviderRegistry

    # 1. Fetch recent pages
    recent_stmt = (
        select(WikiPage.slug, WikiPage.title, WikiPage.status, WikiPage.updated_at)
        .where(
            WikiPage.slug.notin_([INDEX_SLUG, LOG_SLUG, HOT_SLUG]),
            _scope_filter(scope_type, scope_id),
        )
        .order_by(WikiPage.updated_at.desc())
        .limit(10)
    )
    recent_rows = (await session.execute(recent_stmt)).all()
    
    # 2. Fetch seed pages
    seed_stmt = (
        select(WikiPage.slug, WikiPage.title)
        .where(
            WikiPage.slug.notin_([INDEX_SLUG, LOG_SLUG, HOT_SLUG]),
            WikiPage.status == "seed",
            _scope_filter(scope_type, scope_id),
        )
        .order_by(WikiPage.title)
        .limit(20)
    )
    seed_rows = (await session.execute(seed_stmt)).all()

    # 3. Fetch contradiction pages
    contradiction_stmt = (
        select(WikiPage.slug, WikiPage.title, WikiPage.content_md)
        .where(
            WikiPage.slug.notin_([INDEX_SLUG, LOG_SLUG, HOT_SLUG]),
            WikiPage.content_md.contains("[!contradiction]"),
            _scope_filter(scope_type, scope_id),
        )
        .order_by(WikiPage.title)
    )
    contradiction_rows = (await session.execute(contradiction_stmt)).all()

    # 4. Format context natively (Zero LLM cost fallback)
    recent_prose = "\n".join(f"- [[{r.slug}|{r.title}]] (Trạng thái: {r.status}, Cập nhật: {r.updated_at.strftime('%Y-%m-%d %H:%M') if r.updated_at else 'N/A'})" for r in recent_rows) if recent_rows else "- Không có cập nhật gần đây."
    seed_prose = "\n".join(f"- [[{r.slug}|{r.title}]]" for r in seed_rows) if seed_rows else "- Không có trang sơ khởi nào."
    
    contradiction_items = []
    for r in contradiction_rows:
        match = re.search(r">\s*\[!contradiction]\s*(.*?)(?=\n[^>]|\n\n|$)", r.content_md, re.DOTALL | re.IGNORECASE)
        callout_desc = match.group(1).replace(">", "").strip() if match else "Có mâu thuẫn tri thức được phát hiện."
        contradiction_items.append(f"- [[{r.slug}|{r.title}]]: {callout_desc}")
    contradiction_prose = "\n".join(contradiction_items) if contradiction_items else "- Không phát hiện mâu thuẫn tri thức nào."

    new_md = f"""# ⚡ Arkon Briefing Tri Thức Nóng

*(Bản tin trạng thái tri thức tự động của hệ thống)*

## ⚠️ Mâu thuẫn tri thức đang tồn tại
{contradiction_prose}

## 🔄 Tri thức cập nhật gần đây
{recent_prose}

## 🌱 Tri thức sơ khởi (Seed Pages)
{seed_prose}
"""

    page = await get_page_by_slug(session, HOT_SLUG, scope_type=scope_type, scope_id=scope_id)
    if page is None:
        page = WikiPage(
            slug=HOT_SLUG,
            title="Briefing Tri Thức Nóng",
            status="evergreen",
            content_md=new_md,
            summary="Bản tin tóm tắt tự động về tri thức và các mâu thuẫn cần giải quyết",
            knowledge_type_slugs=[],
            source_ids=[],
            scope_type=scope_type,
            scope_id=scope_id,
        )
        session.add(page)
    else:
        page.content_md = new_md
        page.status = "evergreen"
        page.version = (page.version or 1) + 1
    await session.flush()
    return page


async def lint_wiki(
    session: AsyncSession,
    scope_type: str = "global",
    scope_id: Optional[uuid.UUID] = None,
) -> dict:
    """
    Diagnose structural health issues in the wiki:
      - Dead links: wikilinks pointing to non-existent pages in this scope.
      - Orphan pages: pages with zero incoming backlinks.
      - Contradiction nodes: pages containing '[!contradiction]' callouts.
    """
    # 1. Fetch all pages in the current scope
    pages_stmt = select(WikiPage.slug, WikiPage.title, WikiPage.id, WikiPage.status).where(
        _scope_filter(scope_type, scope_id)
    )
    pages_rows = (await session.execute(pages_stmt)).all()
    all_slugs = {r.slug for r in pages_rows}
    slug_to_title = {r.slug: r.title for r in pages_rows}

    # 2. Fetch all links in this scope
    page_ids = [r.id for r in pages_rows]
    links_stmt = select(WikiLink.from_page_id, WikiLink.to_slug)
    if page_ids:
        links_stmt = links_stmt.where(WikiLink.from_page_id.in_(page_ids))
    links_rows = (await session.execute(links_stmt)).all()

    # 3. Find dead links
    dead_links = []
    id_to_slug = {r.id: r.slug for r in pages_rows}
    backlink_counts: dict[str, int] = {slug: 0 for slug in all_slugs}

    for from_page_id, to_slug in links_rows:
        from_slug = id_to_slug.get(from_page_id)
        if not from_slug:
            continue
        
        is_exist = (to_slug in all_slugs)
        
        if not is_exist:
            # Check global scope as fallback
            if scope_type != "global":
                global_exists = (await session.execute(
                    select(WikiPage.id).where(WikiPage.slug == to_slug, WikiPage.scope_type == "global", WikiPage.scope_id.is_(None))
                )).scalar_one_or_none()
                if global_exists:
                    is_exist = True

        if is_exist:
            if to_slug in backlink_counts:
                backlink_counts[to_slug] += 1
        else:
            if to_slug not in (INDEX_SLUG, LOG_SLUG, HOT_SLUG):
                dead_links.append({
                    "from_slug": from_slug,
                    "from_title": slug_to_title.get(from_slug, from_slug),
                    "to_slug": to_slug
                })

    # 4. Find orphan pages (0 incoming backlinks, excluding reserved pages)
    orphans = []
    for r in pages_rows:
        if r.slug in (INDEX_SLUG, LOG_SLUG, HOT_SLUG):
            continue
        if backlink_counts.get(r.slug, 0) == 0:
            orphans.append({
                "slug": r.slug,
                "title": r.title,
                "status": r.status
            })

    # 5. Find pages carrying contradictions
    contradictions_stmt = select(WikiPage.slug, WikiPage.title).where(
        WikiPage.content_md.contains("[!contradiction]"),
        _scope_filter(scope_type, scope_id)
    )
    contradictions_rows = (await session.execute(contradictions_stmt)).all()
    contradiction_nodes = [
        {"slug": r.slug, "title": r.title}
        for r in contradictions_rows
    ]

    return {
        "dead_links": dead_links,
        "orphans": orphans,
        "contradictions": contradiction_nodes
    }
