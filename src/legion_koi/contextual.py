"""Contextual enrichment — prepend document metadata to chunks before embedding.

Each namespace has a preamble template that encodes document-level context
(title, source, date, namespace) into the embedding vector. This helps the
vector index distinguish "a chunk from a journal about X" from "a chunk from
a recording about X" — information that raw passage text alone doesn't carry.

The preamble is prepended at embed time only. Stored chunk_text remains raw.
"""


def extract_preamble(namespace: str, contents: dict) -> str:
    """Build a contextual preamble from document metadata.

    Prepended to each chunk before embedding so the vector encodes
    document-level context (title, source, date, namespace).
    Returns empty string if no meaningful preamble can be extracted.
    """
    fm = contents.get("frontmatter", {})

    if namespace == "legion.claude-journal":
        title = fm.get("title") or ""
        created = fm.get("created") or ""
        if title:
            parts = [f"Journal: {title}"]
            if created:
                parts.append(f"Date: {created}")
            return ". ".join(parts) + "."
        return ""

    if namespace == "legion.claude-recording":
        title = contents.get("title") or contents.get("filename") or ""
        source = contents.get("source") or ""
        date = contents.get("date_recorded") or ""
        if title:
            parts = [f"Recording: {title}"]
            if source:
                parts.append(f"Source: {source}")
            if date:
                parts.append(f"Date: {date}")
            return ". ".join(parts) + "."
        return ""

    if namespace == "legion.claude-web.conversation":
        name = contents.get("name") or ""
        if name:
            return f"Conversation: {name}."
        return ""

    if namespace == "legion.claude-web.project":
        name = contents.get("name") or ""
        desc = (contents.get("description") or "")[:100]
        if name:
            if desc:
                return f"Project: {name}. {desc}"
            return f"Project: {name}."
        return ""

    if namespace == "legion.claude-code":
        cwd = contents.get("cwd") or ""
        date = contents.get("date") or ""
        if cwd:
            parts = [f"Code session: {cwd}"]
            if date:
                parts.append(f"Date: {date}")
            return ". ".join(parts) + "."
        return ""

    if namespace == "legion.claude-venture":
        title = fm.get("title") or ""
        if title:
            return f"Venture: {title}."
        return ""

    if namespace == "legion.claude-message":
        chat_title = contents.get("chat_title") or ""
        date = (contents.get("platform_ts") or "")[:10]
        if chat_title:
            parts = [f"Message: {chat_title}"]
            if date:
                parts.append(f"Date: {date}")
            return ". ".join(parts) + "."
        return ""

    if namespace == "legion.claude-plan":
        title = contents.get("title") or ""
        plan_type = contents.get("plan_type") or "auto"
        if title:
            return f"Plan: {title}. Type: {plan_type}."
        return ""

    if namespace == "legion.claude-research":
        title = fm.get("title") or ""
        created = fm.get("created") or ""
        status = fm.get("status") or ""
        if title:
            parts = [f"Research: {title}"]
            if created:
                parts.append(f"Date: {created}")
            if status:
                parts.append(f"Status: {status}")
            return ". ".join(parts) + "."
        return ""

    if namespace == "legion.claude-logging":
        cwd = contents.get("cwd") or ""
        if cwd:
            return f"Session: {cwd}."
        return ""

    if namespace == "legion.claude-github":
        reference = contents.get("full_name") or contents.get("name") or ""
        if reference:
            return f"Repository: {reference}."
        return ""

    # Default fallback
    return f"Document from {namespace}."


def prepend_preamble(preamble: str, chunk_text: str) -> str:
    """Combine preamble and chunk text for embedding input.

    Returns chunk_text unchanged if preamble is empty.
    """
    if not preamble:
        return chunk_text
    return f"{preamble}\n---\n{chunk_text}"
