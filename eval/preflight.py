#!/usr/bin/env python3
"""
Check the environment before spending a full eval run on it.

Verifies, in increasing order of cost: env vars present, Qdrant reachable with a
populated collection, Groq reachable and the key accepted, embedding model
loadable. Prints one line per check and exits non-zero on the first hard failure.

    python -m eval.preflight
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COLLECTION = "indian_legal_library"
OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
problems: list[str] = []


def report(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f"  -  {detail}" if detail else ""))


def main() -> int:
    env_path = PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        report(WARN, "python-dotenv", "not installed - reading the shell environment only")
    report(OK if env_path.exists() else WARN, ".env file",
           str(env_path) if env_path.exists() else "not found; relying on shell environment")

    # --- 1. required variables --------------------------------------------- #
    for name in ("GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
        value = os.getenv(name)
        if value:
            report(OK, name, f"set ({len(value)} chars)")
        else:
            report(BAD, name, "missing - required")
            problems.append(name)
    if not os.getenv("POSTGRES_URI"):
        report(WARN, "POSTGRES_URI", "unset - fine for eval (in-memory checkpointer), needed by app.py")
    else:
        report(OK, "POSTGRES_URI", "set")
    if os.getenv("LEGISYUKTI_TELEMETRY", "1").lower() in ("0", "false", "no", "off"):
        report(WARN, "LEGISYUKTI_TELEMETRY", "disabled - eval will report no latency numbers")

    if problems:
        print(f"\nFAIL: {len(problems)} required variable(s) missing. Fix these first.")
        return 1

    # --- 2. Qdrant ---------------------------------------------------------- #
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=30)
        names = [c.name for c in client.get_collections().collections]
        report(OK, "Qdrant reachable", f"{len(names)} collection(s)")
        if COLLECTION not in names:
            report(BAD, f"collection '{COLLECTION}'", "does not exist - run the ingestion pipeline")
            problems.append("collection")
        else:
            count = client.get_collection(COLLECTION).points_count
            if count and count > 0:
                report(OK, f"collection '{COLLECTION}'", f"{count} points")
            else:
                report(BAD, f"collection '{COLLECTION}'", "EMPTY - eval would score 0 recall everywhere")
                problems.append("empty collection")
    except Exception as exc:
        report(BAD, "Qdrant", f"{type(exc).__name__}: {exc}")
        problems.append("qdrant")

    # --- 3. Groq ------------------------------------------------------------ #
    # Both configured models are probed, because Groq retires them on a rolling
    # basis and a 404 on one says nothing about the other. On failure the account's
    # actually-available model IDs are listed, so a deprecation is self-service.
    from src.config import GROQ_MODEL_PRIMARY, GROQ_MODEL_FAST
    report(OK, "device", f"{__import__('src.config', fromlist=['DEVICE']).DEVICE}")

    groq_failed = False
    for label, model in (("fast", GROQ_MODEL_FAST), ("primary", GROQ_MODEL_PRIMARY)):
        try:
            from langchain_groq import ChatGroq
            probe = ChatGroq(model_name=model, temperature=0, max_tokens=8,
                             api_key=os.getenv("GROQ_API_KEY"), max_retries=0)
            reply = probe.invoke("Reply with the single word: ok")
            report(OK, f"Groq {label} model", f"{model} replied {reply.content.strip()[:20]!r}")
        except Exception as exc:
            detail = str(exc)
            hint = " - model retired or not enabled on this account" if "model_not_found" in detail or "does not exist" in detail else ""
            report(BAD, f"Groq {label} model", f"{model}: {type(exc).__name__}{hint}")
            groq_failed = True

    if groq_failed:
        problems.append("groq")
        try:
            import httpx
            resp = httpx.get("https://api.groq.com/openai/v1/models",
                             headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                             timeout=20)
            ids = sorted(m["id"] for m in resp.json().get("data", []))
            if ids:
                print("\n  Models your Groq account CAN serve:")
                for mid in ids:
                    print(f"    {mid}")
                print("\n  Set the ones you want in .env, no code change needed:")
                print("    GROQ_MODEL_PRIMARY=<a large instruct model>")
                print("    GROQ_MODEL_FAST=<a small fast model>\n")
        except Exception as exc:
            print(f"  (could not list available models: {type(exc).__name__}: {exc})")

    # --- 4. embedding model ------------------------------------------------- #
    try:
        from src.engine import load_embeddings
        dim = len(load_embeddings().embed_query("test"))
        expected = 384
        if dim == expected:
            report(OK, "embedding model", f"e5-small-v2 loaded, dim={dim}")
        else:
            report(BAD, "embedding model", f"dim={dim}, collection expects {expected}")
            problems.append("embedding dim")
    except Exception as exc:
        report(BAD, "embedding model", f"{type(exc).__name__}: {exc}")
        problems.append("embeddings")

    print()
    if problems:
        print(f"FAIL: {len(problems)} problem(s): {', '.join(problems)}")
        return 1
    print("PASS: environment is ready.  Next:  python -m eval.run_eval --label baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
