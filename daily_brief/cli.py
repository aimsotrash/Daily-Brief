"""Command-line interface.

Everything the GUI can do is available here, so Daily-Brief can be driven from
cron, systemd timers or a terminal without running the web app at all.

    daily-brief run                 start the web app (UI + API + scheduler)
    daily-brief refresh             fetch feeds and run the ingestion pipeline
    daily-brief generate            build today's personalised briefing
    daily-brief brief               print the latest briefing
    daily-brief search "..."        ask a grounded question about the news
    daily-brief configure           view / set / reset interests
    daily-brief sources             list configured sources and their health
    daily-brief status              show system status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from typing import Any

from . import __version__
from .config import ConfigError, load_config
from .service import Application

# --------------------------------------------------------------------------
# Terminal helpers
# --------------------------------------------------------------------------

_ISATTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _ISATTY else text


def bold(text: str) -> str:
    return _c(text, "1")


def dim(text: str) -> str:
    return _c(text, "2")


def cyan(text: str) -> str:
    return _c(text, "36")


def red(text: str) -> str:
    return _c(text, "31")


def green(text: str) -> str:
    return _c(text, "32")


def rule(title: str = "", width: int = 72) -> str:
    if not title:
        return dim("─" * width)
    return dim(f"── {title} " + "─" * max(0, width - len(title) - 4))


def wrap(text: str, indent: str = "  ", width: int = 88) -> str:
    out: list[str] = []
    for block in text.split("\n"):
        if not block.strip():
            out.append("")
            continue
        out.extend(
            textwrap.wrap(
                block, width=width, initial_indent=indent, subsequent_indent=indent
            )
            or [indent]
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_run(app: Application, args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    host = args.host or app.config.server.host
    port = args.port or app.config.server.port

    api = create_app(
        application=app,
        enable_scheduler=not args.no_scheduler,
        run_initial_ingest=not args.no_initial_ingest,
    )
    print(f"{bold('Daily-Brief')} {dim('v' + __version__)}")
    print(f"  UI      {cyan(f'http://{host}:{port}/')}")
    print(f"  API     {dim(f'http://{host}:{port}/docs')}")
    print(f"  Data    {dim(str(app.config.storage.db_path()))}")
    print(f"  Model   {dim(f'{app.provider.name}:{app.provider.model}')}")
    print()

    if args.open_browser or app.config.server.open_browser:
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}/")).start()

    uvicorn.run(api, host=host, port=port, log_level=app.config.logging.level.lower())
    return 0


def cmd_refresh(app: Application, args: argparse.Namespace) -> int:
    report = asyncio.run(app.refresh(args.source or None))
    if args.json:
        print(json.dumps(report.__dict__, default=lambda o: o.__dict__, indent=2))
        return 0

    print(bold("Ingestion complete"))
    print(wrap(report.summary()))
    failures = [r for r in report.per_source if r.error]
    if failures:
        print()
        print(rule(f"{len(failures)} source(s) failed"))
        for entry in failures[:20]:
            print(f"  {red('✗')} {entry.source_name}: {dim(entry.error[:110])}")
    return 0


def cmd_reanalyze(app: Application, args: argparse.Namespace) -> int:
    count = app.reanalyze()
    print(green(f"Re-analysed {count} stored article(s) (topics, entities, bias)."))
    return 0


def cmd_generate(app: Application, args: argparse.Namespace) -> int:
    payload = asyncio.run(app.generate_briefing())
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    _print_briefing(payload)
    return 0


def cmd_brief(app: Application, args: argparse.Namespace) -> int:
    payload = asyncio.run(app.get_briefing(refresh_if_stale=not args.no_refresh))
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    _print_briefing(payload)
    return 0


def _print_briefing(payload: dict[str, Any]) -> None:
    print()
    print(bold("  YOUR DAILY BRIEF"))
    interests = ", ".join(payload.get("interests") or []) or "no interests configured"
    print(dim(f"  {payload.get('date', '')} · {interests} · engine: {payload.get('engine')}"))
    print()

    sections = payload.get("sections") or []
    if not sections:
        reason = payload.get("empty_reason") or "nothing to report"
        print(wrap(f"Nothing in this briefing — {reason}."))
        print(wrap(dim("Try `daily-brief refresh` first, or widen your interests.")))
        return

    for section in sections:
        print(rule(section["title"].upper()))
        for story in section["stories"]:
            bias = story.get("bias") or {}
            lean = bias.get("lean", "unknown")
            tags = []
            if bias.get("is_opinion"):
                tags.append("opinion")
            if lean not in {"unknown", "not-applicable"}:
                tags.append(f"lean: {lean}")
            if story.get("source_count", 1) > 1:
                tags.append(f"{story['source_count']} outlets")
            tag_text = dim(f"  [{' · '.join(tags)}]") if tags else ""

            print(f"  {bold('•')} {bold(story['title'])}{tag_text}")
            print(dim(f"    {story['source']} · {(story.get('published_at') or '')[:16]}"))
            if story.get("summary"):
                print(wrap(story["summary"], indent="    "))
            print(dim(f"    {story['url']}"))
            print()


def cmd_search(app: Application, args: argparse.Namespace) -> int:
    question = " ".join(args.query).strip()
    if not question:
        print(red("error: provide a question, e.g. daily-brief search \"AI news today\""))
        return 2

    result = asyncio.run(app.ask(question, session_id=args.session, use_history=bool(args.session)))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print()
    print(rule("ANSWER"))
    print(wrap(result.answer))
    print()

    if result.sources:
        print(rule(f"SOURCES ({len(result.sources)})"))
        for source in result.sources:
            bias = source.get("bias") or {}
            lean = bias.get("lean", "unknown")
            marker = "" if lean in {"unknown", "not-applicable"} else dim(f" · lean: {lean}")
            print(f"  [{source['n']}] {bold(source['title'])}")
            print(
                dim(f"      {source['source']} · {(source.get('published_at') or 'no date')[:16]}")
                + marker
            )
            print(dim(f"      {source['url']}"))
        print()

    if result.warnings:
        print(rule("NOTES"))
        for warning in result.warnings:
            print(wrap(f"! {warning}"))
        print()

    print(dim(f"engine: {result.engine} · intent: {result.intent} · results: {result.has_results}"))
    return 0 if result.has_results else 1


def cmd_configure(app: Application, args: argparse.Namespace) -> int:
    if args.reset:
        app.preferences.reset()
        print(green("Personalization reset. Daily-Brief will onboard again on next start."))
        return 0

    if args.set:
        prefs = app.preferences.save_interests(" ".join(args.set))
        if not prefs.interests:
            print(red("Could not interpret any interests from that input."))
            return 1
        print(green(f"Saved {len(prefs.interests)} interest(s):"))
        for interest in prefs.interests:
            print(f"  • {interest}")
        return 0

    if args.interactive:
        print(bold("\n  Welcome to Daily-Brief\n"))
        print(wrap("What are you interested in? Tell me what you want to follow."))
        print(
            wrap(
                dim("e.g. AI, Linux, gaming, NVIDIA, geopolitics, US politics, India, technology")
            )
        )
        print()
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
        if not raw:
            print(red("Nothing entered."))
            return 1
        prefs = app.preferences.save_interests(raw)
        print(green(f"\nSaved {len(prefs.interests)} interest(s):"))
        for interest in prefs.interests:
            print(f"  • {interest}")
        return 0

    prefs = app.preferences.load()
    if args.json:
        print(json.dumps(prefs.to_dict(), indent=2))
        return 0

    print(bold("\n  Interests"))
    if not prefs.onboarded:
        print(wrap(dim("Not configured yet. Run `daily-brief configure --interactive`.")))
        return 0
    print(dim(f"  as typed: {prefs.raw_interests_text}"))
    print()
    for interest in app.preferences.interests():
        matched = ", ".join(interest.terms[:8])
        print(f"  • {bold(interest.label)}")
        print(dim(f"      topic={interest.topic or '(free text)'}  matches: {matched}…"))
    return 0


def cmd_sources(app: Application, args: argparse.Namespace) -> int:
    sources = app.source_repo.all()
    state = app.source_repo.all_feed_state()
    if args.json:
        print(json.dumps([s.as_context() | {"url": s.url} for s in sources], indent=2))
        return 0

    print(bold(f"\n  {len(sources)} sources configured "
               f"({sum(1 for s in sources if s.enabled)} enabled)\n"))
    for source in sources:
        st = state.get(source.id, {})
        failures = st.get("consecutive_failures") or 0
        if not source.enabled:
            marker = dim("○")
        elif failures:
            marker = red("✗")
        elif st.get("last_success_at"):
            marker = green("✓")
        else:
            marker = dim("·")
        lean = source.lean if str(source.lean) != "unknown" else "—"
        print(f"  {marker} {bold(source.name):<38} {dim(f'{lean} · {source.source_type}')}")
        if args.verbose:
            print(dim(f"      {source.url}"))
            if source.notes:
                print(wrap(dim(source.notes), indent="      "))
            if failures:
                print(red(f"      {failures} consecutive failures: {st.get('last_error', '')[:90]}"))
    if app.registry_error:
        print(red(f"\n  registry error: {app.registry_error}"))
    return 0


def cmd_status(app: Application, args: argparse.Namespace) -> int:
    payload = asyncio.run(app.status())
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    articles = payload["articles"]
    llm = payload["llm"]
    print(bold("\n  Daily-Brief status\n"))
    print(f"  onboarded    {payload['onboarded']}")
    print(f"  interests    {', '.join(payload['interests']) or dim('none')}")
    print(f"  articles     {articles['total']} stored, {articles['last_24h']} in last 24h")
    print(f"  sources      {payload['sources']['enabled']}/{payload['sources']['total']} enabled")
    available = green("reachable") if llm["available"] else red("unreachable")
    print(f"  model        {llm['provider']}:{llm['model'] or '—'} ({available})")
    if not llm["available"]:
        print(dim("               falling back to the deterministic extractive engine"))
    print(f"  briefing     {payload['briefing']['story_count']} stories, "
          f"generated {payload['briefing']['generated_at'] or dim('never')}")
    print(f"  database     {app.config.storage.db_path()}")
    failing = payload["sources"]["failing"]
    if failing:
        print(f"\n  {red('failing sources:')}")
        for entry in failing:
            print(dim(f"    {entry['id']}: {entry['failures']}x — {entry['error'][:80]}"))
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daily-brief",
        description="Self-hosted, bias-aware personal news briefing and research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              daily-brief configure --interactive
              daily-brief refresh
              daily-brief generate
              daily-brief search "what's happening with NVIDIA today?"
              daily-brief run --open-browser
            """
        ),
    )
    parser.add_argument("--version", action="version", version=f"daily-brief {__version__}")
    parser.add_argument("-c", "--config", help="path to config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="start the web application")
    p_run.add_argument("--host")
    p_run.add_argument("--port", type=int)
    p_run.add_argument("--no-scheduler", action="store_true", help="disable background jobs")
    p_run.add_argument("--no-initial-ingest", action="store_true")
    p_run.add_argument("--open-browser", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_refresh = sub.add_parser("refresh", help="fetch feeds and ingest articles")
    p_refresh.add_argument("--source", action="append", help="limit to a source id (repeatable)")
    p_refresh.add_argument("--json", action="store_true")
    p_refresh.set_defaults(func=cmd_refresh)

    p_reanalyze = sub.add_parser(
        "reanalyze", help="recompute topics/entities/bias for stored articles"
    )
    p_reanalyze.set_defaults(func=cmd_reanalyze)

    p_generate = sub.add_parser("generate", help="generate the personalised daily briefing")
    p_generate.add_argument("--json", action="store_true")
    p_generate.set_defaults(func=cmd_generate)

    p_brief = sub.add_parser("brief", help="print the latest briefing")
    p_brief.add_argument("--no-refresh", action="store_true", help="never regenerate")
    p_brief.add_argument("--json", action="store_true")
    p_brief.set_defaults(func=cmd_brief)

    p_search = sub.add_parser("search", help="ask a grounded question about the news")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--session", help="conversation id, to enable follow-ups")
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_conf = sub.add_parser("configure", help="view or change your interests")
    p_conf.add_argument("--set", nargs="+", metavar="INTEREST", help="replace interests")
    p_conf.add_argument("-i", "--interactive", action="store_true", help="prompt for interests")
    p_conf.add_argument("--reset", action="store_true", help="clear all personalization")
    p_conf.add_argument("--json", action="store_true")
    p_conf.set_defaults(func=cmd_configure)

    p_sources = sub.add_parser("sources", help="list configured news sources")
    p_sources.add_argument("-v", "--verbose", action="store_true")
    p_sources.add_argument("--json", action="store_true")
    p_sources.set_defaults(func=cmd_sources)

    p_status = sub.add_parser("status", help="show system status")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(red(f"configuration error: {exc}"), file=sys.stderr)
        return 2

    app = Application(config)
    try:
        return args.func(app, args)
    except KeyboardInterrupt:
        print()
        return 130
    finally:
        if args.command != "run":
            try:
                asyncio.run(app.aclose())
            except Exception:  # pragma: no cover
                pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
