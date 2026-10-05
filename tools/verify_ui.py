#!/usr/bin/env python3
"""Browser verification for the Daily-Brief frontend.

Drives a real Firefox over the Marionette protocol against a running server and
asserts the things that unit tests cannot see: responsive layout, disclosure
behaviour, bias badges in coverage lists, search state, settings filtering and
absence of horizontal overflow.

Kept out of the pytest suite deliberately -- it needs a live server, a live
browser and (for search) a reachable model, so it is an explicit command rather
than something that runs on every commit.

    python tools/verify_ui.py --base-url http://127.0.0.1:8787 --shots /tmp/shots
    python tools/verify_ui.py --skip-search        # much faster; no LLM needed

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MARIONETTE_PORT = 2829


class Marionette:
    """A very small Marionette client: navigate, evaluate, screenshot."""

    def __init__(self, port: int = MARIONETTE_PORT, timeout: float = 300.0) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.buf = b""
        self.mid = 0
        self._read()
        self.send("WebDriver:NewSession", {"capabilities": {}})

    def _read(self) -> list:
        while b":" not in self.buf:
            self.buf += self.sock.recv(65536)
        length, _, rest = self.buf.partition(b":")
        n = int(length)
        self.buf = rest
        while len(self.buf) < n:
            self.buf += self.sock.recv(65536)
        payload, self.buf = self.buf[:n], self.buf[n:]
        return json.loads(payload)

    def send(self, command: str, params: dict | None = None):
        self.mid += 1
        blob = json.dumps([0, self.mid, command, params or {}]).encode()
        self.sock.sendall(b"%d:%s" % (len(blob), blob))
        response = self._read()
        if response[2]:
            raise RuntimeError(f"{command} failed: {response[2]}")
        return response[3]

    def go(self, url: str) -> None:
        self.send("WebDriver:Navigate", {"url": url})

    def js(self, expression: str):
        return self.send(
            "WebDriver:ExecuteScript", {"script": expression, "args": []}
        )["value"]

    def size(self, width: int, height: int) -> None:
        self.send("WebDriver:SetWindowRect", {"width": width, "height": height})

    def screenshot(self, path: Path, full: bool = False) -> None:
        data = self.send("WebDriver:TakeScreenshot", {"full": full, "hash": False})["value"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data))

    def focus_on(self, selector: str) -> None:
        """Scroll a selector into the middle of the viewport before a shot.

        Screenshots are viewport-sized, so a shot named after a component has to
        actually contain it."""
        self.js(
            f"const el = document.querySelector('{selector}');"
            "if (el) el.scrollIntoView({block: 'center'});"
        )
        time.sleep(0.4)

    def wait(self, expression: str, timeout: float = 240.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.js(f"return ({expression})"):
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, condition, detail: str = "") -> bool:
        ok = bool(condition)
        if ok:
            self.passed += 1
            print(f"  PASS  {name}")
        else:
            self.failed.append(name)
            print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")
        return ok

    def summary(self) -> int:
        total = self.passed + len(self.failed)
        print(f"\n{self.passed}/{total} checks passed")
        for name in self.failed:
            print(f"  failed: {name}")
        return 1 if self.failed else 0


def start_firefox(profile: Path, port: int) -> subprocess.Popen:
    """Launch a private headless Firefox.

    The port comes from a profile pref, not the command line: Firefox ignores
    ``--marionette-port`` and always binds the pref value. ``--no-remote`` plus a
    throwaway profile keeps this completely separate from any Firefox the user
    already has open -- this script never touches their session.
    """
    binary = shutil.which("firefox")
    if not binary:
        sys.exit("firefox not found on PATH")
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "user.js").write_text(
        f'user_pref("marionette.port", {port});\n'
        'user_pref("browser.shell.checkDefaultBrowser", false);\n'
        'user_pref("datareporting.policy.dataSubmissionEnabled", false);\n',
        encoding="utf-8",
    )
    return subprocess.Popen(
        [binary, "--headless", "-marionette", "--profile", str(profile),
         "--no-remote", "about:blank"],
        env={**os.environ, "MOZ_HEADLESS": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def connect(port: int, retries: int = 60) -> Marionette:
    for _ in range(retries):
        try:
            return Marionette(port)
        except OSError:
            time.sleep(1)
    sys.exit(f"could not connect to Marionette on port {port}")


# ---------------------------------------------------------------- the checks


def columns_of(m: Marionette, selector: str) -> int:
    return m.js(
        f"const el = document.querySelector('{selector}');"
        "return el ? getComputedStyle(el).gridTemplateColumns.split(' ').length : 0"
    )


def no_overflow(m: Marionette) -> bool:
    return m.js("return document.documentElement.scrollWidth <= window.innerWidth + 1")


def verify(m: Marionette, base: str, shots: Path, report: Report, skip_search: bool) -> None:
    # ---------------------------------------------------------- desktop wide
    print("\n[desktop 1440x1000]")
    m.size(1440, 1000)
    m.go(base + "/")
    onboarding_open = m.wait("!document.querySelector('#onboarding').hidden", 20)
    if onboarding_open:
        report.check("onboarding shown on a fresh install", True)
        m.screenshot(shots / "onboarding.png")
        print("  (not onboarded — skipping briefing checks)")
        return

    report.check("briefing rendered", m.wait("document.querySelectorAll('.brief-section').length > 0"))
    m.screenshot(shots / "desktop-briefing.png", full=True)

    sections = m.js("return document.querySelectorAll('.brief-section').length")
    stories = m.js("return document.querySelectorAll('.story').length")
    print(f"  ({sections} sections, {stories} stories)")

    report.check("two-column briefing grid on wide desktop",
                 columns_of(m, ".brief-grid") == 2,
                 f"got {columns_of(m, '.brief-grid')}")

    report.check("container uses desktop width",
                 m.js("return document.querySelector('main.wrap').clientWidth") > 1300)
    report.check("briefing columns are wide enough to read in",
                 m.js("return document.querySelector('.brief-section').clientWidth") > 600)

    report.check("headlines still link to the publisher",
                 m.js("return !!document.querySelector('.story-title a[href^=\"http\"]')"))

    report.check("no horizontal overflow (desktop)", no_overflow(m))

    # ------------------------------------------------------ visual hierarchy
    print("\n[hierarchy & density]")
    sizes = m.js("""
      const px = (el, prop) => el ? parseFloat(getComputedStyle(el)[prop]) : 0;
      const story = document.querySelector('.story');
      return {
        heading: px(document.querySelector('.section-heading'), 'fontSize'),
        section: px(document.querySelector('.brief-section-head h3'), 'fontSize'),
        title:   px(story.querySelector('.story-title'), 'fontSize'),
        weight:  px(story.querySelector('.story-title'), 'fontWeight'),
        meta:    px(story.querySelector('.meta-row'), 'fontSize'),
        summary: px(story.querySelector('.story-summary'), 'fontSize'),
      };
    """)
    report.check("page heading outranks section headings",
                 sizes["heading"] > sizes["section"], str(sizes))
    report.check("headline outranks its metadata",
                 sizes["title"] > sizes["meta"], str(sizes))
    report.check("headline is the heaviest weight in a story",
                 sizes["weight"] >= 600, str(sizes["weight"]))
    report.check("summary sits between headline and metadata",
                 sizes["meta"] <= sizes["summary"] < sizes["title"], str(sizes))
    # Wider container, not wider lines: cap the dense summary near 85 characters.
    report.check("summary keeps a readable measure",
                 m.js("return document.querySelector('.story-summary').clientWidth") < 600)
    report.check("summary uses most of its column",
                 m.js("const s = document.querySelector('.story-summary');"
                      "return s.clientWidth / s.closest('.brief-section').clientWidth") > 0.80)
    # Sections are packed into two columns by height rather than paired row by
    # row, so a one-story section can't strand half a row.
    heights = m.js(
        "return [...document.querySelectorAll('.brief-grid > .brief-col')]"
        ".map(c => c.getBoundingClientRect().height)"
    )
    if len(heights) == 2:
        skew = abs(heights[0] - heights[1]) / max(heights)
        report.check("briefing columns are packed to similar heights",
                     skew < 0.30, f"{heights[0]:.0f}px vs {heights[1]:.0f}px")
    report.check("stories carry framing badges",
                 m.js("return document.querySelectorAll('.story .lean').length") > 0)
    report.check("story hover state is defined",
                 m.js("return [...document.styleSheets].some(s => {"
                      "try { return [...s.cssRules].some(r =>"
                      "  r.selectorText && r.selectorText.includes('.story:hover')); }"
                      "catch(e) { return false; } })"))

    # ------------------------------------------------------ topic navigation
    print("\n[topic navigation]")
    tabs = m.js("return document.querySelectorAll('#topic-nav .topic-tab').length")
    report.check("topic navigation rendered", tabs >= 2, f"{tabs} tabs")
    report.check("an 'All' tab is selected by default",
                 m.js("return document.querySelector("
                      "'#topic-nav .topic-tab[data-topic=\"all\"]')"
                      ".getAttribute('aria-pressed') === 'true'"))
    report.check("topic tabs are real buttons",
                 m.js("return document.querySelector('#topic-nav .topic-tab').tagName") == "BUTTON")
    if tabs >= 2:
        m.js("document.querySelectorAll('#topic-nav .topic-tab')[1].click()")
        time.sleep(0.3)
        visible = m.js("return [...document.querySelectorAll('.brief-section')]"
                       ".filter(s => !s.hidden).length")
        report.check("selecting a topic filters the briefing",
                     visible == 1, f"{visible} sections visible")
        report.check("selected topic is visually marked",
                     m.js("return document.querySelectorAll('#topic-nav .topic-tab"
                          "[aria-pressed=\"true\"]').length") == 1)
        report.check("no horizontal overflow while filtered", no_overflow(m))
        m.screenshot(shots / "desktop-topic-filtered.png")
        m.js("document.querySelector('#topic-nav .topic-tab[data-topic=\"all\"]').click()")
        time.sleep(0.3)
        restored = m.js("return [...document.querySelectorAll('.brief-section')]"
                        ".filter(s => !s.hidden).length")
        report.check("'All' restores every section", restored == sections,
                     f"{restored} of {sections}")

    # ------------------------------------------------------------- coverage
    print("\n[coverage disclosure]")
    has_coverage = m.js("return !!document.querySelector('.coverage-details')")
    if has_coverage:
        report.check("coverage collapsed by default",
                     m.js("return !document.querySelector('.coverage-details').open"))
        summary_text = m.js(
            "return document.querySelector('.coverage-details > summary').textContent.trim()"
        )
        report.check("coverage summary names a source count",
                     "source" in summary_text.lower(), summary_text)
        m.js("document.querySelector('.coverage-details').open = true;")
        time.sleep(0.4)
        rows = m.js("return document.querySelectorAll('.coverage-details[open] .coverage-row').length")
        report.check("expanded coverage lists outlets", rows >= 2, f"{rows} rows")

        claimed = m.js(
            "const s=document.querySelector('.coverage-details > summary').textContent;"
            "return parseInt(s.replace(/[^0-9]/g,''),10)"
        )
        report.check("listed outlets match the claimed count", rows == claimed,
                     f"claimed {claimed}, listed {rows}")

        badges = m.js(
            "return document.querySelectorAll('.coverage-details[open] .coverage-row .lean').length"
        )
        report.check("every coverage row carries a classification badge",
                     badges == rows, f"{badges} badges for {rows} rows")

        # Regression guard: an expanded coverage list must not drag its grid
        # column wider than the other one.
        widths = m.js(
            "return [...document.querySelectorAll('.brief-grid > *')]"
            ".map(s => s.getBoundingClientRect().width)"
        )
        if widths and len(set(round(w) for w in widths)) > 0:
            spread = max(widths) - min(widths)
            report.check("columns stay balanced with coverage expanded",
                         spread < 40, f"width spread {spread:.0f}px")
        report.check("coverage outlets are clickable",
                     m.js("return !!document.querySelector('.coverage-row a.coverage-outlet')"))
        report.check("compare-coverage affordance present",
                     m.js("return !!document.querySelector('.coverage-compare')"))
        report.check("coverage names the lead article",
                     m.js("return !!document.querySelector('.coverage-lead-mark')"))
        report.check("coverage headline previews present",
                     m.js("return document.querySelectorAll("
                          "'.coverage-details[open] .coverage-headline').length") == rows)
        # The spread strip is conditional: it only appears when the outlets
        # actually disagree, so its absence is information, not a failure.
        spread = m.js("return [...document.querySelectorAll('.coverage-details')]"
                      ".some(d => d.querySelector('.spread-bar'))")
        print(f"  (framing spread strip present on some story: {bool(spread)})")
        if m.js("return !!document.querySelector('.coverage-details[open] .spread-bar')"):
            report.check("spread strip segments sum to the classified outlets",
                         m.js("return [...document.querySelectorAll("
                              "'.coverage-details[open] .spread-seg')]"
                              ".reduce((n, s) => n + parseFloat(s.style.width), 0)") > 99)
            report.check("spread strip is labelled for assistive tech",
                         m.js("return !!document.querySelector("
                              "'.coverage-details[open] .spread-bar[aria-label]')"))
        m.focus_on(".coverage-details[open]")
        m.screenshot(shots / "desktop-coverage.png")
        m.js("document.querySelector('.coverage-details').open = false;"
             "window.scrollTo(0, 0);")
    else:
        print("  (no multi-outlet story in the current briefing — skipped)")

    # ------------------------------------------------------ bias evidence
    print("\n[bias evidence]")
    if m.js("return !!document.querySelector('.story details.why:not(.coverage-details)')"):
        report.check(
            "classification evidence collapsed by default",
            m.js("return !document.querySelector("
                 "'.story details.why:not(.coverage-details)').open"),
        )
        m.js("document.querySelector('.story details.why:not(.coverage-details)').open = true;")
        time.sleep(0.3)
        body = m.js(
            "return document.querySelector("
            "'.story details.why:not(.coverage-details)[open] .why-body').textContent"
        )
        report.check("evidence shows signals", "Signals found" in body or "Markers" in body)
        report.check("evidence shows the disclaimer", "not a statement of fact" in body)
        report.check("evidence shows subjectivity/method", "Subjectivity" in body)
        report.check("evidence separates article label from source prior",
                     "This article" in body and "Source prior" in body, body[:120])
        report.check("evidence panel is a labelled card",
                     m.js("return !!document.querySelector('.why-body.evidence .evidence-title')"))
        report.check("subjectivity is shown as a meter",
                     m.js("return !!document.querySelector('.evidence .meter > span')"))
        m.focus_on(".story details.why:not(.coverage-details)[open]")
        m.screenshot(shots / "desktop-evidence.png")
        m.js("document.querySelector('.story details.why:not(.coverage-details)').open = false;"
             "window.scrollTo(0, 0);")

    # --------------------------------------------------------------- search
    if not skip_search:
        print("\n[search]")
        report.check("composer is full size before a conversation",
                     not m.js("return document.body.classList.contains('has-chat')"))
        report.check("idle state shows the ask heading",
                     m.js("return document.querySelector('.ask-heading')"
                          ".getBoundingClientRect().height") > 20)
        report.check("idle state offers example queries",
                     m.js("return document.querySelectorAll('#suggestion-chips .chip').length") > 0)
        idle_h = m.js("return document.querySelector('.composer').getBoundingClientRect().height")
        # Regression guard: a component that sets `display` can out-specify the
        # user-agent [hidden] rule and stay on screen.
        report.check("New search hidden until a conversation exists",
                     m.js("return getComputedStyle("
                          "document.querySelector('#clear-chat')).display === 'none'"))
        m.js("""
          const i = document.querySelector('#ask-input');
          i.value = "What's happening with NVIDIA today?";
          document.querySelector('#ask-form').requestSubmit();
        """)
        answered = m.wait("document.querySelectorAll('.source-card').length > 0", 600)
        report.check("search returned sources", answered)
        if answered:
            report.check("composer compacts once a conversation exists",
                         m.js("return document.body.classList.contains('has-chat')"))
            busy_h = m.js("return document.querySelector('.composer')"
                          ".getBoundingClientRect().height")
            report.check("composer physically shrinks in research mode",
                         busy_h < idle_h, f"{idle_h:.0f}px → {busy_h:.0f}px")
            report.check("example queries step aside in research mode",
                         m.js("return getComputedStyle(document.querySelector("
                              "'.try-row')).display === 'none'"))
            report.check("New search action is offered",
                         not m.js("return document.querySelector('#clear-chat').hidden"))
            report.check("result is labelled as a search result",
                         "search result" in m.js(
                             "return document.querySelector('.turn-head').textContent").lower())
            report.check("the question is restated above the answer",
                         m.js("return document.querySelector('.turn-q')"
                              ".textContent.trim().length") > 0)
            report.check("answer carries inline citations",
                         m.js("return document.querySelectorAll('.answer .cite').length") >= 0)
            report.check("grounding badge shown",
                         m.js("return !!document.querySelector('.answer-foot .badge')"))
            report.check("source cards keep publisher links",
                         m.js("return !!document.querySelector('.source-card .source-title a')"))
            report.check("source cards are numbered evidence",
                         m.js("return document.querySelector('.source-num')"
                              ".textContent.trim()") == "01")
            report.check("source cards stay compact",
                         m.js("return document.querySelector('.source-card')"
                              ".getBoundingClientRect().height") < 260)
            report.check("two columns of source cards on wide desktop",
                         columns_of(m, ".source-list-grid") == 2,
                         f"got {columns_of(m, '.source-list-grid')}")
            report.check("answer prose stays within a readable measure",
                         m.js("return document.querySelector('.answer').clientWidth") < 900)
            report.check("no horizontal overflow after search", no_overflow(m))
            m.screenshot(shots / "desktop-search.png")
            m.screenshot(shots / "desktop-search-full.png", full=True)

    # ------------------------------------------------------------- settings
    print("\n[settings]")
    m.js("document.querySelector('#settings-btn').click()")
    report.check("settings opens", m.wait("document.querySelectorAll('#settings-sources .source-row').length > 0"))
    total = m.js("return document.querySelectorAll('#settings-sources .source-row').length")
    report.check("interests editor preserved", m.js("return !!document.querySelector('#settings-interests')"))
    report.check("save action preserved", m.js("return !!document.querySelector('#settings-save')"))
    report.check("reset action preserved", m.js("return !!document.querySelector('#settings-reset')"))
    report.check("system status preserved",
                 m.js("return document.querySelector('#settings-status').textContent.length") > 20)
    report.check("model info lives in settings",
                 "model" in m.js("return document.querySelector('#settings-status').textContent").lower())

    m.js("const s=document.querySelector('#source-search'); s.value='guardian';"
         "s.dispatchEvent(new Event('input'));")
    time.sleep(0.4)
    filtered = m.js("return document.querySelectorAll('#settings-sources .source-row').length")
    report.check("source search filters the list", 0 < filtered < total, f"{filtered} of {total}")

    m.js("const s=document.querySelector('#source-search'); s.value='';"
         "s.dispatchEvent(new Event('input'));"
         "document.querySelector('#settings .seg button[data-filter=\"disabled\"]').click();")
    time.sleep(0.4)
    disabled = m.js("return document.querySelectorAll('#settings-sources .source-row').length")
    report.check("state filter narrows the list", 0 <= disabled < total, f"{disabled} disabled")
    report.check("settings groups its sections",
                 m.js("return document.querySelectorAll('#settings .block-title').length") >= 3)
    m.screenshot(shots / "desktop-settings.png")
    m.js("document.querySelector('#settings .seg button[data-filter=\"all\"]').click();"
         "document.querySelector('#settings-close').click()")

    # ---------------------------------------------------- onboarding preview
    # Shown without touching stored preferences: the overlay is re-opened
    # locally, screenshotted, then dismissed. Nothing is written server-side.
    print("\n[onboarding]")
    m.js("showOnboarding()")
    time.sleep(0.5)
    report.check("onboarding states the value proposition",
                 m.js("return document.querySelector('.onboard-title')"
                      ".textContent.trim().length") > 10)
    report.check("onboarding offers starter interests",
                 m.js("return document.querySelectorAll('#onboard-chips .chip').length") >= 5)
    report.check("onboarding keeps its own keyboard hint",
                 m.js("return getComputedStyle(document.querySelector("
                      "'#onboarding .kbd-hint')).display") != "none")
    report.check("onboarding is a single step",
                 m.js("return document.querySelectorAll("
                      "'#onboarding button[type=\"submit\"]').length") == 1)
    report.check("no horizontal overflow (onboarding)", no_overflow(m))
    m.screenshot(shots / "onboarding-desktop.png")
    m.size(390, 844)
    time.sleep(0.6)
    report.check("onboarding fits a phone", no_overflow(m))
    m.screenshot(shots / "onboarding-mobile.png", full=True)
    m.js("document.querySelector('#onboarding').hidden = true;")

    # ------------------------------------------------------- viewport sweep
    # The laptop sizes matter most: 1366x768 and 1440x900 are where this app
    # actually gets read.
    print("\n[viewport sweep]")
    m.js("window.scrollTo(0, 0);")
    # (label, w, h, expected briefing columns, min share of the viewport the
    #  container must occupy, screenshot). The share is the acceptance test for
    #  desktop space: a laptop must not render into a narrow centre column.
    sweep = [
        ("1920x1080", 1920, 1080, 2, 0.72, "desktop-1920.png"),
        ("1440x900", 1440, 900, 2, 0.90, "desktop-1440.png"),
        ("1366x768", 1366, 768, 2, 0.90, "laptop-1366.png"),
        ("1280x800", 1280, 800, 2, 0.90, "laptop-1280.png"),
        ("1100x800", 1100, 800, 2, 0.90, "laptop-1100.png"),
        ("1024x768", 1024, 768, 1, 0.85, "tablet-1024.png"),
        ("820x1180", 820, 1180, 1, 0.85, "tablet-820.png"),
    ]
    for label, width, height, expect_cols, min_share, shot in sweep:
        m.size(width, height)
        time.sleep(0.8)
        cols = columns_of(m, ".brief-grid")
        report.check(f"{label}: {expect_cols}-column briefing", cols == expect_cols, f"got {cols}")
        report.check(f"{label}: no horizontal overflow", no_overflow(m))
        content = m.js("return document.querySelector('main.wrap').clientWidth")
        share = content / width
        report.check(f"{label}: container uses the window ({share:.0%})",
                     share >= min_share and content <= 1440,
                     f"{content}px of {width}px")
        # One container system: header, main and footer must agree.
        widths = m.js(
            "return [...document.querySelectorAll('.wrap')]"
            ".map(w => Math.round(w.getBoundingClientRect().width))"
        )
        report.check(f"{label}: header/main/footer share one container",
                     len(set(widths)) == 1, f"widths {widths}")
        # And the composer must take that container, not a column inside it.
        composer = m.js("return document.querySelector('.composer').getBoundingClientRect().width")
        report.check(f"{label}: search field spans the container",
                     composer >= content - 2, f"{composer:.0f}px of {content}px")
        # A tab strip that overflows must say so; one that fits must not fade.
        overflows, faded = m.js(
            "const n = document.querySelector('#topic-nav');"
            "return [n.scrollWidth - n.clientWidth - n.scrollLeft > 4,"
            "        n.classList.contains('is-scrollable')];"
        )
        report.check(f"{label}: topic strip signals overflow only when it overflows",
                     overflows == faded, f"overflows={overflows} faded={faded}")
        m.screenshot(shots / shot)

    # --------------------------------------------------------------- tablet
    print("\n[tablet 900x1000]")
    m.size(900, 1000)
    time.sleep(1.0)
    report.check("single column on tablet", columns_of(m, ".brief-grid") == 1,
                 f"got {columns_of(m, '.brief-grid')}")
    report.check("no horizontal overflow (tablet)", no_overflow(m))
    m.screenshot(shots / "tablet-briefing.png", full=True)

    # --------------------------------------------------------------- mobile
    print("\n[mobile 390x844]")
    m.size(390, 844)
    time.sleep(1.0)
    report.check("single column on mobile", columns_of(m, ".brief-grid") == 1,
                 f"got {columns_of(m, '.brief-grid')}")
    report.check("no horizontal overflow (mobile)", no_overflow(m),
                 f"scrollWidth={m.js('return document.documentElement.scrollWidth')}")
    report.check("topic navigation scrolls rather than wraps on mobile",
                 m.js("return getComputedStyle(document.querySelector("
                      "'#topic-nav')).overflowX") in {"auto", "scroll"})
    report.check("send button is full width on mobile",
                 m.js("return document.querySelector('.composer-send')"
                      ".getBoundingClientRect().width") > 200)
    report.check("headlines readable on mobile",
                 m.js("return parseFloat(getComputedStyle("
                      "document.querySelector('.story-title')).fontSize)") >= 14)

    if m.js("return !!document.querySelector('.coverage-details')"):
        m.js("document.querySelector('.coverage-details').open = true;")
        time.sleep(0.4)
        report.check("coverage usable on mobile without overflow",
                     m.js("return document.documentElement.scrollWidth <= window.innerWidth + 1"))
        m.focus_on(".coverage-details[open]")
        m.screenshot(shots / "mobile-coverage.png")
        m.js("document.querySelector('.coverage-details').open = false;"
             "window.scrollTo(0, 0);")

    m.screenshot(shots / "mobile-briefing.png", full=True)

    m.js("document.querySelector('#settings-btn').click()")
    time.sleep(1.0)
    report.check("settings usable on mobile",
                 m.js("return document.documentElement.scrollWidth <= window.innerWidth + 1"))
    m.screenshot(shots / "mobile-settings.png")
    m.js("document.querySelector('#settings-close').click()")

    # ----------------------------------------------------------- light theme
    # The theme toggle is existing functionality; the redesign has to carry it.
    print("\n[light theme]")
    m.size(1440, 1000)
    time.sleep(0.6)
    m.js("document.querySelector('#theme-btn').click()")
    time.sleep(0.6)
    report.check("theme toggle switches the document theme",
                 m.js("return document.documentElement.dataset.theme") == "light")
    report.check("light theme repaints the page background",
                 m.js("return getComputedStyle(document.body).backgroundColor")
                 not in {"rgb(11, 12, 17)", "rgba(0, 0, 0, 0)"})
    report.check("light theme keeps text legible",
                 m.js("return getComputedStyle(document.querySelector('.story-title a'))"
                      ".color") != "rgb(223, 228, 242)")
    report.check("no horizontal overflow (light theme)", no_overflow(m))
    m.screenshot(shots / "desktop-light.png")
    m.js("document.querySelector('#theme-btn').click()")
    time.sleep(0.5)
    report.check("theme toggle returns to dark",
                 m.js("return document.documentElement.dataset.theme") == "dark")

    # ---------------------------------------------------------- design system
    print("\n[design system]")
    tokens = m.js("""
      const s = getComputedStyle(document.documentElement);
      return ['--bg','--surface','--surface-elevated','--surface-hover','--border',
              '--border-subtle','--text','--text-secondary','--text-muted','--accent',
              '--accent-soft','--success','--warning','--danger','--fs-base','--s4',
              '--r-md','--measure','--content-max']
        .filter(name => !s.getPropertyValue(name).trim());
    """)
    report.check("the documented design tokens all resolve", tokens == [], f"missing {tokens}")

    # ------------------------------------------------------ keyboard control
    print("\n[keyboard]")
    m.size(1440, 1000)
    time.sleep(0.8)
    m.js("document.querySelector('#ask-input').blur();"
         "document.dispatchEvent(new KeyboardEvent('keydown', {key: '/', bubbles: true}));")
    time.sleep(0.3)
    report.check("'/' focuses the composer",
                 m.js("return document.activeElement.id") == "ask-input")
    m.js("document.querySelector('#settings-btn').click()")
    time.sleep(0.5)
    m.js("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));")
    time.sleep(0.3)
    report.check("Escape closes settings", m.js("return document.querySelector('#settings').hidden"))
    report.check("topic tabs are keyboard reachable",
                 m.js("return document.querySelector('#topic-nav .topic-tab').tabIndex") >= 0)
    report.check("disclosure summaries are keyboard reachable",
                 m.js("return document.querySelector('details.why > summary').tabIndex") >= 0)

    # -------------------------------------------------------- accessibility
    print("\n[accessibility]")
    report.check("disclosures are native <details>/<summary>",
                 m.js("return document.querySelectorAll('details > summary').length") > 0)
    report.check("briefing sections use headings",
                 m.js("return document.querySelectorAll('.brief-section h3').length") > 0)
    report.check("stories use headings",
                 m.js("return document.querySelectorAll('.story h4').length") > 0)
    report.check("buttons are real buttons",
                 m.js("return document.querySelectorAll('button').length") > 3)
    report.check("status region is announced",
                 m.js("return document.querySelector('#topbar-status')"
                      ".getAttribute('aria-live') === 'polite'"))
    # The onboarding overlay owns its own <h1> when it is up; only one may ever
    # be rendered at a time.
    report.check("exactly one top-level heading is rendered",
                 m.js("return [...document.querySelectorAll('h1')]"
                      ".filter(h => h.getClientRects().length > 0).length") == 1)
    report.check("icon-only controls carry labels",
                 m.js("return [...document.querySelectorAll('#theme-btn, #settings-btn,"
                      " #settings-close, #ask-submit')]"
                      ".every(b => b.getAttribute('aria-label'))"))
    report.check("reduced motion is respected",
                 m.js("return [...document.styleSheets].some(s => {"
                      "try { return [...s.cssRules].some(r =>"
                      "  r.conditionText && r.conditionText.includes('prefers-reduced-motion')); }"
                      "catch(e) { return false; } })"))
    report.check("focus styles defined",
                 m.js("return [...document.styleSheets].some(s => {"
                      "try { return [...s.cssRules].some(r => "
                      "r.selectorText && r.selectorText.includes(':focus-visible')); }"
                      "catch(e){ return false; } })"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--shots", default=None, help="directory for screenshots")
    parser.add_argument("--skip-search", action="store_true",
                        help="skip the search checks (no model needed, much faster)")
    parser.add_argument("--port", type=int, default=MARIONETTE_PORT,
                        help="Marionette port for the throwaway browser instance")
    args = parser.parse_args()

    # Line-buffer so progress is visible when piped to a file or a log.
    sys.stdout.reconfigure(line_buffering=True)

    shots = Path(args.shots) if args.shots else Path(tempfile.mkdtemp(prefix="daily-brief-ui-"))
    profile = Path(tempfile.mkdtemp(prefix="daily-brief-ff-"))
    print(f"screenshots → {shots}")

    firefox = start_firefox(profile, args.port)
    report = Report()
    try:
        m = connect(args.port)
        verify(m, args.base_url.rstrip("/"), shots, report, args.skip_search)
    finally:
        firefox.terminate()
        try:
            firefox.wait(timeout=15)
        except subprocess.TimeoutExpired:
            firefox.kill()
        shutil.rmtree(profile, ignore_errors=True)
    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
