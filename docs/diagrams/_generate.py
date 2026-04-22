"""Generate excalidraw diagrams for the deep-research blog post.

Run:  python docs/diagrams/_generate.py

Emits three .excalidraw files in docs/diagrams/:
  - pipeline.excalidraw       — pipeline + durability (one picture)
  - architecture.excalidraw   — three-layer flow / checkpoints / agents
  - critique.excalidraw       — cross-provider generator / reviewer / judge
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).parent
FONT_SANS = 5   # Excalifont
FONT_MONO = 3   # Cascadia

_counter = itertools.count(1)


def _id() -> str:
    return f"el{next(_counter)}"


def _seed() -> int:
    n = next(_counter)
    return 100000 + n * 7919


# --- Primitives ------------------------------------------------------------

def rect(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    stroke: str = "#1e1e1e",
    bg: str = "transparent",
    rounded: bool = True,
    stroke_width: int = 2,
    stroke_style: str = "solid",
    roughness: int = 0,
) -> dict[str, Any]:
    return {
        "id": _id(),
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": bg,
        "fillStyle": "solid",
        "strokeWidth": stroke_width,
        "strokeStyle": stroke_style,
        "roughness": roughness,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3} if rounded else None,
        "seed": _seed(),
        "versionNonce": _seed(),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }


def diamond(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    stroke: str = "#1e1e1e",
    bg: str = "transparent",
    roughness: int = 0,
) -> dict[str, Any]:
    return {
        "id": _id(),
        "type": "diamond",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": bg,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": roughness,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 2},
        "seed": _seed(),
        "versionNonce": _seed(),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }


def ellipse(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    stroke: str = "#1e1e1e",
    bg: str = "transparent",
    roughness: int = 0,
) -> dict[str, Any]:
    return {
        "id": _id(),
        "type": "ellipse",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": bg,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": roughness,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": _seed(),
        "versionNonce": _seed(),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }


def text(
    x: int,
    y: int,
    content: str,
    *,
    size: int = 16,
    mono: bool = False,
    align: str = "left",
    color: str = "#1e1e1e",
    width: int | None = None,
) -> dict[str, Any]:
    lines = content.split("\n")
    char_w = size * 0.55 if mono else size * 0.58
    est_width = max(int(max(len(l) for l in lines) * char_w), 40)
    est_height = int(size * 1.25 * len(lines))
    return {
        "id": _id(),
        "type": "text",
        "x": x,
        "y": y,
        "width": width if width is not None else est_width,
        "height": est_height,
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": _seed(),
        "versionNonce": _seed(),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
        "text": content,
        "fontSize": size,
        "fontFamily": FONT_MONO if mono else FONT_SANS,
        "textAlign": align,
        "verticalAlign": "top",
        "containerId": None,
        "originalText": content,
        "lineHeight": 1.25,
        "autoResize": True,
    }


def arrow(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    stroke: str = "#1e1e1e",
    dashed: bool = False,
    head: str = "arrow",
    stroke_width: int = 2,
    roughness: int = 0,
) -> dict[str, Any]:
    dx = x2 - x1
    dy = y2 - y1
    return {
        "id": _id(),
        "type": "arrow",
        "x": x1,
        "y": y1,
        "width": max(abs(dx), 1),
        "height": max(abs(dy), 1),
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": stroke_width,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": roughness,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 2},
        "seed": _seed(),
        "versionNonce": _seed(),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
        "points": [[0, 0], [dx, dy]],
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": head,
        "elbowed": False,
    }


# --- High-level helpers ----------------------------------------------------

def labeled_box(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    subtitle: str | None = None,
    *,
    stroke: str = "#1e1e1e",
    bg: str = "transparent",
    title_size: int = 16,
    subtitle_size: int = 12,
    subtitle_mono: bool = True,
    subtitle_color: str | None = None,
) -> list[dict[str, Any]]:
    els = [rect(x, y, w, h, stroke=stroke, bg=bg)]
    if subtitle is None:
        els.append(text(x + 14, y + (h // 2) - (title_size // 2), title, size=title_size, color=stroke, align="left"))
    else:
        els.append(text(x + 14, y + 10, title, size=title_size, color=stroke, align="left"))
        els.append(
            text(
                x + 14,
                y + 10 + int(title_size * 1.4),
                subtitle,
                size=subtitle_size,
                mono=subtitle_mono,
                color=subtitle_color or "#6b7280",
                align="left",
            )
        )
    return els


def checkpoint_badge(cx: int, cy: int, *, color: str = "#2f9e44") -> list[dict[str, Any]]:
    """Small circle with ✓ — marks a phase as a replay boundary."""
    size = 22
    return [
        ellipse(cx - size // 2, cy - size // 2, size, size, stroke=color, bg="#ebfbee"),
        text(cx - 5, cy - 8, "✓", size=14, color=color, align="center"),
    ]


def wait_badge(cx: int, cy: int, *, color: str = "#7048e8") -> list[dict[str, Any]]:
    """Small diamond marker for wait points."""
    size = 22
    return [
        diamond(cx - size // 2, cy - size // 2, size, size, stroke=color, bg="#f3f0ff"),
    ]


def save(filename: str, elements: list[dict[str, Any]]) -> None:
    payload = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#ffffff",
        },
        "files": {},
    }
    path = OUT_DIR / filename
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path.relative_to(OUT_DIR.parent.parent)}  ({len(elements)} elements)")


# --- Palette ---------------------------------------------------------------

INK = "#1e1e1e"
MUTED = "#6b7280"
SUBTLE = "#adb5bd"

BLUE = "#1971c2"
GREEN = "#2f9e44"
VIOLET = "#7048e8"
ORANGE = "#e8590c"
RED = "#c92a2a"
TEAL = "#099268"

BG_BLUE = "#e7f5ff"
BG_GREEN = "#ebfbee"
BG_VIOLET = "#f3f0ff"
BG_ORANGE = "#fff4e6"
BG_YELLOW = "#fff9db"
BG_GRAY = "#f8f9fa"
BG_RED = "#ffe3e3"
BG_TEAL = "#e6fcf5"


# =========================================================================
# Diagram 1: Pipeline + Durability (combined)
# =========================================================================

def build_pipeline() -> list[dict[str, Any]]:
    els: list[dict[str, Any]] = []

    # --- Title block ---
    els.append(text(40, 24, "Research pipeline with durability boundaries", size=28, color=INK))
    els.append(
        text(
            40,
            64,
            "every ✓ is a @checkpoint that caches its typed output · ◆ is a flow-body wait for operator input",
            size=14,
            color=MUTED,
        )
    )

    # --- Main pipeline row ---
    top_y = 130
    phase_h = 80
    phase_w = 150
    gap = 42

    x = 40
    # Entry
    els.append(ellipse(x, top_y + 10, 90, 60, stroke=INK, bg=BG_GRAY))
    els.append(text(x + 20, top_y + 32, "question", size=14, color=INK))
    x += 90 + gap

    # Helper: phase with checkpoint badge
    def phase(px: int, py: int, title: str, kind: str, *, stroke=BLUE, bg=BG_BLUE):
        nodes = []
        nodes += labeled_box(px, py, phase_w, phase_h, title, kind, stroke=stroke, bg=bg, subtitle_size=11)
        # checkpoint badge at top-right
        nodes += checkpoint_badge(px + phase_w - 14, py - 2, color=GREEN)
        return nodes

    els += phase(x, top_y, "Scope", "→ ResearchBrief", stroke=BLUE, bg=BG_BLUE)
    scope_x = x
    x += phase_w + gap

    els += phase(x, top_y, "Plan", "→ ResearchPlan", stroke=BLUE, bg=BG_BLUE)
    plan_x = x
    x += phase_w + gap

    # Wait: plan approval
    wait_w = 130
    els.append(diamond(x, top_y + 5, wait_w, phase_h - 10, stroke=VIOLET, bg=BG_VIOLET))
    els.append(text(x + 18, top_y + 22, "wait", size=14, color=VIOLET))
    els.append(text(x + 14, top_y + 42, "plan approval", size=11, color=VIOLET, mono=True))
    wait_x = x
    x += wait_w + gap

    # --- Iteration loop container ---
    loop_x = x
    loop_y = top_y - 40
    loop_w = 460
    loop_h = 210

    els.append(rect(loop_x, loop_y, loop_w, loop_h, stroke=TEAL, bg=BG_TEAL, stroke_style="dashed"))
    els.append(text(loop_x + 16, loop_y + 10, "Iteration loop", size=16, color=TEAL))
    els.append(
        text(
            loop_x + 16,
            loop_y + 32,
            "until budget · time · supervisor done · max iter",
            size=11,
            color=MUTED,
        )
    )

    # Inside loop: supervisor, fan-out, convergence
    inner_y = loop_y + 70
    sup_w = 130
    sub_w = 130

    els += labeled_box(loop_x + 18, inner_y, sup_w, 90, "Supervisor", "→ SupervisorDecision", stroke=TEAL, bg="#ffffff", subtitle_size=10)
    els += checkpoint_badge(loop_x + 18 + sup_w - 14, inner_y - 2, color=GREEN)

    # Subagents — three stacked cards to suggest fan-out
    fx = loop_x + 18 + sup_w + 30
    for i in range(3):
        els.append(rect(fx + (2 - i) * 6, inner_y + (2 - i) * 6, sub_w, 90, stroke=TEAL, bg="#ffffff"))
    els.append(text(fx + 8, inner_y + 12, "Subagents × N", size=14, color=TEAL))
    els.append(text(fx + 8, inner_y + 36, "search + fetch", size=11, color=MUTED))
    els.append(text(fx + 8, inner_y + 56, "→ SubagentFindings", size=10, color=INK, mono=True))
    els += checkpoint_badge(fx + sub_w - 14, inner_y - 2, color=GREEN)

    # Convergence diamond
    dx = fx + sub_w + 30
    els.append(diamond(dx, inner_y + 10, 120, 70, stroke=TEAL, bg="#ffffff"))
    els.append(text(dx + 18, inner_y + 32, "converge?", size=13, color=TEAL))

    # Inside-loop arrows
    els.append(arrow(loop_x + 18 + sup_w, inner_y + 45, fx, inner_y + 45, stroke=TEAL))
    els.append(arrow(fx + sub_w + 6, inner_y + 45, dx, inner_y + 45, stroke=TEAL))
    # Loop back: no → back to supervisor
    els.append(arrow(dx + 60, inner_y + 80, dx + 60, inner_y + 150, stroke=TEAL))
    els.append(arrow(dx + 60, inner_y + 150, loop_x + 18 + sup_w // 2, inner_y + 150, stroke=TEAL))
    els.append(arrow(loop_x + 18 + sup_w // 2, inner_y + 150, loop_x + 18 + sup_w // 2, inner_y + 90, stroke=TEAL))
    els.append(text(dx - 90, inner_y + 155, "no — iterate", size=11, color=TEAL))

    loop_right = loop_x + loop_w

    # --- Arrows between top-row phases ---
    arrow_y = top_y + phase_h // 2
    els.append(arrow(40 + 90, arrow_y, 40 + 90 + gap, arrow_y))
    els.append(arrow(scope_x + phase_w, arrow_y, scope_x + phase_w + gap, arrow_y))
    els.append(arrow(plan_x + phase_w, arrow_y, plan_x + phase_w + gap, arrow_y))
    els.append(arrow(wait_x + wait_w, arrow_y, wait_x + wait_w + gap, arrow_y))

    # --- Bottom row: draft / critique / finalize / assemble / export ---
    bot_y = top_y + 260
    bx = 40

    def simple_phase(px: int, py: int, title: str, kind: str, *, stroke, bg):
        nodes = labeled_box(px, py, phase_w, phase_h, title, kind, stroke=stroke, bg=bg, subtitle_size=11)
        nodes += checkpoint_badge(px + phase_w - 14, py - 2, color=GREEN)
        return nodes

    els += simple_phase(bx, bot_y, "Draft", "→ DraftReport", stroke=BLUE, bg=BG_BLUE)
    bx += phase_w + gap
    els += simple_phase(bx, bot_y, "Critique", "→ CritiqueReport", stroke=VIOLET, bg=BG_VIOLET)
    critique_x = bx
    bx += phase_w + gap
    els += simple_phase(bx, bot_y, "Finalize", "→ FinalReport", stroke=BLUE, bg=BG_BLUE)
    bx += phase_w + gap
    els += simple_phase(bx, bot_y, "Assemble", "→ Package", stroke=ORANGE, bg=BG_ORANGE)
    bx += phase_w + gap
    els += simple_phase(bx, bot_y, "Export", "filesystem", stroke=ORANGE, bg=BG_ORANGE)

    # Bottom-row connector arrows
    bot_arrow_y = bot_y + phase_h // 2
    for i in range(4):
        bx = 40 + (i + 1) * phase_w + i * gap
        els.append(arrow(bx, bot_arrow_y, bx + gap, bot_arrow_y))

    # Loop → Draft (curved via corner)
    els.append(arrow(loop_x + loop_w // 2, loop_y + loop_h, loop_x + loop_w // 2, bot_y - 30, stroke=TEAL))
    els.append(arrow(loop_x + loop_w // 2, bot_y - 30, 40 + phase_w // 2, bot_y - 30, stroke=TEAL))
    els.append(arrow(40 + phase_w // 2, bot_y - 30, 40 + phase_w // 2, bot_y, stroke=TEAL))
    els.append(text(loop_x + loop_w // 2 - 200, bot_y - 50, "converged — leave loop", size=12, color=TEAL))

    # Supplemental loop: critique → loop
    cx = critique_x + phase_w // 2
    els.append(arrow(cx, bot_y, cx, bot_y - 60, dashed=True, stroke=VIOLET))
    els.append(arrow(cx, bot_y - 60, loop_x + loop_w - 40, bot_y - 60, dashed=True, stroke=VIOLET))
    els.append(arrow(loop_x + loop_w - 40, bot_y - 60, loop_x + loop_w - 40, loop_y + loop_h, dashed=True, stroke=VIOLET))
    els.append(text(cx + 16, bot_y - 80, "supplemental loop — reviewer asks for more", size=11, color=VIOLET))

    # =====================================================================
    # DURABILITY TIMELINE (bottom panel)
    # =====================================================================
    tl_y = bot_y + phase_h + 80
    panel_w = 1200

    els.append(rect(40, tl_y, panel_w, 280, stroke=SUBTLE, bg=BG_GRAY, stroke_style="dashed"))
    els.append(text(56, tl_y + 12, "What survives a crash", size=20, color=INK))
    els.append(
        text(
            56,
            tl_y + 42,
            "Completed @checkpoint results are cached. On replay, only the phase that was in flight re-executes.",
            size=13,
            color=MUTED,
        )
    )

    # Row 1: initial run
    row1_y = tl_y + 80
    els.append(text(56, row1_y, "initial run", size=14, color=BLUE, mono=True))
    # Sequence of blocks
    blocks = [
        ("scope", True),
        ("plan", True),
        ("wait", True),
        ("sup.", True),
        ("sub.", True),
        ("sup.", True),
        ("sub.", False),  # crashes here
    ]
    bx = 180
    by = row1_y - 6
    block_w = 90
    block_h = 36
    for label, done in blocks:
        color = GREEN if done else RED
        bgc = BG_GREEN if done else BG_RED
        els.append(rect(bx, by, block_w, block_h, stroke=color, bg=bgc))
        els.append(text(bx + 10, by + 10, label, size=12, color=INK, mono=True))
        bx += block_w + 8
    els.append(text(bx + 4, by + 8, "⚡ crash", size=14, color=RED))

    # Row 2: replay
    row2_y = row1_y + 80
    els.append(text(56, row2_y, "replay", size=14, color=GREEN, mono=True))
    bx = 180
    by = row2_y - 6
    replay_blocks = [
        ("scope", "cached"),
        ("plan", "cached"),
        ("wait", "cached"),
        ("sup.", "cached"),
        ("sub.", "cached"),
        ("sup.", "cached"),
        ("sub.", "re-run"),
        ("draft", "new"),
        ("crit.", "new"),
    ]
    for label, state in replay_blocks:
        if state == "cached":
            stroke, bgc = GREEN, "#d3f9d8"
        elif state == "re-run":
            stroke, bgc = ORANGE, BG_ORANGE
        else:
            stroke, bgc = BLUE, BG_BLUE
        els.append(rect(bx, by, block_w, block_h, stroke=stroke, bg=bgc))
        els.append(text(bx + 10, by + 10, label, size=12, color=INK, mono=True))
        bx += block_w + 8

    # Legend for replay states
    leg_y = tl_y + 240
    lx = 56
    legend_items = [
        ("cached", GREEN, "#d3f9d8", "completed — returned from Kitaru cache"),
        ("re-run", ORANGE, BG_ORANGE, "was in flight at crash — re-executed"),
        ("new", BLUE, BG_BLUE, "never ran before — executed normally"),
    ]
    for label, stroke, bgc, desc in legend_items:
        els.append(rect(lx, leg_y, 60, 24, stroke=stroke, bg=bgc))
        els.append(text(lx + 12, leg_y + 5, label, size=12, color=INK, mono=True))
        els.append(text(lx + 72, leg_y + 5, desc, size=12, color=MUTED))
        lx += 380

    return els


# =========================================================================
# Diagram 2: Three-layer architecture
# =========================================================================

def build_architecture() -> list[dict[str, Any]]:
    els: list[dict[str, Any]] = []

    els.append(text(40, 24, "Three layers · one concern each", size=28, color=INK))
    els.append(
        text(
            40,
            64,
            "Flow orchestrates · Checkpoints persist · Agents transform. Each layer is replaceable.",
            size=14,
            color=MUTED,
        )
    )

    # Dimensions
    layer_x = 40
    layer_w = 1080
    layer_h = 200
    owns_w = 480
    doesnt_w = 360

    def layer(
        y: int,
        color: str,
        bg: str,
        name: str,
        path: str,
        decorator: str,
        owns: list[str],
        doesnt: list[str],
    ) -> list[dict[str, Any]]:
        nodes = []
        nodes.append(rect(layer_x, y, layer_w, layer_h, stroke=color, bg=bg))

        # Left header block
        nodes.append(text(layer_x + 20, y + 14, name, size=22, color=color))
        nodes.append(text(layer_x + 20, y + 44, path, size=12, color=MUTED, mono=True))
        nodes.append(text(layer_x + 20, y + 62, decorator, size=12, color=MUTED, mono=True))

        # Owns column
        ox = layer_x + 260
        nodes.append(text(ox, y + 20, "owns", size=13, color=color))
        for i, item in enumerate(owns):
            nodes.append(text(ox + 4, y + 48 + i * 26, f"•  {item}", size=13, color=INK))

        # Does-not column
        dx = layer_x + 260 + owns_w + 40
        nodes.append(text(dx, y + 20, "does NOT", size=13, color=MUTED))
        for i, item in enumerate(doesnt):
            nodes.append(text(dx + 4, y + 48 + i * 26, f"–  {item}", size=13, color=MUTED))
        return nodes

    y1 = 110
    els += layer(
        y=y1,
        color=BLUE,
        bg=BG_BLUE,
        name="Flow",
        path="research/flows/deep_research.py",
        decorator="@flow  —  one function",
        owns=[
            "the iteration loop and convergence checks",
            "budget tracking and time limits",
            "flow-body waits for operator input",
            "phase sequence (scope → plan → iter → draft …)",
        ],
        doesnt=[
            "call LLMs directly",
            "parse model output by hand",
            "know agent prompts",
        ],
    )

    y2 = y1 + layer_h + 60
    els += layer(
        y=y2,
        color=GREEN,
        bg=BG_GREEN,
        name="Checkpoints",
        path="research/checkpoints/",
        decorator='@checkpoint(type="llm_call" | "tool_call")',
        owns=[
            "the replay boundary — cached typed results",
            "prompt construction and agent invocation",
            "per-phase error handling",
            "non-determinism (uuid, wall clock) isolated here",
        ],
        doesnt=[
            "drive iteration",
            "evaluate stop rules",
            "aggregate across iterations",
        ],
    )

    y3 = y2 + layer_h + 60
    els += layer(
        y=y3,
        color=VIOLET,
        bg=BG_VIOLET,
        name="Agents",
        path="research/agents/",
        decorator="PydanticAI Agent wrapped in KitaruAgent",
        owns=[
            "one model call with a typed output_type",
            "system prompt (loaded from research/prompts/*.md)",
            "tool bindings (search / fetch / code_exec)",
            "structured output validation (StrictBase)",
        ],
        doesnt=[
            "persist state across calls",
            "know the iteration number",
            "decide when to stop",
            "manage retries or replay",
        ],
    )

    # Vertical arrows (left side: call down, right side: typed result up)
    left_x = layer_x + 100
    right_x = layer_x + 900
    els.append(arrow(left_x, y1 + layer_h, left_x, y2, stroke=INK))
    els.append(text(left_x + 10, y1 + layer_h + 14, "submit()", size=12, color=INK, mono=True))
    els.append(arrow(right_x, y2, right_x, y1 + layer_h, stroke=INK))
    els.append(text(right_x - 100, y1 + layer_h + 14, "load() → typed", size=12, color=INK, mono=True))

    els.append(arrow(left_x, y2 + layer_h, left_x, y3, stroke=INK))
    els.append(text(left_x + 10, y2 + layer_h + 14, "run_sync(prompt)", size=12, color=INK, mono=True))
    els.append(arrow(right_x, y3, right_x, y2 + layer_h, stroke=INK))
    els.append(text(right_x - 130, y2 + layer_h + 14, ".output  (Pydantic)", size=12, color=INK, mono=True))

    # Annotation block — example trace through layers
    ann_x = 40
    ann_y = y3 + layer_h + 60
    ann_w = 1080
    ann_h = 140
    els.append(rect(ann_x, ann_y, ann_w, ann_h, stroke=SUBTLE, bg=BG_GRAY, stroke_style="dashed"))
    els.append(text(ann_x + 20, ann_y + 14, "example trace — one critique call", size=16, color=INK))

    lines = [
        ("Flow", BLUE,  "decides it's time to critique; calls run_critique.submit(draft, plan, ledger, model)"),
        ("Checkpoint", GREEN, "run_critique builds the JSON prompt, invokes build_reviewer_agent(model).run_sync(...)"),
        ("Agent", VIOLET, "PydanticAI + KitaruAgent parse the model reply into CritiqueReport, reject unknown fields"),
        ("↑ back up", MUTED, "CritiqueReport returned to flow — cached by Kitaru so replay skips the call entirely"),
    ]
    for i, (label, color, desc) in enumerate(lines):
        ly = ann_y + 46 + i * 22
        els.append(text(ann_x + 30, ly, label, size=13, color=color, mono=True))
        els.append(text(ann_x + 150, ly, desc, size=12, color=INK))

    return els


# =========================================================================
# Diagram 3: Cross-provider critique
# =========================================================================

def build_critique() -> list[dict[str, Any]]:
    els: list[dict[str, Any]] = []

    els.append(text(40, 24, "Cross-provider review — different models, different blind spots", size=28, color=INK))
    els.append(
        text(
            40,
            64,
            "Generator, reviewer(s), and judge run on different providers. Disagreement is surfaced, not averaged.",
            size=14,
            color=MUTED,
        )
    )

    # Provider colors (brand-ish)
    ANTHROPIC = "#d97706"   # warm amber
    OPENAI = "#059669"      # emerald
    GEMINI = "#2563eb"      # blue

    BG_ANTHROPIC = "#fef3c7"
    BG_OPENAI = "#d1fae5"
    BG_GEMINI = "#dbeafe"

    # --- Evidence ledger (hub, top) ---
    lx, ly = 500, 120
    lw, lh = 320, 96
    els.append(rect(lx, ly, lw, lh, stroke=ORANGE, bg=BG_ORANGE))
    els.append(text(lx + 20, ly + 14, "Evidence ledger", size=18, color=ORANGE))
    els.append(text(lx + 20, ly + 42, "DOI > arXiv ID > canonical URL · dedup", size=12, color=INK, mono=True))
    els.append(text(lx + 20, ly + 62, "append-only · windowed projection", size=12, color=INK, mono=True))

    # Subagents feeding ledger
    sx, sy = 80, 120
    els.append(rect(sx, sy, 260, 96, stroke=GEMINI, bg=BG_GEMINI))
    els.append(text(sx + 20, sy + 14, "Subagents × N", size=18, color=GEMINI))
    els.append(text(sx + 20, sy + 42, "gemini-3.1-flash-lite", size=12, color=INK, mono=True))
    els.append(text(sx + 20, sy + 62, "search + fetch → SubagentFindings", size=11, color=MUTED))
    els.append(arrow(sx + 260, sy + 48, lx, ly + 48, stroke=ORANGE, stroke_width=2))
    els.append(text((sx + 260 + lx) // 2 - 30, sy + 28, "findings", size=11, color=MUTED))

    # --- Generator (Anthropic) ---
    gx, gy = 80, 310
    gw, gh = 280, 130
    els.append(rect(gx, gy, gw, gh, stroke=ANTHROPIC, bg=BG_ANTHROPIC))
    els.append(text(gx + 20, gy + 14, "Generator", size=20, color=ANTHROPIC))
    els.append(text(gx + 20, gy + 44, "anthropic:claude-sonnet-4-6", size=12, color=INK, mono=True))
    els.append(text(gx + 20, gy + 72, "→ DraftReport", size=14, color=INK))
    els.append(text(gx + 20, gy + 100, "run_draft · @checkpoint(llm_call)", size=11, color=MUTED, mono=True))

    # Arrow: ledger → generator
    els.append(arrow(lx + 60, ly + lh, gx + 200, gy, stroke=ORANGE))

    # --- Reviewer A (OpenAI) ---
    rx1, ry1 = 500, 310
    rw, rh = 280, 130
    els.append(rect(rx1, ry1, rw, rh, stroke=OPENAI, bg=BG_OPENAI))
    els.append(text(rx1 + 20, ry1 + 14, "Reviewer", size=20, color=OPENAI))
    els.append(text(rx1 + 20, ry1 + 44, "openai:gpt-5.4-mini", size=12, color=INK, mono=True))
    els.append(text(rx1 + 20, ry1 + 72, "→ CritiqueReport", size=14, color=INK))
    els.append(text(rx1 + 20, ry1 + 100, "run_critique · @checkpoint(llm_call)", size=11, color=MUTED, mono=True))

    # Arrow: ledger → reviewer A
    els.append(arrow(lx + 260, ly + lh, rx1 + 100, ry1, stroke=ORANGE))
    # Arrow: generator → reviewer
    els.append(arrow(gx + gw, gy + gh // 2, rx1, ry1 + gh // 2, stroke=INK, stroke_width=2))
    els.append(text((gx + gw + rx1) // 2 - 40, gy + gh // 2 - 20, "DraftReport", size=11, color=INK, mono=True))

    # --- Reviewer B (Gemini, deep tier) ---
    rx2, ry2 = 500, 480
    els.append(rect(rx2, ry2, rw, rh, stroke=GEMINI, bg=BG_GEMINI, stroke_style="dashed"))
    els.append(text(rx2 + 20, ry2 + 14, "2nd Reviewer", size=20, color=GEMINI))
    els.append(text(rx2 + 20, ry2 + 44, "google-gla:gemini-3.1-pro-preview", size=12, color=INK, mono=True))
    els.append(text(rx2 + 20, ry2 + 72, "→ CritiqueReport", size=14, color=INK))
    els.append(text(rx2 + 20, ry2 + 100, "deep tier only · disagreement merged", size=11, color=MUTED))

    # Arrow: draft → 2nd reviewer
    els.append(arrow(gx + gw, gy + gh, rx2, ry2 + gh // 2, stroke=INK, dashed=True, stroke_width=2))
    els.append(arrow(lx + 260, ly + lh, rx2 + 100, ry2, stroke=ORANGE, dashed=True))

    # --- Merge block ---
    mx, my = 900, 390
    mw, mh = 240, 120
    els.append(rect(mx, my, mw, mh, stroke=INK, bg="#ffffff"))
    els.append(text(mx + 20, my + 14, "merge", size=18, color=INK))
    els.append(text(mx + 20, my + 44, "per-dimension", size=12, color=INK))
    els.append(text(mx + 20, my + 62, "disagreement > threshold", size=12, color=INK))
    els.append(text(mx + 20, my + 82, "→ reviewer_disagreements[]", size=11, color=MUTED, mono=True))
    els.append(text(mx + 20, my + 100, "(deep tier)", size=11, color=MUTED))

    els.append(arrow(rx1 + rw, ry1 + gh // 2, mx, my + mh // 2, stroke=INK))
    els.append(arrow(rx2 + rw, ry2 + gh // 2, mx, my + mh // 2, stroke=INK, dashed=True))

    # --- Rebuttal / supplemental loop ---
    els.append(arrow(rx1, ry1 + gh - 20, gx + gw, gy + gh - 20, stroke=RED, dashed=True, stroke_width=2))
    els.append(text(gx + gw + 18, gy + gh - 38, "require_more_research", size=11, color=RED, mono=True))
    els.append(text(gx + gw + 18, gy + gh - 22, "→ supplemental loop", size=11, color=RED))

    # --- Judge (council only) ---
    jx, jy = 900, 570
    jw, jh = 240, 120
    els.append(rect(jx, jy, jw, jh, stroke=VIOLET, bg=BG_VIOLET, stroke_style="dashed"))
    els.append(text(jx + 20, jy + 14, "Judge", size=20, color=VIOLET))
    els.append(text(jx + 20, jy + 44, "google-gla:gemini-3.1-pro-preview", size=12, color=INK, mono=True))
    els.append(text(jx + 20, jy + 72, "→ CouncilComparison", size=14, color=INK))
    els.append(text(jx + 20, jy + 98, "council mode only", size=11, color=MUTED))

    # --- Side panel: why this matters ---
    px, py = 1200, 120
    pw, ph = 280, 570
    els.append(rect(px, py, pw, ph, stroke=SUBTLE, bg=BG_GRAY, stroke_style="dashed"))
    els.append(text(px + 18, py + 14, "Why cross-provider?", size=18, color=INK))

    bullets = [
        ("Self-eval is biased.", "A model reviewing its own output agrees with itself more than with ground truth."),
        ("Shared blind spots.", "Same-provider models share training data, tokenizers, hallucination patterns."),
        ("Disagreement as signal.", "Surfaced per-dimension. Not averaged away."),
        ("Typed contracts.", "StrictBase extra=\"forbid\" rejects malformed critiques at the boundary."),
    ]
    for i, (head, body) in enumerate(bullets):
        by_ = py + 60 + i * 120
        els.append(text(px + 18, by_, head, size=13, color=INK))
        els.append(text(px + 18, by_ + 24, body, size=11, color=MUTED, width=250))

    # --- Bottom: tier behavior strip ---
    tsx, tsy = 40, 720
    els.append(text(tsx, tsy, "By tier", size=16, color=INK))
    rows = [
        ("quick / standard", "single reviewer"),
        ("deep", "dual reviewer · disagreement tracked"),
        ("exhaustive", "dual reviewer · 10 parallel subagents · breadth-biased supervisor"),
        ("council", "N parallel generators · judge compares · operator selects via wait()"),
    ]
    for i, (tier, desc) in enumerate(rows):
        els.append(text(tsx, tsy + 30 + i * 24, f"•  {tier}", size=13, color=INK, mono=True))
        els.append(text(tsx + 220, tsy + 30 + i * 24, desc, size=13, color=MUTED))

    return els


# --- Run -------------------------------------------------------------------

def main() -> None:
    global _counter
    _counter = itertools.count(1); save("pipeline.excalidraw", build_pipeline())
    _counter = itertools.count(1); save("architecture.excalidraw", build_architecture())
    _counter = itertools.count(1); save("critique.excalidraw", build_critique())

    # Clean up stale file from previous version
    stale = OUT_DIR / "durability.excalidraw"
    if stale.exists():
        stale.unlink()
        print(f"removed stale {stale.name}")


if __name__ == "__main__":
    main()
