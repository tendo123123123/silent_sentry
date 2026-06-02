#!/usr/bin/env python3

"""Generate architecture review diagrams as SVG and PNG.

The generator is designed for source-controlled, publication-style figures that
can be regenerated as the repository evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import textwrap
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT
CANVAS_W = 1880
CANVAS_H = 1240
MARGIN = 48
PANEL_H = 138
PANEL_Y = CANVAS_H - PANEL_H - 36


PALETTE = {
    "bg_top": "#f3f6fb",
    "bg_bottom": "#ffffff",
    "grid": "#d7dfeb",
    "title": "#1f2937",
    "subtitle": "#526072",
    "text": "#243042",
    "muted": "#64748b",
    "border": "#314258",
    "arrow": "#4b5d78",
    "shadow": "#8fa1b8",
    "topic": "#0f766e",
    "sim": "#dcecff",
    "control": "#fff0b8",
    "loc": "#dff5d7",
    "perception": "#f8dced",
    "tf": "#e2e9ff",
    "risk": "#ffe6dc",
    "math": "#e3f8ee",
    "sensor": "#def1ff",
    "panel": "#ffffff",
}


LEGEND_ITEMS: list[tuple[str, str]] = [
    ("Simulation / robot model", PALETTE["sim"]),
    ("Control / actuation", PALETTE["control"]),
    ("Localization / estimation", PALETTE["loc"]),
    ("Perception / planning", PALETTE["perception"]),
    ("TF / composed pose", PALETTE["tf"]),
    ("Diagnostics / critique", PALETTE["risk"]),
]


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    title: str
    body: str
    fill: str
    tag: str = ""


@dataclass
class Arrow:
    start: tuple[int, int]
    end: tuple[int, int]
    label: str = ""
    color: str = PALETTE["arrow"]
    dashed: bool = False


@dataclass
class Diagram:
    filename: str
    title: str
    subtitle: str
    badge: str
    summary: str
    boxes: list[Box] = field(default_factory=list)
    arrows: list[Arrow] = field(default_factory=list)
    footer: str = "Generated from current repo structure and active launch paths"


def load_font(size: int, bold: bool = False):
    candidates: list[str] = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT = load_font(40, bold=True)
SUBTITLE_FONT = load_font(20)
BOX_TITLE_FONT = load_font(22, bold=True)
BOX_BODY_FONT = load_font(15)
TAG_FONT = load_font(13, bold=True)
PANEL_TITLE_FONT = load_font(17, bold=True)
PANEL_BODY_FONT = load_font(14)
FOOTER_FONT = load_font(13)


def rgb_from_hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def rgba_tuple(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    red, green, blue = rgb_from_hex(value)
    return red, green, blue, alpha


def blend(color_a: str, color_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    rgb_a = rgb_from_hex(color_a)
    rgb_b = rgb_from_hex(color_b)
    mixed = tuple(int(rgb_a[index] + (rgb_b[index] - rgb_a[index]) * ratio) for index in range(3))
    return "#%02x%02x%02x" % mixed


def lighten(color: str, amount: float) -> str:
    return blend(color, "#ffffff", amount)


def darken(color: str, amount: float) -> str:
    return blend(color, "#0f172a", amount)


def estimate_text_width(text: str, font_size: int, scale: float = 0.62) -> int:
    return max(40, int(len(text) * font_size * scale))


def wrap_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if not line:
            lines.append("")
            continue
        if line.startswith("- "):
            bullet = line[2:].strip()
            wrapped = textwrap.wrap(
                bullet,
                width=max(10, width - 2),
                initial_indent="- ",
                subsequent_indent="  ",
                break_long_words=False,
            )
            lines.extend(wrapped if wrapped else ["- "])
            continue
        wrapped = textwrap.wrap(line, width=width, break_long_words=False)
        lines.extend(wrapped if wrapped else [line])
    return lines


def draw_background_png(image: Image.Image):
    draw = ImageDraw.Draw(image)
    top = rgb_from_hex(PALETTE["bg_top"])
    bottom = rgb_from_hex(PALETTE["bg_bottom"])
    for y in range(CANVAS_H):
        ratio = y / (CANVAS_H - 1)
        color = tuple(int(top[index] + (bottom[index] - top[index]) * ratio) for index in range(3)) + (255,)
        draw.line([(0, y), (CANVAS_W, y)], fill=color)

    dot_fill = rgba_tuple(PALETTE["grid"], 92)
    for x in range(MARGIN, CANVAS_W - MARGIN, 40):
        for y in range(126, PANEL_Y - 24, 40):
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=dot_fill)

    draw.ellipse([CANVAS_W - 560, 28, CANVAS_W - 140, 420], fill=rgba_tuple(lighten(PALETTE["tf"], 0.22), 116))
    draw.ellipse([90, 760, 430, 1100], fill=rgba_tuple(lighten(PALETTE["sim"], 0.18), 98))
    draw.rounded_rectangle([0, 0, CANVAS_W, 108], radius=0, fill=rgba_tuple("#ffffff", 138))


def draw_pill_png(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fill: str,
    text_color: str,
    outline: str,
    font: ImageFont.FreeTypeFont,
) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = (bbox[2] - bbox[0]) + 20
    height = (bbox[3] - bbox[1]) + 10
    draw.rounded_rectangle(
        [x, y, x + width, y + height],
        radius=10,
        fill=rgba_tuple(fill, 244),
        outline=rgba_tuple(outline, 255),
        width=1,
    )
    draw.text((x + 10, y + 5), text, font=font, fill=text_color)
    return width


def draw_arrow_png(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    dashed: bool = False,
):
    x1, y1 = start
    x2, y2 = end
    width = 4
    if dashed:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        unit_x = dx / length
        unit_y = dy / length
        step = 20
        dash = 11
        pos = 0.0
        while pos < length - 16:
            seg_start = pos
            seg_end = min(pos + dash, length - 16)
            draw.line(
                [(x1 + unit_x * seg_start, y1 + unit_y * seg_start), (x1 + unit_x * seg_end, y1 + unit_y * seg_end)],
                fill=rgba_tuple(color, 255),
                width=width,
            )
            pos += step
    else:
        draw.line([start, end], fill=rgba_tuple(color, 255), width=width)

    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_len = 18
    left = (
        x2 - arrow_len * math.cos(angle - math.pi / 6),
        y2 - arrow_len * math.sin(angle - math.pi / 6),
    )
    right = (
        x2 - arrow_len * math.cos(angle + math.pi / 6),
        y2 - arrow_len * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=rgba_tuple(color, 255))


def draw_box_png(draw: ImageDraw.ImageDraw, box: Box):
    accent = darken(box.fill, 0.34)
    outline = darken(box.fill, 0.56)
    draw.rounded_rectangle(
        [box.x + 8, box.y + 10, box.x + box.w + 8, box.y + box.h + 10],
        radius=22,
        fill=rgba_tuple(PALETTE["shadow"], 52),
    )
    draw.rounded_rectangle(
        [box.x, box.y, box.x + box.w, box.y + box.h],
        radius=22,
        fill=rgba_tuple(lighten(box.fill, 0.06), 248),
        outline=rgba_tuple(outline, 255),
        width=3,
    )
    draw.rounded_rectangle(
        [box.x + 4, box.y + 4, box.x + box.w - 4, box.y + 20],
        radius=18,
        fill=rgba_tuple(accent, 255),
    )
    draw.text((box.x + 16, box.y + 26), box.title, font=BOX_TITLE_FONT, fill=PALETTE["title"])

    if box.tag:
        pill_width = estimate_text_width(box.tag, 13, 0.66) + 18
        pill_x = box.x + box.w - pill_width - 16
        draw_pill_png(
            draw,
            pill_x,
            box.y + 20,
            box.tag,
            lighten(accent, 0.78),
            accent,
            lighten(accent, 0.4),
            TAG_FONT,
        )

    divider_y = box.y + 62
    draw.line(
        [(box.x + 16, divider_y), (box.x + box.w - 16, divider_y)],
        fill=rgba_tuple(lighten(accent, 0.32), 255),
        width=2,
    )
    body_lines = wrap_lines(box.body, max(14, int((box.w - 34) / 9.3)))
    draw.multiline_text(
        (box.x + 16, divider_y + 12),
        "\n".join(body_lines),
        font=BOX_BODY_FONT,
        fill=PALETTE["text"],
        spacing=5,
    )


def draw_arrow_label_png(draw: ImageDraw.ImageDraw, arrow: Arrow):
    if not arrow.label:
        return
    center_x = (arrow.start[0] + arrow.end[0]) / 2
    center_y = (arrow.start[1] + arrow.end[1]) / 2
    label = arrow.label
    bbox = draw.textbbox((0, 0), label, font=FOOTER_FONT)
    label_w = bbox[2] - bbox[0]
    label_h = bbox[3] - bbox[1]
    pad_x = 10
    pad_y = 6
    draw.rounded_rectangle(
        [
            center_x - label_w / 2 - pad_x + 2,
            center_y - label_h / 2 - pad_y + 2,
            center_x + label_w / 2 + pad_x + 2,
            center_y + label_h / 2 + pad_y + 2,
        ],
        radius=10,
        fill=rgba_tuple(PALETTE["shadow"], 34),
    )
    draw.rounded_rectangle(
        [
            center_x - label_w / 2 - pad_x,
            center_y - label_h / 2 - pad_y,
            center_x + label_w / 2 + pad_x,
            center_y + label_h / 2 + pad_y,
        ],
        radius=10,
        fill=rgba_tuple(lighten(PALETTE["topic"], 0.88), 248),
        outline=rgba_tuple(lighten(PALETTE["topic"], 0.35), 255),
        width=1,
    )
    draw.text(
        (center_x - label_w / 2, center_y - label_h / 2),
        label,
        font=FOOTER_FONT,
        fill=PALETTE["topic"],
    )


def draw_footer_panel_png(draw: ImageDraw.ImageDraw, diagram: Diagram):
    panel_x = MARGIN
    panel_w = CANVAS_W - 2 * MARGIN
    draw.rounded_rectangle(
        [panel_x + 8, PANEL_Y + 10, panel_x + panel_w + 8, PANEL_Y + PANEL_H + 10],
        radius=24,
        fill=rgba_tuple(PALETTE["shadow"], 40),
    )
    draw.rounded_rectangle(
        [panel_x, PANEL_Y, panel_x + panel_w, PANEL_Y + PANEL_H],
        radius=24,
        fill=rgba_tuple(PALETTE["panel"], 244),
        outline=rgba_tuple(lighten(PALETTE["border"], 0.52), 255),
        width=2,
    )
    draw.rounded_rectangle(
        [panel_x + 6, PANEL_Y + 6, panel_x + panel_w - 6, PANEL_Y + 18],
        radius=18,
        fill=rgba_tuple(PALETTE["topic"], 255),
    )

    draw.text((panel_x + 20, PANEL_Y + 26), "Interpretation", font=PANEL_TITLE_FONT, fill=PALETTE["title"])
    summary_lines = wrap_lines(diagram.summary, 74)
    draw.multiline_text(
        (panel_x + 20, PANEL_Y + 54),
        "\n".join(summary_lines),
        font=PANEL_BODY_FONT,
        fill=PALETTE["text"],
        spacing=4,
    )
    draw.text(
        (panel_x + 20, PANEL_Y + 104),
        "Solid arrows = primary runtime flow    Dashed arrows = commands, fallbacks, or diagnostic-only links",
        font=FOOTER_FONT,
        fill=PALETTE["muted"],
    )

    legend_x = panel_x + 910
    draw.text((legend_x, PANEL_Y + 26), "Legend", font=PANEL_TITLE_FONT, fill=PALETTE["title"])
    for index, (label, color) in enumerate(LEGEND_ITEMS):
        col = index // 2
        row = index % 2
        item_x = legend_x + col * 255
        item_y = PANEL_Y + 58 + row * 34
        draw.rounded_rectangle(
            [item_x, item_y, item_x + 24, item_y + 16],
            radius=5,
            fill=rgba_tuple(lighten(color, 0.06), 255),
            outline=rgba_tuple(darken(color, 0.36), 255),
            width=1,
        )
        draw.text((item_x + 34, item_y - 2), label, font=PANEL_BODY_FONT, fill=PALETTE["text"])

    footer_bbox = draw.textbbox((0, 0), diagram.footer, font=FOOTER_FONT)
    footer_x = panel_x + panel_w - (footer_bbox[2] - footer_bbox[0]) - 22
    draw.text((footer_x, PANEL_Y + PANEL_H - 26), diagram.footer, font=FOOTER_FONT, fill=PALETTE["muted"])


def render_png(diagram: Diagram):
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 255))
    draw_background_png(image)
    draw = ImageDraw.Draw(image)

    draw.text((MARGIN, 28), diagram.title, font=TITLE_FONT, fill=PALETTE["title"])
    draw.text((MARGIN, 78), diagram.subtitle, font=SUBTITLE_FONT, fill=PALETTE["subtitle"])
    draw.line([(MARGIN, 112), (CANVAS_W - MARGIN, 112)], fill=rgba_tuple(PALETTE["grid"], 255), width=2)

    badge_width = estimate_text_width(diagram.badge, 13, 0.68) + 20
    draw_pill_png(
        draw,
        CANVAS_W - MARGIN - badge_width,
        36,
        diagram.badge,
        lighten(PALETTE["topic"], 0.86),
        PALETTE["topic"],
        lighten(PALETTE["topic"], 0.34),
        TAG_FONT,
    )

    for arrow in diagram.arrows:
        draw_arrow_png(draw, arrow.start, arrow.end, arrow.color, dashed=arrow.dashed)

    for box in diagram.boxes:
        draw_box_png(draw, box)

    for arrow in diagram.arrows:
        draw_arrow_label_png(draw, arrow)

    draw_footer_panel_png(draw, diagram)
    image.convert("RGB").save(OUT_DIR / f"{diagram.filename}.png")


def svg_text_block(
    x: int,
    y: int,
    lines: list[str],
    font_size: int,
    fill: str,
    weight: str = "400",
) -> str:
    out = [
        f'<text x="{x}" y="{y}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{font_size}" font-weight="{weight}" fill="{fill}">'
    ]
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else int(font_size * 1.38)
        out.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    out.append("</text>")
    return "\n".join(out)


def svg_pill(x: int, y: int, text: str, fill: str, text_color: str, outline: str) -> str:
    width = estimate_text_width(text, 13, 0.68) + 20
    height = 24
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" ry="10" fill="{fill}" stroke="{outline}" stroke-width="1" />',
            svg_text_block(x + 10, y + 16, [text], 13, text_color, weight="700"),
        ]
    )


def render_svg(diagram: Diagram):
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">',
        "<defs>",
        f'<linearGradient id="bgGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{PALETTE["bg_top"]}" /><stop offset="100%" stop-color="{PALETTE["bg_bottom"]}" /></linearGradient>',
        f'<pattern id="dotGrid" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.2" fill="{PALETTE["grid"]}" /></pattern>',
        f'<filter id="cardShadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="8" stdDeviation="6" flood-color="{PALETTE["shadow"]}" flood-opacity="0.22" /></filter>',
        f'<marker id="arrowhead" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto"><polygon points="0 0, 12 4, 0 8" fill="{PALETTE["arrow"]}" /></marker>',
        "</defs>",
        f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" fill="url(#bgGradient)" />',
        f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" fill="url(#dotGrid)" opacity="0.55" />',
        f'<ellipse cx="{CANVAS_W - 350}" cy="210" rx="220" ry="190" fill="{lighten(PALETTE["tf"], 0.22)}" opacity="0.55" />',
        '<ellipse cx="260" cy="930" rx="180" ry="170" fill="#edf7ff" opacity="0.55" />',
        f'<rect x="0" y="0" width="{CANVAS_W}" height="108" fill="#ffffff" opacity="0.58" />',
        svg_text_block(MARGIN, 54, [diagram.title], 40, PALETTE["title"], weight="700"),
        svg_text_block(MARGIN, 88, [diagram.subtitle], 20, PALETTE["subtitle"]),
        f'<line x1="{MARGIN}" y1="112" x2="{CANVAS_W - MARGIN}" y2="112" stroke="{PALETTE["grid"]}" stroke-width="2" />',
        svg_pill(CANVAS_W - MARGIN - estimate_text_width(diagram.badge, 13, 0.68) - 20, 36, diagram.badge, lighten(PALETTE["topic"], 0.86), PALETTE["topic"], lighten(PALETTE["topic"], 0.34)),
    ]

    for arrow in diagram.arrows:
        dash_attr = 'stroke-dasharray="12,8" ' if arrow.dashed else ''
        svg.append(
            f'<line x1="{arrow.start[0]}" y1="{arrow.start[1]}" x2="{arrow.end[0]}" y2="{arrow.end[1]}" stroke="{arrow.color}" stroke-width="4" {dash_attr}marker-end="url(#arrowhead)" />'
        )

    for box in diagram.boxes:
        accent = darken(box.fill, 0.34)
        outline = darken(box.fill, 0.56)
        svg.extend(
            [
                '<g filter="url(#cardShadow)">',
                f'<rect x="{box.x}" y="{box.y}" rx="22" ry="22" width="{box.w}" height="{box.h}" fill="{lighten(box.fill, 0.06)}" stroke="{outline}" stroke-width="3" />',
                f'<rect x="{box.x + 4}" y="{box.y + 4}" rx="18" ry="18" width="{box.w - 8}" height="16" fill="{accent}" />',
                '</g>',
                svg_text_block(box.x + 16, box.y + 42, [box.title], 22, PALETTE["title"], weight="700"),
                f'<line x1="{box.x + 16}" y1="{box.y + 62}" x2="{box.x + box.w - 16}" y2="{box.y + 62}" stroke="{lighten(accent, 0.32)}" stroke-width="2" />',
            ]
        )
        if box.tag:
            pill_w = estimate_text_width(box.tag, 13, 0.68) + 18
            svg.append(
                svg_pill(
                    box.x + box.w - pill_w - 16,
                    box.y + 20,
                    box.tag,
                    lighten(accent, 0.78),
                    accent,
                    lighten(accent, 0.42),
                )
            )
        body_lines = wrap_lines(box.body, max(14, int((box.w - 34) / 9.3)))
        svg.append(svg_text_block(box.x + 16, box.y + 86, body_lines, 15, PALETTE["text"]))

    for arrow in diagram.arrows:
        if not arrow.label:
            continue
        center_x = int((arrow.start[0] + arrow.end[0]) / 2)
        center_y = int((arrow.start[1] + arrow.end[1]) / 2)
        label_w = estimate_text_width(arrow.label, 13, 0.66) + 20
        svg.extend(
            [
                f'<rect x="{center_x - label_w // 2 + 2}" y="{center_y - 14 + 2}" width="{label_w}" height="28" rx="10" ry="10" fill="{PALETTE["shadow"]}" opacity="0.18" />',
                f'<rect x="{center_x - label_w // 2}" y="{center_y - 14}" width="{label_w}" height="28" rx="10" ry="10" fill="{lighten(PALETTE["topic"], 0.88)}" stroke="{lighten(PALETTE["topic"], 0.35)}" stroke-width="1" />',
                svg_text_block(center_x - label_w // 2 + 10, center_y + 5, [arrow.label], 13, PALETTE["topic"], weight="400"),
            ]
        )

    panel_x = MARGIN
    panel_w = CANVAS_W - 2 * MARGIN
    svg.extend(
        [
            '<g filter="url(#cardShadow)">',
            f'<rect x="{panel_x}" y="{PANEL_Y}" rx="24" ry="24" width="{panel_w}" height="{PANEL_H}" fill="{PALETTE["panel"]}" stroke="{lighten(PALETTE["border"], 0.52)}" stroke-width="2" opacity="0.97" />',
            '</g>',
            f'<rect x="{panel_x + 6}" y="{PANEL_Y + 6}" rx="18" ry="18" width="{panel_w - 12}" height="12" fill="{PALETTE["topic"]}" />',
            svg_text_block(panel_x + 20, PANEL_Y + 44, ["Interpretation"], 17, PALETTE["title"], weight="700"),
            svg_text_block(panel_x + 20, PANEL_Y + 72, wrap_lines(diagram.summary, 74), 14, PALETTE["text"]),
            svg_text_block(panel_x + 20, PANEL_Y + 122, ["Solid arrows = primary runtime flow    Dashed arrows = commands, fallbacks, or diagnostic-only links"], 13, PALETTE["muted"]),
            svg_text_block(panel_x + 910, PANEL_Y + 44, ["Legend"], 17, PALETTE["title"], weight="700"),
        ]
    )

    for index, (label, color) in enumerate(LEGEND_ITEMS):
        col = index // 2
        row = index % 2
        item_x = panel_x + 910 + col * 255
        item_y = PANEL_Y + 58 + row * 34
        svg.extend(
            [
                f'<rect x="{item_x}" y="{item_y}" width="24" height="16" rx="5" ry="5" fill="{lighten(color, 0.06)}" stroke="{darken(color, 0.36)}" stroke-width="1" />',
                svg_text_block(item_x + 34, item_y + 14, [label], 14, PALETTE["text"]),
            ]
        )

    footer_x = panel_x + panel_w - estimate_text_width(diagram.footer, 13, 0.61) - 22
    svg.append(svg_text_block(footer_x, PANEL_Y + PANEL_H - 12, [diagram.footer], 13, PALETTE["muted"]))
    svg.append("</svg>")
    (OUT_DIR / f"{diagram.filename}.svg").write_text("\n".join(svg), encoding="utf-8")


def system_architecture() -> Diagram:
    boxes = [
        Box(70, 150, 320, 240, "Gazebo + Robot Model", "Role:\n- world physics and robot spawn\n- IMU, LiDAR, GT pose\n- gz model and joint topics", PALETTE["sim"], tag="SIM"),
        Box(455, 150, 320, 240, "Robot Interface", "Boundary layer:\n- ros_gz_bridge\n- robot_state_publisher\n- controller_manager\n- Emcon HW interface", PALETTE["control"], tag="ROS"),
        Box(840, 88, 380, 320, "Localization Stack", "Active deployed estimator:\n- Madgwick attitude\n- terramechanic odom\n- GTSAM factor graph\n- local DEM frontend\n- TRN global correction", PALETTE["loc"], tag="ACTIVE"),
        Box(1285, 108, 515, 280, "Planning + EMCON", "High-level command path:\n- camera -> terrain class\n- SBLP primitive choice\n- EMCON gating / scaling\n- final /cmd_vel output", PALETTE["perception"], tag="PLAN"),
        Box(315, 500, 485, 280, "Control Conversion", "Low-level actuation:\n- ackermann_twist_controller\n- steering split\n- rear wheel velocity split\n- forward controllers", PALETTE["control"], tag="ACT"),
        Box(900, 500, 390, 280, "TF + Outputs", "Frame ownership:\n- TRN: map->odom\n- factor graph: odom->base_footprint\n- RSP: base and sensor tree", PALETTE["tf"], tag="TF"),
        Box(1360, 500, 390, 280, "Diagnostics", "Operator-facing tools:\n- odom visualizer\n- GT comparator\n- TRN quality plots\n- error logging", PALETTE["risk"], tag="OBS"),
    ]
    arrows = [
        Arrow((390, 270), (455, 270), "gz topics + model"),
        Arrow((775, 270), (840, 270), "ROS topics / TF"),
        Arrow((615, 390), (615, 500), "joint states + /cmd_vel"),
        Arrow((1220, 255), (1285, 255), "terrain class / goal"),
        Arrow((800, 640), (900, 640), "odom + TF"),
        Arrow((1290, 640), (1360, 640), "metrics + plots"),
        Arrow((1540, 388), (700, 500), "/cmd_vel", dashed=True),
    ]
    return Diagram(
        filename="01_system_architecture",
        title="Silent Sentry - System Architecture",
        subtitle="Current active stack in the repo: Gazebo, native ros2_control, terramechanic/TRN localization, and VLM-assisted command selection",
        badge="SYSTEM VIEW",
        summary="The deployed system is a wheel and IMU local estimator with DEM-based global correction. No bundled LiDAR-inertial SLAM package remains in the active workspace path.",
        boxes=boxes,
        arrows=arrows,
    )


def runtime_data_flow() -> Diagram:
    boxes = [
        Box(70, 140, 240, 220, "IMU", "Source topic:\n- /imu\n- ang vel + accel\n- bridged from Gazebo", PALETTE["sensor"], tag="SRC"),
        Box(70, 400, 240, 220, "Joint States", "Source topic:\n- /joint_states\n- wheel speed feedback\n- steering position", PALETTE["sensor"], tag="SRC"),
        Box(70, 660, 240, 220, "LiDAR", "Source topic:\n- /scan/points\n- PointCloud2 sweep\n- used for DEM only", PALETTE["sensor"], tag="SRC"),
        Box(380, 140, 275, 220, "Madgwick", "Attitude prefilter:\nInputs:\n- /imu\nOutput:\n- /imu/data_filtered", PALETTE["loc"], tag="PREFILT"),
        Box(380, 400, 275, 220, "Terramechanic Odom", "Wheel-centric motion:\nInputs:\n- joint states\n- filtered IMU\nOutput:\n- /terramechanic_odom", PALETTE["loc"], tag="ODOM"),
        Box(380, 660, 275, 220, "Local DEM Builder", "LiDAR frontend:\nInputs:\n- points + IMU + odom\nOutputs:\n- local DEM grid\n- DEM metadata", PALETTE["loc"], tag="DEM"),
        Box(745, 220, 320, 280, "Factor Graph Fuser", "Local DR backend:\nInputs:\n- terramechanic odom\n- filtered IMU\n- TRN quality hint\nOutputs:\n- /odometry/filtered\n- odom->base_footprint", PALETTE["loc"], tag="FG"),
        Box(745, 600, 320, 240, "TRN MCL", "Global correction:\nInputs:\n- local DEM\n- odom prior\nOutputs:\n- map->odom\n- /trn diagnostics", PALETTE["loc"], tag="TRN"),
        Box(1145, 220, 320, 240, "Composed Pose", "Consumer-visible pose:\n- map->odom oplus odom->base\n- full localized body pose", PALETTE["tf"], tag="TF"),
        Box(1145, 560, 320, 240, "Visualizer / Comparator", "Diagnostic consumers:\n- GT alignment\n- localized vs raw odom\n- plots and logs", PALETTE["risk"], tag="DIAG"),
        Box(1545, 220, 260, 240, "Navigation Consumers", "Downstream use:\n- RViz\n- benchmark tools\n- future nav modules", PALETTE["control"], tag="NAV"),
    ]
    arrows = [
        Arrow((310, 250), (380, 250), "/imu"),
        Arrow((310, 510), (380, 510), "/joint_states"),
        Arrow((310, 770), (380, 770), "/scan/points"),
        Arrow((655, 250), (745, 310), "/imu/data_filtered"),
        Arrow((655, 510), (745, 380), "/terramechanic_odom"),
        Arrow((655, 770), (745, 710), "/elevation_map/local_float"),
        Arrow((1065, 360), (1145, 340), "odom->base"),
        Arrow((1065, 720), (1145, 340), "map->odom"),
        Arrow((1465, 340), (1545, 340), "map->base"),
        Arrow((1065, 720), (1145, 670), "/trn/*"),
        Arrow((1465, 670), (1545, 470), "metrics"),
        Arrow((655, 510), (1145, 670), "raw odom", dashed=True),
    ]
    return Diagram(
        filename="02_runtime_data_flow",
        title="Silent Sentry - Runtime Data Flow",
        subtitle="Topic-level flow for the active localization pipeline",
        badge="TOPIC FLOW",
        summary="Local pose is formed first in odom. Map-frame localization only appears after TRN publishes map->odom and consumers compose that transform with odom->base_footprint.",
        boxes=boxes,
        arrows=arrows,
    )


def sensor_processing() -> Diagram:
    boxes = [
        Box(60, 120, 395, 380, "IMU Processing", "Pipeline:\n- raw IMU -> Madgwick attitude\nEstimator use:\n- gravity compensation\n- stall and ZUPT logic\n- roll and pitch priors + PIM", PALETTE["sensor"], tag="IMU"),
        Box(515, 120, 395, 380, "Wheel / Joint Processing", "Pipeline:\n- joint states -> wheel speed and steer angle\n- sinkage and slip estimation\n- gyro-aided yaw-rate fusion\nOutputs:\n- /terramechanic_odom", PALETTE["sensor"], tag="WHEEL"),
        Box(970, 120, 395, 380, "LiDAR Processing", "Pipeline:\n- azimuth timing\n- deskew\n- self-hit filtering\n- gravity alignment\n- RANSAC ground\n- rolling DEM raster", PALETTE["sensor"], tag="LIDAR"),
        Box(1425, 120, 385, 380, "Camera Processing", "Pipeline:\n- /camera/image_raw\n- CLIP zero-shot prompts\nOutputs:\n- terrain class\n- traversability context", PALETTE["sensor"], tag="CAM"),
        Box(270, 610, 455, 270, "Estimator Use", "Localization contribution:\n- terramechanics drives local motion\n- factor graph stabilizes local pose\n- TRN corrects global drift with DEM matching", PALETTE["loc"], tag="EST"),
        Box(930, 610, 560, 270, "Planning Use", "Behavior contribution:\n- terrain class changes SBLP scenario\n- localization remains mostly diagnostic today\n- planning branch is lighter than the estimation stack", PALETTE["perception"], tag="PLAN"),
    ]
    arrows = [
        Arrow((255, 500), (450, 610), "inertial priors"),
        Arrow((710, 500), (540, 610), "wheel motion"),
        Arrow((1165, 500), (560, 610), "local DEM / TRN"),
        Arrow((1615, 500), (1210, 610), "terrain class"),
        Arrow((725, 745), (930, 745), "pose / context"),
    ]
    return Diagram(
        filename="03_sensor_processing",
        title="Silent Sentry - Sensor Processing and Use",
        subtitle="How each sensor is transformed before it affects estimation or planning",
        badge="SENSOR USE",
        summary="Each sensing modality serves a distinct role: IMU stabilizes attitude, joints drive local motion, LiDAR supports DEM/TRN localization, and camera changes planning context rather than pose estimation.",
        boxes=boxes,
        arrows=arrows,
    )


def tf_formation() -> Diagram:
    boxes = [
        Box(770, 80, 320, 160, "map", "Global correction frame\npublished by TRN", PALETTE["tf"], tag="FRAME"),
        Box(770, 320, 320, 160, "odom", "Local drift frame\nheld by factor graph", PALETTE["tf"], tag="FRAME"),
        Box(770, 560, 320, 160, "base_footprint", "Localized body frame\nroot for body kinematics", PALETTE["tf"], tag="FRAME"),
        Box(230, 300, 400, 200, "TRN SLAM Node", "Authority:\n- DEM and particle matching\nOutput:\n- map->odom", PALETTE["loc"], tag="PRODUCER"),
        Box(230, 540, 400, 200, "Factor Graph Fuser", "Authority:\n- wheel + IMU dead reckoning\nOutput:\n- odom->base_footprint", PALETTE["loc"], tag="PRODUCER"),
        Box(1260, 540, 470, 240, "Robot State Publisher", "Authority:\n- URDF kinematic tree\nOutputs:\n- base_link\n- wheels / steering\n- imu / lidar / camera frames", PALETTE["control"], tag="KIN"),
        Box(1260, 220, 470, 200, "Consumers", "Consumers:\n- odom visualizer\n- GT comparator\n- local DEM builder\n- RViz and nav users", PALETTE["risk"], tag="CONSUMER"),
    ]
    arrows = [
        Arrow((630, 400), (770, 400), "map->odom"),
        Arrow((630, 640), (770, 640), "odom->base"),
        Arrow((930, 240), (930, 320), "TRN correction"),
        Arrow((930, 480), (930, 560), "local pose"),
        Arrow((1090, 640), (1260, 640), "base->links"),
        Arrow((1090, 400), (1260, 320), "map->base composed"),
    ]
    return Diagram(
        filename="04_tf_formation",
        title="Silent Sentry - TF Formation",
        subtitle="Which nodes contribute to the runtime frame tree and how consumers obtain the final localized pose",
        badge="FRAME OWNERSHIP",
        summary="Frame authority is intentionally split: the factor graph owns odom->base_footprint, TRN owns map->odom, and robot_state_publisher owns the body and sensor tree below the localized base.",
        boxes=boxes,
        arrows=arrows,
    )


def math_methods() -> Diagram:
    boxes = [
        Box(60, 120, 370, 280, "Vehicle Kinematics", "Ackermann geometry:\n- delta = atan(L * omega / v)\n- inner / outer steer split\n- wheel speed split", PALETTE["math"], tag="KIN"),
        Box(500, 120, 370, 280, "Terramechanics", "Desert odom model:\n- Bekker sinkage\n- effective wheel radius\n- slip ratio\n- understeer model\n- gyro bias KF", PALETTE["math"], tag="TERRA"),
        Box(940, 120, 370, 280, "Factor Graph", "GTSAM backend:\n- Pose3 / Rot3 / NavState\n- ImuFactor + wheel factor\n- bias priors\n- iSAM2 update", PALETTE["math"], tag="GTSAM"),
        Box(1380, 120, 370, 280, "Local DEM Frontend", "LiDAR preprocessing:\n- azimuth-time deskew\n- gravity alignment\n- self-hit masking\n- ground segmentation\n- rasterization", PALETTE["math"], tag="DEM"),
        Box(260, 540, 440, 280, "TRN Matching", "Terrain localization:\n- dynamic ROI over DEM\n- particle motion update\n- MAD likelihood\n- ESS resampling\n- recovery injection", PALETTE["math"], tag="TRN"),
        Box(860, 540, 440, 280, "Learning Component", "Scene classifier:\n- CLIP image-text scoring\n- prompt set over terrain types\n- scenario selection input", PALETTE["math"], tag="VLM"),
        Box(1460, 540, 320, 280, "LiDAR-Inertial Gap", "Current state:\n- no bundled LIO-SAM package\n- active stack is terramechanic + TRN\n- LiDAR is used for DEM matching", PALETTE["risk"], tag="ALT"),
    ]
    arrows = [
        Arrow((430, 260), (500, 260), "wheel motion model"),
        Arrow((870, 260), (940, 260), "local state estimation"),
        Arrow((1310, 260), (1380, 260), "LiDAR frontend"),
        Arrow((700, 400), (500, 540), "odom prior"),
        Arrow((1565, 400), (500, 540), "DEM evidence"),
        Arrow((1080, 400), (1080, 540), "perception branch"),
    ]
    return Diagram(
        filename="05_math_methods",
        title="Silent Sentry - Mathematical Methods in the Codebase",
        subtitle="Algorithms actually implemented in the active stack and their relationship",
        badge="ALGORITHM STACK",
        summary="The active architecture is a hybrid of Ackermann kinematics, terramechanics, GTSAM dead reckoning, DEM-based terrain matching, and zero-shot terrain classification. It is not SE(3)-LIO.",
        boxes=boxes,
        arrows=arrows,
    )


def build_all() -> list[Diagram]:
    return [
        system_architecture(),
        runtime_data_flow(),
        sensor_processing(),
        tf_formation(),
        math_methods(),
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diagrams = build_all()
    for diagram in diagrams:
        render_svg(diagram)
        render_png(diagram)
    print(f"Generated {len(diagrams)} diagram sets in {OUT_DIR}")


if __name__ == "__main__":
    main()
