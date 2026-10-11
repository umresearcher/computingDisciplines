"""TownSquare title-screen city, v2: a 12x12 isometric town on a block of earth.

x runs down-right, y runs down-left, z is up. Ground is z = 0; the earth hangs below.
Writes one SVG cropped to its content. Usage: python3 tools/city.py assets/city.svg  (animation classes + keyframes are written into the SVG)
"""
import math
import random
import sys

random.seed(7)
S = 28
COS, SIN = math.cos(math.radians(30)), math.sin(math.radians(30))
N, DEPTH = 12, 4.2

NIGHT, PANEL, PAPER, LILAC = "#17163A", "#232259", "#F6F2E7", "#B9B2FF"
SIGNAL, CORAL, TEAL, TEAL_D = "#FFD23F", "#FF6B4A", "#3DD6C4", "#1E8C80"

ground_layer, objects, overlay = [], [], []   # draw order: ground, sorted objects, overlay
_cur = None                                   # list currently being written to
bbox = [1e9, 1e9, -1e9, -1e9]


def p(x, y, z=0.0):
    X, Y = (x - y) * COS * S, (x + y) * SIN * S - z * S
    bbox[0], bbox[1] = min(bbox[0], X), min(bbox[1], Y)
    bbox[2], bbox[3] = max(bbox[2], X), max(bbox[3], Y)
    return X, Y


def emit(s):
    _cur.append(s)


def poly(points, fill, extra=""):
    emit('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in points) + f'" fill="{fill}"{extra}/>')


def quad(a, b, c, d, fill, extra=""):
    poly([p(*a), p(*b), p(*c), p(*d)], fill, extra)


def shade(c, k):
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    if k <= 1:
        r, g, b = (int(v * k) for v in (r, g, b))
    else:
        t = k - 1
        r, g, b = (int(v + (255 - v) * t) for v in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def ground_rect(x0, y0, x1, y1, fill, z=0.0):
    quad((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), fill)


def ground_ellipse(cx, cy, rx, ry, fill, n=28, extra=""):
    pts = [p(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n), 0) for i in range(n)]
    poly(pts, fill, extra)


# ---------------------------------------------------------------- animation helpers
# Every moving part gets a class; the keyframes live in the <style> block written at the end.
_arnd = random.Random(99)


def anim_style(dur_lo, dur_hi, rnd=None):
    """Random duration + negative delay so identical parts never move in sync."""
    r = rnd or _arnd
    d = r.uniform(dur_lo, dur_hi)
    return f'animation-duration:{d:.1f}s;animation-delay:-{r.uniform(0, d):.1f}s'


def window(pts, lit, lit_color, dark_color):
    """Draw one window. Some lit windows switch off for a moment and some dark ones switch on:
    a dark pane is always drawn, and an animated lit pane sits on top of it."""
    roll = _arnd.random()
    if lit and roll < 0.2:
        quad(*pts, dark_color)
        quad(*pts, lit_color, f' class="w-off" style="{anim_style(6, 14)}"')
    elif not lit and roll < 0.14:
        quad(*pts, dark_color)
        quad(*pts, lit_color, f' class="w-on" style="{anim_style(7, 16)}"')
    else:
        quad(*pts, lit_color if lit else dark_color)


def group(cls, style=""):
    emit(f'<g class="{cls}"' + (f' style="{style}"' if style else "") + ">")


def end():
    emit("</g>")


# ---------------------------------------------------------------- 3D pieces
def box(x0, y0, w, d, h, color, z0=0.0, windows=None, door=False, roof_inset=True):
    x1, y1, top = x0 + w, y0 + d, z0 + h
    quad((x0, y1, z0), (x1, y1, z0), (x1, y1, top), (x0, y1, top), shade(color, 0.84))
    quad((x1, y0, z0), (x1, y1, z0), (x1, y1, top), (x1, y0, top), shade(color, 0.64))
    quad((x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top), color)
    if roof_inset and w > 0.8 and d > 0.8 and h > 0.9:
        m = 0.12
        quad((x0 + m, y0 + m, top), (x1 - m, y0 + m, top), (x1 - m, y1 - m, top), (x0 + m, y1 - m, top), shade(color, 0.92))
    if windows:
        rows, cols, seed = windows
        rnd = random.Random(seed)
        for r in range(rows):
            za = z0 + 0.4 + r * (h - 0.55) / rows
            zb = za + min(0.34, (h - 0.55) / rows * 0.6)
            for c in range(cols):
                xa = x0 + 0.18 + c * (w - 0.25) / cols
                xb = xa + min(0.26, (w - 0.25) / cols * 0.6)
                lit = rnd.random() < 0.62
                window(((xa, y1, za), (xb, y1, za), (xb, y1, zb), (xa, y1, zb)), lit, SIGNAL, shade(color, 0.5))
            for c in range(max(1, int(cols * d / max(w, 0.1)))):
                ncols = max(1, int(cols * d / max(w, 0.1)))
                ya = y0 + 0.18 + c * (d - 0.25) / ncols
                yb = ya + min(0.26, (d - 0.25) / ncols * 0.6)
                lit = rnd.random() < 0.5
                window(((x1, ya, za), (x1, yb, za), (x1, yb, zb), (x1, ya, zb)), lit, "#E8B92A", shade(color, 0.42))
    if door:
        xa = x0 + w * 0.42
        quad((xa, y1, z0), (xa + 0.32, y1, z0), (xa + 0.32, y1, z0 + 0.55), (xa, y1, z0 + 0.55), shade(color, 0.42))


def pyramid(x0, y0, w, d, z, h, color):
    x1, y1 = x0 + w, y0 + d
    apex = p((x0 + x1) / 2, (y0 + y1) / 2, z + h)
    poly([p(x0, y1, z), p(x1, y1, z), apex], shade(color, 0.88))
    poly([p(x1, y0, z), p(x1, y1, z), apex], shade(color, 0.66))


def gable(x0, y0, w, d, z, h, color):
    """Pitched roof with the ridge along x."""
    x1, y1, ym = x0 + w, y0 + d, y0 + d / 2
    quad((x0, y1, z), (x1, y1, z), (x1, ym, z + h), (x0, ym, z + h), color)
    poly([p(x1, y0, z), p(x1, y1, z), p(x1, ym, z + h)], shade(color, 0.7))


def tree(x, y, size=0.42, kind="round"):
    bx, by = p(x, y, 0)
    _, ty = p(x, y, 0.85)
    emit(f'<rect x="{bx - 1.8:.1f}" y="{ty:.1f}" width="3.6" height="{by - ty:.1f}" fill="#5A3420"/>')
    r = size * S
    group("sway", anim_style(4, 7))
    if kind == "pine":
        emit(f'<path d="M{bx:.1f} {ty - r * 1.6:.1f} L{bx + r * 0.8:.1f} {ty + r * 0.3:.1f} L{bx - r * 0.8:.1f} {ty + r * 0.3:.1f} Z" fill="#1E7A63"/>')
        emit(f'<path d="M{bx:.1f} {ty - r * 1.6:.1f} L{bx + r * 0.8:.1f} {ty + r * 0.3:.1f} L{bx:.1f} {ty + r * 0.3:.1f} Z" fill="#17614F"/>')
    else:
        emit(f'<ellipse cx="{bx:.1f}" cy="{ty:.1f}" rx="{r:.1f}" ry="{r * 0.92:.1f}" fill="#1E8C80"/>')
        emit(f'<ellipse cx="{bx + r * 0.2:.1f}" cy="{ty + r * 0.15:.1f}" rx="{r * 0.7:.1f}" ry="{r * 0.6:.1f}" fill="#197869"/>')
        emit(f'<ellipse cx="{bx - r * 0.28:.1f}" cy="{ty - r * 0.32:.1f}" rx="{r * 0.42:.1f}" ry="{r * 0.32:.1f}" fill="#3DD6C4"/>')
    end()


def lamp(x, y):
    bx, by = p(x, y, 0)
    _, ty = p(x, y, 1.15)
    emit(f'<rect x="{bx - 1.2:.1f}" y="{ty:.1f}" width="2.4" height="{by - ty:.1f}" fill="#6F6AA8"/>')
    emit(f'<circle cx="{bx:.1f}" cy="{ty:.1f}" r="7" fill="{SIGNAL}" fill-opacity="0.25" class="halo" style="{anim_style(3, 5)}"/>')
    emit(f'<circle cx="{bx:.1f}" cy="{ty:.1f}" r="3.6" fill="{SIGNAL}"/>')


def lamp_glow(x, y):
    ground_ellipse(x, y, 0.55, 0.55, SIGNAL, extra=f' fill-opacity="0.13" class="glow" style="{anim_style(3, 5)}"')


def car(x, y, along_x, color, rear=False):
    LIGHT = CORAL if rear else SIGNAL
    if along_x:
        box(x, y, 0.62, 0.3, 0.2, color, roof_inset=False)
        box(x + 0.14, y + 0.03, 0.3, 0.24, 0.14, shade(color, 1.35), z0=0.2, roof_inset=False)
        quad((x + 0.62, y + 0.04, 0.06), (x + 0.62, y + 0.1, 0.06), (x + 0.62, y + 0.1, 0.12), (x + 0.62, y + 0.04, 0.12), LIGHT)
        quad((x + 0.62, y + 0.2, 0.06), (x + 0.62, y + 0.26, 0.06), (x + 0.62, y + 0.26, 0.12), (x + 0.62, y + 0.2, 0.12), LIGHT)
    else:
        box(x, y, 0.3, 0.62, 0.2, color, roof_inset=False)
        box(x + 0.03, y + 0.14, 0.24, 0.3, 0.14, shade(color, 1.35), z0=0.2, roof_inset=False)
        quad((x + 0.04, y + 0.62, 0.06), (x + 0.1, y + 0.62, 0.06), (x + 0.1, y + 0.62, 0.12), (x + 0.04, y + 0.62, 0.12), LIGHT)
        quad((x + 0.2, y + 0.62, 0.06), (x + 0.26, y + 0.62, 0.06), (x + 0.26, y + 0.62, 0.12), (x + 0.2, y + 0.62, 0.12), LIGHT)


def person(x, y, color, cls="idle"):
    bx, by = p(x, y, 0)
    group(cls, anim_style(1.6, 3.2) if cls == "idle" else "")
    emit(f'<rect x="{bx - 2.6:.1f}" y="{by - 11:.1f}" width="5.2" height="9" rx="2.6" fill="{color}"/>')
    emit(f'<circle cx="{bx:.1f}" cy="{by - 14:.1f}" r="3" fill="#E0AC85"/>')
    end()


def flag(x, y, z, color):
    bx, by = p(x, y, z)
    emit(f'<rect x="{bx - 1:.1f}" y="{by - 26:.1f}" width="2" height="26" fill="{PAPER}"/>')
    emit(f'<path class="flag" d="M{bx + 1:.1f} {by - 26:.1f} L{bx + 15:.1f} {by - 22:.1f} L{bx + 1:.1f} {by - 17:.1f} Z" fill="{color}"/>')


def bubble(x, y, z, color, icon, delay=0.0):
    bx, by = p(x, y, z)
    group("bub", f"animation-delay:-{delay:.1f}s")
    w, h = 44, 30
    left, top = bx - w / 2, by - h - 10
    p(x, y, z + (h + 14) / S)  # extend bbox
    emit(f'<rect x="{left + 3:.1f}" y="{top + 3:.1f}" width="{w}" height="{h}" rx="9" fill="#0D0C26"/>')
    emit(f'<rect x="{left:.1f}" y="{top:.1f}" width="{w}" height="{h}" rx="9" fill="{color}"/>')
    emit(f'<path d="M{bx - 6:.1f} {top + h:.1f} L{bx:.1f} {top + h + 8:.1f} L{bx + 6:.1f} {top + h:.1f} Z" fill="{color}"/>')
    cx, cy = bx, top + h / 2
    if icon == "heart":
        emit(f'<path d="M{cx:.1f} {cy + 7:.1f} C{cx - 12:.1f} {cy - 1:.1f} {cx - 7:.1f} {cy - 10:.1f} {cx:.1f} {cy - 4:.1f} C{cx + 7:.1f} {cy - 10:.1f} {cx + 12:.1f} {cy - 1:.1f} {cx:.1f} {cy + 7:.1f} Z" fill="{NIGHT}"/>')
    elif icon == "check":
        emit(f'<path d="M{cx - 8:.1f} {cy:.1f} L{cx - 2:.1f} {cy + 6:.1f} L{cx + 9:.1f} {cy - 6:.1f}" stroke="{NIGHT}" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    elif icon == "bars":
        for i, bh in enumerate((6, 12, 9)):
            emit(f'<rect x="{cx - 9 + i * 7:.1f}" y="{cy + 7 - bh:.1f}" width="5" height="{bh}" rx="1" fill="{NIGHT}"/>')
    elif icon == "dots":
        for i in (-8, 0, 8):
            emit(f'<circle cx="{cx + i:.1f}" cy="{cy:.1f}" r="3" fill="{NIGHT}"/>')
    elif icon == "news":
        emit(f'<rect x="{cx - 10:.1f}" y="{cy - 7:.1f}" width="20" height="14" rx="2" fill="{NIGHT}"/>')
        emit(f'<rect x="{cx - 7:.1f}" y="{cy - 4:.1f}" width="6" height="5" fill="{color}"/>')
        emit(f'<rect x="{cx + 1:.1f}" y="{cy - 4:.1f}" width="6" height="2" fill="{color}"/>')
        emit(f'<rect x="{cx + 1:.1f}" y="{cy}" width="6" height="2" fill="{color}"/>')
    end()


# ================================================================ EARTH (drawn first)
_cur = ground_layer


def wavy(n_pts, base, amp, seed):
    rnd = random.Random(seed)
    return [base + amp * math.sin(i * 1.3 + seed) + rnd.uniform(-amp, amp) * 0.5 for i in range(n_pts)]


def earth_face(left):
    """left=True: face y=N (x from 0..N); else face x=N (y from 0..N)."""
    k = 1.0 if left else 0.78
    pt = (lambda u, z: p(u, N, z)) if left else (lambda u, z: p(N, u, z))
    us = [i * N / 24 for i in range(25)]
    layers = [  # (top boundary z values, color)
        ([-0.0] * 25, "#2F8F6A"),                               # grass lip
        (wavy(25, -0.32, 0.05, 1 if left else 2), "#6B4428"),    # topsoil
        (wavy(25, -1.35, 0.12, 3 if left else 4), "#8A5A34"),    # clay
        (wavy(25, -2.55, 0.16, 5 if left else 6), "#5B4A5E"),    # rock
        (wavy(25, -3.55, 0.12, 7 if left else 8), "#463B52"),    # bedrock
    ]
    for i, (tops, color) in enumerate(layers):
        bottoms = layers[i + 1][0] if i + 1 < len(layers) else [-DEPTH] * 25
        pts = [pt(u, z) for u, z in zip(us, tops)] + [pt(u, z) for u, z in reversed(list(zip(us, bottoms)))]
        poly(pts, shade(color, k))
    rnd = random.Random(11 if left else 12)
    # stones
    for _ in range(26):
        u, z = rnd.uniform(0.4, N - 0.4), rnd.uniform(-DEPTH + 0.3, -0.7)
        r = rnd.uniform(0.12, 0.3)
        c = "#A37650" if z > -2.5 else "#776A80"
        pts = [pt(u + r * math.cos(a) * 1.3, z + r * math.sin(a) * 0.8) for a in [j * math.pi / 4 for j in range(8)]]
        poly(pts, shade(c, k))
    # roots under trees near the edge
    for u0 in ([1.0, 4.6, 7.4, 10.6] if left else [2.2, 6.1, 9.2]):
        z, u = -0.3, u0
        pts = [pt(u, z)]
        for _ in range(5):
            u += rnd.uniform(-0.25, 0.25)
            z -= rnd.uniform(0.18, 0.3)
            pts.append(pt(u, z))
        emit('<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="none" stroke="{shade("#3A2418", k)}" stroke-width="2" stroke-linecap="round"/>')
    # one thin fibre cable running through the clay
    zc = -1.95
    cable = [pt(u, zc + 0.08 * math.sin(u * 1.7)) for u in [i * N / 30 for i in range(31)]]
    pts_s = " ".join(f"{a:.1f},{b:.1f}" for a, b in cable)
    emit(f'<polyline points="{pts_s}" fill="none" stroke="{shade(TEAL, k * 0.9)}" stroke-width="2.4" stroke-opacity="0.8"/>')
    emit(f'<polyline class="data" points="{pts_s}" pathLength="100" fill="none" stroke="#9FF0E6" stroke-width="3.4" stroke-linecap="round" style="animation-duration:{5 if left else 4.2}s"/>')
    for u in ([2.5, 6.5, 10.0] if left else [3.5, 8.0]):
        x, y = pt(u, zc + 0.08 * math.sin(u * 1.7))
        emit(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{SIGNAL}"/>')


earth_face(True)
earth_face(False)

# ================================================================ GROUND
ground_rect(0, 0, N, N, "#3A3880")
ROAD = "#24235A"
R1, R2 = (3.6, 4.2), (7.8, 8.4)
for a, b in (R1, R2):
    ground_rect(a, 0, b, N, ROAD)
    ground_rect(0, a, N, b, ROAD)
# sidewalks (thin lighter edges along roads)
SIDE = "#4A4790"
for a, b in (R1, R2):
    for (lo, hi) in ((0, R1[0]), (R1[1], R2[0]), (R2[1], N)):
        ground_rect(a - 0.12, lo, a, hi, SIDE); ground_rect(b, lo, b + 0.12, hi, SIDE)
        ground_rect(lo, a - 0.12, hi, a, SIDE); ground_rect(lo, b, hi, b + 0.12, SIDE)
# lane dashes
for a, b in (R1, R2):
    m = (a + b) / 2
    for t in range(24):
        s0, s1 = t * 0.5 + 0.1, t * 0.5 + 0.32
        if any(lo - 0.15 < s0 < hi + 0.05 for lo, hi in (R1, R2)):
            continue
        ground_rect(m - 0.025, s0, m + 0.025, s1, "#6F6AA8")
        ground_rect(s0, m - 0.025, s1, m + 0.025, "#6F6AA8")
# crosswalks at the four intersections
for a, b in (R1, R2):
    for c, d in (R1, R2):
        for i in range(4):
            u = c + 0.08 + i * 0.14
            ground_rect(u, a - 0.32, u + 0.07, a - 0.14, "#DCD8FF")
            ground_rect(a - 0.32, u, a - 0.14, u + 0.07, "#DCD8FF")
# blocks: lawns and plaza
ground_rect(0.3, 0.3, 3.45, 3.45, "#1E5F63")            # back park
ground_rect(4.35, 4.35, 7.65, 7.65, "#4A4790")          # plaza
ground_rect(4.75, 4.75, 7.25, 7.25, "#5B58A8")
ground_rect(8.55, 8.55, 11.7, 11.7, "#33316F")          # front lot
ground_ellipse(1.9, 1.7, 0.9, 0.65, "#2BA597")           # pond
ground_ellipse(1.9, 1.7, 0.6, 0.4, TEAL)
# parking lines in the front lot
for i in range(4):
    u = 9.0 + i * 0.6
    ground_rect(u, 8.7, u + 0.04, 9.6, "#6F6AA8")
# fountain basin on the plaza
ground_ellipse(6.0, 6.0, 0.75, 0.75, "#6F6AA8")
ground_ellipse(6.0, 6.0, 0.6, 0.6, "#2BA597")
for lx, ly in ((4.27, 2.2), (2.2, 4.27), (8.47, 6.0), (6.0, 8.47), (4.27, 10.2), (10.2, 4.27), (8.47, 10.4)):
    lamp_glow(lx, ly)

# ================================================================ OBJECTS (painter-sorted)
items = []


def add(key, fn):
    items.append((key, fn))


# back park: trees around the pond
for tx, ty, k in ((0.7, 0.7, "pine"), (3.0, 0.7, "round"), (0.7, 3.0, "round"), (3.1, 2.6, "pine"), (2.6, 3.1, "round"), (1.2, 2.9, "pine")):
    add(tx + ty, lambda tx=tx, ty=ty, k=k: tree(tx, ty, 0.4, k))
add(3.2, lambda: box(2.9, 0.4, 0.5, 0.18, 0.18, "#8A5A34", roof_inset=False))  # bench

# right-back block (x 4.2-7.8, y 0.3-3.6): row of houses + water tower
for i, (hx, col) in enumerate(((4.4, PAPER), (5.5, "#DCD8FF"), (6.6, "#FFE9A8"))):
    add(hx + 0.6 + i * 0.01, lambda hx=hx, col=col: (box(hx, 0.6, 0.9, 1.2, 1.0, col, windows=(1, 2, int(hx * 10)), door=True), gable(hx, 0.6, 0.9, 1.2, 1.0, 0.55, CORAL)))
add(6.9, lambda: (box(5.0, 2.3, 1.6, 1.0, 1.5, "#5B58A8", windows=(2, 3, 33), door=True)))
def water_tower():
    x, y = 7.0, 2.6
    for dx, dy in ((0, 0), (0.4, 0), (0, 0.4), (0.4, 0.4)):
        bx, by = p(x + dx, y + dy, 0)
        _, ty = p(x + dx, y + dy, 1.6)
        emit(f'<rect x="{bx - 1:.1f}" y="{ty:.1f}" width="2" height="{by - ty:.1f}" fill="#6F6AA8"/>')
    box(x - 0.1, y - 0.1, 0.6, 0.6, 0.7, "#8A5A34", z0=1.6, roof_inset=False)
    pyramid(x - 0.1, y - 0.1, 0.6, 0.6, 2.3, 0.35, "#6B4428")
add(9.8, water_tower)

# right corner block (x 8.4-11.7, y 0.3-3.6): campus clinic + campus hall
add(10.2, lambda: box(8.7, 0.5, 2.8, 2.0, 2.8, TEAL, windows=(3, 5, 41), door=True))
add(10.3, lambda: (box(9.6, 1.0, 0.8, 0.8, 0.4, "#2BA597", z0=2.8, roof_inset=False)))
add(12.6, lambda: (box(8.8, 2.75, 1.5, 0.75, 1.0, PAPER, windows=(1, 3, 42), door=True)))
add(12.9, lambda: tree(11.2, 3.2, 0.38))
add(13.0, lambda: tree(10.6, 3.3, 0.34, "pine"))

# left-back block (x 0.3-3.6, y 4.2-7.8): Leo's café + bakery + outdoor tables
def cafe():
    box(0.5, 4.5, 2.0, 1.6, 1.6, CORAL, windows=(1, 3, 51), door=True)
    for i in range(8):  # striped awning on the visible front (y = 6.1)
        xa = 0.5 + i * 0.25
        quad((xa, 6.1, 1.05), (xa + 0.25, 6.1, 1.05), (xa + 0.25, 6.35, 0.85), (xa, 6.35, 0.85), PAPER if i % 2 == 0 else CORAL)
add(5.6, cafe)
add(6.0, lambda: box(2.7, 4.6, 0.8, 1.2, 1.2, "#FFE9A8", windows=(1, 2, 52), door=True))
for tx in (0.8, 1.6):
    add(tx + 6.9, lambda tx=tx: (box(tx, 6.65, 0.35, 0.35, 0.3, PAPER, roof_inset=False), person(tx + 0.5, 6.8, LILAC)))
add(8.4, lambda: tree(3.2, 7.3, 0.36))

# plaza (x 4.2-7.8, y 4.2-7.8): feed screen, fountain jet, benches, people
def feed_screen():
    x0, y0, w, d, h = 4.6, 4.9, 1.8, 0.3, 2.6
    box(x0, y0, w, d, h, SIGNAL, roof_inset=False)
    yy = y0 + d
    for i, z in enumerate((1.95, 1.35, 0.75)):
        quad((x0 + 0.1, yy, z), (x0 + w - 0.1, yy, z), (x0 + w - 0.1, yy, z + 0.42), (x0 + 0.1, yy, z + 0.42), "#B8901A")
        quad((x0 + 0.1, yy, z), (x0 + w - 0.1, yy, z), (x0 + w - 0.1, yy, z + 0.42), (x0 + 0.1, yy, z + 0.42), NIGHT, f' class="row" style="animation-delay:{i * 1.2:.1f}s"')
        quad((x0 + 0.18, yy, z + 0.12), (x0 + 0.45, yy, z + 0.12), (x0 + 0.45, yy, z + 0.3), (x0 + 0.18, yy, z + 0.3), CORAL if i == 0 else SIGNAL)
        for j, (a, b) in enumerate(((0.55, 1.5), (0.55, 1.15))):  # headline lines
            zz = z + 0.27 - j * 0.12
            quad((x0 + a, yy, zz), (x0 + b, yy, zz), (x0 + b, yy, zz + 0.05), (x0 + a, yy, zz + 0.05), PAPER, ' fill-opacity="0.75"')
add(9.9, feed_screen)
def jet():
    bx, by = p(6.0, 6.0, 0)
    group("jet")
    emit(f'<path d="M{bx - 3:.1f} {by:.1f} Q{bx:.1f} {by - 34:.1f} {bx + 3:.1f} {by:.1f}" fill="{TEAL}"/>')
    emit(f'<ellipse cx="{bx:.1f}" cy="{by - 30:.1f}" rx="9" ry="5" fill="#9FF0E6"/>')
    end()
    for i, dx in enumerate((-7, 6, -2)):  # droplets
        emit(f'<circle class="drop" cx="{bx + dx:.1f}" cy="{by - 26:.1f}" r="2" fill="#9FF0E6" style="animation-delay:{i * 0.4:.1f}s;--dx:{dx * 0.8:.0f}px"/>')
add(12.0, jet)
add(12.4, lambda: box(7.0, 5.0, 0.18, 0.6, 0.18, "#8A5A34", roof_inset=False))
add(12.6, lambda: box(5.0, 7.1, 0.6, 0.18, 0.18, "#8A5A34", roof_inset=False))
for px, py, c in ((5.2, 6.6, CORAL), (6.9, 6.8, SIGNAL), (7.2, 5.5, PAPER), (5.4, 7.4, TEAL)):
    add(px + py, lambda px=px, py=py, c=c: person(px, py, c))
for tx, ty in ((4.5, 7.5), (7.5, 4.5)):
    add(tx + ty, lambda tx=tx, ty=ty: tree(tx, ty, 0.36))

# right-front block (x 8.4-11.7, y 4.2-7.8): apartment tower + low shop
def apartments():
    box(8.8, 4.5, 2.0, 2.0, 3.6, "#4A4790", windows=(5, 4, 61), door=True)
    for ax, ay in ((9.1, 4.8), (9.9, 5.4)):
        box(ax, ay, 0.4, 0.35, 0.25, "#8F8BC7", z0=3.6, roof_inset=False)
    bx, by = p(10.5, 4.8, 3.6)
    emit(f'<rect x="{bx - 1:.1f}" y="{by - 24:.1f}" width="2" height="24" fill="#8F8BC7"/>')
    emit(f'<circle class="beacon" cx="{bx:.1f}" cy="{by - 24:.1f}" r="2.8" fill="{CORAL}"/>')
add(14.0, apartments)
add(16.4, lambda: box(9.1, 6.9, 1.6, 0.7, 0.9, "#DCD8FF", windows=(1, 3, 62), door=True))
add(17.9, lambda: tree(11.3, 7.4, 0.34))

# left corner block (x 0.3-3.6, y 8.4-11.7): city hall with roof, columns and flag
def city_hall():
    box(0.6, 8.7, 2.6, 2.4, 2.2, LILAC, windows=(2, 4, 71), door=True)
    for i in range(5):  # columns on the visible front (y = 11.1)
        xa = 0.85 + i * 0.5
        quad((xa, 11.1, 0.05), (xa + 0.14, 11.1, 0.05), (xa + 0.14, 11.1, 1.0), (xa, 11.1, 1.0), PAPER)
    pyramid(0.6, 8.7, 2.6, 2.4, 2.2, 1.0, LILAC)
    flag(1.9, 9.9, 3.2, CORAL)
add(10.5, city_hall)
add(15.0, lambda: tree(3.3, 11.4, 0.36, "pine"))

# left-front block (x 4.2-7.8, y 8.4-11.7): Sam's newsstand, houses, bus stop
add(13.6, lambda: (box(4.6, 8.8, 1.0, 1.2, 1.1, PAPER, windows=(1, 2, 81), door=True), gable(4.6, 8.8, 1.0, 1.2, 1.1, 0.5, "#5B58A8")))
add(14.6, lambda: (box(6.0, 8.8, 1.2, 1.0, 1.3, "#FFE9A8", windows=(1, 2, 82), door=True), gable(6.0, 8.8, 1.2, 1.0, 1.3, 0.5, CORAL)))
def newsstand():
    box(5.0, 10.6, 0.9, 0.6, 0.7, TEAL_D, roof_inset=False)
    quad((5.0, 11.2, 0.7), (5.9, 11.2, 0.7), (5.9, 11.45, 0.55), (5.0, 11.45, 0.55), SIGNAL)
add(16.3, newsstand)
add(17.2, lambda: person(6.2, 11.3, LILAC))
def bus_stop():
    box(6.6, 10.9, 0.9, 0.25, 0.04, "#8F8BC7", z0=0.75, roof_inset=False)
    for dx in (0.05, 0.8):
        bx, by = p(6.6 + dx, 11.0, 0)
        _, ty = p(6.6 + dx, 11.0, 0.75)
        emit(f'<rect x="{bx - 1:.1f}" y="{ty:.1f}" width="2" height="{by - ty:.1f}" fill="#8F8BC7"/>')
add(17.6, bus_stop)

# front block (x 8.4-11.7, y 8.4-11.7): small shops + parked cars
add(18.2, lambda: box(10.0, 9.9, 1.4, 1.4, 1.0, "#5B58A8", windows=(1, 3, 91), door=True))
add(18.0, lambda: car(9.05, 9.0, False, CORAL))
add(18.2, lambda: car(9.65, 9.0, False, LILAC))
add(21.5, lambda: tree(9.0, 11.3, 0.34))

# street lamps
for lx, ly in ((4.27, 2.2), (2.2, 4.27), (8.47, 6.0), (6.0, 8.47), (4.27, 10.2), (10.2, 4.27), (8.47, 10.4)):
    add(lx + ly + 0.2, lambda lx=lx, ly=ly: lamp(lx, ly))

# ================================================================ TRAFFIC (moves; drawn under the buildings)
traffic = []
_cur = traffic
keyframes = []
CAR_L = 0.62


def screen_vec(along_x, dist):
    return (dist * COS * S, dist * SIN * S) if along_x else (-dist * COS * S, dist * SIN * S)


def moving(name, draw, along_x, dist, dur, delay, fade=True, alternate=False):
    dx, dy = screen_vec(along_x, dist)
    if fade:
        keyframes.append(f"@keyframes {name}{{0%{{transform:translate(0,0);opacity:0}}5%{{opacity:1}}95%{{opacity:1}}"
                         f"100%{{transform:translate({dx:.1f}px,{dy:.1f}px);opacity:0}}}}")
    else:
        keyframes.append(f"@keyframes {name}{{from{{transform:translate(0,0)}}to{{transform:translate({dx:.1f}px,{dy:.1f}px)}}}}")
    style = f"animation:{name} {dur}s linear -{delay}s infinite" + (" alternate" if alternate else "")
    group("mover", style)
    draw()
    end()


# (road axis is x?, lane coordinate, +1 toward the viewer / -1 away, color, seconds, start offset)
LANES = [(True, 3.88, 1, SIGNAL, 14, 2), (True, 3.62, -1, PAPER, 17, 9), (True, 8.08, 1, CORAL, 12, 7), (True, 7.82, -1, TEAL, 16, 3),
         (False, 3.88, 1, LILAC, 15, 11), (False, 3.62, -1, PAPER, 19, 4), (False, 8.08, 1, SIGNAL, 13, 5), (False, 7.82, -1, CORAL, 18, 12)]
# draw the far lanes first so near cars cover them at crossings
for i, (ax, lane, sign, col, dur, off) in enumerate(sorted(LANES, key=lambda l: l[1])):
    start = 0.0 if sign > 0 else N - CAR_L
    dist = (N - CAR_L) * sign
    if ax:
        moving(f"car{i}", lambda start=start, lane=lane, col=col, sign=sign: car(start, lane, True, col, rear=sign < 0), True, dist, dur, off)
    else:
        moving(f"car{i}", lambda start=start, lane=lane, col=col, sign=sign: car(lane, start, False, col, rear=sign < 0), False, dist, dur, off)

# people walking on sidewalks, back and forth
WALKERS = [(0.5, 4.27, True, 2.8, CORAL, 9, 0), (8.47, 4.5, False, 3.0, PAPER, 10, 4), (4.5, 8.47, True, 3.0, TEAL, 8, 2), (0.6, 7.7, True, 2.4, SIGNAL, 7, 5)]
for i, (wx, wy, ax, dist, col, dur, off) in enumerate(WALKERS):
    moving(f"walk{i}", lambda wx=wx, wy=wy, col=col: person(wx, wy, col, cls="step"), ax, dist, dur, off, fade=False, alternate=True)

_cur = objects
for _, fn in sorted(items, key=lambda it: it[0]):
    fn()

# ================================================================ BUBBLES
_cur = overlay
bubble(1.5, 5.3, 2.4, CORAL, "heart", 0)        # Leo's café
bubble(10.1, 1.5, 3.9, TEAL, "check", 2.6)        # campus clinic
bubble(1.9, 9.9, 3.9, LILAC, "dots", 5.2)         # city hall
bubble(9.8, 5.5, 4.4, SIGNAL, "bars", 1.3)        # apartments
bubble(5.45, 10.9, 1.4, PAPER, "news", 3.9)       # Sam's newsstand
# little reactions floating up from the plaza feed screen
for i, (u, col) in enumerate(((4.9, CORAL), (5.6, SIGNAL), (6.2, CORAL), (5.2, PAPER))):
    hx, hy = p(u, 5.2, 2.7)
    emit(f'<path class="like" style="animation-delay:{i * 1.1:.1f}s" d="M{hx:.1f} {hy + 5:.1f} C{hx - 8:.1f} {hy - 1:.1f} {hx - 5:.1f} {hy - 7:.1f} {hx:.1f} {hy - 3:.1f} C{hx + 5:.1f} {hy - 7:.1f} {hx + 8:.1f} {hy - 1:.1f} {hx:.1f} {hy + 5:.1f} Z" fill="{col}"/>')
    p(u, 5.2, 2.7 + 70 / S)

# ================================================================ WRITE
pad = 6
x0, y0, x1, y1 = bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad
W, H = x1 - x0, y1 - y0
body = "".join(ground_layer + traffic + objects + overlay)
# All motion lives here, so the SVG animates on its own (also inside <img>) and stops for reduced motion.
CSS = """
.w-off{animation:w-off 9s steps(1) infinite}
@keyframes w-off{0%,78%{opacity:1}80%,92%{opacity:0}94%,100%{opacity:1}}
.w-on{opacity:0;animation:w-on 11s steps(1) infinite}
@keyframes w-on{0%,60%{opacity:0}62%,90%{opacity:.9}92%,100%{opacity:0}}
.sway{transform-box:fill-box;transform-origin:50% 100%;animation:sway 5s ease-in-out infinite}
@keyframes sway{0%,100%{transform:rotate(-2deg)}50%{transform:rotate(2deg)}}
.glow,.halo{animation:glow 4s ease-in-out infinite}
@keyframes glow{0%,100%{opacity:.55}50%{opacity:1}}
.idle{animation:idle 2.4s ease-in-out infinite}
@keyframes idle{0%,100%{transform:translateY(0)}50%{transform:translateY(-1.5px)}}
.step{animation:step .5s ease-in-out infinite}
@keyframes step{0%,100%{transform:translateY(0)}50%{transform:translateY(-2.5px)}}
.flag{transform-box:fill-box;transform-origin:0 50%;animation:flag 1.6s ease-in-out infinite}
@keyframes flag{0%,100%{transform:skewY(0) scaleX(1)}50%{transform:skewY(10deg) scaleX(.85)}}
.bub{transform-box:fill-box;transform-origin:50% 100%;animation:bub 6.5s ease-in-out infinite}
@keyframes bub{0%{opacity:0;transform:translateY(8px) scale(.6)}8%{opacity:1;transform:translateY(0) scale(1.06)}12%{transform:scale(1)}
  45%{transform:translateY(-6px)}70%{opacity:1;transform:translateY(-2px)}82%,100%{opacity:0;transform:translateY(-14px) scale(.9)}}
.row{opacity:0;animation:row 3.6s steps(1) infinite}
@keyframes row{0%{opacity:1}33.3%,100%{opacity:0}}
.jet{transform-box:fill-box;transform-origin:50% 100%;animation:jet 1.4s ease-in-out infinite}
@keyframes jet{0%,100%{transform:scaleY(1)}50%{transform:scaleY(.82)}}
.drop{animation:drop 1.2s ease-in infinite}
@keyframes drop{0%{transform:translate(0,0);opacity:1}100%{transform:translate(var(--dx),22px);opacity:0}}
.beacon{animation:beacon 1.8s steps(1) infinite}
@keyframes beacon{0%{opacity:1}50%,100%{opacity:.15}}
.data{stroke-dasharray:2.5 22.5;animation:data 5s linear infinite}
@keyframes data{to{stroke-dashoffset:-100}}
.like{opacity:0;transform-box:fill-box;transform-origin:50% 50%;animation:like 4.4s ease-out infinite}
@keyframes like{0%{opacity:0;transform:translateY(0) scale(.5)}10%{opacity:1;transform:translateY(-6px) scale(1)}
  60%{opacity:.9}100%{opacity:0;transform:translateY(-62px) scale(.8)}}
""" + "\n".join(keyframes) + """
@media (prefers-reduced-motion:reduce){*{animation:none!important}.w-on,.like{opacity:0}.bub{opacity:1}}
"""
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.1f} {y0:.1f} {W:.1f} {H:.1f}" width="{W:.0f}" height="{H:.0f}">'
       f'<title>TownSquare: a small town on a block of earth</title><style>{CSS}</style>{body}</svg>')
open(sys.argv[1], "w").write(svg)
gx, gy = p(0, 0, 0)
print(f"size {W:.0f}x{H:.0f}  ground-top-vertex at y={gy - y0:.0f}  bytes {len(svg)}")
