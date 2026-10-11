"""Off-screen 3-D rendering for the book's figures (three.js in headless Chromium).

Python computes every piece of geometry (field lines, coils, surfaces); the browser
only shades and rasterises it.  Text is NOT drawn in 3-D: labelled points are
projected to screen coordinates and the labels are typeset afterwards by
matplotlib with the same fonts as every other figure.

Requirements: the Python package `playwright` with its Chromium build
(`python -m playwright install chromium`).  three.js (MIT licence) is vendored in
figsrc/three/.

Typical use inside a figure function::

    from render3d import Scene, render, compose
    s = Scene(width=900, height=560)
    s.camera(position=(5, -6, 4), target=(0, 0, 0))
    s.tube(points, radius=0.02, color=MAG, arrows=[0.5])
    s.label("B", (0, 0, 1.2))
    img, px = render(s)
    fig, ax = compose(img, px, s)
    put(ax, px, "B", r"$\\mathbf{B}$", color=MAG, dx=8)
    return fig
"""
from __future__ import annotations

import functools
import http.server
import json
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
THREE_DIR = HERE / "three"


class Scene:
    def __init__(self, width: int = 900, height: int = 560, background: str = "#ffffff"):
        self.d: dict = {"width": width, "height": height, "background": background,
                        "objects": [], "labels": {}, "camera": {"position": [5, -6, 4]}}

    # -- camera and lights -------------------------------------------------------
    def camera(self, position, target=(0, 0, 0), up=(0, 0, 1), fov=30.0, ortho=False, height=None):
        self.d["camera"] = {"position": list(map(float, position)), "target": list(map(float, target)),
                            "up": list(map(float, up)), "fov": fov, "ortho": ortho}
        if height is not None:
            self.d["camera"]["height"] = height
        return self

    def lights(self, ambient=0.55, key=0.65, fill=0.25, key_dir=(3, -4, 6), fill_dir=(-5, 2, 2)):
        self.d.update(ambient=ambient, key=key, fill=fill, key_dir=list(key_dir), fill_dir=list(fill_dir))
        return self

    # -- primitives --------------------------------------------------------------
    def _add(self, **o):
        from matplotlib.colors import to_hex
        for key in ("color", "edge_color"):
            if key in o and not isinstance(o[key], str):
                o[key] = to_hex(o[key])
        self.d["objects"].append(o)
        return self

    def tube(self, points, radius=0.02, color="#444444", arrows=(), closed=False, head_len=None,
             head_radius=None, **kw):
        pts = np.asarray(points, float).tolist()
        o = dict(type="tube", points=pts, radius=radius, color=color, arrows=list(arrows), closed=closed, **kw)
        if head_len is not None:
            o["head_len"] = head_len
        if head_radius is not None:
            o["head_radius"] = head_radius
        return self._add(**o)

    def arrow(self, start, end, radius=0.025, color="#222222", head_len=0.18, head_radius=0.07, **kw):
        return self._add(type="arrow", **{"from": list(map(float, start)), "to": list(map(float, end))},
                         radius=radius, color=color, head_len=head_len, head_radius=head_radius, **kw)

    def sphere(self, center, radius, color="#cccccc", **kw):
        return self._add(type="sphere", center=list(map(float, center)), radius=radius, color=color, **kw)

    def box(self, center, size, color="#cccccc", **kw):
        return self._add(type="box", center=list(map(float, center)), size=list(map(float, size)), color=color, **kw)

    def cylinder(self, start, end, radius, color="#cccccc", **kw):
        return self._add(type="cylinder", **{"from": list(map(float, start)), "to": list(map(float, end))},
                         radius=radius, color=color, **kw)

    def torus(self, center, radius, tube=0.03, normal=(0, 0, 1), color="#444444", **kw):
        return self._add(type="torus", center=list(map(float, center)), radius=radius, tube=tube,
                         normal=list(map(float, normal)), color=color, **kw)

    def mesh(self, vertices, faces, color="#cccccc", vertex_colors=None, **kw):
        o = dict(type="mesh", vertices=np.asarray(vertices, float).tolist(),
                 faces=np.asarray(faces, int).tolist(), color=color, **kw)
        if vertex_colors is not None:
            o["vertex_colors"] = np.asarray(vertex_colors, float)[:, :3].tolist()
        return self._add(**o)

    def surface(self, X, Y, Z, color="#cccccc", colors=None, **kw):
        """Structured grid surface; `colors` is an (ny, nx, 3|4) array in 0..1."""
        ny, nx = X.shape
        V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        idx = np.arange(nx * ny).reshape(ny, nx)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
        F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        vc = None if colors is None else np.asarray(colors).reshape(-1, np.asarray(colors).shape[-1])
        return self.mesh(V, F, color=color, vertex_colors=vc, **kw)

    def polyline(self, points, color="#666666", dashed=False, **kw):
        return self._add(type="polyline", points=np.asarray(points, float).tolist(), color=color, dashed=dashed, **kw)

    def label(self, name, point):
        self.d["labels"][name] = list(map(float, point))
        return self


# -- rendering -------------------------------------------------------------------
class _Server:
    def __init__(self):
        handler = functools.partial(_QuietHandler, directory=str(THREE_DIR))
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # silence request logging
        pass


_STATE: dict = {}


def _page(scale: int):
    if "pw" not in _STATE:
        from playwright.sync_api import sync_playwright
        _STATE["server"] = _Server()
        _STATE["pw"] = sync_playwright().start()
        _STATE["browser"] = _STATE["pw"].chromium.launch(
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    return _STATE["browser"].new_page(viewport={"width": 100, "height": 100}, device_scale_factor=scale)


def render(scene: Scene, scale: int = 3):
    """Render a scene; returns (RGB uint8 image, {label: (x_px, y_px)} in image pixels)."""
    from io import BytesIO
    from PIL import Image
    d = scene.d
    page = _page(scale)
    page.set_viewport_size({"width": d["width"], "height": d["height"]})
    page.goto(f"http://127.0.0.1:{_STATE['server'].port}/render.html")
    page.wait_for_function("window.READY === true", timeout=60000)
    page.evaluate("S => window.renderScene(S)", d)
    page.wait_for_function("window.DONE === true", timeout=120000)
    labels = page.evaluate("window.LABELS")
    png = page.screenshot(clip={"x": 0, "y": 0, "width": d["width"], "height": d["height"]})
    page.close()
    img = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    px = {k: (v[0] * scale, v[1] * scale) for k, v in labels.items()}
    return img, px


def trim(img, px, pad=12):
    """Crop white margins (keeps label coordinates consistent)."""
    nonwhite = np.any(img < 250, axis=2)
    ys, xs = np.nonzero(nonwhite)
    if len(xs) == 0:
        return img, px
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    return img[y0:y1, x0:x1], {k: (x - x0, y - y0) for k, (x, y) in px.items()}


def compose(img, px, width_in: float, extra_left=0.0, extra_right=0.0, extra_top=0.0, extra_bottom=0.0):
    """Place a rendered image in a matplotlib figure of the given width.

    Returns (fig, ax, to_data) where ax uses image-pixel coordinates (y down) so that
    labels can be put at the projected positions in `px`.  Extra margins (inches) leave
    room for labels outside the picture.
    """
    import matplotlib.pyplot as plt
    h_px, w_px = img.shape[:2]
    img_w_in = width_in - extra_left - extra_right
    img_h_in = img_w_in * h_px / w_px
    H = img_h_in + extra_top + extra_bottom
    fig = plt.figure(figsize=(width_in, H))
    ax = fig.add_axes([extra_left / width_in, extra_bottom / H, img_w_in / width_in, img_h_in / H])
    ax.imshow(img, interpolation="none")
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)
    ax.axis("off")
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def compose_row(images, width_in, gap_in=0.15, titles=None, title_colors=None, title_size=9.8, top_in=0.0):
    """Several rendered images side by side (scaled to a common height). Returns (fig, axes)."""
    import matplotlib.pyplot as plt
    aspects = [im.shape[1] / im.shape[0] for im in images]
    n = len(images)
    title_h = 0.28 if titles else 0.0
    h_in = (width_in - gap_in * (n - 1)) / sum(aspects)
    H = h_in + title_h + top_in
    fig = plt.figure(figsize=(width_in, H))
    axes = []
    x = 0.0
    for k, (im, asp) in enumerate(zip(images, aspects)):
        w = h_in * asp
        ax = fig.add_axes([x / width_in, 0, w / width_in, h_in / H])
        ax.imshow(im, interpolation="none")
        ax.set_xlim(0, im.shape[1]); ax.set_ylim(im.shape[0], 0); ax.axis("off")
        if titles:
            fig.text((x + w / 2) / width_in, (h_in + 0.06) / H, titles[k], ha="center", va="bottom",
                     fontsize=title_size, fontweight="bold",
                     color=(title_colors[k] if title_colors else "#1E2328"))
        axes.append(ax)
        x += w + gap_in
    return fig, axes


def put(ax, px, name, text, dx=0.0, dy=0.0, color="#1E2328", size=10.0, ha="center", va="center", halo=True, **kw):
    """Typeset a label next to a projected 3-D point (dx, dy in image pixels, y down)."""
    import matplotlib.patheffects as pe
    x, y = px[name]
    t = ax.text(x + dx, y + dy, text, color=color, fontsize=size, ha=ha, va=va, zorder=10, **kw)
    if halo:
        t.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
    return t


def put_leader(ax, px, name, text, dx, dy, color="#1E2328", size=10.0, ha=None, va="center", lw=0.8, gap=6, **kw):
    """Label placed at an offset with a thin leader line back to the projected point."""
    import matplotlib.patheffects as pe
    x, y = px[name]
    tx, ty = x + dx, y + dy
    ha = ha or ("left" if dx > 0 else "right" if dx < 0 else "center")
    ax.plot([x, tx - np.sign(dx) * gap], [y, ty], color=color, lw=lw, zorder=9, solid_capstyle="round")
    ax.plot([x], [y], "o", ms=2.6, color=color, zorder=9)
    t = ax.text(tx, ty, text, color=color, fontsize=size, ha=ha, va=va, zorder=10, **kw)
    t.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
    return t


def close():
    if "browser" in _STATE:
        _STATE["browser"].close()
        _STATE["pw"].stop()
        _STATE["server"].httpd.shutdown()
        _STATE.clear()
