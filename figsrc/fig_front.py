"""Front-matter figure: the roadmap of the seven parts and 24 chapters.

roadmap  each part is a row; chapters are pills (filled = main line, dashed = optional depth);
         amber badges mark the four capability checkpoints of the preface.
"""
from __future__ import annotations

from style import ACCENT, AMBER, FIG_W, INK, MUTED, T, ZH, arrow, canvas, figure, label, rbox, tint

PARTS = [
    (T("第一部分", "Part I"), T("场的语言", "Language"),
     [(1, T("一个问题", "Question"), 1), (2, T("场的几何", "Geometry"), 1)], None),
    (T("第二部分", "Part II"), T("电与磁", "E and B"),
     [(3, T("电荷与电势", "Charge, φ"), 1), (4, T("磁场与感应", "Induction"), 1),
      (5, T("物质中的场", "Matter"), 1), (6, T("界面", "Interfaces"), 1),
      (7, T("四个系统", "Systems"), 1)], 1),
    (T("第三部分", "Part III"), T("时间与表示", "Time, potentials"),
     [(8, T("快与慢", "Fast/slow"), 1), (9, T("势函数", "Potentials"), 1)], None),
    (T("第四部分", "Part IV"), T("核心场问题", "Core problems"),
     [(10, "TES", 1), (11, "TMS", 1), (12, T("涡流", "Eddy"), 0), (13, T("铁磁材料", "Iron B–H"), 0),
      (14, T("集总磁路", "Circuits"), 0), (15, T("磁体", "Magnets"), 0)], 2),
    (T("第五部分", "Part V"), T("有限元", "Finite elements"),
     [(16, T("标量有限元", "Scalar FEM"), 1), (17, T("矢量有限元", "Vector FEM"), 0)], 3),
    (T("第六部分", "Part VI"), T("应用", "Applications"),
     [(18, T("统一框架", "Framework"), 1), (19, T("MRI 场", "MRI fields"), 0), (20, T("线圈设计", "Coils"), 0),
      (21, T("神经响应", "Nerves"), 0), (22, T("射频与 SAR", "RF, SAR"), 0)], None),
    (T("第七部分", "Part VII"), T("综合与验证", "Synthesis"),
     [(23, T("综合项目", "Projects"), 1), (24, T("验证与决策", "Verification"), 1)], 4),
]


@figure("roadmap")
def roadmap():
    H = 4.65
    fig, ax = canvas(FIG_W, H, xlim=(0, 10.4), ylim=(0, H * 10.4 / FIG_W))
    top = H * 10.4 / FIG_W - 0.08
    rh, gap = 0.78, 0.25
    xl, wl = 0.05, 2.0
    x0, pw, pg = 2.3, 1.15, 0.1
    for i, (pnum, pname, chaps, check) in enumerate(PARTS):
        y = top - (i + 1) * rh - i * gap
        rbox(ax, xl, y, wl, rh, fc=tint(ACCENT, 0.88), ec="none", r=0.1)
        label(ax, xl + wl / 2, y + rh * 0.68, pnum, color="white", size=8.0)
        label(ax, xl + wl / 2, y + rh * 0.3, pname, color="white", size=8.8, weight="bold")
        for k, (n, title, main) in enumerate(chaps):
            x = x0 + k * (pw + pg)
            if main:
                rbox(ax, x, y, pw, rh, fc=tint(ACCENT, 0.13), ec=tint(ACCENT, 0.6), lw=0.9, r=0.1)
            else:
                rbox(ax, x, y, pw, rh, fc="white", ec=tint(MUTED, 0.75), lw=0.9, r=0.1, ls=(0, (3, 2)))
            label(ax, x + pw / 2, y + rh * 0.7, T(f"第 {n} 章", f"Ch. {n}"), size=7.6, color=ACCENT if main else MUTED)
            label(ax, x + pw / 2, y + rh * 0.3, title, size=8.0 if ZH else 7.5, color=INK if main else MUTED)
        if check:
            cx = 10.4 - 0.27
            ax.add_patch(__import__("matplotlib").patches.Circle((cx, y + rh / 2), 0.22, fc=AMBER, ec="none", zorder=5))
            label(ax, cx, y + rh / 2, str(check), color="white", size=9, weight="bold", z=6)
        if i < len(PARTS) - 1:
            arrow(ax, (xl + wl / 2, y - 0.02), (xl + wl / 2, y - gap + 0.02), color=ACCENT, lw=1.3, ms=8)
    # legend
    yl = 0.12
    rbox(ax, 0.1, yl, 0.55, 0.3, fc=tint(ACCENT, 0.13), ec=tint(ACCENT, 0.6), lw=0.9, r=0.06)
    label(ax, 0.75, yl + 0.15, T("主线：建议按顺序读", "main line: read in order"), size=8.2, ha="left")
    rbox(ax, 3.55, yl, 0.55, 0.3, fc="white", ec=tint(MUTED, 0.75), lw=0.9, r=0.06, ls=(0, (3, 2)))
    label(ax, 4.2, yl + 0.15, T("按需深入：依应用选读", "optional depth: by application"), size=8.2, ha="left", color=MUTED)
    ax.add_patch(__import__("matplotlib").patches.Circle((7.25, yl + 0.15), 0.17, fc=AMBER, ec="none"))
    label(ax, 7.25, yl + 0.15, "1", color="white", size=8, weight="bold")
    label(ax, 7.5, yl + 0.15, T("能力检查点（见正文）", "capability checkpoint"), size=8.2, ha="left")
    return fig
