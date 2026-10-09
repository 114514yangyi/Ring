"""Build the Chinese PDF report describing the defects of the self-collected 200 Hz dataset.

The report is written with matplotlib's PDF backend (no LaTeX / browser needed) and embeds the
figures produced by ``visualize_all_samples.py`` / ``visualize_lab_results.py``.

Usage::

    python -m writing_state.make_dataset_report --output outputs/lab_eval/RingLab_200Hz_report.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

PAGE_W, PAGE_H = 8.27, 11.69          # A4 portrait, inches
LEFT, RIGHT = 0.078, 0.078
TOP, BOTTOM = 0.055, 0.062
CONTENT_W = 1.0 - LEFT - RIGHT
BODY = 9.6
SMALL = 8.4
TINY = 7.6
ACCENT = "#1f4e79"
INK = "#1a1a1a"
MUTED = "#5b6770"
GREEN = "#1e8449"
AMBER = "#b7791f"
RED = "#c0392b"
LIGHT = "#eef3f8"


def configure_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


class Report:
    def __init__(self, path: Path, preview_dir: Path | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf = PdfPages(str(path))
        self.preview_dir = preview_dir
        if preview_dir is not None:
            preview_dir.mkdir(parents=True, exist_ok=True)
        self.fig = None
        self.y = 0.0
        self.page = 0
        self.heading_text = ""
        self.chapter = ""
        self._width_cache: dict[float, dict[str, float]] = {}

    # ---------------------------------------------------------------- pages
    def new_page(self, heading: str = "") -> None:
        self._close_page()
        self.page += 1
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.y = 1.0 - TOP
        if heading:
            self.chapter = heading
        self.heading_text = heading or self.chapter
        self._draw_header(self.heading_text)

    def _draw_header(self, heading: str) -> None:
        self.fig.text(LEFT, 1.0 - TOP + 0.022, heading, fontsize=14.5, color=ACCENT,
                      weight="bold", va="bottom")
        self.fig.add_artist(plt.Line2D([LEFT, 1 - RIGHT], [1.0 - TOP + 0.014] * 2,
                                       color=ACCENT, lw=1.1))
        self.y = 1.0 - TOP - 0.012

    def _close_page(self) -> None:
        if self.fig is None:
            return
        self.fig.text(LEFT, BOTTOM - 0.022, "自采 200 Hz 数据集 · 缺陷评估报告", fontsize=TINY,
                      color=MUTED, va="top")
        self.fig.text(1 - RIGHT, BOTTOM - 0.022, "第 %d 页" % self.page, fontsize=TINY,
                      color=MUTED, va="top", ha="right")
        self.fig.add_artist(plt.Line2D([LEFT, 1 - RIGHT], [BOTTOM - 0.014] * 2,
                                       color="#c9d3dd", lw=0.8))
        self.pdf.savefig(self.fig)
        if self.preview_dir is not None:
            self.fig.savefig(self.preview_dir / ("page_%02d.png" % self.page), dpi=110)
        plt.close(self.fig)
        self.fig = None

    def save(self) -> None:
        self._close_page()
        self.pdf.close()

    # ---------------------------------------------------------------- text
    def _width(self, char: str, size: float) -> float:
        cache = self._width_cache.setdefault(size, {})
        if char not in cache:
            text = self.fig.text(0, 0, char, fontsize=size)
            cache[char] = text.get_window_extent(self.fig.canvas.get_renderer()).width / self.fig.bbox.width
            text.remove()
        return cache[char]

    def measure(self, text: str, size: float) -> float:
        return sum(self._width(ch, size) for ch in text)

    def wrap(self, text: str, size: float, width: float) -> list[str]:
        lines: list[str] = []
        for paragraph in text.split("\n"):
            current, used = "", 0.0
            for char in paragraph:
                step = self._width(char, size)
                if used + step > width and current:
                    lines.append(current)
                    current, used = char, step
                else:
                    current += char
                    used += step
            lines.append(current)
        return lines

    def ensure(self, height: float) -> None:
        if self.y - height < BOTTOM:
            self.new_page()

    def text(self, content: str, size: float = BODY, color: str = INK, weight: str = "normal",
             indent: float = 0.0, gap: float = 1.45, space_after: float = 0.010,
             width: float | None = None) -> None:
        width = width if width is not None else CONTENT_W - indent
        lines = self.wrap(content, size, width)
        line_h = size * gap / 72.0 / PAGE_H
        self.ensure(len(lines) * line_h + space_after)
        for line in lines:
            self.fig.text(LEFT + indent, self.y, line, fontsize=size, color=color,
                          weight=weight, va="top")
            self.y -= line_h
        self.y -= space_after

    def bullets(self, items: Sequence[str], size: float = BODY, indent: float = 0.012,
                space_after: float = 0.008) -> None:
        for item in items:
            marker_w = 0.016
            lines = self.wrap(item, size, CONTENT_W - indent - marker_w)
            line_h = size * 1.45 / 72.0 / PAGE_H
            self.ensure(len(lines) * line_h + space_after)
            self.fig.text(LEFT + indent, self.y, "•", fontsize=size, color=ACCENT, va="top")
            for index, line in enumerate(lines):
                self.fig.text(LEFT + indent + marker_w, self.y, line, fontsize=size,
                              color=INK, va="top")
                self.y -= line_h
            self.y -= space_after

    def section(self, title: str) -> None:
        self.ensure(0.075)
        self.y -= 0.012
        self.fig.text(LEFT, self.y, title, fontsize=12, color=ACCENT, weight="bold", va="top")
        self.y -= 0.026
        self.fig.add_artist(plt.Line2D([LEFT, LEFT + 0.09], [self.y + 0.008] * 2,
                                       color="#9fb6cb", lw=2.0))
        self.y -= 0.006

    def callout(self, title: str, body: str, color: str = ACCENT, size: float = SMALL) -> None:
        lines = self.wrap(body, size, CONTENT_W - 0.045)
        box_h = 0.030 + len(lines) * size * 1.45 / 72.0 / PAGE_H
        self.ensure(box_h + 0.012)
        self.fig.patches.append(plt.Rectangle((LEFT, self.y - box_h), CONTENT_W, box_h,
                                              transform=self.fig.transFigure, facecolor=LIGHT,
                                              edgecolor="none", zorder=-1))
        self.fig.add_artist(plt.Line2D([LEFT, LEFT], [self.y, self.y - box_h], color=color, lw=2.6))
        self.fig.text(LEFT + 0.014, self.y - 0.010, title, fontsize=size + 0.6, color=color,
                      weight="bold", va="top")
        self.y -= 0.028
        for line in lines:
            self.fig.text(LEFT + 0.014, self.y, line, fontsize=size, color=INK, va="top")
            self.y -= size * 1.45 / 72.0 / PAGE_H
        self.y -= 0.012

    def table(self, header: Sequence[str], rows: Sequence[Sequence[str]],
              widths: Sequence[float], size: float = SMALL, align: Sequence[str] | None = None,
              colors: Sequence[Sequence[str]] | None = None, space_after: float = 0.014,
              row_colors: Sequence[str] | None = None) -> None:
        align = align or ["left"] * len(header)
        line_h = size * 1.42 / 72.0 / PAGE_H
        pad = 0.007
        header_h = line_h + 2 * pad

        def cell_line(text: str, column: int) -> list[str]:
            return self.wrap(text, size, CONTENT_W * widths[column] - 0.016)

        blocks = []
        for row in rows:
            wrapped = [cell_line(str(value), index) for index, value in enumerate(row)]
            blocks.append((wrapped, max(len(item) for item in wrapped) * line_h + 2 * pad))

        self.ensure(header_h + sum(block[1] for block in blocks) + space_after)
        x_positions, cursor = [], LEFT
        for width in widths:
            x_positions.append(cursor)
            cursor += CONTENT_W * width

        self.fig.patches.append(plt.Rectangle((LEFT, self.y - header_h), CONTENT_W, header_h,
                                              transform=self.fig.transFigure, facecolor="#dce6f1",
                                              edgecolor="none", zorder=-1))
        for column, label in enumerate(header):
            x = x_positions[column] + 0.008
            if align[column] == "center":
                x = x_positions[column] + CONTENT_W * widths[column] / 2
            self.fig.text(x, self.y - pad, label, fontsize=size, color=ACCENT, weight="bold",
                          va="top", ha="center" if align[column] == "center" else "left")
        self.y -= header_h

        for index, (wrapped, height) in enumerate(blocks):
            if row_colors is not None and index % 2 == 1:
                self.fig.patches.append(plt.Rectangle((LEFT, self.y - height), CONTENT_W, height,
                                                      transform=self.fig.transFigure,
                                                      facecolor="#f5f8fb", edgecolor="none",
                                                      zorder=-1))
            for column, lines in enumerate(wrapped):
                x = x_positions[column] + 0.008
                ha = "left"
                if align[column] == "center":
                    x = x_positions[column] + CONTENT_W * widths[column] / 2
                    ha = "center"
                for line_index, line in enumerate(lines):
                    color = INK
                    if colors is not None and column < len(colors[index]):
                        color = colors[index][column]
                    self.fig.text(x, self.y - pad - line_index * line_h, line, fontsize=size,
                                  color=color, va="top", ha=ha)
            self.y -= height
            self.fig.add_artist(plt.Line2D([LEFT, 1 - RIGHT], [self.y] * 2, color="#dbe3ea", lw=0.6))
        self.y -= space_after

    def image(self, path: Path, caption: str = "", width: float = CONTENT_W,
              max_height: float = 0.52) -> None:
        if not Path(path).exists():
            self.text("[缺少配图 %s]" % Path(path).name, size=SMALL, color=RED)
            return
        image = mpimg.imread(str(path))
        aspect = image.shape[0] / image.shape[1]
        height = width * PAGE_W * aspect / PAGE_H
        if height > max_height:
            height = max_height
            width = height * PAGE_H / (aspect * PAGE_W)
        caption_h = 0.0
        if caption:
            caption_h = len(self.wrap(caption, TINY, width)) * TINY * 1.4 / 72.0 / PAGE_H + 0.006
        self.ensure(height + caption_h + 0.016)
        x = LEFT + (CONTENT_W - width) / 2
        axes = self.fig.add_axes([x, self.y - height, width, height])
        axes.imshow(image)
        axes.axis("off")
        self.y -= height
        if caption:
            for line in self.wrap(caption, TINY, width):
                self.fig.text(x, self.y - 0.004, line, fontsize=TINY, color=MUTED, va="top")
                self.y -= TINY * 1.4 / 72.0 / PAGE_H
        self.y -= 0.014

    def stat_row(self, stats: Sequence[tuple[str, str, str]], height: float = 0.085) -> None:
        self.ensure(height + 0.016)
        gap = 0.012
        width = (CONTENT_W - gap * (len(stats) - 1)) / len(stats)
        for index, (value, label, color) in enumerate(stats):
            x = LEFT + index * (width + gap)
            self.fig.patches.append(plt.Rectangle((x, self.y - height), width, height,
                                                  transform=self.fig.transFigure,
                                                  facecolor="#f7f9fc", edgecolor="#d5e0ea",
                                                  lw=0.8, zorder=-1))
            self.fig.text(x + width / 2, self.y - 0.016, value, fontsize=17, color=color,
                          weight="bold", ha="center", va="top")
            for line in self.wrap(label, TINY, width - 0.03)[:2]:
                self.fig.text(x + width / 2, self.y - height + 0.020, line, fontsize=TINY,
                              color=MUTED, ha="center", va="bottom")
        self.y -= height + 0.018


def build(output: Path, figures: Path, preview: Path | None) -> None:
    configure_fonts()
    report = Report(output, preview)

    # ------------------------------------------------------------------ cover
    report.new_page()
    report.y = 0.93
    report.fig.patches.append(plt.Rectangle((0, 0.60), 1.0, 0.40, transform=report.fig.transFigure,
                                            facecolor="#12446e", edgecolor="none", zorder=-2))
    report.fig.text(LEFT, 0.895, "RING 项目 · 数据质量评估", fontsize=11, color="#bcd4ea")
    report.fig.text(LEFT, 0.845, "自采 200 Hz 戒指书写数据集", fontsize=25, color="white",
                    weight="bold")
    report.fig.text(LEFT, 0.800, "缺陷清单、量化影响与改进行动", fontsize=14, color="#dbe9f6")
    report.fig.text(LEFT, 0.755, "2026-10-07", fontsize=10.5, color="#a9c6e0")
    report.y = 0.545
    report.stat_row([("29%", "试次把整个字母完整录下来", GREEN),
                     ("42%", "试次缺失一半以上的书写过程", RED),
                     ("82% → 15%", "识别率随录制覆盖率的落差", AMBER)])
    report.section("本报告回答三个问题")
    report.bullets([
        "这份数据到底有哪些缺陷，每一条有多严重；",
        "每条缺陷对轨迹重建和字母识别的影响是多少；",
        "哪些缺陷能靠重新训练解决，哪些必须回到采集端解决。",
    ])
    report.section("结论先行")
    report.bullets([
        "数据格式、方向、IMU 流本身没有问题：1580/1580 无丢帧、时间戳单调、平均 199.8 Hz；"
        "存储方向与官方数据一致（原始方向识别率 82.5%，上下/左右翻转后只剩 5–12%）。",
        "最主要的缺陷是录制覆盖率：只有 29% 的试次完整记录了整个书写过程，42% 缺失一半以上；"
        "这直接把识别率从 82%（录全）拉到 15%（缺一半以上）。",
        "第二个缺陷是书写风格差异：26 个字母全部一笔连写，官方数据平均 1.55 个接触段/字符，"
        "导致 T、X、F、H、K 五个字母明显吃亏——但这一条靠重训已经基本解决。",
        "只靠清洗数据、重新训练无法解决覆盖率问题：在录制完整的样本上，现有模型已经达到 99.0%；"
        "而只用干净样本重训（训练量减少三分之二）反而让全量测试从 76.2% 掉到 52.8%。",
        "因此改进的优先级是：补采（提高覆盖率、增加样本量）> 采集流程质检 > 模型与训练策略。",
    ])

    # --------------------------------------------------------------- chapter 1
    report.new_page("一、数据构成与评测口径")
    report.section("1.1 评估依据")
    report.table(["项目", "说明"], [
        ["原始数据", "/home/huyang/data/trae_projects/raw/200hz（1580 试次）"],
        ["参照模型", "官方 clean 数据训练的字母分类器（用户无关划分）与本仓库轨迹前端"],
        ["覆盖率定义", "一个试次的落笔总时长中，落在戒指录制窗内的比例"],
        ["构造数据集", "lab200_v4（1516 样本，物理 mm + 尖峰对齐）、lab200_v5（覆盖 ≥0.85 的 504 样本）"],
        ["重训模型", "lab_gt_long（全部数据）、lab_v4clean_gt（只用录全样本，本报告新增实验）"],
    ], widths=[0.20, 0.80], size=SMALL, row_colors=["#f5f8fb"] * 5)
    report.section("1.2 数据规模")
    report.table(["项目", "内容"], [
        ["试次数", "1580（4 个条件 × 26 字母 × 2 位采集者）"],
        ["条件", "3×3 cm 抬腕 523 / 3×3 cm 靠腕 537 / 5×5 cm 抬腕 260 / 5×5 cm 靠腕 260"],
        ["采集者与设备", "fk：手机画布，1040 试次（4 个条件全有）；yjx：平板画布 1080×682，540 试次（只有两个 3×3 条件）"],
        ["文件构成", "每个试次一对 imu.csv（6 轴 200 Hz）＋ gt_raw.csv / gt_100hz.csv（平板笔迹）；_meta/manifest.csv 汇总 20 列元信息（条件、字母、时钟差 delta、帧数、落笔事件数等）"],
        ["形状统计", "每试次恰 1 个落笔事件；平板记录点数中位 262（p5–p95：114–427）"],
    ], widths=[0.17, 0.83], size=SMALL, row_colors=["#f5f8fb"] * 5)
    report.section("1.3 覆盖率：本报告最核心的指标")
    report.text("覆盖率 = 该试次的落笔时长里，有多少比例落在戒指的录制窗之内。"
                "覆盖率 1.0 表示写字的全过程都被录到了；0.5 表示只录到一半；"
                "0.1 表示戒指几乎什么都没录到，但平板上仍然留下了一个完整的字母。"
                "计算方式：把每个落笔段的起止时间减去该试次的对齐偏移 delta，再与 IMU 录制窗求交集，"
                "交集总时长除以落笔总时长。")
    report.callout("为什么它决定上限",
                   "戒指只能看到它录制的那一段时间里手的运动。如果录制窗只盖住字母的后半段，"
                   "那么模型看到的信号里根本不存在前半段的信息——无论怎样训练、怎样换模型，都不可能把没录到的东西恢复出来。",
                   color=RED)
    report.section("1.4 评测参照")
    report.text("为了让结论不依赖本仓库自己训练的模型，本报告大量使用“官方 clean 数据训练的字母分类器”"
                "作为裁判：它对书写方向、字形风格很敏感，因此既能用来判断方向是否正确，"
                "也能量化“录制截断”对识别的影响。此外也使用了在本数据上重训的模型（lab_gt_long 等）"
                "来回答“重训能不能解决”这一问题。")

    # --------------------------------------------------------------- chapter 2
    report.new_page("二、缺陷总览")
    report.text("下表按严重度排序。严重度依据“受影响的样本比例 × 对最终指标的影响”综合判断。")
    report.table(["#", "缺陷", "受影响比例", "严重度", "能否靠重训解决"], [
        ["1", "录制覆盖率不足：录制窗没有覆盖整个书写过程", "42% 的试次缺失一半以上", "高", "不能（信息未录到）"],
        ["2", "建集后退化样本：按录制窗裁剪后只剩下一条线段", "176/1516（11.6%）", "高", "不能（与缺陷 1 同源）"],
        ["3", "书写风格与笔顺差异：全部一笔连写", "T/X/F/H/K 五个字母", "中", "能（已验证）"],
        ["4", "坐标系与设备差异：CSS 像素、两种画布尺度、两台设备", "全部样本（可标定）", "低", "不需要"],
        ["5", "设备时钟偏移：试次间 delta 从 1 秒到 12.9 小时不等", "全部样本（可逐试次校正）", "低", "不需要"],
        ["6", "采集分布不均：第二位采集者只有两个条件", "540/1580 试次", "低", "需补采"],
    ], widths=[0.045, 0.395, 0.2, 0.10, 0.26], size=SMALL, row_colors=["#f5f8fb"] * 6,
        colors=[[INK, INK, INK, RED, MUTED],
                [INK, INK, INK, RED, MUTED],
                [INK, INK, INK, AMBER, GREEN],
                [INK, INK, INK, GREEN, MUTED],
                [INK, INK, INK, GREEN, MUTED],
                [INK, INK, INK, GREEN, MUTED]])

    # --------------------------------------------------------------- chapter 3
    report.new_page("三、缺陷一：录制覆盖率不足")
    report.text("这是整份数据里最严重、也最值得先修的一条。把 1580 个试次按覆盖率分成三档：")
    report.table(["分档", "样本数", "占比", "含义"], [
        ["录全（覆盖率 ≥ 0.9）", "464", "29%", "字写过程的 90% 以上都在录制窗内"],
        ["缺一点（0.6 – 0.9）", "447", "28%", "少了开头或结尾的一小段"],
        ["缺一半以上（< 0.6）", "669", "42%", "录制窗只盖到字母的一小部分"],
    ], widths=[0.34, 0.12, 0.10, 0.44], size=SMALL,
        colors=[[INK, INK, GREEN, INK], [INK, INK, AMBER, INK], [INK, INK, RED, INK]])
    report.section("3.1 逐字母分布")
    report.image(figures / "coverage_bands_per_letter.png",
                 "图 1：每个字母的样本按覆盖率分档的占比。绿色 = 录全，黄色 = 缺一点，红色 = 缺一半以上。"
                 "最差的是 B 与 K（各只有 11.7% 的样本录全），最好是 Z（48.3%）、V（45.0%）、X（42.6%）。",
                 max_height=0.30)
    report.section("3.2 全部 1580 个样本")
    report.image(figures / "all_samples_overview.png",
                 "图 2：全部样本的缩略总览（每行一个字母，按覆盖率从好到差排列）。"
                 "绿色部分（左侧）是完整的字母，越往右红色越多，代表录到的笔画越不完整。",
                 max_height=0.34)

    # --------------------------------------------------------------- chapter 3b
    report.new_page("三、缺陷一（续）：对识别性能的影响")
    report.text("把“被录制窗截断后的真值”直接喂给官方 clean 数据训练的字母分类器（n=1516），"
                "可以干净地测出覆盖率与识别率的因果关系：")
    report.table(["测试输入", "样本数", "Top-1 识别率"], [
        ["覆盖率 ≥ 0.9（录全）", "460", "82.2%"],
        ["覆盖率 0.6 – 0.9（缺一点）", "447", "52.1%"],
        ["覆盖率 < 0.6（缺一半以上）", "609", "15.4%"],
        ["未截断的原始平板轨迹（参照上界）", "1580", "82.5%"],
    ], widths=[0.50, 0.16, 0.34], size=SMALL, align=["left", "center", "center"],
        colors=[[INK, INK, GREEN], [INK, INK, AMBER], [INK, INK, RED], [INK, INK, ACCENT]])
    report.callout("结论",
                   "录全时，识别率（82.2%）与“完全没有截断”的上界（82.5%）几乎持平——说明模型和字形本身都没有问题；"
                   "一旦录制窗切掉一半字母，识别率掉到 15%，此时模型面对的其实是“输入与标签不对应”的样本。",
                   color=ACCENT)
    report.section("3.3 逐字母视角：录全比例与最终识别率高度相关")
    report.table(["字母", "录全占比", "重训后识别率", "字母", "录全占比", "重训后识别率"], [
        ["B", "11.7%", "18%", "V", "45.0%", "93%"],
        ["K", "11.7%", "44%", "X", "42.6%", "100%"],
        ["G", "21.3%", "39%", "Y", "41.0%", "80%"],
        ["L", "21.7%", "44%", "R", "40.3%", "86%"],
        ["N", "26.7%", "45%", "Z", "48.3%", "100%"],
    ], widths=[0.10, 0.17, 0.23, 0.10, 0.17, 0.23], size=SMALL,
        align=["center", "center", "center", "center", "center", "center"])
    report.text("26 个字母上，“录全占比”与“识别率”的相关系数 r = 0.674，说明覆盖率是当前识别率的"
                "主要瓶颈；录得越完整的字母，识别率越高。")
    report.image(figures / "data_quality.png",
                 "图 3：覆盖率的整体分布（左）与逐字母平均覆盖率（右）。", max_height=0.22)

    # --------------------------------------------------------------- chapter 4
    report.new_page("四、缺陷二：建集后退化样本（与缺陷一同源）")
    report.text("在构建训练集（lab200_v4）时，平板轨迹会被裁剪到戒指的录制窗内。"
                "当录制窗只覆盖字母的一小段时，样本就会退化成一条近乎直线的线段："
                "排除字母 I 之后，共有 176/1516（11.6%）的样本落在这一档（长宽比 < 0.12）。")
    report.image(figures / "build_truncation_B.png",
                 "图 4：同一批字母 B 样本的两行对比。上排（绿）是平板自己的记录——笔形完整；"
                 "下排（红）是建集裁剪后的轨迹——只剩一条线段。可见截断发生在处理流程里，而不是采集设备上。",
                 max_height=0.235)
    report.callout("重要澄清：这不是平板的问题",
                   "逐条核对原始数据后确认，这些样本的平板记录是完整的——例如 00012_b_fk_1 的平板原始轨迹有 569 个点、"
                   "字形完整（宽高比 0.46），但构建后的轨迹只剩 110 个点。也就是说，退化发生在“按录制窗裁剪”这一步，"
                   "与缺陷一完全同源；修好录制覆盖率，这一条会自动消失。",
                   color=ACCENT)
    report.text("顺带说明另一件事：字母 B 在“完整平板真值”上有 6/60 个样本被官方模型判成 D，"
                "这 6 个样本的轨迹都是完整的（300–486 个点），属于分类器对这种连笔 b 的风格混淆，"
                "而不是方向或镜像问题——全库只有 19 个样本“左右翻转后反而变好”，且集中在 O/X/K/I 这类近似对称的字母上。")

    # --------------------------------------------------------------- chapter 5
    report.new_page("五、缺陷三：书写风格与笔顺差异")
    report.text("本数据集的 26 个字母全部一笔连写：1580/1580 个试次恰好只有 1 个落笔事件。"
                "而官方 clean 数据集的字母平均包含 1.55 个接触段（多笔画字母 2–3 段）。"
                "这本身不是“错误”，但它会让“直接拿官方模型来用”这件事失败，并且影响 T、X、F、H、K 五个字母。")
    report.image(figures / "all_samples/T.png",
                 "图 5：字母 T 的全部 61 个样本。所有样本都是一笔画成（先横后竖、中间带一个圈），"
                 "与官方数据的“横、竖两笔”拓扑完全不同，因此官方模型会把它们判成 E。", max_height=0.36)
    report.table(["字母", "官方模型（未重训）", "自采数据重训后", "说明"], [
        ["T", "0%（判成 E）", "86%", "一笔连写 vs 官方两笔"],
        ["X", "0%（判成 Y/K）", "100%", "一笔连写 vs 官方两笔"],
        ["F", "9%", "86%", "连笔带圈，与官方字形差异大"],
        ["H", "48%", "57%", "连笔写法，提升有限"],
        ["K", "55%", "44%", "与其他字母（如 R）拉不开距离"],
        ["其余 21 个字母", "≥ 95%", "≥ 76%（多数 ≥ 90%）", "风格一致，无系统性问题"],
    ], widths=[0.16, 0.26, 0.24, 0.34], size=SMALL, row_colors=["#f5f8fb"] * 6,
        colors=[[INK, RED, GREEN, INK], [INK, RED, GREEN, INK], [INK, RED, GREEN, INK],
                [INK, AMBER, AMBER, INK], [INK, AMBER, AMBER, INK], [INK, GREEN, GREEN, INK]])
    report.callout("结论：这一条可以靠重训解决",
                   "用自采数据重训后，T 从 0% 升到 86%、X 从 0% 升到 100%、F 从 9% 升到 86%。"
                   "说明模型完全有能力学习这套书写风格，前提是训练数据里要有足够的同风格样本。",
                   color=GREEN)

    # --------------------------------------------------------------- chapter 6
    report.new_page("六、缺陷四：坐标系、单位与设备差异")
    report.text("平板记录的是 CSS 像素坐标（原点在左上角、y 轴向下），不是物理单位；"
                "两位采集者使用的设备画布也不同。这些差异需要标定，但都已经量化并验证可以处理。")
    report.table(["项目", "实际情况", "处理方式"], [
        ["坐标单位", "CSS 像素（非物理单位），y 轴向下", "按已知书写框（3×3 / 5×5 cm）标定 mm/像素，再除以 240 mm 对齐官方尺度"],
        ["设备尺度", "fk 0.176409 mm/px；yjx 0.250143 mm/px（相差 1.42 倍）", "逐设备标定；同一设备的 3 cm / 5 cm 两组数据自洽"],
        ["画布尺寸", "fk 出现 3 种（894×390、956×370、956×390）；yjx 固定 1080×682", "实测同一字母的像素尺寸跨画布一致（如 O：160×181 vs 171×180 px），说明画布变化未改变书写比例"],
        ["纵横比", "官方按 240×169.5 mm 各向异性归一化，本数据按各向同性标定", "把 y 拉伸 1.416 倍后识别率只差 0.8 个百分点，影响可忽略"],
        ["存储方向", "与官方一致（屏幕坐标 y 向下）", "绘图时需 invert_yaxis()，与官方可视化代码一致"],
    ], widths=[0.15, 0.45, 0.40], size=SMALL, row_colors=["#f5f8fb"] * 5)
    report.callout("关于“轨迹看起来上下颠倒”的澄清",
                   "这是绘图约定问题，不是数据问题：用官方 clean 数据训练的模型判原始方向，识别率 82.5%；"
                   "一旦上下翻转只剩 11.9%、左右翻转 6.6%、旋转 180° 只剩 5.1%。数据里存的方向是正确的，"
                   "画图时把 y 轴翻过来即可（官方可视化代码一直这么做）。",
                   color=ACCENT)
    report.image(figures / "orientation_explained.png",
                 "图 6：同一批真值轨迹的两种画法。上排是绘图工具默认坐标轴（看起来颠倒），"
                 "下排是按采集约定翻转 y 轴之后的结果。数据本身没有任何改动。", max_height=0.20)

    # --------------------------------------------------------------- chapter 7
    report.new_page("七、缺陷五：设备时钟偏移")
    report.text("戒指与平板各自计时，两者之间在每次录制时都有一个不同的固定偏移（manifest 里的 delta 字段）。"
                "这个偏移不是常数，必须逐试次校正：")
    report.table(["指标", "数值", "含义"], [
        ["delta 的分布", "p5 = 1.4 s，中位 4.6 s，p95 = 857 s", "大多数试次在几秒量级"],
        ["delta 的极值", "最大 46461 s（约 12.9 小时）", "存在跨天级别的偏移，绝不能用一个全局常数对齐"],
        ["received_at 抖动", "相邻采样间隔的抖动标准差约 6.6 ms（最大 9.6 ms）", "可用于粗对齐，不能用于样本级精确对齐"],
        ["IMU 内部时间戳", "1580/1580 单调递增，平均 199.80 Hz（p5–p95：199.8–200.1）", "设备自身计时可靠，对齐应基于它"],
    ], widths=[0.22, 0.42, 0.36], size=SMALL, row_colors=["#f5f8fb"] * 4)
    report.callout("处理约定（已落地）",
                   "逐试次使用 manifest 的 delta 做粗对齐，再用论文所述的“尖峰对齐”——把戒指的运动能量包络与平板的笔尖速度尖峰"
                   "（落笔/抬笔瞬间）做相关，粗搜 0.02 s、细搜 0.005 s。测试集上逐试次最优残余位移中位约 1 帧。"
                   "不要使用全局统一位移，也不要用 received_at 替代设备时间戳。",
                   color=ACCENT)

    # --------------------------------------------------------------- chapter 8
    report.section("八、缺陷六：采集分布不均")
    report.table(["采集者", "设备/画布", "试次数", "覆盖的条件"], [
        ["fk", "手机画布（894×390 / 956×370 / 956×390）", "1040", "3×3 与 5×5，抬腕与靠腕共 4 个条件"],
        ["yjx", "平板画布 1080×682", "540", "只有 3×3 的两个条件"],
    ], widths=[0.13, 0.42, 0.14, 0.31], size=SMALL, row_colors=["#f5f8fb"] * 2)
    report.text("影响：无法做“用户 × 条件”的完整交叉评估，跨用户泛化实验的结论会受采集者分布影响"
                "（此前测得的“留一人测试”误差 0.38，就明显高于同用户划分的 0.19–0.26）。")
    report.section("九、未发现缺陷的项：这些是好的")
    report.table(["检查项", "结果", "结论"], [
        ["平板轨迹内部自洽", "419,919 个步长中仅 1 处 |步长 − dx/dy| 不一致（6.02 px），其余偏差 0.0000", "通过"],
        ["时间戳单调性", "7 处首点时间戳重复（同一毫秒的两个事件）", "无影响"],
        ["IMU 丢帧", "1580/1580 个文件 frame_id 步进恒定，无跳变", "通过"],
        ["IMU 采样率", "中位 199.80 Hz，p5–p95 为 199.8–200.1 Hz", "通过"],
        ["文件完整性", "1580 个试次的 IMU / 平板文件全部存在，无缺失", "通过"],
    ], widths=[0.20, 0.55, 0.25], size=SMALL, row_colors=["#f5f8fb"] * 5,
        colors=[[INK, INK, GREEN]] * 5)

    # --------------------------------------------------------------- chapter 10
    report.new_page("十、重训能解决哪一半？")
    report.text("为了直接回答“能不能在新的数据上重新训练来解决”，我们做了对照实验："
                "在同一数据划分、同一批测试样本下，只更换训练数据——")
    report.bullets([
        "A 组：现有模型 lab_gt_long，用全部 1061 条训练样本训练；",
        "B 组：只保留“录制完整（覆盖率 ≥ 0.85）”的 349 条训练样本重训。",
    ])
    report.table(["测试样本", "A：全部数据训练（现有）", "B：只用干净数据重训"], [
        ["录制完整的样本（n = 100）", "99.0%", "97.0%"],
        ["全部测试样本（n = 303）", "76.2%", "52.8%"],
    ], widths=[0.40, 0.30, 0.30], size=SMALL, align=["left", "center", "center"],
        colors=[[INK, GREEN, GREEN], [INK, GREEN, RED]])
    report.callout("两条结论",
                   "① 在录制完整的样本上，现有模型已经有 99.0% 的识别率，此前最弱的 B/K/G/L/N 全部满分——"
                   "差的不是模型，而是三分之二被截断的测试样本。"
                   "② 只留干净样本重训并没有变好：训练量减少三分之二后，全量测试从 76.2% 掉到 52.8%，"
                   "干净测试集也只从 99.0% 变成 97.0%（差异在噪声范围内）。",
                   color=ACCENT)
    report.text("因此正确的路线是：把采集覆盖率修好、把样本量做上去，再在新旧合并的数据上重训，"
                "而不是靠清洗数据重训。")
    report.section("仍然存在的真实差距")
    report.table(["指标", "本数据集（录全子集）", "论文（WritingRing）"], [
        ["轨迹重建归一化误差", "0.195", "0.073"],
        ["轨迹重建 mm 误差", "6.7 mm", "1.64 mm"],
        ["字母识别（端到端，全量测试）", "51.2%", "88.7%（Google IME，协议不同）"],
    ], widths=[0.42, 0.29, 0.29], size=SMALL, align=["left", "center", "center"])
    report.text("即便只看录制完整的样本，轨迹重建精度仍与论文有约 2.7 倍的差距。"
                "这部分不能只归因于覆盖率，还需要更多数据、以及设备佩戴与标定的进一步对齐，是本项目后续要攻的方向。")

    # --------------------------------------------------------------- chapter 11
    report.new_page("十一、行动建议")
    report.section("11.1 采集端（优先级最高）")
    report.bullets([
        "先起录、再落笔；抬笔后再停录——把录制窗开在书写动作两侧各留 0.5 s 以上余量，目标覆盖率 ≥ 0.95。",
        "每次书写前留 1–2 s 静止段，用于时钟对齐与质检，也便于事后确认录制窗是否覆盖完整。",
        "每个字母每个条件把样本量翻倍（当前每字母每条件约 3–4 条），并让第二位采集者补齐 5×5 条件。",
        "统一设备与画布尺寸；如无法统一，至少保证同一会话内画布尺寸不变。",
        "采集当场做质检：覆盖率 + 字形完整性（记录点数、包围盒长宽比）两条门限，不合格的当场重采。",
    ])
    report.section("11.2 处理端")
    report.bullets([
        "对齐：逐试次使用 manifest 的 delta 做粗对齐，再用尖峰对齐精调；禁用 received_at 做样本级对齐。",
        "绘图：统一使用 invert_yaxis()，与官方可视化约定保持一致。",
        "训练：保留全部样本（含被截断的），不要为了“干净”而丢数据；覆盖率过滤只用于评测与误差分析。",
        "报告指标时同时给出“录全子集”和“全量测试集”两个数字，避免把采集问题误读成模型问题。",
    ])
    report.section("11.3 汇报口径")
    report.bullets([
        "当前可直接汇报：真值轨迹识别 76.2%（全量）/ 99.0%（录全子集）；端到端识别 51.2%。",
        "与论文 88.7% 不可直接比较：协议不同（论文用 Google IME 做最终解码，且官方数据录制完整、样本量数千）。",
        "轨迹重建误差应同时给出归一化值与 mm 值，并注明测试样本的覆盖率分布。",
    ])

    # --------------------------------------------------------------- appendix
    report.new_page("附录 A：文件与脚本清单")
    report.table(["路径", "说明"], [
        ["raw/200hz/_meta/manifest.csv", "1580 试次 × 20 列元信息（条件、字母、delta、帧数、落笔事件数等）"],
        ["raw/200hz/*_imu.csv", "6 轴 IMU 200 Hz：timestamp、received_at、frame_id、加速度、角速度、四元数等 23 列"],
        ["raw/200hz/*_gt_raw.csv", "平板原始笔迹：timestamp、x、y、dx、dy、width、height、event_type 等"],
        ["datasets/RingLab/coverage_all.csv", "逐试次覆盖率表（本报告核心依据）"],
        ["datasets/RingLab/lab200_v4 / v5", "构建后的训练集（1516 / 504 样本）"],
        ["outputs/lab_eval/", "本报告全部配图与统计表（all_samples/ 下为逐字母样本表）"],
        ["models_character/lab_v4clean_gt/", "“只用干净样本重训”实验的分类器与报告"],
    ], widths=[0.34, 0.66], size=SMALL, row_colors=["#f5f8fb"] * 7)
    report.section("附录 B：复现命令")
    for command in [
        "# 逐字母样本表 + 覆盖总分档图 + 汇总表",
        "python -m writing_state.visualize_all_samples --output-dir outputs/lab_eval",
        "",
        "# 轨迹/识别配图（真值 vs 重建、识别样例、混淆矩阵）",
        "python -m writing_state.visualize_lab_results --data-root datasets/RingLab/lab200_v4 \\",
        "    --trajectory-checkpoint models_lab/traj_f_aux1/paper_trajectory.pt \\",
        "    --character-report models_character/lab_both_f/character_report.json \\",
        "    --output-dir outputs/lab_eval",
        "",
        "# 方向核对（官方分类器判四种坐标约定）",
        "python outputs/lab_eval/orientation_check.py",
    ]:
        report.text(command if command else " ", size=7.4, color="#25415c" if command.startswith("python") else MUTED,
                    gap=1.25, space_after=0.001, indent=0.006)

    report.save()
    print("wrote", output)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the dataset defect report PDF")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/lab_eval/RingLab_200Hz_dataset_report.pdf"))
    parser.add_argument("--figures", type=Path, default=Path("outputs/lab_eval"))
    parser.add_argument("--preview", type=Path, default=None,
                        help="optional directory for per-page PNG previews")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    build(args.output, args.figures, args.preview)


if __name__ == "__main__":
    main()
