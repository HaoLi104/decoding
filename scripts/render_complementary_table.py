"""
生成「通用大模型 vs 领域小模型」互补性分析表，便于插入 PPT。

输出：
  figures/complementary_analysis_table.png  （推荐：插入 PPT 为图片）
  figures/complementary_analysis_table.svg  （矢量，部分 PPT 版本可插入）

另可在终端打印 TSV，复制到 Excel 再全选复制到 PPT 可得到可编辑表格。

用法：
  /opt/anaconda3/bin/python3 scripts/render_complementary_table.py
  /opt/anaconda3/bin/python3 scripts/render_complementary_table.py --tsv   # 只打印 TSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"


def build_rows():
    # 图片导出：纯中文，避免 matplotlib 缺字形；要 ✅❌⚠️ 请用 TSV 或 HTML 复制到 PPT
    return [
        ["维度", "通用大模型（32B）", "领域小模型（3B 微调）"],
        ["复杂推理与任务理解", "强", "弱"],
        ["语言生成流利度", "强", "中等"],
        ["领域事实与规范一致性", "不足", "强"],
        ["分布外泛化", "强", "弱"],
        ["推理速度（单卡）", "慢（约 27 t/s）", "快（约 200 t/s）"],
    ]


def build_rows_emoji():
    """供 TSV / HTML：与原始需求一致的 ✅ ❌ ⚠️"""
    return [
        ["维度", "通用大模型（32B）", "领域小模型（3B 微调）"],
        ["复杂推理与任务理解", "✅ 强", "❌ 弱"],
        ["语言生成流利度", "✅ 强", "⚠️ 中等"],
        ["领域事实与规范一致性", "❌ 不足", "✅ 强"],
        ["分布外泛化", "✅ 强", "❌ 弱"],
        ["推理速度（单卡）", "❌ 慢（~27 t/s）", "✅ 快（~200 t/s）"],
    ]


def print_tsv(rows):
    for r in rows:
        print("\t".join(r))


def render_figure(rows, png_path: Path, svg_path: Path | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 中文 + Emoji：macOS 上 Apple Color Emoji 可补齐 ✅❌⚠️
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Apple Color Emoji",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        loc="center",
        cellLoc="center",
        colWidths=[0.34, 0.33, 0.33],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.05, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#1f3a68")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f8f9fb" if row % 2 else "#ffffff")

    fig.suptitle(
        "互补性分析：通用大模型 vs 领域小模型",
        fontsize=14,
        fontweight="bold",
        color="#1f3a68",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    if svg_path:
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_html(path: Path, rows_emoji):
    """浏览器打开后全选表格，复制到 Word/PPT 往往保留表格结构。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in r) + "</tr>"
        for r in rows_emoji[1:]
    )
    thead = "<tr>" + "".join(f"<th>{c}</th>" for c in rows_emoji[0]) + "</tr>"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<style>
  body {{ font-family: "PingFang SC", "Microsoft YaHei", sans-serif; padding: 24px; }}
  h2 {{ color: #1f3a68; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 720px; }}
  th, td {{ border: 1px solid #333; padding: 10px 12px; text-align: center; }}
  thead th {{ background: #1f3a68; color: #fff; font-weight: bold; }}
  tbody tr:nth-child(even) {{ background: #f8f9fb; }}
</style>
</head>
<body>
<h2>互补性分析：通用大模型 vs 领域小模型</h2>
<table>
<thead>{thead}</thead>
<tbody>{tbody}</tbody>
</table>
<p style="color:#666;font-size:12px;">提示：拖选表格 Cmd+C → 粘贴到 Excel 或 Word → 再复制到 PPT，可得到可编辑表格。</p>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", action="store_true", help="仅输出 TSV 到 stdout")
    args = parser.parse_args()

    rows_img = build_rows()
    rows_emoji = build_rows_emoji()
    if args.tsv:
        print_tsv(rows_emoji)
        return

    png = OUT_DIR / "complementary_analysis_table.png"
    svg = OUT_DIR / "complementary_analysis_table.svg"
    html = OUT_DIR / "complementary_analysis_table.html"
    render_figure(rows_img, png, svg_path=svg)
    write_html(html, rows_emoji)
    print(f"[OK] {png}")
    print(f"[OK] {svg}")
    print(f"[OK] {html}")
    print("\n--- TSV（✅❌⚠️ 版，复制到 Excel，再复制到 PPT）---")
    print_tsv(rows_emoji)


if __name__ == "__main__":
    main()
