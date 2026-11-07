"""生成 assets/icon.png（NuGet 包图标 + README 头部图标）。

设计：深色圆角底 + 青紫渐变 "Ox" 字标 + 下方 "CPU" 小字（强调只有 CPU 后端）。
用 4x 超采样后 LANCZOS 缩小，避免锯齿。

用法：python assets/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 256
SS = 4                      # 超采样倍数
W = SIZE * SS
OUT = Path(__file__).with_name("icon.png")

BG_TOP = (15, 23, 42)       # slate-900
BG_BOTTOM = (30, 41, 59)    # slate-800
MARK_LEFT = (34, 211, 238)  # cyan-400
MARK_RIGHT = (167, 139, 250)  # violet-400
CAPTION = (148, 163, 184)   # slate-400

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\consolab.ttf",
]


def lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit("找不到可用字体，请修改 FONT_CANDIDATES")


# 背景竖向渐变
back = Image.new("RGB", (W, W))
draw = ImageDraw.Draw(back)
for y in range(W):
    draw.line([(0, y), (W, y)], fill=lerp(BG_TOP, BG_BOTTOM, y / (W - 1)))

# 圆角遮罩
rounded = Image.new("L", (W, W), 0)
ImageDraw.Draw(rounded).rounded_rectangle(
    [0, 0, W - 1, W - 1], radius=round(0.23 * W), fill=255
)

icon = Image.new("RGBA", (W, W), (0, 0, 0, 0))
icon.paste(back, (0, 0), rounded)

# 横向渐变，用来填充字标
mark = Image.new("RGB", (W, W))
mark_draw = ImageDraw.Draw(mark)
for x in range(W):
    mark_draw.line([(x, 0), (x, W)], fill=lerp(MARK_LEFT, MARK_RIGHT, x / (W - 1)))

mark_font = load_font(round(0.46 * W))
mark_mask = Image.new("L", (W, W), 0)
ImageDraw.Draw(mark_mask).text(
    (W / 2, W * 0.44), "Ox", font=mark_font, fill=255, anchor="mm"
)
icon.paste(mark, (0, 0), mark_mask)

# 底部 "CPU" 小字，字母间加空隙
cap_font = load_font(round(0.095 * W))
cap_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
cap_draw = ImageDraw.Draw(cap_layer)
letters = "CPU"
gap = 0.018 * W
widths = [cap_draw.textlength(ch, font=cap_font) for ch in letters]
total = sum(widths) + gap * (len(letters) - 1)
x = (W - total) / 2
for ch, cw in zip(letters, widths):
    cap_draw.text((x, W * 0.745), ch, font=cap_font, fill=CAPTION, anchor="lm")
    x += cw + gap
icon.alpha_composite(cap_layer)

icon.resize((SIZE, SIZE), Image.LANCZOS).save(OUT, optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {SIZE}x{SIZE})")
