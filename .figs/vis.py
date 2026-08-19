# -*- coding: utf-8 -*-
import io, re
src = io.open('.figs/gen.py', encoding='utf-8').read()
head = src.split("f={}")[0] + "f={}\n"
exec(head)

X0, PW, PG = 150, 46, 3
def px(i): return X0 + i*(PW+PG)
def pages(y, kind, h=28):
    return "\n".join('  <rect x="%d" y="%d" width="%d" height="%d" rx="3" fill="%s" stroke="%s"%s/>'
        % (px(i), y, PW, h, *kind(i)[:2], ' stroke-dasharray="3 3"' if kind(i)[2] else '') for i in range(12))
GREY = ('#F1F3F5', '#CED4DA', True)
HOLE = 6

# ---------- fig1：三种层在同一条 token 轴上的数据布局完全不同 ----------
b = []
b.append('  <text x="28" y="76" font-size="13" font-weight="bold" fill="#C92A2A">全注意力层</text>')
b.append(pages(62, lambda i: REDB if i == HOLE else BLUE))
b.append('  <polyline points="%d,100 %d,100 %d,94" fill="none" stroke="#E03131" stroke-width="1.8"/>' % (X0, px(HOLE)-6, px(HOLE)-6))
b.append('  <polygon points="%d,88 %d,96 %d,96" fill="#E03131"/>' % (px(HOLE)+2, px(HOLE)-8, px(HOLE)+2))
b.append('  <text x="%d" y="116" font-size="11.5" fill="#C92A2A">重算依赖左边全部 KV，页都在，可以补</text>' % (X0+10))

b.append('  <text x="28" y="152" font-size="13" font-weight="bold" fill="#087F5B">滑窗层</text>')
b.append(pages(138, lambda i: GREY if i < 8 else GRN))
b.append('  <text x="%d" y="157" font-size="15" fill="#0CA678" text-anchor="middle">✓</text>' % (px(HOLE)+PW/2))
b.append('  <polyline points="%d,176 %d,176" fill="none" stroke="#0CA678" stroke-width="2"/>' % (px(8), px(11)+PW))
b.append('  <text x="%d" y="192" font-size="11.5" fill="#0CA678">窗口 W</text>' % (px(8)+16))
b.append('  <text x="%d" y="192" font-size="11.5" fill="#868E96">窗口外的 K 已被环形缓冲覆写，掩码也不看它</text>' % (X0+2))

b.append('  <text x="28" y="228" font-size="13" font-weight="bold" fill="#E8590C">线性注意力层</text>')
b.append('  <polyline points="%d,228 %d,228" fill="none" stroke="#CED4DA" stroke-width="2"/>' % (X0, px(11)+PW))
for i in (0, 5):
    b.append('  <polygon points="%d,220 %d,228 %d,236 %d,228" fill="#FFD8A8" stroke="#F08C00"/>' % (px(i)+8, px(i), px(i)+8, px(i)+16))
    b.append('  <text x="%d" y="252" font-size="10.5" fill="#E8590C" text-anchor="middle">快照</text>' % (px(i)+8))
b.append('  <text x="%d" y="214" font-size="15" fill="#E03131" text-anchor="middle">✕</text>' % (px(HOLE)+PW/2))
b.append('  <rect x="%d" y="214" width="70" height="28" rx="4" fill="url(#lo)" stroke="#F08C00"/>' % (px(11)+PW-70))
b.append('  <text x="%d" y="233" font-size="11" fill="#E8590C" text-anchor="middle">循环状态</text>' % (px(11)+PW-35))
b.append('  <polyline points="%d,268 %d,268" fill="none" stroke="#F08C00" stroke-width="2"/>' % (px(5)+8, px(11)+PW))
b.append('  <polygon points="%d,262 %d,268 %d,274" fill="#F08C00"/>' % (px(11)+PW-12, px(11)+PW, px(11)+PW-12))
b.append('  <text x="%d" y="286" font-size="11.5" font-weight="bold" fill="#E8590C">从最近快照重放，上界 8192 token，与空洞长度无关</text>' % (px(5)+16))
b.append('  <text x="%d" y="252" font-size="10.5" fill="#868E96">没有逐 token 的 KV</text>' % (px(7)))
f['fig1'] = card('三类层在同一条 token 轴上的布局完全不同', '决定「只补一块」成不成立', 'orange', "\n".join(b), H=310)

# ---------- s9：窗口掩码 vs softmax 质量 ----------
b = []
b.append('  <text x="28" y="70" font-size="12.5" font-weight="bold" fill="#087F5B">滑窗层：丢的块在窗口外</text>')
for i in range(8):
    k = ('#F1F3F5', '#CED4DA', True) if i < 5 else ('url(#par)', '#0CA678', False)
    if i == 2: k = ('url(#bad)', '#E03131', True)
    b.append('  <rect x="%d" y="82" width="34" height="26" rx="3" fill="%s" stroke="%s"%s/>' % (28+i*38, k[0], k[1], ' stroke-dasharray="3 3"' if k[2] else ''))
b.append('  <polyline points="218,118 332,118" fill="none" stroke="#0CA678" stroke-width="2"/>')
b.append('  <text x="240" y="134" font-size="11.5" fill="#0CA678">窗口 W 只覆盖这几块</text>')
b.append('  <text x="96" y="134" font-size="11.5" fill="#C92A2A">丢的在这外面</text>')
b.append('  <text x="28" y="160" font-size="12.5" font-weight="bold" fill="#087F5B">跳过与补上逐 token 相同</text>')
b.append('  <polyline points="368,60 368,168" fill="none" stroke="#DEE2E6" stroke-width="1.5"/>')
b.append('  <text x="396" y="70" font-size="12.5" font-weight="bold" fill="#C92A2A">全注意力层：丢的块本来有权重</text>')
w = [16, 26, 40, 30, 22, 34, 18, 14]
xx = 396
for i, v in enumerate(w):
    fill = 'url(#bad)' if i == 3 else 'url(#ok)'
    st = '#E03131' if i == 3 else '#4C6EF5'
    b.append('  <rect x="%d" y="%d" width="34" height="%d" rx="2" fill="%s" stroke="%s"%s/>' % (xx+i*40, 118-v, v, fill, st, ' stroke-dasharray="3 3"' if i == 3 else ''))
b.append('  <polyline points="396,120 716,120" fill="none" stroke="#ADB5BD" stroke-width="1.5"/>')
b.append('  <text x="%d" y="98" font-size="12" font-weight="bold" fill="#C92A2A" text-anchor="middle">δ</text>' % (xx+3*40+17))
b.append('  <text x="396" y="140" font-size="11.5" fill="#868E96">每块在 softmax 里占的注意力质量</text>')
b.append('  <text x="396" y="162" font-size="12.5" font-weight="bold" fill="#C92A2A">误差 ≤ M·δ(2−δ)/(1−δ)，δ 可测</text>')
f['s9'] = card('⑨ 跳过空洞', '需 kernel 改造 · 分情况', 'orange', "\n".join(b), H=190)

# ---------- s5：中段 block 没有独立的名字 ----------
b = []
for j, (nm, dead) in enumerate([('实例 B', False), ('实例 C', False), ('实例 A', True)]):
    x = 28 + j*244
    b.append('  <rect x="%d" y="62" width="228" height="56" rx="6" fill="%s" stroke="%s"%s/>'
             % (x, '#FFF5F5' if dead else '#E3FAFC', '#E03131' if dead else '#1098AD', ' stroke-dasharray="4 3"' if dead else ''))
    b.append('  <text x="%d" y="80" font-size="12" fill="%s">%s</text>' % (x+12, '#C92A2A' if dead else '#0B7285', nm))
    for i in range(8):
        k = ('url(#bad)', '#E03131', True) if (dead and i == 4) else ('url(#ok)', '#4C6EF5', False)
        b.append('  <rect x="%d" y="90" width="22" height="18" rx="2" fill="%s" stroke="%s"%s/>'
                 % (x+12+i*26, k[0], k[1], ' stroke-dasharray="3 2"' if k[2] else ''))
b.append('  <polyline points="528,128 %d,128" fill="none" stroke="#E03131" stroke-width="2"/>' % (528+4*26+6))
b.append('  <polygon points="%d,122 %d,128 %d,134" fill="#E03131"/>' % (528+4*26-6, 528+4*26+8, 528+4*26-6))
b.append('  <text x="528" y="148" font-size="11.5" fill="#C92A2A">名字依赖左边全部 token</text>')
b.append('  <text x="28" y="172" font-size="12.5" font-weight="bold" fill="#C92A2A">中段 block 没有跨实例可独立计算的名字，补一个全局索引解决不了这件事</text>')
f['s5'] = card('⑤ 对等实例恢复', '缺命名方案 · 开销未知', 'red', "\n".join(b), H=196)

for k, v in f.items():
    if k in ('fig1', 's9', 's5'):
        io.open('.figs/%s.svg' % k, 'w', encoding='utf-8').write(v)
print('redrew fig1 / s9 / s5')
