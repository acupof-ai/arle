# -*- coding: utf-8 -*-
import io
W = 760
PAL = {'green':('#0CA678','#087F5B','#E6FCF5'),'red':('#E03131','#C92A2A','#FFF5F5'),'orange':('#F08C00','#E8590C','#FFF4E6')}
DEFS = '''  <defs>
    <linearGradient id="ok" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#EDF2FF"/><stop offset="1" stop-color="#DBE4FF"/></linearGradient>
    <linearGradient id="bad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFF5F5"/><stop offset="1" stop-color="#FFC9C9"/></linearGradient>
    <linearGradient id="fix" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFF9DB"/><stop offset="1" stop-color="#FFE066"/></linearGradient>
    <linearGradient id="par" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#E6FCF5"/><stop offset="1" stop-color="#96F2D7"/></linearGradient>
    <linearGradient id="lo" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFF4E6"/><stop offset="1" stop-color="#FFD8A8"/></linearGradient>
  </defs>
'''
def tw(s, size): return sum((size if ord(c) > 0x2E80 else size*0.56) for c in s)

def card(title, badge, color, body, H=176):
    stroke, tcol, bg = PAL[color]
    bw = tw(badge, 12.5) + 30; bx = W - 22 - bw
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">\n%s'
      '  <rect x="0" y="0" width="%d" height="%d" fill="#FFFFFF"/>\n'
      '  <rect x="8" y="8" width="%d" height="%d" rx="10" fill="#FFFFFF" stroke="%s" stroke-width="2"/>\n'
      '  <text x="28" y="42" font-size="17" font-weight="bold" fill="%s">%s</text>\n'
      '  <rect x="%d" y="24" width="%d" height="26" rx="13" fill="%s" stroke="%s"/>\n'
      '  <text x="%d" y="42" font-size="12.5" font-weight="bold" fill="%s" text-anchor="middle">%s</text>\n'
      '%s\n</svg>\n') % (W,H,W,H,DEFS,W,H,W-16,H-16,stroke,tcol,title,bx,bw,bg,stroke,bx+bw/2,tcol,badge,body)

def bar(x, y, n, w, gap, kind, h=30):
    o=[]
    for i in range(n):
        f,s,d = kind(i); dash=' stroke-dasharray="4 3"' if d else ''
        o.append('  <rect x="%d" y="%d" width="%d" height="%d" rx="4" fill="%s" stroke="%s"%s/>'%(x+i*(w+gap),y,w,h,f,s,dash))
    return "\n".join(o)
BLUE=('url(#ok)','#4C6EF5',False); REDB=('url(#bad)','#E03131',True)
YEL=('url(#fix)','#F08C00',False); GRN=('url(#par)','#0CA678',False); ORG=('url(#lo)','#F08C00',False)
f={}

f['fig0'] = card('丢块场景', '右侧的 KV 仍然有效', 'red',
  bar(28,74,12,52,6,lambda i: REDB if i==6 else BLUE) + '''
  <text x="416" y="94" font-size="12.5" fill="#C92A2A">丢失</text>
  <polyline points="28,120 372,120" fill="none" stroke="#4C6EF5" stroke-width="2"/>
  <text x="120" y="140" font-size="12.5" fill="#3B5BDB">重算要用它</text>
  <polyline points="404,120 730,120" fill="none" stroke="#4C6EF5" stroke-width="2"/>
  <text x="500" y="140" font-size="12.5" fill="#3B5BDB">已算完，不受影响</text>''', H=164)

f['fig1'] = card('三类层的修复语义完全不同', '决定「只补一块」成不成立', 'orange', '''
  <rect x="28" y="64" width="704" height="52" rx="6" fill="#FFF8F8" stroke="#E03131"/>
  <text x="46" y="86" font-size="13.5" font-weight="bold" fill="#C92A2A">全注意力层</text>
  <text x="150" y="86" font-size="12.5" fill="#495057">逐 token 的 K/V 都在页里，可按位置独立重算</text>
  <text x="46" y="106" font-size="12.5" font-weight="bold" fill="#C92A2A">缺块必须补</text>
  <rect x="28" y="126" width="704" height="52" rx="6" fill="#F4FCF8" stroke="#0CA678"/>
  <text x="46" y="148" font-size="13.5" font-weight="bold" fill="#087F5B">滑窗层</text>
  <text x="150" y="148" font-size="12.5" fill="#495057">窗口外的 K 已被环形缓冲覆写，但掩码本来也不看它</text>
  <text x="46" y="168" font-size="12.5" font-weight="bold" fill="#087F5B">窗口外缺块无影响，窗口内只依赖左边 W 个 token</text>
  <rect x="28" y="188" width="704" height="52" rx="6" fill="#FFF8F0" stroke="#F08C00"/>
  <text x="46" y="210" font-size="13.5" font-weight="bold" fill="#E8590C">线性注意力层</text>
  <text x="180" y="210" font-size="12.5" fill="#495057">只存最前沿一份循环状态，不可逆，无法从任意位置起算</text>
  <text x="46" y="230" font-size="12.5" font-weight="bold" fill="#E8590C">只能回退到最近快照重放，上界 8192 token</text>''', H=264)

f['s1'] = card('① 空洞局部重算', '开销 0 · 仅全注意力页', 'green',
  bar(28,72,12,52,6,lambda i: YEL if i==6 else BLUE) + '''
  <polyline points="40,124 356,124 380,110" fill="none" stroke="#F08C00" stroke-width="2"/>
  <polygon points="376,102 390,109 376,116" fill="#F08C00"/>
  <text x="40" y="146" font-size="12.5" fill="#495057">读左侧已在显存的 KV</text>
  <text x="410" y="146" font-size="12.5" font-weight="bold" fill="#E8590C">只前向 h 个 token</text>''')

cells=[]
for i in range(10):
    k = GRN if i>=8 else (REDB if i==2 else BLUE)
    cells.append('  <rect x="%d" y="72" width="60" height="38" rx="4" fill="%s" stroke="%s"%s/>'%(28+i*70,k[0],k[1],' stroke-dasharray="4 3"' if k[2] else ''))
    cells.append('  <text x="%d" y="126" font-size="11" fill="%s" text-anchor="middle">DN%d</text>'%(58+i*70,'#C92A2A' if i==2 else '#868E96',i+1))
f['s2'] = card('② 纠删码 RS(8,2)', '开销 ≥25% · 只处理已定位的丢失', 'orange', "\n".join(cells) + '''
  <text x="28" y="64" font-size="12" fill="#3B5BDB">8 个数据块</text>
  <text x="590" y="64" font-size="12" fill="#087F5B">2 个校验块</text>
  <polyline points="58,142 58,156 730,156 730,142" fill="none" stroke="#0CA678" stroke-width="1.6"/>
  <polyline points="158,142 158,156" fill="none" stroke="#0CA678" stroke-width="1.6"/>
  <polygon points="152,148 164,148 158,136" fill="#0CA678"/>
  <text x="320" y="176" font-size="12.5" font-weight="bold" fill="#087F5B">任意 8 块可重建</text>''', H=196)

f['s3'] = card('③ 多副本', '开销 100%/份 · 需全量覆盖', 'orange', '''
  <rect x="60" y="70" width="160" height="44" rx="6" fill="url(#ok)" stroke="#4C6EF5"/>
  <text x="140" y="98" font-size="13" fill="#3B5BDB" text-anchor="middle">DN1 副本</text>
  <rect x="300" y="70" width="160" height="44" rx="6" fill="url(#bad)" stroke="#E03131" stroke-dasharray="4 3"/>
  <text x="380" y="98" font-size="13" fill="#C92A2A" text-anchor="middle">DN2 失效</text>
  <rect x="540" y="70" width="160" height="44" rx="6" fill="url(#ok)" stroke="#4C6EF5"/>
  <text x="620" y="98" font-size="13" fill="#3B5BDB" text-anchor="middle">DN3 副本</text>
  <polyline points="620,126 620,146 380,146" fill="none" stroke="#4C6EF5" stroke-width="2"/>
  <polygon points="386,140 386,152 374,146" fill="#4C6EF5"/>
  <text x="400" y="150" font-size="12.5" font-weight="bold" fill="#3B5BDB">一次单点读</text>''')

f['s5'] = card('⑤ 对等实例恢复', '缺命名方案 · 开销未知', 'red', '''
  <rect x="28" y="76" width="150" height="52" rx="6" fill="#E3FAFC" stroke="#1098AD"/>
  <text x="103" y="100" font-size="13" fill="#0B7285" text-anchor="middle">实例 B</text>
  <text x="103" y="119" font-size="11" fill="#868E96" text-anchor="middle">HBM 有同前缀</text>
  <rect x="288" y="76" width="184" height="52" rx="6" fill="#FFF5F5" stroke="#E03131" stroke-dasharray="4 3"/>
  <text x="380" y="100" font-size="13" fill="#C92A2A" text-anchor="middle">全局索引</text>
  <text x="380" y="119" font-size="11" fill="#C92A2A" text-anchor="middle">中段 block 无法命名</text>
  <rect x="582" y="76" width="150" height="52" rx="6" fill="url(#bad)" stroke="#E03131" stroke-dasharray="4 3"/>
  <text x="657" y="100" font-size="13" fill="#C92A2A" text-anchor="middle">实例 A</text>
  <text x="657" y="119" font-size="11" fill="#868E96" text-anchor="middle">缺页</text>
  <polyline points="186,102 280,102" fill="none" stroke="#ADB5BD" stroke-width="1.6" stroke-dasharray="5 4"/>
  <polyline points="480,102 574,102" fill="none" stroke="#ADB5BD" stroke-width="1.6" stroke-dasharray="5 4"/>''')

f['s6'] = card('⑥ 低精度影子副本', '开销随主副本精度变 · 有损', 'red', '''
  <rect x="28" y="72" width="164" height="42" rx="6" fill="url(#bad)" stroke="#E03131" stroke-dasharray="4 3"/>
  <text x="110" y="98" font-size="12.5" fill="#C92A2A" text-anchor="middle">主副本 bf16　丢失</text>
  <rect x="222" y="72" width="140" height="42" rx="6" fill="url(#lo)" stroke="#F08C00"/>
  <text x="292" y="98" font-size="12.5" fill="#E8590C" text-anchor="middle">影子 INT4</text>
  <polyline points="370,93 410,93" fill="none" stroke="#F08C00" stroke-width="2"/>
  <polygon points="406,87 418,93 406,99" fill="#F08C00"/>
  <rect x="428" y="72" width="120" height="42" rx="6" fill="#FFFFFF" stroke="#F08C00"/>
  <text x="488" y="98" font-size="12.5" fill="#E8590C" text-anchor="middle">反量化</text>
  <polyline points="556,93 588,93" fill="none" stroke="#E03131" stroke-width="2"/>
  <polygon points="584,87 596,93 584,99" fill="#E03131"/>
  <rect x="604" y="72" width="128" height="42" rx="6" fill="url(#bad)" stroke="#E03131"/>
  <text x="668" y="98" font-size="12.5" fill="#C92A2A" text-anchor="middle">带量化误差</text>
  <polyline points="200,93 214,93" fill="none" stroke="#ADB5BD" stroke-width="1.6"/>''')

f['s7'] = card('⑦ 全量重算', '开销 0 · 无损', 'green',
  bar(28,72,12,52,6,lambda i: YEL) + '''
  <polyline points="40,128 716,128" fill="none" stroke="#F08C00" stroke-width="2"/>
  <polygon points="712,122 726,128 712,134" fill="#F08C00"/>
  <text x="40" y="150" font-size="12.5" fill="#495057">从位置 0 起整条重建，前缀 KV 全部作废</text>''')

f['s8'] = card('⑧ 异步修复加降级', '需 kernel 改造 · 窗口内有损', 'orange', '''
  <polyline points="40,104 720,104" fill="none" stroke="#ADB5BD" stroke-width="1.6"/>
  <circle cx="70" cy="104" r="6" fill="#E03131"/>
  <text x="46" y="84" font-size="12" fill="#C92A2A">t0 缺页</text>
  <rect x="96" y="88" width="240" height="32" rx="5" fill="url(#bad)" stroke="#E03131"/>
  <text x="216" y="109" font-size="12.5" fill="#C92A2A" text-anchor="middle">降级窗口：照常出 token</text>
  <circle cx="366" cy="104" r="6" fill="#F08C00"/>
  <text x="330" y="84" font-size="12" fill="#E8590C">t1 补齐热插</text>
  <rect x="392" y="88" width="320" height="32" rx="5" fill="url(#par)" stroke="#0CA678"/>
  <text x="552" y="109" font-size="12.5" fill="#087F5B" text-anchor="middle">恢复精确</text>
  <text x="46" y="146" font-size="12.5" font-weight="bold" fill="#C92A2A">t0 到 t1 之间发出的 token 无法撤回</text>''')

f['s9'] = card('⑨ 跳过空洞', '需 kernel 改造 · 分情况', 'orange', '''
  <rect x="28" y="66" width="340" height="82" rx="8" fill="#F4FCF8" stroke="#0CA678"/>
  <text x="48" y="92" font-size="13.5" font-weight="bold" fill="#087F5B">丢在滑窗层的窗口外</text>
  <text x="48" y="122" font-size="13" font-weight="bold" fill="#087F5B">硬保证 · 逐 token 精确</text>
  <rect x="392" y="66" width="340" height="82" rx="8" fill="#FFF8F8" stroke="#E03131"/>
  <text x="412" y="92" font-size="13.5" font-weight="bold" fill="#C92A2A">丢在全注意力层</text>
  <text x="412" y="122" font-size="13" font-weight="bold" fill="#C92A2A">只有统计保证 · 误差随 workload 变</text>''')

for k,v in f.items(): io.open('.figs/%s.svg'%k,'w',encoding='utf-8').write(v)
print('wrote', len(f))
