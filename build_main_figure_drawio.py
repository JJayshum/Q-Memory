"""Recreate the supplied main figure with native, editable draw.io objects."""

from pathlib import Path
import xml.etree.ElementTree as ET


OUT = Path(__file__).resolve().parent
BLUE = '#001bba'
RED = '#ff1717'
BLACK = '#151515'
mx = ET.Element('mxfile', host='drawio', version='31.0.2')
diagram = ET.SubElement(mx, 'diagram', id='qmemory-overview', name='Main figure')
model = ET.SubElement(diagram, 'mxGraphModel', dx='1536', dy='780', grid='1',
                      gridSize='10', page='1', pageScale='1', pageWidth='1536',
                      pageHeight='780', background='#ffffff', math='0', shadow='0')
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')
seq = 1


def cell(value, x, y, w, h, style='', parent='1'):
    global seq
    seq += 1
    c = ET.SubElement(root, 'mxCell', id=str(seq), value=value, vertex='1',
                      parent=parent, style='html=1;whiteSpace=wrap;fontFamily=Arial;'
                      'fontSize=18;fontColor=#080808;spacing=2;' + style)
    ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h),
                  attrib={'as': 'geometry'})
    return str(seq)


def box(x, y, w, h, fill='#ffffff', stroke='#555555', value='', parent='1', extra=''):
    return cell(value, x, y, w, h, 'rounded=1;arcSize=8;absoluteArcSize=1;'
                f'fillColor={fill};strokeColor={stroke};strokeWidth=1;' + extra, parent)


def text(value, x, y, w, h, size=18, color=BLACK, parent='1', bold=False, align='center'):
    return cell(value, x, y, w, h, 'text;fillColor=none;strokeColor=none;'
                f'fontSize={size};fontColor={color};fontStyle={1 if bold else 0};'
                f'align={align};verticalAlign=middle;', parent)


def line(points, color=BLACK, arrow=False, width=1.3, parent='1', dashed=False):
    global seq
    seq += 1
    c = ET.SubElement(root, 'mxCell', id=str(seq), value='', edge='1', parent=parent,
                      style='html=1;rounded=0;orthogonalLoop=1;jettySize=auto;'
                      f'strokeColor={color};strokeWidth={width};'
                      f'endArrow={"block" if arrow else "none"};endFill=1;'
                      f'endSize=9;dashed={int(dashed)};')
    g = ET.SubElement(c, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
    for name, pt in [('sourcePoint', points[0]), ('targetPoint', points[-1])]:
        ET.SubElement(g, 'mxPoint', x=str(pt[0]), y=str(pt[1]), attrib={'as': name})
    if len(points) > 2:
        a = ET.SubElement(g, 'Array', attrib={'as': 'points'})
        for x, y in points[1:-1]:
            ET.SubElement(a, 'mxPoint', x=str(x), y=str(y))


def math(s):
    return '<font face="Times New Roman"><i>' + s + '</i></font>'


def swatch(x, y, fill, size=25, parent='1'):
    return cell('', x, y, size, size,
                f'rounded=0;fillColor={fill};strokeColor=#333333;strokeWidth=1.2;', parent)


palette = ['#f8cdcd', '#fff1bb', '#d5ecc4', '#c6eff2', '#c6d8fa', '#e4d8ef', '#e7e7e7']
history = box(10, 10, 385, 375, '#f5faff', '#325979', extra='container=1;pointerEvents=0;')
memory = box(560, 10, 420, 375, '#fbfff7', '#318552', extra='container=1;pointerEvents=0;')
deploy = box(1015, 10, 510, 375, '#f2fbff', '#377aac', extra='container=1;pointerEvents=0;')
replay = box(10, 400, 1105, 208, '#fff8f8', '#e34848', extra='container=1;pointerEvents=0;')

text('1) HISTORY PREFIX INPUT', 12, 9, 360, 31, 23, parent=history, bold=True)
text('History prefix ' + math('h<sub>t</sub><sup>&minus;</sup>'), 20, 43, 345, 31, 21, parent=history)
table = box(15, 81, 355, 211, parent=history, extra='container=1;pointerEvents=0;')
for yy in [60, 110, 160]:
    line([(0, yy), (355, yy)], '#777777', parent=table)
for xx in [62, 240]:
    line([(xx, 0), (xx, 211)], '#777777', parent=table)
rows = [
    (math('o<sub>1</sub>'), 'Observation<br>(decision-relevant cue)', 'Lang<sub>1</sub><br>&ldquo;go to the key&rdquo;', BLUE, '#4c168a'),
    (math('a<sub>1</sub>'), 'Action', math('a<sub>1</sub>') + ' (left)', BLUE, '#087917'),
    (math('r<sub>1</sub>'), 'Reward', math('r<sub>1</sub>'), '#cc580d', '#cc580d'),
    ('Lang<sub>1</sub>', 'Language cue', 'Lang<sub>1</sub><br>&ldquo;go to the key&rdquo;', BLUE, '#4c168a'),
]
for i, (symbol, label, example, sc, ec) in enumerate(rows):
    yy, hh = (0, 60) if i == 0 else (60 + (i-1)*50, 50)
    text(symbol, 2, yy, 58, hh, 22, sc, table)
    text(label, 64, yy, 174, hh, 16, parent=table, bold=True)
    text(example, 242, yy, 110, hh, 16, ec, table)
text('Current observation  ' + math('o<sub>t</sub>'), 20, 306, 345, 43, 21, parent=history)

box(433, 118, 100, 164, '#f0f5ff', '#254be0')
text('Encoder', 437, 163, 92, 34, 21, bold=True)
text(math('E<sub>&phi;</sub>'), 445, 200, 76, 45, 34)
line([(395, 193), (433, 193)], BLUE, True, 3)
line([(533, 193), (560, 193)], BLUE, True, 3)
line([(980, 193), (1015, 193)], BLUE, True, 3)

text('2) DISCRETE MEMORY BOTTLENECK', 8, 9, 404, 31, 22, parent=memory, bold=True)
text('Discrete code  ' + math('m<sub>t</sub>') + '  (low-rate persistent)', 12, 56, 396, 37, 20, parent=memory)
for x, color in zip([118, 201, 283], [palette[0], palette[2], palette[4]]):
    swatch(x, 107, color, parent=memory)
line([(22, 151), (400, 151)], '#777777', parent=memory, dashed=True)
text('Finite codebook  ' + math('M = {m<sup>(1)</sup>, m<sup>(2)</sup>, &hellip;, m<sup>(K)</sup>}'),
     15, 164, 390, 43, 19, parent=memory)
for i, color in enumerate(palette):
    swatch(30+i*49, 217, color, parent=memory)
text('&hellip;', 367, 208, 40, 40, 28, parent=memory)
line([(22, 264), (400, 264)], '#777777', parent=memory, dashed=True)
text('Rate  ' + math('R(m<sub>t</sub>) = &minus;log<sub>2</sub> P(m<sub>t</sub>)') + '  (bits)',
     18, 272, 386, 42, 21, BLUE, memory)
box(12, 325, 396, 38, '#f9fff6', '#5b9171',
    'same code =&gt; similar centered action advantages', memory, 'fontSize=17;fontColor=#155124;')

text('3) DEPLOYMENT', 10, 9, 490, 31, 23, parent=deploy, bold=True)
for x, w, label in [(16, 124, math('m<sub>t</sub>')), (150, 107, math('o<sub>t</sub>')),
                    (268, 107, '&Pi;<sub>ref</sub>'), (386, 112, math('A<sub>cov</sub>(o<sub>t</sub>)'))]:
    box(x, 49, w, 64, value='', parent=deploy)
    text(label, x, 51, w, 34 if x == 16 else 60, 25, BLUE, deploy)
for i, color in enumerate([palette[0], palette[2], palette[4]]):
    swatch(36+i*34, 86, color, 19, deploy)
line([(78, 113), (78, 132), (442, 132), (442, 113)], parent=deploy)
line([(203, 113), (203, 132)], arrow=True, parent=deploy)
line([(321, 113), (321, 132)], arrow=True, parent=deploy)
line([(139, 132), (139, 148)], arrow=True, parent=deploy)
box(16, 148, 269, 48, '#edf4fe', value='Decoder  ' + math('D<sub>&psi;</sub>'), parent=deploy, extra='fontSize=23;')
text('Centered advantages  ' + math('A&#770;<sup>&pi;</sup>(o<sub>t</sub>, a)'), 15, 203, 272, 36, 18, parent=deploy)
# The inset bars preserve the original illustrative values; they are not new results.
zero, axis_y = 160, 327
line([(68, 237), (68, axis_y), (276, axis_y)], '#444444', parent=deploy)
line([(zero, 237), (zero, axis_y)], '#444444', parent=deploy)
for yy, label, xx, width, color in [(247, 'left', 105, 55, palette[0]),
                                    (276, 'right', zero, 38, palette[2]),
                                    (305, 'forward', zero, 98, palette[4])]:
    text(label, 17, yy-5, 47, 27, 15, parent=deploy, align='right')
    cell('', xx, yy, width, 18, 'rounded=0;strokeColor=#222222;fillColor='+color+';', deploy)
for xx, label in [(68, '&minus;1.0'), (114, '&minus;0.5'), (160, '0'), (206, '0.5'), (252, '1.0')]:
    line([(xx, axis_y), (xx, axis_y+5)], parent=deploy)
    text(label, xx-22, axis_y+6, 44, 24, 14, parent=deploy)
text('Centered advantage (higher is better)', 22, 356, 265, 17, 13, parent=deploy)
box(318, 210, 180, 84, value=math('a<sub>t</sub> = argmax<sub>a</sub>') + '<br>' +
    math('D<sub>&psi;</sub>(m<sub>t</sub>, o<sub>t</sub>, &Pi;<sub>ref</sub>, A<sub>cov</sub>)'),
    parent=deploy, extra='fontSize=18;')
line([(285, 246), (318, 246)], BLUE, True, 2.5, deploy)
line([(408, 294), (408, 336)], BLUE, True, 2, deploy)
box(333, 336, 150, 38, value='Decision  ' + math('a<sub>t</sub>'), parent=deploy, extra='fontSize=22;')

text('TRAINING ONLY: COUNTERFACTUAL REPLAY-Q', 325, 4, 590, 31, 22, '#c90000', replay)
box(15, 35, 210, 132, value='', parent=replay)
text('History-consistent<br>snapshot at time ' + math('t') + '<br>(from dataset)', 23, 43, 194, 70, 17, parent=replay)
text(math('o<sub>t</sub> &nbsp; a<sub>1:t&minus;1</sub> &nbsp; r<sub>1:t&minus;1</sub> &nbsp; Lang<sub>1:t</sub>'),
     22, 119, 196, 40, 17, BLUE, replay)
text('Fork<br>candidate<br>actions', 237, 64, 72, 71, 16, parent=replay)
for yy, label in [(35, math('a<sub>1</sub>')+' (left)'), (78, math('a<sub>2</sub>')+' (right)'),
                  (143, math('a<sub>k</sub>')+' (forward)')]:
    box(343, yy, 96, 32, '#f1fae9', '#729c56', label, replay, 'fontSize=17;')
    line([(309, 101), (321, 101), (321, yy+16), (343, yy+16)], arrow=True, parent=replay)
    line([(439, yy+16), (481, yy+16)], arrow=True, parent=replay)
text('&vellip;', 369, 109, 40, 28, 26, parent=replay)
box(481, 43, 151, 132, '#eff4ff', '#5f7cd1', value='', parent=replay)
text('Frozen continuation<br>policy  ' + math('&pi;<sub>thereafter</sub>') + '<br>(frozen)',
     488, 64, 137, 90, 17, '#071954', replay)
for yy, label in [(43, '1'), (85, '2'), (147, 'k')]:
    box(674, yy, 66, 32, '#fff5ed', '#d18f56', math('G<sup>('+label+')</sup>'), replay, 'fontSize=23;')
    line([(632, yy+16), (674, yy+16)], arrow=True, parent=replay)
    line([(740, yy+16), (762, yy+16), (762, 108), (781, 108)], arrow=True, parent=replay)
text('&vellip;', 689, 116, 36, 29, 26, parent=replay)
box(781, 53, 152, 116, value='', parent=replay)
text('Aggregate returns<br>(discounted)', 788, 62, 138, 44, 17, parent=replay)
text(math('Q&#770;<sup>&pi;</sup>(a<sub>i</sub>, o<sub>t</sub>, a)'), 787, 111, 140, 39, 24, BLUE, replay)
line([(933, 108), (967, 108)], arrow=True, parent=replay)
box(967, 53, 126, 116, value='', parent=replay)
text('Centering', 973, 65, 114, 28, 19, parent=replay)
text(math('Q&#770;<sup>&pi;</sup> &rarr; A&#770;<sup>&pi;</sup>'), 973, 96, 114, 36, 24, BLUE, replay)
text('(zero-mean)', 973, 140, 114, 23, 16, parent=replay)

box(1205, 448, 302, 104, '#fff5f5', '#ef1717')
text('Loss (training objective)', 1212, 456, 288, 36, 21, bold=True)
text('Huber loss + ' + math('&beta; R(m<sub>t</sub>)'), 1215, 498, 282, 37, 25, BLUE)
line([(1103, 508), (1205, 508)], RED, True, 2.5)
line([(1150, 508), (1150, 385)], RED, True, 2.5)
line([(1178, 582), (1228, 582)], RED, True, 2.5)
text('Supervision (training only)', 1243, 568, 281, 28, 17, align='left')
line([(1178, 607), (1228, 607)], BLUE, True, 2.5)
text('Data flow (deployment &amp; training)', 1243, 593, 285, 28, 16, align='left')

box(14, 634, 381, 77, '#f3edff', '#8662d3',
    'Evidence: Binary<br>2 codes: 100%', extra='fontSize=21;fontColor=#200898;')
box(420, 634, 381, 77, '#eef4ff', '#5c82cf',
    'Evidence: Frozen Qwen2.5<br>150/150 decisions preserved', extra='fontSize=21;fontColor=#0022ad;')
box(813, 634, 445, 77, '#e9fafd', '#3c9ca8',
    'Evidence: MiniGrid 4-code Q-memory<br>99.76% on S13, 98.04% on S17', extra='fontSize=21;fontColor=#006c77;')
box(14, 729, 1510, 40, '#ffffff', '#777777',
    '<b>BOUNDARY NOTE:</b> Claim is conditional on the declared policy class &Pi;<sub>ref</sub> and covered action set '
    + math('A<sub>cov</sub>') + '; not universal state sufficiency.', extra='fontSize=17;')

ET.indent(mx)
ET.ElementTree(mx).write(OUT / 'policy_class_q_memory_main_figure.drawio', encoding='utf-8', xml_declaration=True)
print(f'Wrote {seq - 1} editable objects to policy_class_q_memory_main_figure.drawio')
