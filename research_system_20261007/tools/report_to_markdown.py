from pathlib import Path
import re, json, hashlib
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[1]
ASSETS = BASE / 'assets'
ASSETS.mkdir(exist_ok=True)
SRC = (BASE / 'MENTOR_REPORT.tex').read_text(encoding='utf-8')
source_hash = hashlib.sha256(SRC.encode('utf-8')).hexdigest()
BODY = SRC.split(r'\begin{document}', 1)[1].split(r'\end{document}', 1)[0]
TOKENS = {}
REFS = {}
EQUATIONS = []
FIGURES = []
TABLES = []

def protect(text):
    key = f'ZZTOKEN{len(TOKENS)}ZZ'
    TOKENS[key] = text
    return key

def argument(text, command):
    start = text.index(command + '{') + len(command) + 1
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{': depth += 1
        if text[i] == '}': depth -= 1
        if depth == 0: return text[start:i]
    raise ValueError(command)

def mathfix(text):
    text = re.sub(r'\\label\{[^}]+\}', '', text).strip()
    text = text.replace(r'\med', r'\operatorname{median}').replace(r'\Std', r'\operatorname{std}')
    return text

def inline(text):
    text = re.sub(r'\$([^$]+)\$', lambda m: protect(' $' + mathfix(m[1]) + '$ '), text)
    text = re.sub(r'\\code\{([^{}]+)\}', lambda m: protect(chr(96) + m[1].replace(r'\_', '_') + chr(96)), text)
    text = re.sub(r'\\textbf\{([^{}]+)\}', lambda m: ' **' + m[1] + '** ', text)
    text = text.replace(r'\%', '%').replace(r'\_', '_').replace('~', ' ')
    return text.strip()

def font(size):
    for name in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc']:
        if Path(name).exists(): return ImageFont.truetype(name, size)
    raise RuntimeError('Chinese font missing')

def centered(draw, pos, text, size=30, color='#253347'):
    draw.text(pos, text, font=font(size), fill=color, anchor='mm')

def arrow(draw, points, dashed=False):
    import math
    for a, b in zip(points, points[1:]):
        if dashed:
            dist=math.dist(a,b)
            for k in range(0,int(dist),20):
                t0=k/dist; t1=min((k+11)/dist,1)
                draw.line([(a[0]+(b[0]-a[0])*t0,a[1]+(b[1]-a[1])*t0),(a[0]+(b[0]-a[0])*t1,a[1]+(b[1]-a[1])*t1)],fill='#245a9a',width=4)
        else: draw.line([a,b], fill='#245a9a',width=4)
    a,b=points[-2:]; ang=math.atan2(b[1]-a[1],b[0]-a[0]); length=19
    draw.polygon([b,(b[0]-length*math.cos(ang-.45),b[1]-length*math.sin(ang-.45)),(b[0]-length*math.cos(ang+.45),b[1]-length*math.sin(ang+.45))],fill='#245a9a')

def workflow(path):
    im=Image.new('RGB',(1800,760),'white'); d=ImageDraw.Draw(im)
    boxes=[(150,180,590,310),(680,180,1120,310),(1210,180,1650,310),(1210,480,1650,610),(680,480,1120,610),(150,480,590,610)]
    labels=[['窗口输入','256 × 20'],['单体相对中位数残差','32维统计特征'],['中位数 / MAD','稳健标准化'],['LOF新样本评分','正常校准分位数阈值'],['连续5窗超阈值','确认报警'],['确认报警输出','可疑单体排序']]
    for box,lines in zip(boxes,labels):
        d.rounded_rectangle(box,radius=12,fill='#eef5fc',outline='#245a9a',width=3)
        centered(d,((box[0]+box[2])/2,box[1]+43),lines[0],30)
        centered(d,((box[0]+box[2])/2,box[1]+89),lines[1],29)
    for pts in [[(590,245),(680,245)],[(1120,245),(1210,245)],[(1430,310),(1430,480)],[(1210,545),(1120,545)],[(680,545),(590,545)]]: arrow(d,pts)
    arrow(d,[(1430,180),(1430,95),(80,95),(80,545),(150,545)],True)
    d.rectangle((560,64,1260,120),fill='white'); centered(d,(910,92),'独立的标准化残差证据',29)
    im.save(path,dpi=(300,300))

def chart(path, series, names, legends, ymax, ylabel):
    im=Image.new('RGB',(1800,1050),'white');d=ImageDraw.Draw(im)
    l,r,t,b=160,1740,160,780;colors=['#3c78b4','#e49a34','#51a36d']; step=20 if ymax==100 else 5
    for value in range(0,int(ymax)+1,step):
        y=b-(b-t)*value/ymax
        d.line([(l,y),(r,y)],fill='#e0e5ec',width=2)
        d.text((l-25,y),str(value),font=font(27),fill='#44546a',anchor='rm')
    d.line([(l,t),(l,b),(r,b)],fill='#708090',width=3)
    centered(d,((l+r)/2,40),ylabel,31)
    group=(r-l)/len(names); bw=min(85,group/(len(series)+2)); span=len(series)*bw
    for i,name in enumerate(names):
        x=l+(i+.5)*group
        for j,values in enumerate(series):
            val=values[i]; xx=x-span/2+j*bw; yy=b-(b-t)*val/ymax
            d.rectangle((xx+5,yy,xx+bw-5,b),fill=colors[j] if len(series)>1 else '#cc5d63')
            centered(d,(xx+bw/2,yy-22),f'{val:.1f}' if ymax==100 else f'{val:.2f}',25)
        centered(d,(x,b+43),name,25)
    if legends:
        for j,lab in enumerate(legends):
            x=220+j*520;d.rectangle((x,97,x+32,127),fill=colors[j]);d.text((x+47,110),lab,font=font(29),fill='#253347',anchor='lm')
    else: centered(d,((l+r)/2,b+96),'留出包',30)
    im.save(path,dpi=(300,300))

def table_block(m):
    env=m[0]; no=len(TABLES)+1; caption=argument(env,r'\caption');label=argument(env,r'\label')
    REFS[label]=str(no)
    rows=[]
    for line in env.splitlines():
        if '&' in line:
            line=line.strip().removesuffix(chr(92)*2).strip()
            rows.append([inline(cell.strip()) for cell in line.split('&')])
    assert rows and len(set(map(len,rows)))==1
    TABLES.append({'label':label,'rows':rows,'caption':caption})
    out='**表'+str(no)+' '+inline(caption)+'**\n\n'
    out+='| '+' | '.join(rows[0])+' |\n| '+' | '.join([':---:']*len(rows[0]))+' |\n'
    out+='\n'.join('| '+' | '.join(row)+' |' for row in rows[1:])
    return '\n\n'+protect(out)+'\n\n'

def figure_block(m):
    env=m[0]; no=len(FIGURES)+1;caption=argument(env,r'\caption');label=argument(env,r'\label');REFS[label]=str(no)
    name=['fig_workflow.png','fig_development.png','fig_healthy.png'][no-1]
    path=ASSETS/name
    coordinates=[]
    if no==1: workflow(path)
    else:
        coordinate_sets=re.findall(r'coordinates\s*\{([^}]+)\}',env)
        coordinates=[[ (key,float(value)) for key,value in re.findall(r'\(([^,]+),([^\)]+)\)',group)] for group in coordinate_sets]
        if no==2:
            assert len(coordinates)==3
            names=[c[0] for c in coordinates[0]]
            names=[{'LOF':'lof_peer','Anchor':'peer_anchor','Multi':'peer_multiscale','PCA':'pca_peer','SVM':'svm_peer'}[key] for key in names]
            chart(path,[[x[1] for x in group] for group in coordinates],names,['新增确认 A','最差形态 B','全事件 Top-1 C'],100,'事件比例（%）')
        else:
            chart(path,[[x[1] for x in coordinates[0]]],[x[0] for x in coordinates[0]],[],22,'确认报警窗口比例（%）')
    FIGURES.append({'label':label,'caption':caption,'image':str(path),'coordinates':coordinates})
    return '\n\n'+protect(f'![图{no}](assets/{name})\n\n**图{no} '+inline(caption)+'**')+'\n\n'

BODY=re.sub(r'\\begin\{table\}.*?\\end\{table\}',table_block,BODY,flags=re.S)
BODY=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',figure_block,BODY,flags=re.S)

def equation_block(m):
    env=m[1]; content=m[2]; label=re.search(r'\\label\{([^}]+)\}',content)
    no=len(EQUATIONS)+1
    if label: REFS[label[1]]=str(no)
    content=mathfix(content)
    if env=='align': content=r'\begin{aligned}'+'\n'+content+'\n'+r'\end{aligned}'
    EQUATIONS.append({'label':label[1] if label else '', 'math':content})
    return '\n\n'+protect('$$\n'+content+'\n$$\n\n（'+str(no)+'）')+'\n\n'
BODY=re.sub(r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}',equation_block,BODY,flags=re.S)

BODY=re.sub(r'\\begin\{abstract\}', '\n\n**摘要**\n\n',BODY)
BODY=BODY.replace(r'\end{abstract}','\n\n')
section_count=0;sub_count=0

def heading(m):
    global section_count,sub_count
    if m[1]=='section':
        section_count+=1;sub_count=0;return f'\n\n## {section_count} {m[2]}\n\n'
    sub_count+=1;return f'\n\n### {section_count}.{sub_count} {m[2]}\n\n'
BODY=re.sub(r'\\(section|subsection)\{([^}]+)\}',heading,BODY)
BODY=re.sub(r'\\(?:eqref|ref)\{([^}]+)\}',lambda m: '('+REFS[m[1]]+')' if m[0].startswith(r'\eqref') else REFS[m[1]],BODY)

def description(m):
    pieces=[]
    for item in re.split(r'\\item\[',m[1])[1:]:
        lab,text=item.split(']',1);pieces.append('- **'+inline(lab)+'** '+inline(''.join(text.strip().splitlines())))
    return '\n\n'+protect('\n'.join(pieces))+'\n\n'
BODY=re.sub(r'\\begin\{description\}(?:\[[^]]*\])?(.*?)\\end\{description\}',description,BODY,flags=re.S)

def enumerate_list(m):
    items=[inline(''.join(s.strip().splitlines())) for s in re.split(r'\\item\s*',m[1])[1:]]
    return '\n\n'+protect('\n'.join(f'{i}. {s}' for i,s in enumerate(items,1)))+'\n\n'
BODY=re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}',enumerate_list,BODY,flags=re.S)
BODY=inline(BODY)
BODY=re.sub(r'\\(?:maketitle|noindent|centering|small|FloatBarrier)\b','',BODY)
BODY=BODY.replace(r'\noindent','').replace(r'\thispagestyle{plain}','').replace(r'\begin{center}','\n\n').replace(r'\end{center}','\n\n').replace(r'\par','\n\n').replace(r'\qquad',' ').replace(r'\quad',' ')
BODY=re.sub(r'\\allowbreak\{\}','',BODY)
# Join source-wrapped prose without inserting spaces between Chinese characters.
BODY=re.sub(r'(?<!\n)\n(?!\n)',lambda m:'\n' if BODY[max(0,m.start()-1):m.start()]=='#' else '',BODY)
# Headings and tokens are already separated by blank lines; restore protected material.
for _ in range(3):
    BODY=re.sub(r'ZZTOKEN\d+ZZ',lambda m:TOKENS[m[0]],BODY)
TITLE=argument(SRC,r'\title').replace('\\\\',' ')
DATE=argument(SRC,r'\date')
MD='# '+TITLE+'\n\n**日期：** '+DATE+'\n\n'+BODY.strip()+'\n'
MD=re.sub(r'\n{3,}','\n\n',MD)
MD=re.sub(r'\*\*([^*\n]+)\*\*', lambda m: '**' + m[1].strip() + '**', MD)
assert '下一阶段应优先完成冻结方案' not in MD
assert 'ZZTOKEN' not in MD
assert not re.search(r'\\(?:begin\{(?:tikzpicture|table|figure)|caption|label|section|subsection)',MD)
(BASE/'MENTOR_REPORT.md').write_text(MD,encoding='utf-8')
(BASE/'assets'/'report_conversion_audit.json').write_text(json.dumps({'source_sha256':source_hash,'tables':TABLES,'figures':FIGURES,'equations':EQUATIONS,'refs':REFS},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'markdown':str(BASE/'MENTOR_REPORT.md'),'tables':len(TABLES),'figures':len(FIGURES),'equations':len(EQUATIONS),'source_sha256':source_hash},ensure_ascii=False))
