#!/usr/bin/env python3
"""Minimal, doc-specific Markdown -> LaTeX for the summary-of-changes document."""
import re, sys

SRC = sys.argv[1]
OUT = sys.argv[2]

SPECIAL = {
    '\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$',
    '&': r'\&', '#': r'\#', '_': r'\_', '%': r'\%',
    '^': r'\textasciicircum{}', '~': r'\textasciitilde{}', '|': r'\textbar{}',
}
UNI = {
    '\u2014': '---', '\u2013': '--', '\u2192': r'$\to$', '\u03c1': r'$\rho$',
    '\u2248': r'$\approx$', '\u03c4': r'$\tau$', '\u00b1': r'$\pm$',
    '\u2212': r'$-$', '\u00d7': r'$\times$', '\u00a7': r'\S{}',
    '\u2026': r'\ldots{}', '\u2208': r'$\in$', '\u0394': r'$\Delta$',
    '\u2265': r'$\ge$',
}

def esc(t):
    return ''.join(SPECIAL.get(c, UNI.get(c, c)) for c in t)

def code(c):
    e = esc(c)  # allow long paths/identifiers to break
    e = e.replace('/', '/\\allowbreak{}').replace('\\_', '\\_\\allowbreak{}')
    return r'\texttt{' + e + '}'

def inline(t):
    codes = []
    def stash(m):
        codes.append(m.group(1)); return f'\x00{len(codes)-1}\x00'
    t = re.sub(r'`([^`]+)`', stash, t)
    t = re.sub(r'"([^"]*)"', lambda m: '``' + m.group(1) + "''", t)
    t = esc(t)
    t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t)
    t = re.sub(r'\*(.+?)\*', r'\\textit{\1}', t)
    t = re.sub(r'\x00(\d+)\x00', lambda m: code(codes[int(m.group(1))]), t)
    return t

def is_table_sep(line):
    return bool(re.match(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$', line))

def split_row(line):
    line = line.strip().strip('|')
    line = line.replace(r'\|', '\x01')
    cells = [c.strip().replace('\x01', '|') for c in line.split('|')]
    return cells

lines = open(SRC, encoding='utf-8').read().split('\n')
out = []
i = 0
n = len(lines)
while i < n:
    line = lines[i]
    s = line.strip()

    # code fence
    if s.startswith('```'):
        i += 1
        buf = []
        while i < n and not lines[i].strip().startswith('```'):
            buf.append(lines[i]); i += 1
        i += 1
        out.append(r'\begin{quote}\ttfamily\footnotesize')
        for b in buf:
            out.append(esc(b) + r'\\')   # natural spaces so long lines wrap
        out.append(r'\end{quote}')
        continue

    # horizontal rule
    if s == '---':
        out.append(r'\vspace{2pt}\hrule\vspace{6pt}')
        i += 1; continue

    # heading
    m = re.match(r'^(#{1,4})\s+(.*)$', s)
    if m:
        lvl = len(m.group(1)); txt = inline(m.group(2))
        cmd = {1: r'\section*', 2: r'\subsection*', 3: r'\subsubsection*',
               4: r'\paragraph'}[lvl]
        out.append(f'{cmd}{{{txt}}}')
        i += 1; continue

    # table
    if s.startswith('|') and i + 1 < n and is_table_sep(lines[i+1]):
        header = split_row(lines[i]); i += 2
        rows = []
        while i < n and lines[i].strip().startswith('|'):
            rows.append(split_row(lines[i])); i += 1
        ncol = len(header)
        out.append(r'\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}')
        out.append(r'\begin{tabular}{' + 'l' * ncol + '}')
        out.append(r'\toprule')
        out.append(' & '.join(inline(c) for c in header) + r' \\')
        out.append(r'\midrule')
        for r in rows:
            r = (r + [''] * ncol)[:ncol]
            out.append(' & '.join(inline(c) for c in r) + r' \\')
        out.append(r'\bottomrule')
        out.append(r'\end{tabular}')
        out.append(r'\end{center}')
        continue

    # blockquote
    if s.startswith('>'):
        buf = []
        while i < n and lines[i].strip().startswith('>'):
            buf.append(re.sub(r'^\s*>\s?', '', lines[i])); i += 1
        out.append(r'\begin{quote}\itshape')
        out.append(inline(' '.join(x.strip() for x in buf if x.strip())))
        out.append(r'\end{quote}')
        continue

    # unordered list
    if re.match(r'^-\s+', s):
        out.append(r'\begin{itemize}\setlength{\itemsep}{2pt}')
        while i < n:
            ls = lines[i]
            if re.match(r'^-\s+', ls.strip()) and not ls.startswith('  '):
                item = [re.sub(r'^-\s+', '', ls.strip())]; i += 1
                while i < n and lines[i].startswith('  ') and lines[i].strip():
                    item.append(lines[i].strip()); i += 1
                out.append(r'\item ' + inline(' '.join(item)))
            elif ls.strip() == '':
                # peek: continue only if next non-blank is another item
                j = i + 1
                while j < n and lines[j].strip() == '':
                    j += 1
                if j < n and re.match(r'^-\s+', lines[j].strip()):
                    i += 1; continue
                break
            else:
                break
        out.append(r'\end{itemize}')
        continue

    # blank
    if s == '':
        out.append(''); i += 1; continue

    # paragraph (gather consecutive plain lines)
    buf = [line]; i += 1
    while i < n:
        nx = lines[i].strip()
        if (nx == '' or nx.startswith('#') or nx.startswith('>') or nx == '---'
                or nx.startswith('```') or (lines[i].strip().startswith('|'))
                or re.match(r'^-\s+', nx)):
            break
        buf.append(lines[i]); i += 1
    out.append(inline(' '.join(x.strip() for x in buf)))
    out.append('')

BODY = '\n'.join(out)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amssymb}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\usepackage{enumitem}
\setlist{leftmargin=1.4em}
\begin{document}
"""
open(OUT, 'w', encoding='utf-8').write(PREAMBLE + BODY + '\n\\end{document}\n')
print("wrote", OUT)
