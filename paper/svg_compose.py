"""Stack SVGs into one page without flattening either of them.

WHY NOT COMPOSITE AS PDF. The obvious route -- lay the artwork and the drawn
panels into one PDF, then convert -- destroys the thing that matters:
`pdftocairo` outlines every glyph, so the drawn panels arrive with no editable
text. Composing at the SVG level keeps matplotlib's `<text>` as text. The
artwork half is still outlined, because it enters as a converted PDF, but that
half is the one whose original `.ai` the author already has.

IDs ARE GLOBAL IN SVG. Two files each defining `#glyph0-1` or a clip path of
the same name will silently cross-reference after a naive merge, and marks
vanish or take the wrong shape. Every id is prefixed per source.
"""
from __future__ import annotations

import re
from pathlib import Path

SVG_NS = 'http://www.w3.org/2000/svg'
XLINK_NS = 'http://www.w3.org/1999/xlink'


def _size(root):
    def num(v):
        return float(re.sub(r'[a-z]+$', '', v)) if v else None
    w, h = num(root.get('width')), num(root.get('height'))
    if w and h:
        return w, h
    vb = (root.get('viewBox') or '').split()
    return (float(vb[2]), float(vb[3])) if len(vb) == 4 else (None, None)


def _prefix_ids(root, pre):
    import xml.etree.ElementTree as ET
    ids = {el.get('id') for el in root.iter() if el.get('id')}
    for el in root.iter():
        if el.get('id') in ids:
            el.set('id', pre + el.get('id'))
        for attr in (f'{{{XLINK_NS}}}href', 'href'):
            v = el.get(attr)
            if v and v.startswith('#') and v[1:] in ids:
                el.set(attr, '#' + pre + v[1:])
        for attr in ('clip-path', 'fill', 'stroke', 'mask', 'filter'):
            v = el.get(attr)
            if v and v.startswith('url(#') and v[5:-1] in ids:
                el.set(attr, f'url(#{pre}{v[5:-1]})')
        st = el.get('style')
        if st and 'url(#' in st:
            el.set('style', re.sub(r'url\(#([^)]+)\)',
                                   lambda m: f'url(#{pre}{m.group(1)})'
                                   if m.group(1) in ids else m.group(0), st))


def stack(parts, out_path, gap=0.0):
    """Stack SVGs top to bottom on one page. `parts` are paths, in order."""
    import xml.etree.ElementTree as ET
    ET.register_namespace('', SVG_NS)
    ET.register_namespace('xlink', XLINK_NS)

    loaded = []
    for i, p in enumerate(parts):
        r = ET.parse(str(p)).getroot()
        _prefix_ids(r, f's{i}_')
        loaded.append((r, *_size(r)))

    width = max(w for _, w, _ in loaded)
    height = sum(h for _, _, h in loaded) + gap * (len(loaded) - 1)
    out = ET.Element(f'{{{SVG_NS}}}svg', {
        'width': f'{width}pt', 'height': f'{height}pt',
        'viewBox': f'0 0 {width} {height}', 'version': '1.1'})
    y = 0.0
    for r, w, h in loaded:
        g = ET.SubElement(out, f'{{{SVG_NS}}}g',
                          {'transform': f'translate(0 {y:.6g})'})
        for child in list(r):
            g.append(child)
        y += h + gap
    ET.ElementTree(out).write(out_path, xml_declaration=True, encoding='utf-8')
    return Path(out_path), width, height
