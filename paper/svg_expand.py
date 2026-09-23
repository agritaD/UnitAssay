"""Turn matplotlib's `<use>` marker instances into standalone `<path>` elements.

WHY THIS EXISTS. The SVG backend draws repeated markers efficiently: one
`<path>` in `<defs>` and a `<use xlink:href="#...">` per point. That is valid
vector -- no raster anywhere -- but Illustrator imports `<use>` as symbol
INSTANCES, which behave like a single uneditable object until expanded. A
figure whose scatter cannot be edited is no better than a picture to whoever
has to finish it.

Expanding costs file size and nothing else: each `<use>` becomes the referenced
path with a translate.
"""
from __future__ import annotations

import re
from pathlib import Path

SVG_NS = 'http://www.w3.org/2000/svg'
XLINK_NS = 'http://www.w3.org/1999/xlink'


def expand_use(svg_path, out_path=None):
    """Rewrite every `<use>` as the `<path>` it points at. Returns the output."""
    import xml.etree.ElementTree as ET

    ET.register_namespace('', SVG_NS)
    ET.register_namespace('xlink', XLINK_NS)
    svg_path = Path(svg_path)
    out_path = Path(out_path or svg_path)
    tree = ET.parse(svg_path)
    root = tree.getroot()

    symbols = {}
    for el in root.iter(f'{{{SVG_NS}}}path'):
        i = el.get('id')
        if i and el.get('d'):
            symbols[i] = el.get('d')

    n = 0
    for parent in root.iter():
        for i, el in enumerate(list(parent)):
            if el.tag != f'{{{SVG_NS}}}use':
                continue
            ref = (el.get(f'{{{XLINK_NS}}}href') or el.get('href') or '').lstrip('#')
            d = symbols.get(ref)
            if d is None:
                continue
            x, y = float(el.get('x', 0)), float(el.get('y', 0))
            new = ET.Element(f'{{{SVG_NS}}}path', {'d': d})
            if x or y:
                new.set('transform', f'translate({x:.6g} {y:.6g})')
            if el.get('style'):
                new.set('style', el.get('style'))
            for a in ('fill', 'stroke', 'stroke-width', 'clip-path', 'opacity'):
                if el.get(a):
                    new.set(a, el.get(a))
            parent.remove(el)
            parent.insert(i, new)
            n += 1
    tree.write(out_path, xml_declaration=True, encoding='utf-8')
    return out_path, n


def report(svg_path):
    """What a reader would find in the file: rasters, instances, real marks."""
    t = Path(svg_path).read_text(errors='ignore')
    return dict(images=len(re.findall(r'<image', t)),
                base64=len(re.findall(r'data:image', t)),
                use_instances=len(re.findall(r'<use', t)),
                paths=len(re.findall(r'<path', t)),
                text=len(re.findall(r'<text', t)))


def drop_clips(svg_path, out_path=None):
    """Remove clipping masks, so nothing has to be entered before it is editable.

    matplotlib clips each axes' contents to its own box. That is correct for
    display, but Illustrator imports every clip as a clipping GROUP: the marks
    inside are editable, yet a click selects the mask first and the group has
    to be entered. On a figure whose data already sits inside its axes the
    clips buy nothing.

    DO NOT APPLY BLIND. Tried on Figure 3 C/D it BROKE the figure: the
    correlogram is on a log y-axis, so its bars are drawn to an enormous
    negative extent and only the clip keeps them inside the panel. Without it
    they flood the panel below. Render the result and look at it before
    keeping it.
    """
    import xml.etree.ElementTree as ET

    ET.register_namespace('', SVG_NS)
    ET.register_namespace('xlink', XLINK_NS)
    svg_path = Path(svg_path)
    out_path = Path(out_path or svg_path)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    n = 0
    for el in root.iter():
        if el.get('clip-path'):
            del el.attrib['clip-path']
            n += 1
    for parent in root.iter():
        for c in list(parent):
            if c.tag == f'{{{SVG_NS}}}clipPath':
                parent.remove(c)
    tree.write(out_path, xml_declaration=True, encoding='utf-8')
    return out_path, n
