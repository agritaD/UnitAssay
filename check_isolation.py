#!/usr/bin/env python3
"""check_isolation.py — prove the two trees do not import from each other.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).

    python3 check_isolation.py

TWO COPIES OF THIS CODE EXIST while unitassay is being prepared for release: the
lab tree it was extracted from, and this package. They already drifted once
within hours -- unitassay was missing an ambiguity guard the lab tree had -- so
"they are the same" is a claim that has to be re-checked, not assumed.

WHAT THIS CHECKS, and it is not the same as diffing the files:

  1. IMPORT PROVENANCE. Each tree is imported in a CLEAN subprocess and every
     module it loads is traced back to a file. If any module of one tree
     resolves inside the other, the trees are entangled and a change to one
     silently changes the other.
  2. LOGIC EQUIVALENCE. Every shared module is normalised (docstrings, comments
     and package prefixes removed) and compared. Only the deliberate
     differences are allowed, and they are listed here rather than tolerated
     silently.

Exit code 0 = isolated and equivalent. Non-zero = say what changed before
anyone runs either tree.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = Path('/bdz/pesaranlab/Troopa_Behavior_Ephys_Hexa/matlab/mfiles/agrita/'
           'SpikeSortingPaper/claudeSS/code')

# Differences that are CORRECT and must not be "fixed" into agreement:
# the lab tree pins mind_snag because three copies sit on its sys.path;
# unitassay contains exactly one copy, so the pin has nothing to disambiguate.
ALLOWED = {'curation.py': {'from Utils.mind_snag_pin import pin', 'pin()'}}

PROBE = """
import sys, json
sys.path.insert(0, {root!r})
{imp}
mods = {{}}
for name, m in list(sys.modules.items()):
    f = getattr(m, '__file__', None)
    if f and ('unitassay' in f or 'claudeSS' in f):
        mods[name] = f
print(json.dumps(mods))
"""


def provenance(root, imp, tag):
    """Every project module a tree loads, and where it came from."""
    r = subprocess.run([sys.executable, '-c', PROBE.format(root=root, imp=imp)],
                       capture_output=True, text=True, cwd='/tmp')
    if r.returncode != 0:
        print(f'  {tag}: FAILED to import\n{r.stderr[-500:]}')
        return None
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


def norm(p):
    t = p.read_text()
    t = re.sub(r'""".*?"""', '', t, flags=re.S)
    t = re.sub(r'#.*', '', t)
    t = re.sub(r'\bunitassay\.curate\b', 'Utils', t)
    t = re.sub(r'\bunitassay\.isolate\b', 'mind_snag', t)
    t = re.sub(r'\bcurate/', 'Utils/', t)
    return [l.strip() for l in t.splitlines() if l.strip()]


def main():
    bad = 0

    print('1. IMPORT PROVENANCE — does either tree reach into the other?\n')
    checks = [
        ('unitassay', str(HERE), 'import unitassay; from unitassay import curate',
         'unitassay', 'claudeSS'),
        ('lab tree', str(LAB), 'from Utils.curation import curate_session',
         'claudeSS', 'unitassay'),
    ]
    for tag, root, imp, own, foreign in checks:
        mods = provenance(root, imp, tag)
        if mods is None:
            bad += 1
            continue
        leak = {k: v for k, v in mods.items() if foreign in v}
        print(f'  {tag:<11} loaded {len(mods):>3} project modules, '
              f'{len(leak)} from the other tree')
        for k, v in list(leak.items())[:5]:
            print(f'      LEAK {k} <- {v}')
        bad += bool(leak)

    print('\n2. LOGIC EQUIVALENCE — same rules in both?\n')
    U = HERE / 'unitassay' / 'curate'
    for f in sorted(U.glob('*.py')):
        g = LAB / 'Utils' / f.name
        if not g.exists():
            print(f'  {f.name:<20} unitassay only (package __init__)')
            continue
        import difflib
        d = [l[1:].strip() for l in difflib.unified_diff(norm(g), norm(f), n=0)
             if l[0] in '+-' and not l.startswith(('---', '+++'))]
        extra = [l for l in d if l not in ALLOWED.get(f.name, set())]
        if extra:
            bad += 1
            print(f'  {f.name:<20} {len(extra)} UNEXPECTED differing lines')
            for l in extra[:4]:
                print(f'      {l[:90]}')
        else:
            note = f'  ({len(d)} allowed)' if d else ''
            print(f'  {f.name:<20} equivalent{note}')

    print('\n' + ('ISOLATED AND EQUIVALENT' if not bad else
                  f'{bad} PROBLEM(S) — resolve before running either tree'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
