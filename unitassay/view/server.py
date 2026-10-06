"""view/server.py — a local HTTP server for the viewer. Standard library only.

    unitassay view <out_dir> [--port 8765] [--events times.npy]

Serves one HTML page (view/static/index.html) and a small read-only JSON API
over a unitassay output directory. No new dependency: the package has seven and
a viewer is not a reason for an eighth.

BINDS TO 127.0.0.1 BY DEFAULT. The data are unpublished recordings; on a shared
server, reach it from a laptop with an SSH tunnel
(`ssh -L 8765:localhost:8765 <server>`), not by opening a port.

API (all GET, all JSON):
    /api/sessions                       one line per session under the root
    /api/session/<name>                 verdict rows, pairs, thresholds, geometry
    /api/cluster/<name>/<clu>           layer 1 + layer 2 for one cluster
    /api/ccg/<name>/<a>/<b>             a pair's CCG and rule 5's statistic on it
    /api/psth/<name>/<clu>              only when --events was given

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
import json
import sys
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

from unitassay.view.data import Run

STATIC = Path(__file__).parent / 'static'


def _handler(run):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):          # quiet: one line per error only
            pass

        def _send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(_finite(obj), default=_default,
                                        allow_nan=False))

        def do_GET(self):
            parts = [unquote(p) for p in urlparse(self.path).path.split('/') if p]
            try:
                if not parts or parts == ['index.html']:
                    return self._send(200, (STATIC / 'index.html').read_bytes(),
                                      'text/html; charset=utf-8')
                if parts[0] != 'api':
                    return self._json({'error': 'not found'}, 404)
                kind, args = parts[1] if len(parts) > 1 else '', parts[2:]
                if kind == 'sessions':
                    return self._json(dict(root=str(run.root),
                                           sessions=run.sessions()))
                s = run.session(args[0])
                if kind == 'session':
                    return self._json(s.overview())
                if kind == 'cluster':
                    return self._json(s.cluster(int(args[1])))
                if kind == 'ccg':
                    return self._json(s.ccg(int(args[1]), int(args[2])))
                if kind == 'psth':
                    return self._json(s.psth(int(args[1])))
                return self._json({'error': f'unknown endpoint {kind!r}'}, 404)
            except KeyError as e:
                return self._json({'error': f'no session {e}'}, 404)
            except (IndexError, ValueError) as e:
                return self._json({'error': f'bad request: {e}'}, 400)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
    return H


def _finite(o):
    """NaN/inf -> None, recursively. curated.json rows carry NaN (a dip p-value
    that could not be formed, say), and a browser's JSON.parse rejects NaN."""
    if isinstance(o, float):
        return o if o == o and o not in (float('inf'), float('-inf')) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o).__name__)


def serve(root, host='127.0.0.1', port=8765, events=None):
    run = Run(root, events=events)
    httpd = ThreadingHTTPServer((host, port), _handler(run))
    n = len(run.dirs)
    print(f'unitassay view: {n} session{"s" * (n != 1)} under {run.root}')
    print(f'  open http://{host}:{port}/   (Ctrl-C to stop)')
    if host in ('127.0.0.1', 'localhost'):
        print(f'  from another machine: ssh -L {port}:localhost:{port} <this host>')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
