"""Fresh loopback REST state per task and arm. No reset endpoint is exposed to the model."""
from __future__ import annotations

import copy
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .common import schema, strict_json, tool

S = {'type': 'string', 'maxLength': 128}
Q = {'type': 'integer', 'minimum': 1}
TOOLS = [tool('list_items', 'List all inventory items with live stock and integer unit prices.', {}),
         tool('get_item', 'Read live item stock, name and unit price.', {'item': S}),
         tool('reserve', 'Reserve available stock into a new named order. Decreases stock.', {'item': S, 'qty': Q, 'order': S}),
         tool('cancel', 'Cancel a reserved order, returning its units to stock.', {'order': S}),
         tool('ship', 'Mark a reserved order shipped; stock already decreased at reservation.', {'order': S}),
         tool('restock', 'Add positive units to stock.', {'item': S, 'qty': Q}),
         tool('rename', 'Change an item display name.', {'item': S, 'name': S})]


class Store:
    def __init__(self, initial):
        self.state = copy.deepcopy(initial)
        self.trace = []

    def execute(self, name, args):
        definition = next((t for t in TOOLS if t['function']['name'] == name), None)
        if definition is None:
            raise ValueError('unknown tool')
        schema(args, definition['function']['parameters'])
        items, orders = self.state['items'], self.state['orders']
        if name == 'list_items':
            result = copy.deepcopy(items)
        elif name in ('get_item', 'reserve', 'restock', 'rename'):
            if args['item'] not in items:
                raise ValueError('unknown item')
            item = items[args['item']]
            if name == 'reserve':
                if args['order'] in orders or args['qty'] > item['stock']:
                    raise ValueError('duplicate order or insufficient stock')
                item['stock'] -= args['qty']
                orders[args['order']] = {'item': args['item'], 'qty': args['qty'], 'status': 'reserved',
                                        'total': args['qty'] * item['price']}
                result = copy.deepcopy(orders[args['order']])
            else:
                if name == 'restock':
                    item['stock'] += args['qty']
                if name == 'rename':
                    item['name'] = args['name']
                result = copy.deepcopy(item)
        else:
            order = orders.get(args['order'])
            if order is None or order['status'] != 'reserved':
                raise ValueError('order is not reserved')
            if name == 'cancel':
                items[order['item']]['stock'] += order['qty']
                order['status'] = 'cancelled'
            else:
                order['status'] = 'shipped'
            result = copy.deepcopy(order)
        self.trace.append({'name': name, 'args': copy.deepcopy(args), 'result': result})
        return result


class MockREST:
    tools = TOOLS

    def __init__(self, initial):
        self.store = Store(initial)
        store = self.store
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    if not self.path.startswith('/tool/'):
                        raise ValueError('unknown endpoint')
                    size = int(self.headers.get('Content-Length', '0'))
                    if size < 0 or size > 65536:
                        raise ValueError('invalid body size')
                    args = strict_json(self.rfile.read(size).decode('utf-8'))
                    result, status = store.execute(self.path.removeprefix('/tool/'), args), 200
                except (ValueError, KeyError, TypeError) as exc:
                    result, status = {'error': str(exc)}, 400
                data = json.dumps(result).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def execute(self, name, args):
        # Reject names before URL construction; only known loopback routes can be reached.
        if name not in {t['function']['name'] for t in self.tools}:
            raise ValueError('unknown tool')
        request = urllib.request.Request(f'http://127.0.0.1:{self.server.server_port}/tool/{name}',
                                         json.dumps(args).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return strict_json(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise ValueError(strict_json(exc.read().decode()).get('error')) from exc

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


def validate_tools(task, store, answer):
    expected = Store(task['initial'])
    for name, args in task['mutations']:
        expected.execute(name, args)
    mutations = [(r['name'], r['args']) for r in store.trace if r['name'] not in ('get_item', 'list_items')]
    required = [(name, args) for name, args in task['mutations']]
    read_first = bool(store.trace) and store.trace[0]['name'] in ('get_item', 'list_items')
    try:
        answer_ok = strict_equal(strict_json(answer), task['answer'])
    except (ValueError, TypeError):
        answer_ok = False
    return {'ok': read_first and mutations == required and store.state == expected.state and answer_ok,
            'read_first': read_first, 'sequence_ok': mutations == required,
            'state_ok': store.state == expected.state, 'answer_ok': answer_ok,
            'final_state': copy.deepcopy(store.state)}


def strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b
