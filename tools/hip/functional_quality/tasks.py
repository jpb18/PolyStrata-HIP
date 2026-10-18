"""Predefined tasks and executable oracles. Nothing here is sent except prompt/files/tools.

Reference implementations exist solely for CPU validation before freeze. Hidden cases
never enter the evaluated model's message history or readable project files.
"""
from __future__ import annotations

import copy

CODE_SYSTEM = '''Work on the isolated Python project using the supplied tools. Read solution.py,
edit it, and run the public tests before finishing. Only solution.py is writable.
Use pure Python without imports, I/O, introspection, decorators, classes or private names.
Available builtins: abs, all, any, bool, dict, enumerate, float, int, isinstance, len, list,
max, min, range, reversed, round, set, sorted, str, sum, tuple, zip, ValueError, TypeError.
Ordinary public list/dict/set/string methods are supported. Hidden tests run only after
you finish. Return a brief completion message; source code in the message is not applied.'''


def code_task(n, description, signature, reference, public, hidden):
    name = signature.split('(')[0]
    return {'id': f'C{n:02}', 'category': 'code', 'system': CODE_SYSTEM,
            'prompt': description, 'files': {'solution.py': f'def {signature}:\n    raise ValueError("TODO")\n'},
            'function': name, 'public': public, 'hidden': hidden, 'reference': reference,
            'max_turns': 10, 'max_tokens': 2048}


def case(args, expected=None, raises=None):
    return {'args': args, **({'raises': raises} if raises else {'expected': expected})}


def code_tasks():
    tasks = [
        code_task(1, 'Implement clamp(x, low, high). Clamp inclusively; low > high raises ValueError.',
                  'clamp(x, low, high)', 'def clamp(x, low, high):\n    if low > high: raise ValueError()\n    return min(max(x, low), high)\n',
                  [case([7, 0, 5], 5)], [case([-9, -4, 6], -4), case([2, 2, 2], 2), case([0, 4, 1], raises='ValueError')]),
        code_task(2, 'Implement chunks(values, size): consecutive lists of at most size. Empty input gives []. Nonpositive size raises ValueError.',
                  'chunks(values, size)', 'def chunks(values, size):\n    if size <= 0: raise ValueError()\n    return [values[i:i+size] for i in range(0,len(values),size)]\n',
                  [case([[1, 2, 3], 2], [[1, 2], [3]])], [case([[], 2], []), case([[1], 5], [[1]]), case([[1], 0], raises='ValueError')]),
        code_task(3, 'Implement unique(values): stable deduplication by equality, including unhashable nested lists. Do not mutate input.',
                  'unique(values)', 'def unique(values):\n    out=[]\n    for x in values:\n        if x not in out: out.append(x)\n    return out\n',
                  [case([[3, 1, 3]], [3, 1])], [case([[[1], [1], [2]]], [[1], [2]]), case([[]], [])]),
        code_task(4, 'Implement pairs(text): parse semicolon-separated key=value pairs, trimming whitespace; ignore empty segments. Reject missing =, empty keys and duplicate keys with ValueError. Values may contain =.',
                  'pairs(text)', 'def pairs(text):\n    out={}\n    for part in text.split(";"):\n        if not part.strip(): continue\n        if "=" not in part: raise ValueError()\n        k,v=part.split("=",1)\n        k=k.strip()\n        if not k or k in out: raise ValueError()\n        out[k]=v.strip()\n    return out\n',
                  [case(['a=1; b=2'], {'a': '1', 'b': '2'})], [case(['; x=a=b;;'], {'x': 'a=b'}), case(['x=1;x=2'], raises='ValueError'), case(['=2'], raises='ValueError'), case(['oops'], raises='ValueError')]),
        code_task(5, 'Implement merge(intervals): sort and merge closed integer intervals including touching endpoints. Reject reversed intervals. Return lists, do not mutate input.',
                  'merge(intervals)', 'def merge(intervals):\n    out=[]\n    for a,b in sorted(intervals):\n        if a>b: raise ValueError()\n        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)\n        else: out.append([a,b])\n    return out\n',
                  [case([[[1, 3], [2, 4]]], [[1, 4]])], [case([[[7, 8], [1, 2], [2, 2]]], [[1, 2], [7, 8]]), case([[]], []), case([[[3, 1]]], raises='ValueError')]),
        code_task(6, 'Implement topo(graph): graph maps each node to its prerequisites; include nodes appearing only as prerequisites. Return lexicographically smallest available node at each step (Kahn order). Cycle raises ValueError.',
                  'topo(graph)', 'def topo(graph):\n    nodes=set(graph)\n    for deps in graph.values(): nodes.update(deps)\n    out=[]\n    while nodes:\n        ready=sorted(n for n in nodes if all(d in out for d in graph.get(n,[])))\n        if not ready: raise ValueError()\n        n=ready[0]\n        out.append(n)\n        nodes.remove(n)\n    return out\n',
                  [case([{'b': ['a'], 'a': []}], ['a', 'b'])], [case([{'c': ['a'], 'b': []}], ['a', 'b', 'c']), case([{'a': ['b'], 'b': ['a']}], raises='ValueError'), case([{}], [])]),
        code_task(7, 'Implement rle(values): consecutive equal values become [value,count] lists. Empty returns []. Do not mutate input.',
                  'rle(values)', 'def rle(values):\n    out=[]\n    for x in values:\n        if out and out[-1][0]==x: out[-1][1]+=1\n        else: out.append([x,1])\n    return out\n',
                  [case([[1, 1, 2]], [[1, 2], [2, 1]])], [case([[]], []), case([[1, 2, 1]], [[1, 1], [2, 1], [1, 1]]), case([[None, None]], [[None, 2]])]),
        code_task(8, 'Implement transpose(rows): rectangular matrix transpose returning lists. [] and rows of zero width return []. Ragged rows raise ValueError.',
                  'transpose(rows)', 'def transpose(rows):\n    if not rows: return []\n    w=len(rows[0])\n    if any(len(r)!=w for r in rows): raise ValueError()\n    return [[r[i] for r in rows] for i in range(w)]\n',
                  [case([[[1, 2], [3, 4]]], [[1, 3], [2, 4]])], [case([[]], []), case([[[], []]], []), case([[[1], [2, 3]]], raises='ValueError')]),
        code_task(9, 'Implement deep_merge(a,b): recursively merge dicts; on conflicts use b unless both values are dicts. No input mutation. Inputs contain JSON values.',
                  'deep_merge(a, b)', 'def deep_merge(a,b):\n    out=dict(a)\n    for k,v in b.items():\n        if k in out and isinstance(out[k],dict) and isinstance(v,dict): out[k]=deep_merge(out[k],v)\n        else: out[k]=v\n    return out\n',
                  [case([{'x': 1}, {'y': 2}], {'x': 1, 'y': 2})], [case([{'x': {'a': 1, 'b': 2}}, {'x': {'b': 9}}], {'x': {'a': 1, 'b': 9}}), case([{'x': [1]}, {'x': None}], {'x': None})]),
        code_task(10, 'Implement rolling(values,k): sums of all complete consecutive windows of length k. k<=0 raises ValueError; k>len gives [].',
                  'rolling(values, k)', 'def rolling(values,k):\n    if k<=0: raise ValueError()\n    return [sum(values[i:i+k]) for i in range(len(values)-k+1)]\n',
                  [case([[1, 2, 3], 2], [3, 5])], [case([[-1, 2, -3], 1], [-1, 2, -3]), case([[1], 2], []), case([[], 0], raises='ValueError')]),
        code_task(11, 'Implement backoff(n,base,cap): n delays base*2**i capped at cap. n<0 or base<0 or cap<0 raises ValueError.',
                  'backoff(n, base, cap)', 'def backoff(n,base,cap):\n    if n<0 or base<0 or cap<0: raise ValueError()\n    return [min(base*2**i,cap) for i in range(n)]\n',
                  [case([4, 2, 5], [2, 4, 5, 5])], [case([0, 1, 9], []), case([3, 0, 1], [0, 0, 0]), case([-1, 1, 2], raises='ValueError')]),
        code_task(12, 'Implement duration(text): unsigned integer followed by ms, s, m or h. Return integer milliseconds. Reject whitespace, decimals, signs, empty input and unknown suffix with ValueError.',
                  'duration(text)', 'def duration(text):\n    units={"ms":1,"s":1000,"m":60000,"h":3600000}\n    for u in units:\n        if text.endswith(u):\n            n=text[:-len(u)]\n            if not n or any(c not in "0123456789" for c in n): raise ValueError()\n            return int(n)*units[u]\n    raise ValueError()\n',
                  [case(['2s'], 2000)], [case(['15ms'], 15), case(['0h'], 0), case(['-2m'], raises='ValueError'), case([' 2s'], raises='ValueError'), case(['2.5s'], raises='ValueError'), case(['s'], raises='ValueError')]),
        code_task(13, 'Implement normalize(path): normalize an absolute POSIX path: collapse // and ., resolve .. without going above root. Remove trailing / except root. Relative paths raise ValueError.',
                  'normalize(path)', 'def normalize(path):\n    if not path.startswith("/"): raise ValueError()\n    out=[]\n    for x in path.split("/"):\n        if x=="..":\n            if out: out.pop()\n        elif x and x!=".": out.append(x)\n    return "/"+"/".join(out)\n',
                  [case(['/a/../b//'], '/b')], [case(['/../../'], '/'), case(['/a/./b'], '/a/b'), case(['a'], raises='ValueError')]),
        code_task(14, 'Implement apply(stock,changes): changes is [item,delta] pairs applied in order to a COPY of stock; unknown item starts at 0. Any intermediate negative count raises ValueError. Never mutate inputs, even on failure.',
                  'apply(stock, changes)', 'def apply(stock,changes):\n    out=dict(stock)\n    for k,v in changes:\n        out[k]=out.get(k,0)+v\n        if out[k]<0: raise ValueError()\n    return out\n',
                  [case([{'a': 3}, [['a', -2]]], {'a': 1})], [case([{}, [['b', 2]]], {'b': 2}), case([{'a': 1}, [['a', -2], ['a', 3]]], raises='ValueError')]),
        code_task(15, 'Implement median(values). Sort a copy. Odd: middle value; even: arithmetic mean. Empty raises ValueError.',
                  'median(values)', 'def median(values):\n    if not values: raise ValueError()\n    v=sorted(values)\n    n=len(v)\n    return v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2\n',
                  [case([[3, 1, 2]], 2)], [case([[9, 1]], 5), case([[]], raises='ValueError')]),
        code_task(16, 'Implement brackets(text): True iff (), [], {} are correctly nested and balanced. Ignore other characters.',
                  'brackets(text)', 'def brackets(text):\n    stack=[]\n    closes={")":"(","]":"[","}":"{"}\n    for c in text:\n        if c in "([{": stack.append(c)\n        elif c in closes:\n            if not stack or stack.pop()!=closes[c]: return False\n    return not stack\n',
                  [case(['([])'], True)], [case(['([)]'], False), case(['abc'], True), case(['('], False)]),
        code_task(17, 'Implement lower_bound(values,x) on sorted integers: first index with value >= x, else len.',
                  'lower_bound(values, x)', 'def lower_bound(values,x):\n    lo,hi=0,len(values)\n    while lo<hi:\n        mid=(lo+hi)//2\n        if values[mid]<x: lo=mid+1\n        else: hi=mid\n    return lo\n',
                  [case([[1, 3, 3], 3], 1)], [case([[], 2], 0), case([[1, 3], 4], 2), case([[1], 0], 0)]),
        code_task(18, 'Implement frequencies(words): case-insensitive string counts; ignore empty strings. Return dict.',
                  'frequencies(words)', 'def frequencies(words):\n    out={}\n    for w in words:\n        if w: out[w.lower()]=out.get(w.lower(),0)+1\n    return out\n',
                  [case([['A', 'a', 'B']], {'a': 2, 'b': 1})], [case([['', 'X', 'x']], {'x': 2}), case([[]], {})]),
        code_task(19, 'Implement flatten(values): recursively flatten nested lists only; preserve non-list values in order.',
                  'flatten(values)', 'def flatten(values):\n    out=[]\n    for x in values:\n        if isinstance(x,list): out.extend(flatten(x))\n        else: out.append(x)\n    return out\n',
                  [case([[1, [2]]], [1, 2])], [case([[[], [None, [3]], 'x']], [None, 3, 'x']), case([[]], [])]),
        code_task(20, 'Implement runs(values): lengths of consecutive strictly increasing runs of numbers; equality ends a run; empty gives [].',
                  'runs(values)', 'def runs(values):\n    if not values: return []\n    out=[1]\n    for i in range(1,len(values)):\n        if values[i]>values[i-1]: out[-1]+=1\n        else: out.append(1)\n    return out\n',
                  [case([[1, 2, 0, 1]], [2, 2])], [case([[2, 2, 3]], [1, 2]), case([[]], []), case([[4, 3, 2]], [1, 1, 1])]),
    ]
    # Include repair/refactoring work as well as implementations; public and hidden
    # criteria remain the same for the stated behavior.
    repairs = {
        1: ('Fix clamp: reversed bounds currently pass silently.',
            'def clamp(x, low, high):\n    return min(max(x, low), high)\n'),
        4: ('Fix the parser so it handles the specified validation and embedded equals.',
            'def pairs(text):\n    return dict(part.strip().split("=") for part in text.split(";"))\n'),
        8: ('Fix transpose: zip currently silently truncates ragged rows.',
            'def transpose(rows):\n    return [list(column) for column in zip(*rows)]\n'),
        14: ('Refactor apply to remove input mutation and validate every intermediate count.',
             'def apply(stock, changes):\n    for item, delta in changes:\n        stock[item] = stock.get(item, 0) + delta\n    return stock\n'),
    }
    for n, (intro, source) in repairs.items():
        tasks[n-1]['prompt'] = intro + '\n' + tasks[n-1]['prompt']
        tasks[n-1]['files']['solution.py'] = source
    return tasks


def initial_store():
    return {'items': {'A': {'name': 'adapter', 'stock': 8, 'price': 12},
                      'B': {'name': 'cable', 'stock': 3, 'price': 5},
                      'C': {'name': 'sensor', 'stock': 0, 'price': 20}}, 'orders': {}}


def tool_tasks():
    # Mutations are the required ordered subsequence; reads are permitted anywhere.
    specs = [
        ('Read item A stock and unit price.', [], {'stock': 8, 'price': 12}),
        ('Reserve 2 units of A as order o1.', [('reserve', {'item': 'A', 'qty': 2, 'order': 'o1'})], {'order': 'o1', 'status': 'reserved'}),
        ('Reserve 1 B as o2, then cancel o2.', [('reserve', {'item': 'B', 'qty': 1, 'order': 'o2'}), ('cancel', {'order': 'o2'})], {'order': 'o2', 'status': 'cancelled'}),
        ('Reserve 3 A as o3, then ship o3.', [('reserve', {'item': 'A', 'qty': 3, 'order': 'o3'}), ('ship', {'order': 'o3'})], {'order': 'o3', 'status': 'shipped'}),
        ('Restock C by 4, then reserve 2 C as o4.', [('restock', {'item': 'C', 'qty': 4}), ('reserve', {'item': 'C', 'qty': 2, 'order': 'o4'})], {'item': 'C', 'stock': 2}),
        ('Read B: if stock is less than 5, add exactly enough to reach 5.', [('restock', {'item': 'B', 'qty': 2})], {'item': 'B', 'stock': 5}),
        ('Read A: if stock is below 4, restock to 4; otherwise leave it unchanged.', [], {'item': 'A', 'stock': 8}),
        ('Rename B to braided cable, then reserve 1 B as o5.', [('rename', {'item': 'B', 'name': 'braided cable'}), ('reserve', {'item': 'B', 'qty': 1, 'order': 'o5'})], {'name': 'braided cable', 'stock': 2}),
        ('Reserve 2 A as o6, then 1 B as o7. Report total cost.', [('reserve', {'item': 'A', 'qty': 2, 'order': 'o6'}), ('reserve', {'item': 'B', 'qty': 1, 'order': 'o7'})], {'total': 29}),
        ('Find the cheapest in-stock item; reserve 2 units as o8.', [('reserve', {'item': 'B', 'qty': 2, 'order': 'o8'})], {'item': 'B', 'total': 10}),
        ('With a budget of 30, reserve as many A as possible as o9; do not exceed stock or budget.', [('reserve', {'item': 'A', 'qty': 2, 'order': 'o9'})], {'qty': 2, 'spent': 24, 'remaining': 6}),
        ('Reserve 2 A as o10; cancel it; reserve 1 A as o11; ship o11.', [('reserve', {'item': 'A', 'qty': 2, 'order': 'o10'}), ('cancel', {'order': 'o10'}), ('reserve', {'item': 'A', 'qty': 1, 'order': 'o11'}), ('ship', {'order': 'o11'})], {'stock': 7, 'shipped': 'o11'}),
    ]
    result = []
    for n, (prompt, mutations, answer) in enumerate(specs, 1):
        result.append({'id': f'T{n:02}', 'category': 'tools',
                       'system': 'Use the supplied inventory tools to accomplish the goal. Inspect live state first; do not invent tool results. End with only a JSON object.',
                       'prompt': prompt + ' Final JSON keys: ' + ', '.join(answer) + '.',
                       'initial': copy.deepcopy(initial_store()), 'mutations': mutations,
                       'answer': answer, 'max_turns': 10, 'max_tokens': 1024})
    return result


def answer_task(id_, category, prompt, expected, validator='exact'):
    return {'id': id_, 'category': category, 'system': 'Follow the task exactly. No markdown or commentary.',
            'prompt': prompt, 'expected': expected, 'validator': validator, 'max_turns': 1, 'max_tokens': 1024}


def reasoning_tasks():
    return [
        answer_task('R01', 'reasoning', 'A KV cache has 11 layers, 2 K/V tensors, 2 heads, 256 values per head and 49152 positions. FP16 uses 2 bytes/value. Q8 uses 32 codes plus a 2-byte scale per 32 values. Return JSON integer keys fp16_bytes,q8_bytes,saved_bytes.', {'fp16_bytes': 1107296256, 'q8_bytes': 588251136, 'saved_bytes': 519045120}),
        answer_task('R02', 'reasoning', 'Return JSON key order, a list: Kahn topological order for prerequisites {"deploy":["test"],"test":["build"],"build":["fetch"],"lint":["fetch"],"fetch":[]}. Always pick the lexicographically smallest available node.', {'order': ['fetch', 'build', 'lint', 'test', 'deploy']}),
        answer_task('R03', 'reasoning', 'LRU cache capacity 3, initially empty. Access A,B,C,A,D,B,E. Return JSON keys misses (integer), lru_to_mru (list).', {'misses': 6, 'lru_to_mru': ['D', 'B', 'E']}),
        answer_task('R04', 'reasoning', 'Undirected weighted edges: A-B=4,A-C=1,C-B=2,B-D=1,C-D=7. Return shortest A-to-D path and cost as JSON keys path,cost.', {'path': ['A', 'C', 'B', 'D'], 'cost': 4}),
        answer_task('R05', 'reasoning', 'An exclusive resource is held on half-open intervals [2,5),[7,9),[11,15). A new job needs 3 consecutive units, cannot start before 4. Return earliest start and end as JSON integer keys start,end.', {'start': 15, 'end': 18}),
        answer_task('R06', 'reasoning', 'Exactly one of A,B,C is true. Implications: A implies B; not B implies C; C implies not A. Return JSON keys A,B,C with booleans. If several assignments satisfy this, return the one with C true.', {'A': False, 'B': False, 'C': True}),
    ]


def instruction_tasks():
    return [
        answer_task('I01', 'instructions', 'Return only a JSON object with exactly keys id,enabled,tags. id is integer 17, enabled is boolean false; tags is ["hip","q8"] in this order. No other keys.', {'id': 17, 'enabled': False, 'tags': ['hip', 'q8']}),
        answer_task('I02', 'instructions', 'Return exactly three lines with no blank lines: alpha=3 then beta=1 then gamma=2. No spaces, bullets, quotes or other text.', 'alpha=3\nbeta=1\ngamma=2', 'lines'),
        answer_task('I03', 'instructions', 'Filter words ["pear","kiwi","fig","apple","plum"] to length four, sort ascending, uppercase. Return a JSON array only. No key or prose.', ['KIWI', 'PEAR', 'PLUM']),
        answer_task('I04', 'instructions', 'Return one line of CSV without header: fields x,y,z in reverse order; values x="red,blue", y="ok", z="done". Quote only fields containing a comma. No JSON, spaces, code fences or commentary.', 'done,ok,"red,blue"', 'lines'),
    ]


def pilot():
    return code_tasks()[:14] + tool_tasks() + reasoning_tasks() + instruction_tasks()
