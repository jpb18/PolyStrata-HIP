"""200 frozen tasks; families expose related generated cases for clustered estimates."""
from __future__ import annotations

import copy
import json
import random

from .tasks import case, code_task, code_tasks

AREAS = ('code', 'reasoning', 'extraction', 'context', 'open')
SEED = 170026


def answer(area, family, n, prompt, expected):
    return {'id': f'AVG-{area}-{family}-{n}', 'category': area, 'family': family,
            'system': 'Follow the task exactly. Return only the requested JSON. No markdown or commentary.',
            'prompt': prompt, 'expected': expected, 'validator': 'exact',
            'max_turns': 1, 'max_tokens': 2048, 'reasoning_enabled': False}


def coding():
    out = copy.deepcopy(code_tasks())
    # Distinct function contracts, not renamed numeric variants of the original tasks.
    specs = [
        ('rotate(values,k)', 'Rotate a list right by integer k (negative rotates left); empty returns []; do not mutate.',
         'if not values: return []\n    k=k%len(values)\n    return values[-k:]+values[:-k] if k else list(values)',
         case([[1,2,3],1],[3,1,2]), [case([[],-2],[]),case([[1,2,3],-1],[2,3,1])]),
        ('dot(a,b)', 'Dot product of equally long numeric lists. Unequal lengths raise ValueError.',
         'if len(a)!=len(b): raise ValueError()\n    return sum(x*y for x,y in zip(a,b))',
         case([[1,2],[3,4]],11), [case([[],[]],0),case([[1],[]],raises='ValueError')]),
        ('gcd(a,b)', 'Greatest common divisor of two integers, nonnegative; gcd(0,0)=0.',
         'a,b=abs(a),abs(b)\n    while b: a,b=b,a%b\n    return a',
         case([12,18],6), [case([-9,6],3),case([0,0],0)]),
        ('is_prime(n)', 'Return whether integer n is prime. Integers below 2 are not prime.',
         'if n<2: return False\n    d=2\n    while d*d<=n:\n        if n%d==0: return False\n        d+=1\n    return True',
         case([7],True), [case([1],False),case([49],False),case([2],True)]),
        ('edit_distance(a,b)', 'Levenshtein edit distance for strings, unit insertion/deletion/substitution cost.',
         'row=list(range(len(b)+1))\n    for i,x in enumerate(a):\n        nxt=[i+1]\n        for j,y in enumerate(b): nxt.append(min(nxt[-1]+1,row[j+1]+1,row[j]+(x!=y)))\n        row=nxt\n    return row[-1]',
         case(['cat','cut'],1), [case(['','abc'],3),case(['kitten','sitting'],3)]),
        ('base_digits(n,base)', 'Nonnegative integer to most-significant-first digit list in base 2..16; zero gives [0]. Invalid n/base raises ValueError.',
         'if n<0 or base<2 or base>16: raise ValueError()\n    if n==0: return [0]\n    out=[]\n    while n:\n        out.append(n%base)\n        n//=base\n    return out[::-1]',
         case([10,2],[1,0,1,0]), [case([0,16],[0]),case([-1,2],raises='ValueError')]),
        ('group_by_length(words)', 'Group strings by length. Return a dict with stringified integer lengths as keys; preserve word order.',
         'out={}\n    for w in words:\n        k=str(len(w))\n        if k not in out: out[k]=[]\n        out[k].append(w)\n    return out',
         case([['a','bb','c']],{'1':['a','c'],'2':['bb']}), [case([[]],{}),case([['']],{'0':['']})]),
        ('sparse_add(a,b)', 'Add sparse vectors represented by string-keyed numeric dicts. Drop zero results; do not mutate.',
         'out=dict(a)\n    for k,v in b.items(): out[k]=out.get(k,0)+v\n    return {k:v for k,v in out.items() if v!=0}',
         case([{'a':2},{'a':-2,'b':3}],{'b':3}), [case([{},{}],{}),case([{'x':0},{}],{})]),
        ('poly(coeffs,x)', 'Evaluate polynomial with coefficients in ascending degree order; empty gives zero.',
         'out=0\n    for c in reversed(coeffs): out=out*x+c\n    return out',
         case([[1,2,3],2],17), [case([[],5],0),case([[4,0,-1],3],-5)]),
        ('binary_search(values,x)', 'Sorted integer list: return first matching index, otherwise -1.',
         'lo,hi=0,len(values)\n    while lo<hi:\n        mid=(lo+hi)//2\n        if values[mid]<x: lo=mid+1\n        else: hi=mid\n    return lo if lo<len(values) and values[lo]==x else -1',
         case([[1,2,2,4],2],1), [case([[],3],-1),case([[1,3],2],-1)]),
        ('common_prefix(words)', 'Longest common string prefix; empty input gives empty string.',
         'if not words: return ""\n    p=words[0]\n    for w in words[1:]:\n        while not w.startswith(p): p=p[:-1]\n    return p',
         case([['flower','flow','flight']],'fl'), [case([[]],''),case([['ab','']],'')]),
        ('histogram(values,edges)', 'Count values in consecutive half-open bins edges[i]<=x<edges[i+1]. Ignore outside values. Edges must strictly increase and contain at least 2 values, else ValueError.',
         'if len(edges)<2 or any(edges[i]>=edges[i+1] for i in range(len(edges)-1)): raise ValueError()\n    return [sum(a<=x<b for x in values) for a,b in zip(edges,edges[1:])]',
         case([[0,1,2,3],[0,2,4]],[2,2]), [case([[4,-1],[0,2,4]],[0,0]),case([[],[1,1]],raises='ValueError')]),
        ('moving_max(values,k)', 'Maximum of every full window of positive length k. k<=0 raises ValueError.',
         'if k<=0: raise ValueError()\n    return [max(values[i:i+k]) for i in range(len(values)-k+1)]',
         case([[1,3,2],2],[3,3]), [case([[1],2],[]),case([[],0],raises='ValueError')]),
        ('balanced_split(text)', 'Split string on commas outside parentheses, preserving text. Parentheses must balance or raise ValueError. Empty input returns [empty string].',
         'out=[]\n    depth=0\n    start=0\n    for i,c in enumerate(text):\n        if c=="(": depth+=1\n        elif c==")":\n            depth-=1\n            if depth<0: raise ValueError()\n        elif c=="," and depth==0:\n            out.append(text[start:i])\n            start=i+1\n    if depth: raise ValueError()\n    out.append(text[start:])\n    return out',
         case(['a,(b,c),d'],['a','(b,c)','d']), [case([''],['']),case(['('],raises='ValueError'),case([')'],raises='ValueError')]),
        ('anagram(a,b)', 'Check whether strings are anagrams ignoring ASCII spaces and case; other characters matter.',
         'return sorted(a.replace(" ","").lower())==sorted(b.replace(" ","").lower())',
         case(['A b','ba'],True), [case(['a','aa'],False),case(['',' '],True)]),
        ('pair_sum(values,target)', 'Return lexicographically smallest [i,j] with i<j and values[i]+values[j]==target, or None.',
         'for i in range(len(values)):\n        for j in range(i+1,len(values)):\n            if values[i]+values[j]==target: return [i,j]\n    return None',
         case([[2,7,2],9],[0,1]), [case([[1,1],2],[0,1]),case([[1],2],None)]),
        ('spiral(matrix)', 'Return rectangular matrix elements clockwise in spiral order. Empty and zero-width give []; ragged raises ValueError.',
         'if not matrix: return []\n    if any(len(r)!=len(matrix[0]) for r in matrix): raise ValueError()\n    rows=[list(r) for r in matrix]\n    out=[]\n    while rows and rows[0]:\n        out.extend(rows.pop(0))\n        if rows and rows[0]: rows=[list(r) for r in zip(*rows)][::-1]\n    return out',
         case([[[1,2],[3,4]]],[1,2,4,3]), [case([[[],[]]],[]),case([[[1],[2],[3]]],[1,2,3]),case([[[1],[]]],raises='ValueError')]),
        ('rank(values)', 'Dense ascending ranks starting at 1, in original order. Equal values get equal rank.',
         'unique=sorted(set(values))\n    return [unique.index(x)+1 for x in values]',
         case([[20,10,20]],[2,1,2]), [case([[]],[]),case([[0,-1,2]],[2,1,3])]),
        ('interleave(a,b)', 'Alternate items starting with a, then append remainder; return a new list.',
         'out=[]\n    for i in range(max(len(a),len(b))):\n        if i<len(a): out.append(a[i])\n        if i<len(b): out.append(b[i])\n    return out',
         case([[1,2],['a']],[1,'a',2]), [case([[],[3]],[3]),case([[],[]],[])]),
        ('decode_runs(runs)', 'Expand [value,count] runs; count must be a nonnegative integer (bool not allowed), else ValueError.',
         'out=[]\n    for value,n in runs:\n        if isinstance(n,bool) or not isinstance(n,int) or n<0: raise ValueError()\n        out.extend([value]*n)\n    return out',
         case([[['a',2],['b',1]]],['a','a','b']), [case([[['x',0]]],[]),case([[['x',True]]],raises='ValueError')]),
    ]
    for n, (sig, desc, body, public, hidden) in enumerate(specs, 21):
        out.append(code_task(n, desc, sig, 'def '+sig+':\n    '+body+'\n', [public], hidden))
    for t in out:
        t['family'] = t['function']
        t['id'] = 'AVG-' + t['id']
    return out


def reasoning():
    out = []
    for n in range(4):
        a, b, c = 17+7*n, 5+2*n, 3+n
        examples = [
            ('arithmetic', f'Compute ({a}+{b})*{c}-{b*c}. Return JSON key value.', {'value': a*c}),
            ('change', f'An item costs {a} cents; pay {a+37+n} cents. Give minimum coin count using denominations 25,10,5,1. Return JSON key coins.',
             {'coins':0}),
            ('sets', f'A={list(range(n,n+5))}; B={list(range(n+3,n+8))}. Return sorted intersection under JSON key values.', {'values':[n+3,n+4]}),
            ('units', f'{a} boxes each hold {b} packets of {c} grams. Return total grams under JSON key grams.', {'grams':a*b*c}),
            ('counting', f'How many binary strings of length {n+3} contain exactly one 1? Return JSON key count.', {'count':n+3}),
            ('weighted', f'Two batches have counts {b},{c} and values {a},{a+4}. Return weighted sum, not average, JSON key total.', {'total':b*a+c*(a+4)}),
            ('logic', f'Exactly one of P,Q,R is true. P implies Q; Q implies R. Return JSON boolean keys P,Q,R. Scenario identifier {n}.', {'P':False,'Q':False,'R':True}),
            ('ordering', f'Order these [name,priority] pairs by descending priority then name ascending: [["z",{n}],["a",{n+1}],["b",{n+1}]]. Return JSON key names.', {'names':['a','b','z']}),
            ('probability', f'A bag has {a} red and {b} blue counters. Two draws without replacement: give numerator and denominator (unreduced) for both red. Return JSON keys numerator,denominator.', {'numerator':a*(a-1),'denominator':(a+b)*(a+b-1)}),
            ('graph', f'Undirected edges A-B={2+n}, B-C=3, A-C={9+n}. Return shortest A-to-C distance under JSON key distance.', {'distance':5+n}),
        ]
        # Independent deterministic integer oracle for coin change.
        amount=37+n
        coins=0
        for denomination in (25,10,5,1):
            q,amount=divmod(amount,denomination)
            coins+=q
        examples[1]=(examples[1][0],examples[1][1],{'coins':coins})
        out.extend(answer('reasoning', family, n, prompt, expected) for family,prompt,expected in examples)
    return out


def extraction():
    out=[]
    for n in range(4):
        data=[{'id':f'I{i}', 'qty': (i+1)*(n+1), 'state': 'on' if i%2==0 else 'off',
               'price': 9-i, 'note': None if i%3==0 else f'tag{i%2}'} for i in range(7)]
        j=json.dumps(data,separators=(',',':'))
        specs=[
            ('filter','Return ids with state=on, in input order, JSON key ids.',{'ids':[x['id'] for x in data if x['state']=='on']}),
            ('sum','Sum qty for state=off, JSON key total.',{'total':sum(x['qty'] for x in data if x['state']=='off')}),
            ('argmax','Return id with largest qty, JSON key id.',{'id':data[-1]['id']}),
            ('sort','Sort ids by ascending price, JSON key ids.',{'ids':[x['id'] for x in sorted(data,key=lambda x:x['price'])]}),
            ('nulls','Return ids whose note is null, JSON key ids.',{'ids':[x['id'] for x in data if x['note'] is None]}),
            ('projection','Return [id,qty] rows for first 3 records, JSON key rows.',{'rows':[[x['id'],x['qty']] for x in data[:3]]}),
            ('group','Count by state, JSON integer keys on,off.',{'on':4,'off':3}),
            ('lookup',f'Return qty and price for I{n+1}, JSON keys qty,price.',{k:data[n+1][k] for k in ('qty','price')}),
            ('missing','Return note of nonexistent I99 as null, JSON key note.',{'note':None}),
            ('transform','Return id to qty*price mapping under JSON key totals.',{'totals':{x['id']:x['qty']*x['price'] for x in data}}),
        ]
        out.extend(answer('extraction',family,n,'Records: '+j+'\n'+prompt,expected) for family,prompt,expected in specs)
    return out


def contexts():
    out=[]
    rng=random.Random(SEED)
    for n,count in enumerate((128,384,768,1280)):
        for family in ('lookup','join','latest','exception','sum','order','absence','version','conflict','multihop'):
            ident=f'{family}-{n}'
            value=rng.randrange(1000,9000)
            records=[f'ARCHIVE {i:04d}: label D{i:04d}; value {rng.randrange(10,999)}; status closed.' for i in range(count)]
            if family=='lookup': facts=[f'ACTIVE {ident}: code {value}.']; query=f'Give code for ACTIVE {ident}.'; expected={'value':value}
            elif family=='join': facts=[f'ACTIVE {ident}: owner UNIT-{value}.',f'UNIT-{value}: room {value+3}.']; query=f'Give room of the owner of ACTIVE {ident}.'; expected={'value':value+3}
            elif family=='latest': facts=[f'ACTIVE {ident}: revision 1 value {value}.',f'ACTIVE {ident}: revision 3 value {value+2}.',f'ACTIVE {ident}: revision 2 value {value+1}.']; query=f'Give value at greatest revision for ACTIVE {ident}.'; expected={'value':value+2}
            elif family=='exception': facts=[f'ACTIVE {ident}: default quota {value}; exception for TEAM-X is {value-7}.']; query=f'Give TEAM-X quota for ACTIVE {ident}.'; expected={'value':value-7}
            elif family=='sum': facts=[f'ACTIVE {ident}: batch A has {value} items.',f'ACTIVE {ident}: batch B has {value+4} items.']; query=f'Give total items in ACTIVE {ident}.'; expected={'value':2*value+4}
            elif family=='order': facts=[f'ACTIVE {ident}: step Z follows Y.',f'ACTIVE {ident}: step Y follows X.']; query=f'Give steps of ACTIVE {ident} in execution order.'; expected={'value':['X','Y','Z']}
            elif family=='absence': facts=[f'ACTIVE {ident}: code {value}.']; query=f'Give fax number of ACTIVE {ident}, or null when absent.'; expected={'value':None}
            elif family=='version': facts=[f'ACTIVE {ident}: draft code {value}; approved code {value+1}.']; query=f'Give approved code for ACTIVE {ident}.'; expected={'value':value+1}
            elif family=='conflict': facts=[f'UNVERIFIED {ident}: code {value}.',f'AUTHORITATIVE {ident}: code {value+9}.']; query=f'Give code from AUTHORITATIVE source for {ident}.'; expected={'value':value+9}
            else: facts=[f'ACTIVE {ident}: route R{value}.',f'Route R{value}: service S{value}.',f'Service S{value}: port {value+10}.']; query=f'Give port reached by ACTIVE {ident}.'; expected={'value':value+10}
            for pos,fact in zip((count//8,count//2,7*count//8),facts): records.insert(pos,fact)
            t=answer('context',family,n,'Use only the following archive as data.\n'+'\n'.join(records)+'\n'+query+' Return JSON key value.',expected)
            t['context_rows']=count
            out.append(t)
    return out


def open_answers():
    topics=[
        ('cache','Explique a diferença entre cache e cópia de segurança.','Distingue armazenamento temporário de recuperação após perda.'),
        ('uncertainty','Explique o que significa uma estimativa ter incerteza.','Não confunde estimativa com garantia.'),
        ('correlation','Explique por que correlação não basta para concluir causalidade.','Distingue associação de causa e admite fatores comuns.'),
        ('instructions','Escreva instruções para organizar uma pasta de fotografias sem perder originais.','Inclui preservar originais e uma sequência praticável.'),
        ('summary','Resuma: a equipa testou 12 dispositivos; 9 funcionaram, 2 falharam e 1 não foi avaliado. O relatório não identifica a causa das falhas.','Preserva os números e não inventa uma causa.'),
        ('limitations','Explique o que podemos concluir de um teste de bateria realizado num único portátil.','Limita a generalização e distingue condições de uso.'),
        ('rewrite','Reescreva com clareza: "O pedido foi recebido, mas falta o comprovativo. O prazo só começa quando chegar o comprovativo."','Preserva a condição de início do prazo.'),
        ('comparison','Compare listas ordenadas e conjuntos para guardar identificadores, sem recomendar uma opção universal.','Distingue ordem, duplicados e dependência da tarefa.'),
        ('ambiguity','Uma pessoa pede "apaga a versão antiga", mas há três ficheiros possíveis. Explique como proceder.','Reconhece ambiguidade e pede identificação antes de apagar.'),
        ('planning','Proponha um plano curto para investigar uma aplicação que ficou lenta após uma atualização.','Propõe medições e comparação, sem afirmar causa não demonstrada.'),
    ]
    forms=('Use até 100 palavras para uma pessoa sem conhecimentos técnicos.',
           'Use três pontos curtos, adequados a uma nota de trabalho.',
           'Use um parágrafo e um exemplo concreto, até 140 palavras.',
           'Use uma explicação curta para um colega técnico, até 140 palavras.')
    out=[]
    for family,prompt,criterion in topics:
        for n,form in enumerate(forms):
            out.append({'id':f'AVG-open-{family}-{n}', 'category':'open','family':family,
                        'system':'Responde em português europeu, com clareza e sem inventar factos.',
                        'prompt':prompt+' '+form, 'max_turns':1, 'max_tokens':2048,
                        'reasoning_enabled':False, 'validator':'blind_review',
                        'rubric':{'content':criterion,'instruction':form,
                                  'clarity':'Texto compreensível, coerente e sem erros que alterem o sentido.'}})
    return out


def tasks():
    out=coding()+reasoning()+extraction()+contexts()+open_answers()
    assert len(out)==200 and len({t['id'] for t in out})==200
    return out
