"""Opt-in 200-task paired quality extension, clustered estimates and blind review."""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import uuid
from pathlib import Path

from .average_tasks import AREAS, tasks
from .common import ROOT, append, digest, now, rows, sha, strict_json, write_json
from .comparison import compare_tasks

DEFAULT = ROOT / 'bench/results/2026-10-17-hip-kv-q8-average-quality'
POLICY = 'KV-Q8-average-v1-2026-10-17'


def validate_weights(weights):
    if not isinstance(weights, dict) or set(weights) != set(AREAS):
        raise ValueError('weights must specify exactly the five areas')
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in weights.values()):
        raise ValueError('weights must be finite nonnegative numbers')
    total=sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError('positive finite weight sum required')
    return {k:weights[k]/total for k in AREAS}


def validate_plan(plan, selected):
    if plan.get('policy') != POLICY or plan.get('task_digest') != digest(selected):
        raise ValueError('plan and task catalog differ; create a new plan')
    validate_weights(plan['weights'])


def context_admission(selected, tokenizer_path):
    from tools.strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    tokenizer=Tokenizer.from_dir(tokenizer_path)
    template=ChatTemplate(tokenizer_path/'chat_template.jinja')
    counts={}
    for t in selected:
        if t['category']=='code': continue  # Multi-turn code budgets are handled by the existing harness.
        prompt=template.render([{'role':'system','content':t['system']},{'role':'user','content':t['prompt']}],
                               tools=[],enable_thinking=False,reasoning_effort='none')
        count=len(tokenizer.encode(prompt,parse_special=True))
        counts[t['id']]=count
        if count+t['max_tokens']+256 > 49152:
            raise ValueError('task exceeds context admission margin: '+t['id'])
    return {'method':'Actual frozen tokenizer/template; 256-token margin for server additions',
            'max_context':49152,'prompt_tokens':counts}


def plan(directory, weights=None):
    selected=tasks()
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory/name).exists() for name in ('PLAN.json','MANIFEST.json','TASKS.jsonl','CASEBOOK.jsonl')):
        raise RuntimeError('existing plan/freeze must be preserved; use a new campaign')
    value={'policy':POLICY, 'at':now(), 'status':'PLANNED_NOT_RUN',
           'task_digest':digest(selected), 'task_count':len(selected), 'planned_generations':2*len(selected),
           'weights':validate_weights(weights if weights is not None else {k:1 for k in AREAS}),
           'areas':dict(collections.Counter(t['category'] for t in selected)),
           'families':{a:len({t['family'] for t in selected if t['category']==a}) for a in AREAS},
           'reasoning':'disabled', 'repetitions':1,
           'meaning':'Synthetic casebook estimate, not a population or general-model guarantee.'}
    write_json(directory/'PLAN.json',value)
    (directory/'CASEBOOK.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in selected),encoding='utf-8')
    print(json.dumps(value,indent=2))


def measures(pairs):
    n=len(pairs)
    if not n:
        return None
    fp=sum(x for x,y in pairs); q8=sum(y for x,y in pairs)
    losses=sum(x and not y for x,y in pairs); gains=sum(y and not x for x,y in pairs)
    return {'n':n,'fp16_pass':fp,'q8_pass':q8,'fp16_accuracy':fp/n,'q8_accuracy':q8/n,
            'q8_losses':losses,'q8_gains':gains,'shared_failures':sum(not x and not y for x,y in pairs),
            'loss_rate':losses/n,'gain_rate':gains/n,'net_delta_pp':100*(q8-fp)/n,
            'average_loss_pp':100*(fp-q8)/n,
            'relative_accuracy_loss':(fp-q8)/fp if fp else None}


def estimate(selected, scores, weights, draws=4000, seed=170026):
    weights=validate_weights(weights)
    if len({t['id'] for t in selected})!=len(selected):
        raise ValueError('duplicate task identity')
    if any(type(v) is not bool for v in scores.values()):
        raise ValueError('quality scores must be booleans')
    if draws < 100:
        raise ValueError('at least 100 bootstrap draws required')
    by_area={}; clusters={}
    for area in AREAS:
        group=[t for t in selected if t['category']==area]
        missing=[t['id'] for t in group if any((t['id'],a) not in scores for a in ('fp16','q8'))]
        pairs=[(scores[t['id'],'fp16'],scores[t['id'],'q8']) for t in group
               if all((t['id'],a) in scores for a in ('fp16','q8'))]
        by_area[area]={'planned':len(group),'scored_pairs':len(pairs),'missing':missing,
                       'complete':bool(group) and not missing,'descriptive_completed_only':measures(pairs)}
        if group and not missing:
            grouped=collections.defaultdict(list)
            for t in group:
                grouped[t['family']].append((scores[t['id'],'fp16'],scores[t['id'],'q8']))
            clusters[area]=list(grouped.values())
    required=[a for a,w in weights.items() if w>0]
    complete=all(by_area[a]['complete'] for a in required)
    result={'policy':POLICY,'status':'COMPLETE' if complete else 'INCOMPLETE',
            'weights':weights,'areas':by_area,'weighted':None,
            'interval_method':'95% percentile bootstrap, paired outcomes, family clusters resampled within area',
            'draws':draws,'seed':seed,'general_equivalence_proven':False,
            'warning':'Synthetic casebook; family independence is an assumption. Zero observed loss does not prove zero risk.'}
    if not complete:
        return result
    weighted={k:sum(weights[a]*by_area[a]['descriptive_completed_only'][k] for a in required)
              for k in ('fp16_accuracy','q8_accuracy','loss_rate','gain_rate','net_delta_pp','average_loss_pp')}
    weighted['relative_accuracy_loss']=((weighted['fp16_accuracy']-weighted['q8_accuracy'])/weighted['fp16_accuracy']
                                        if weighted['fp16_accuracy'] else None)
    rng=random.Random(seed); samples={k:[] for k in ('net_delta_pp','loss_rate','gain_rate')}
    for _ in range(draws):
        one={k:0.0 for k in samples}
        for area in required:
            groups=clusters[area]
            sampled=[p for _ in groups for p in rng.choice(groups)]
            m=measures(sampled)
            for k in one: one[k]+=weights[area]*m[k]
        for k in one: samples[k].append(one[k])
    weighted['ci95']={k:[sorted(v)[int(.025*(draws-1))], sorted(v)[int(.975*(draws-1))]]
                      for k,v in samples.items()}
    lo,hi=weighted['ci95']['net_delta_pp']
    weighted['ci95']['average_loss_pp']=[-hi,-lo]
    result['weighted']=weighted
    return result


def load(directory):
    from .campaign import audit, registry
    manifest=audit(directory)
    experiment=strict_json((directory/'EXPERIMENT.json').read_text(encoding='utf-8'))
    if experiment.get('task_set') != 'average':
        raise ValueError('not an average-quality campaign')
    selected=rows(directory/'TASKS.jsonl')
    validate_plan(strict_json((directory/'PLAN.json').read_text(encoding='utf-8')),selected)
    records=registry(directory/'results-pilot.jsonl',manifest['freeze_id'])
    return manifest,experiment,selected,records


def export_blind(directory):
    manifest,experiment,selected,records=load(directory)
    for name in ('BLIND_PACKET.json','BLIND_KEY.private.json'):
        if (directory/name).exists():
            raise RuntimeError('blind packet already exists; preserve mapping')
    items=[]; mapping={}; rng=random.SystemRandom()
    for t in selected:
        if t['category']!='open': continue
        arms=[records.get((t['id'],a)) for a in ('fp16','q8')]
        if any(not r or r['status'] not in ('REVIEW_PENDING','FAIL') for r in arms):
            raise RuntimeError('all open-answer generations must finish before blind export')
        order=['fp16','q8']; rng.shuffle(order)
        item=uuid.uuid4().hex
        items.append({'item_id':item,'prompt':t['prompt'],'rubric':t['rubric'],
                      'A':records[t['id'],order[0]].get('final') or '',
                      'B':records[t['id'],order[1]].get('final') or ''})
        mapping[item]={'task_id':t['id'],'A':order[0],'B':order[1],
                       'result_sha':{a:records[t['id'],a]['result_sha'] for a in order}}
    packet={'version':1,'instructions':'Judge A and B independently against every rubric criterion. '
            'Mark each content/instruction/clarity criterion true or false; give preference A, B or TIE and a rationale. '
            'Use only this packet; do not open the private key or engine records.', 'items':items}
    write_json(directory/'BLIND_PACKET.json',packet)
    write_json(directory/'BLIND_KEY.private.json',{'freeze_id':manifest['freeze_id'],
                                                'packet_sha':sha(directory/'BLIND_PACKET.json'),'mapping':mapping})
    write_json(directory/'REVIEW_TEMPLATE.json',{'packet_sha':sha(directory/'BLIND_PACKET.json'),
               'reviewer':'', 'reviews':[{'item_id':i['item_id'],'A':{k:None for k in i['rubric']},
                                        'B':{k:None for k in i['rubric']},'preference':None,'rationale':''} for i in items]})


def validate_reviews(value, key, selected, records):
    if value.get('packet_sha')!=key['packet_sha'] or not isinstance(value.get('reviewer'),str) or not value['reviewer'].strip():
        raise ValueError('reviewer identity and matching packet hash required')
    if not isinstance(value.get('reviews'),list) or not value['reviews']:
        raise ValueError('nonempty review list required')
    tasks_by_id={t['id']:t for t in selected}; out=[]; seen=set()
    for review in value['reviews']:
        item=review.get('item_id')
        if item not in key['mapping'] or item in seen:
            raise ValueError('unknown or duplicate review item')
        seen.add(item); info=key['mapping'][item]; t=tasks_by_id[info['task_id']]
        if any(records[t['id'],a]['result_sha']!=h for a,h in info['result_sha'].items()):
            raise ValueError('review refers to changed inference records')
        if review.get('preference') not in ('A','B','TIE') or not isinstance(review.get('rationale'),str) or not review['rationale'].strip():
            raise ValueError('preference and nonempty rationale required')
        scores={}
        for side in ('A','B'):
            marks=review.get(side)
            if not isinstance(marks,dict) or set(marks)!=set(t['rubric']) or any(type(v) is not bool for v in marks.values()):
                raise ValueError('every rubric criterion needs an explicit boolean for both answers')
            scores[info[side]]=all(marks.values())
        preference=info.get(review['preference'], 'TIE')
        out.append({'task_id':t['id'],'scores':scores,'preference':preference,
                    'review':review,'reviewer':value['reviewer'],'result_sha':info['result_sha'],
                    'packet_sha':key['packet_sha'],'freeze_id':key['freeze_id']})
    return out


def read_reviews(directory, manifest, selected, records):
    ledger=rows(directory/'REVIEWS.jsonl')
    if not ledger: return {}
    key=strict_json((directory/'BLIND_KEY.private.json').read_text(encoding='utf-8'))
    if key['freeze_id']!=manifest['freeze_id'] or key['packet_sha']!=sha(directory/'BLIND_PACKET.json'):
        raise ValueError('blind packet/freeze identity mismatch')
    result={}
    for rec in ledger:
        if rec.get('review_sha')!=digest({k:v for k,v in rec.items() if k!='review_sha'}):
            raise ValueError('review journal hash mismatch')
        rebuilt=validate_reviews({'packet_sha':rec['packet_sha'],'reviewer':rec['reviewer'],
                                  'reviews':[rec['review']]},key,selected,records)[0]
        if any(rec.get(k)!=v for k,v in rebuilt.items()):
            raise ValueError('review journal does not match blind adjudication')
        if rec['task_id'] in result: raise ValueError('duplicate adjudication')
        result[rec['task_id']]=rec
    return result


def import_reviews(directory,path):
    manifest,experiment,selected,records=load(directory)
    existing=read_reviews(directory,manifest,selected,records)
    key=strict_json((directory/'BLIND_KEY.private.json').read_text(encoding='utf-8'))
    if key['freeze_id']!=manifest['freeze_id'] or key['packet_sha']!=sha(directory/'BLIND_PACKET.json'):
        raise ValueError('blind packet/freeze identity mismatch')
    decisions=validate_reviews(strict_json(path.read_text(encoding='utf-8')),key,selected,records)
    if any(d['task_id'] in existing for d in decisions):
        raise ValueError('refusing to overwrite an adjudication')
    for d in decisions:
        d.update(at=now(),source_sha=sha(path)); d['review_sha']=digest(d)
        append(directory/'REVIEWS.jsonl',d)


def report(directory):
    manifest,experiment,selected,records=load(directory)
    reviews=read_reviews(directory,manifest,selected,records)
    judged={}; scores={}
    for t in selected:
        for arm in ('fp16','q8'):
            rec=records.get((t['id'],arm))
            if not rec: continue
            value=dict(rec)
            if t['category']=='open':
                if rec['status']=='FAIL': scores[t['id'],arm]=False
                elif rec['status']=='REVIEW_PENDING' and t['id'] in reviews:
                    scores[t['id'],arm]=reviews[t['id']]['scores'][arm]
                    value['status']='PASS' if scores[t['id'],arm] else 'FAIL'
                elif rec['status']=='REVIEW_PENDING':
                    continue
            elif rec['status'] in ('PASS','FAIL'):
                scores[t['id'],arm]=rec['status']=='PASS'
            judged[t['id'],arm]=value
    settings=experiment['average_policy']
    result=estimate(selected,scores,settings['weights'],settings['bootstrap_draws'],settings['bootstrap_seed'])
    objective_weights={a:(0 if a=='open' else result['weights'][a]) for a in AREAS}
    result['objective_subset']=None
    if sum(objective_weights.values())>0:
        subset=estimate(selected,scores,objective_weights,settings['bootstrap_draws'],settings['bootstrap_seed'])
        result['objective_subset']={'meaning':'Renormalized objective areas only; not the complete five-area average',
                                    'status':subset['status'],'weights':subset['weights'],'weighted':subset['weighted']}
    paired=compare_tasks(selected,judged)
    result.update(freeze_id=manifest['freeze_id'],paired=paired,
                  review_pairs=len(reviews),generation_records=len(records),planned_generations=2*len(selected),
                  general_equivalence_proven=False,q8_promotion='NOT_EVALUATED')
    # Unreviewed generations are missing quality scores, not infrastructure failures.
    invalid=[r for r in records.values() if r['status'] not in ('PASS','FAIL','REVIEW_PENDING')]
    invalid += [r for (task_id,arm),r in records.items() if task_id not in {t['id'] for t in selected}]
    config_errors=[p for p in paired['pairs'] if p['classification']=='CONFIG_OR_INTEGRITY_ERROR']
    if invalid or config_errors:
        result.update(status='ERROR',weighted=None,objective_subset=None)
    elif result['status']=='COMPLETE':
        result['status']='COMPLETE_FOR_REVIEW'
    result['pending_blind_pairs']=[t['id'] for t in selected if t['category']=='open' and
                                 any((t['id'],a) not in scores for a in ('fp16','q8'))]
    prefs=collections.Counter(v['preference'] for v in reviews.values())
    result['blind_preferences']=dict(prefs)
    lines=['# Paired average quality: Q8 versus FP16','',f'Status: {result["status"]}',
           '', 'Positive average loss means Q8 is worse; net delta is Q8 minus FP16.', '',
           '| Area | Scored / planned pairs | FP16 correct | Q8 correct | Q8-only losses | Q8-only gains |',
           '|---|---:|---:|---:|---:|---:|']
    for area,v in result['areas'].items():
        m=v['descriptive_completed_only']
        values=[str(m[k]) if m else '—' for k in ('fp16_pass','q8_pass','q8_losses','q8_gains')]
        lines.append(f'| {area} | {v["scored_pairs"]}/{v["planned"]} | '+ ' | '.join(values)+' |')
    if result['weighted'] is None:
        lines+=['','Overall average unavailable: required paired scores/reviews are missing or inputs are incompatible.']
        subset=result['objective_subset']
        if subset and subset['weighted']:
            lines+=[f'Objective-only subset loss: {subset["weighted"]["average_loss_pp"]:.2f} percentage points. This excludes open-answer quality and is not the overall average.']
    else:
        w=result['weighted']; ci=w['ci95']['average_loss_pp']
        lines+=['',f'Weighted average loss: {w["average_loss_pp"]:.2f} percentage points; empirical 95% family-bootstrap interval [{ci[0]:.2f}, {ci[1]:.2f}].',
                f'Weighted loss rate: {100*w["loss_rate"]:.2f}%; gain rate: {100*w["gain_rate"]:.2f}%.']
    lines+=['','Weights: '+json.dumps(result['weights']), '',result['warning'],
            'Missing reviews never become shared failures or automatic passes. R05 diagnostic variants and historical runs are not pooled.',
            'No universal acceptable-loss threshold or Q8 promotion is inferred.']
    write_json(directory/'AVERAGE_SUMMARY.json',result)
    (directory/'AVERAGE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'report':str(directory/'AVERAGE_RESULTS.md')}))


def main():
    from . import campaign
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('plan','prepare','preflight','run','report','audit','export-blind','import-reviews'))
    ap.add_argument('--campaign',type=Path,default=DEFAULT)
    ap.add_argument('--weights',type=Path,help='JSON weights for all five areas; plan only')
    ap.add_argument('--reviews',type=Path)
    ap.add_argument('--retry-infra',action='store_true')
    args=ap.parse_args(); directory=args.campaign.resolve()
    if args.weights and args.command!='plan': ap.error('--weights is only allowed before freezing, with plan')
    if args.command=='plan': plan(directory,strict_json(args.weights.read_text(encoding='utf-8')) if args.weights else None)
    elif args.command=='prepare': campaign.prepare(directory,task_set='average')
    elif args.command=='run': campaign.run(directory,'pilot',args.retry_infra)
    elif args.command=='report': report(directory)
    elif args.command=='audit': load(directory)
    elif args.command=='export-blind': export_blind(directory)
    elif args.command=='import-reviews':
        if not args.reviews: ap.error('--reviews is required')
        import_reviews(directory,args.reviews); report(directory)
    else:
        from .resources import admission,snapshot
        state=snapshot(); reasons=admission(state)
        write_json(directory/'PREFLIGHT.json',{'at':now(),'memory':state,'refused':reasons})
        print(json.dumps({'memory':state,'refused':reasons},indent=2))
        return 2 if reasons else 0
    return 0


if __name__=='__main__':
    raise SystemExit(main())
