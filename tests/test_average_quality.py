"""CPU-only validation of the casebook, estimators, blind review and resume semantics."""
import collections
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.hip.functional_quality import average_quality as aq
from tools.hip.functional_quality.average_tasks import AREAS, coding, tasks
from tools.hip.functional_quality.campaign import COMPLETED, registry, schedule
from tools.hip.functional_quality.code import run_code
from tools.hip.functional_quality.common import append, digest, scratch, sha, write_json
from tools.hip.functional_quality.evaluate import evaluate


def record(t,arm,status='PASS'):
    r={'task_id':t['id'],'arm':arm,'status':status,'freeze_id':'fixture',
       'final':'Uma resposta clara.', 'requests':[],
       'config':{'exe':'same.exe','args':['--expert-profile',arm+'.txt']+(['--kv','q8_0'] if arm=='q8' else [])}}
    r['result_sha']=digest(r)
    return r


class AverageTests(unittest.TestCase):
    def test_catalog_and_counterbalanced_order(self):
        selected=tasks()
        self.assertEqual(len(selected),200)
        self.assertEqual(collections.Counter(t['category'] for t in selected),{a:40 for a in AREAS})
        self.assertEqual(digest(selected),digest(tasks()))
        self.assertEqual(len({t['id'] for t in selected}),200)
        self.assertFalse(any(t.get('reasoning_enabled') for t in selected))
        self.assertFalse(any('R05V' in t['id'] for t in selected))
        for arm in ('fp16','q8'):
            ids=[i for block in schedule(selected) if block['arm']==arm for i in block['task_ids']]
            self.assertCountEqual(ids,[t['id'] for t in selected])
        self.assertEqual(sum(len({t['family'] for t in selected if t['category']==a}) for a in AREAS),80)

    def test_all_40_code_oracles_execute_and_reject_wrong_solution(self):
        for t in coding():
            with self.subTest(task=t['id']):
                cases=t['public']+t['hidden']
                self.assertTrue(run_code(t['reference'],t['function'],cases)['ok'])
                self.assertFalse(run_code('def '+t['function']+'(*args): return None',t['function'],cases)['ok'])

    def test_oracle_spot_checks_and_context_length_strata(self):
        selected=tasks(); by_id={t['id']:t for t in selected}
        self.assertEqual(by_id['AVG-reasoning-change-0']['expected'],{'coins':4})
        self.assertEqual(by_id['AVG-reasoning-change-3']['expected'],{'coins':3})
        self.assertEqual(by_id['AVG-extraction-sum-2']['expected'],{'total':36})
        self.assertEqual(by_id['AVG-context-order-3']['expected'],{'value':['X','Y','Z']})
        self.assertEqual(by_id['AVG-context-absence-3']['expected'],{'value':None})
        self.assertEqual({t['context_rows'] for t in selected if t['category']=='context'},{128,384,768,1280})
        self.assertGreater(len(by_id['AVG-context-join-3']['prompt']),60000)

    def test_weights_and_plan_tamper_detection(self):
        for bad in ({'code':1},{a:0 for a in AREAS},{a:float('nan') for a in AREAS},
                    {a:True for a in AREAS},{a:-1 for a in AREAS}):
            with self.assertRaises(ValueError): aq.validate_weights(bad)
        p={'policy':aq.POLICY,'task_digest':digest(tasks()),'weights':{a:1 for a in AREAS}}
        aq.validate_plan(p,tasks())
        with self.assertRaises(ValueError): aq.validate_plan(p,tasks()[:-1])

    def test_empty_weights_and_changed_reviewable_casebook_are_refused(self):
        from tools.hip.functional_quality import campaign
        with scratch() as folder:
            p=Path(folder)
            with self.assertRaises(ValueError): aq.plan(p,{})
            aq.plan(p)
            (p/'CASEBOOK.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'casebook differs'): campaign.prepare(p,task_set='average')

    def test_context_budget_refuses_overflow_before_inference(self):
        selected=[{'id':'context','category':'context','system':'s','prompt':'p','max_tokens':2048}]
        with patch('tools.strata_tokenizer.Tokenizer.from_dir') as loader, patch('serve.frontend.ChatTemplate'):
            loader.return_value.encode.return_value=[0]*47000
            with self.assertRaisesRegex(ValueError,'context admission'): aq.context_admission(selected,Path('.'))
            loader.return_value.encode.return_value=[0]*32000
            self.assertEqual(aq.context_admission(selected,Path('.'))['prompt_tokens']['context'],32000)

    def test_paired_metrics_do_not_cancel_losses(self):
        m=aq.measures([(True,False),(False,True),(True,True),(False,False)])
        self.assertEqual(m['net_delta_pp'],0)
        self.assertEqual(m['loss_rate'],.25)
        self.assertEqual(m['gain_rate'],.25)
        self.assertEqual(m['shared_failures'],1)
        self.assertIsNone(aq.measures([(False,False)])['relative_accuracy_loss'])

    def test_weighted_mean_and_missing_review_never_complete(self):
        selected=[{'id':a,'category':a,'family':'one'} for a in AREAS]
        scores={(t['id'],arm):True for t in selected for arm in ('fp16','q8')}
        scores['code','q8']=False
        result=aq.estimate(selected,scores,{a:1 for a in AREAS},draws=100)
        self.assertAlmostEqual(result['weighted']['average_loss_pp'],20)
        del scores['open','q8']
        result=aq.estimate(selected,scores,{a:1 for a in AREAS},draws=100)
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertIsNone(result['weighted'])
        self.assertEqual(result['areas']['open']['missing'],['open'])

    def test_family_repetition_does_not_narrow_bootstrap_interval(self):
        def data(reps):
            selected=[{'id':f'{family}-{i}','category':'code','family':family} for family in ('a','b') for i in range(reps)]
            scores={(t['id'],arm):(arm=='fp16' or t['family']=='b') for t in selected for arm in ('fp16','q8')}
            return selected,scores
        weights={a:int(a=='code') for a in AREAS}
        one=aq.estimate(*data(1),weights,draws=500)
        repeated=aq.estimate(*data(20),weights,draws=500)
        self.assertEqual(one['weighted']['ci95'],repeated['weighted']['ci95'])

    def test_open_generation_awaits_review_and_truncation_is_failure(self):
        t=next(t for t in tasks() if t['category']=='open')
        class Fake:
            def __init__(self,finish): self.finish=finish
            def complete(self,*args,**kwargs):
                return {'message':{'role':'assistant','content':'Resposta.'},'finish_reason':self.finish,
                        'events':[],'request':{},'usage':{'completion_tokens':3}}
        with scratch() as folder:
            p=Path(folder)
            self.assertEqual(evaluate(t,Fake('stop'),p/'project',p/'one.jsonl')['status'],'REVIEW_PENDING')
            self.assertEqual(evaluate(t,Fake('length'),p/'project',p/'two.jsonl')['status'],'FAIL')
        self.assertIn('REVIEW_PENDING',COMPLETED)

    def test_registry_refuses_regenerating_review_pending_answer(self):
        t=next(t for t in tasks() if t['category']=='open')
        r=record(t,'fp16','REVIEW_PENDING')
        with scratch() as folder:
            p=Path(folder)/'results.jsonl'; append(p,r); append(p,r)
            with self.assertRaisesRegex(RuntimeError,'duplicate'): registry(p,'fixture')

    def test_blind_packet_and_review_validation(self):
        t=next(t for t in tasks() if t['category']=='open')
        records={(t['id'],arm):record(t,arm,'REVIEW_PENDING') for arm in ('fp16','q8')}
        fixture=({'freeze_id':'fixture'},{},[t],records)
        with scratch() as folder:
            p=Path(folder)
            with patch.object(aq,'load',return_value=fixture): aq.export_blind(p)
            packet=json.loads((p/'BLIND_PACKET.json').read_text(encoding='utf-8'))
            key=json.loads((p/'BLIND_KEY.private.json').read_text())
            self.assertEqual(set(packet['items'][0]),{'item_id','prompt','rubric','A','B'})
            review=json.loads((p/'REVIEW_TEMPLATE.json').read_text())
            with self.assertRaises(ValueError): aq.validate_reviews(review,key,[t],records)
            review['reviewer']='human reviewer'
            row=review['reviews'][0]
            row.update(A={k:True for k in t['rubric']},B={k:False for k in t['rubric']},preference='A',rationale='Content is correct only in A.')
            parsed=aq.validate_reviews(review,key,[t],records)
            mapping=key['mapping'][row['item_id']]
            self.assertTrue(parsed[0]['scores'][mapping['A']])
            self.assertFalse(parsed[0]['scores'][mapping['B']])
            bad=copy.deepcopy(review); bad['reviews'].append(row)
            with self.assertRaises(ValueError): aq.validate_reviews(bad,key,[t],records)
            bad=copy.deepcopy(review); bad['packet_sha']='stale'
            with self.assertRaises(ValueError): aq.validate_reviews(bad,key,[t],records)
            with patch.object(aq,'load',return_value=fixture):
                source=p/'reviews.json'; write_json(source,review)
                aq.import_reviews(p,source)
                with self.assertRaises(ValueError): aq.import_reviews(p,source)
            self.assertEqual(len(aq.read_reviews(p,fixture[0],[t],records)),1)
            records[t['id'],'fp16']['result_sha']='changed'
            with self.assertRaises(ValueError): aq.validate_reviews(review,key,[t],records)

    def test_report_keeps_unreviewed_quality_missing_and_config_errors_visible(self):
        selected=[next(t for t in tasks() if t['category']==a) for a in AREAS]
        records={(t['id'],a):record(t,a,'REVIEW_PENDING' if t['category']=='open' else 'PASS')
                 for t in selected for a in ('fp16','q8')}
        settings={'weights':{a:1 for a in AREAS},'bootstrap_seed':1,'bootstrap_draws':100}
        with scratch() as folder:
            p=Path(folder)
            with patch.object(aq,'load',return_value=({'freeze_id':'fixture'},{'average_policy':settings},selected,records)):
                aq.report(p)
                result=json.loads((p/'AVERAGE_SUMMARY.json').read_text())
                self.assertEqual(result['status'],'INCOMPLETE')
                self.assertEqual(result['paired']['status'],'INCOMPLETE')
                self.assertIsNone(result['weighted'])
                records[selected[0]['id'],'q8']['config']['exe']='wrong.exe'
                aq.report(p)
                result=json.loads((p/'AVERAGE_SUMMARY.json').read_text())
                self.assertEqual(result['status'],'ERROR')
                self.assertIsNone(result['weighted'])

    def test_report_dispatch_avoids_missing_tool_category(self):
        from tools.hip.functional_quality import campaign
        with scratch() as folder:
            p=Path(folder); write_json(p/'EXPERIMENT.json',{'task_set':'average'})
            with patch.object(aq,'report') as called:
                campaign.report(p)
                called.assert_called_once_with(p)

    def test_complete_blind_review_enters_weighted_mean_without_rewriting_records(self):
        selected=[next(t for t in tasks() if t['category']==a) for a in AREAS]
        records={(t['id'],a):record(t,a,'REVIEW_PENDING' if t['category']=='open' else 'PASS')
                 for t in selected for a in ('fp16','q8')}
        settings={'weights':{a:1 for a in AREAS},'bootstrap_seed':1,'bootstrap_draws':100}
        fixture=({'freeze_id':'fixture'},{'average_policy':settings},selected,records)
        before=copy.deepcopy(records)
        with scratch() as folder:
            p=Path(folder)
            with patch.object(aq,'load',return_value=fixture):
                aq.export_blind(p)
                review=json.loads((p/'REVIEW_TEMPLATE.json').read_text())
                review['reviewer']='human'
                item=review['reviews'][0]
                item.update(A={k:True for k in item['A']},B={k:False for k in item['B']},
                            preference='A',rationale='One answer meets the frozen rubric.')
                source=p/'review.json'; write_json(source,review)
                aq.import_reviews(p,source); aq.report(p)
                result=json.loads((p/'AVERAGE_SUMMARY.json').read_text())
                self.assertEqual(result['status'],'COMPLETE_FOR_REVIEW')
                self.assertAlmostEqual(abs(result['weighted']['average_loss_pp']),20)
                self.assertEqual(result['objective_subset']['weighted']['average_loss_pp'],0)
                self.assertEqual(result['pending_blind_pairs'],[])
                self.assertEqual(records,before)
                ledger=p/'REVIEWS.jsonl'
                corrupted=json.loads(ledger.read_text()); corrupted['scores']={'fp16':True,'q8':True}
                ledger.write_text(json.dumps(corrupted)+'\n')
                with self.assertRaisesRegex(ValueError,'hash mismatch'): aq.report(p)


if __name__=='__main__': unittest.main()
