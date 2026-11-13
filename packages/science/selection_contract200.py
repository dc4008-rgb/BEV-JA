"""Immutable non-test 200-member selection; no model/attack-based filtering."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from collections import Counter
from source_bindings import DATA_MANIFEST_SHA256
NATIVE_FIELDS = ('sample_id','case_id','dataset','source_split','source_path','source_sha256','road_quad_norm','width','height')
OLD_MASTER_SHA = '0538c8ab84521abb088e780be6706e7c340f3b47eddabaf482023d4d6b76c23b'
DIAGNOSTIC_IDS = ('kitti_000007','nuscenes_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404')
def require(ok,text):
    if not ok: raise ValueError(text)
def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def digest(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value),'explicit selection SHA required');return value

def validate_selection(document, *, allow_candidate=False):
    require(document.get('schema')=='periodic-foa-non-test-200-sample-selection-v1' and document.get('status') in
        (('frozen_source_image_qa_passed','candidate_pending_source_image_qa') if allow_candidate else ('frozen_source_image_qa_passed',)),
        'new200 source/road QA must be frozen before formal execution')
    require(document.get('source_manifest_sha256')==DATA_MANIFEST_SHA256 and document.get('old_master40_sha256')==OLD_MASTER_SHA,
        'original non-test240 and original40 identities must remain immutable')
    require(document.get('sample_count')==200 and document.get('counts_by_dataset')=={'kitti':100,'nuscenes':100} and
        document.get('counts_by_selection_role')=={'retained_validation40':40,'added_train160':160},'200 selected sample counts differ')
    require(all(document.get(k) is False for k in ('test_data_opened','attack_outputs_used_for_selection','held_out_image_generalization')) and
        document.get('optimization_on_original_validation') is True,'no test/result selection or held-out generalization claim allowed')
    require(document.get('diagnostic_sample_ids')==list(DIAGNOSTIC_IDS),'diagnostic exclusion registry differs')
    rows=document.get('samples');require(isinstance(rows,list) and len(rows)==200,'exact200 master required')
    require(len({r['sample_id'] for r in rows})==len(rows) and len({r['case_id'] for r in rows})==len(rows) and len({r['source_sha256'] for r in rows})==len(rows),
        'duplicate source/member forbidden')
    require(rows==sorted(rows,key=lambda r:(r['dataset'],r['sample_id'])),'canonical dataset/sample master order required')
    require(canonical_sha(rows)==digest(document['master_samples_sha256']),'independent200 master SHA differs')
    counts=Counter((r['dataset'],r['source_split'],r['selection_role']) for r in rows)
    require(counts=={('kitti','validation','retained_validation40'):20,('nuscenes','validation','retained_validation40'):20,
        ('kitti','train','added_train160'):80,('nuscenes','train','added_train160'):80},'old40/new160 strata differ or test was selected')
    require(not set(DIAGNOSTIC_IDS)&{r['sample_id'] for r in rows},'fixed diagnostic inputs cannot enter200')
    old=[{k:r[k] for k in NATIVE_FIELDS} for r in rows if r['selection_role']=='retained_validation40']
    require(canonical_sha(old)==OLD_MASTER_SHA,'old40 original native fields changed')
    for row in rows:
        require(row['case_id']==row['sample_id'].replace('+','_plus_') and re.fullmatch('[A-Za-z0-9_-]{1,180}',row['case_id']), 'safe own-sample case identity required')
        digest(row['source_sha256']);require(Path(row['source_path']).is_absolute(),'original source path required')
        require(all(type(row[k]) is int and row[k]>0 for k in ('width','height')),'positive original dimensions required')
    for key in ('source_input_proof_binding','manual_qa_binding'):
        b=document.get(key)
        if allow_candidate and document['status']=='candidate_pending_source_image_qa' and b is None:continue
        require(isinstance(b,dict) and Path(b['path']).is_absolute(),'actual source/QA binding required: '+key)
        digest(b['sha256'])
    return document

def validate_selection_binding(binding, *, allow_candidate=False):
    require(isinstance(binding,dict) and Path(binding.get('path','')).is_absolute() and isinstance(binding.get('document'),dict),
        'explicit independent200 selection binding required')
    digest(binding.get('sha256'));require(canonical_sha(binding['document'])==binding.get('document_sha256'),
        'embedded200 selection changed')
    return validate_selection(binding['document'],allow_candidate=allow_candidate)

def validate_selection_support(plan):
    """Open explicit new-namespace proof copies, never semantic historical paths."""
    import importlib.util
    from source_bindings import sha256
    root=Path(__file__).resolve().parent
    selection=plan['selection_binding']['document']
    support=plan.get('selection_support_bindings',{})
    require(set(support)=={'source_input_proof','manual_qa'},'both actual new source/QA copies required')
    expected={'source_input_proof':selection['source_input_proof_binding']['sha256'],
        'manual_qa':selection['manual_qa_binding']['sha256']}
    documents={}
    for key,b in support.items():
        p=Path(b['path'])
        require(p.is_absolute() and p.is_relative_to(root) and p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)) and
            sha256(p)==b['sha256']==expected[key] and canonical_sha(b['document'])==b['document_sha256'],
            'actual new source/QA byte binding differs: '+key)
        actual=json.loads(p.read_text());require(actual==b['document'],'source/QA embedded document differs')
        documents[key]=actual
    helper_path=root/'source_selection_helper.py'
    require(sha256(helper_path)=='e3d7487020196b7b534f117bc5af279449b059f546e75ae4f04ecb92772e6cf2',
        'approved source-only selection validator changed')
    spec=importlib.util.spec_from_file_location('_bound200_source_rgb_selection',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    # Runtime's original non-test manifest is supplied explicitly by the plan
    # creator; no original image or model is loaded by this pure proof gate.
    source=plan['source_manifest_binding'];p=Path(source['path'])
    require(p.is_file() and sha256(p)==source['sha256']==DATA_MANIFEST_SHA256,'actual bound non-test240 CSV differs')
    rows=helper.load_source(p)
    helper.validate_input_proof(documents['source_input_proof'],rows)
    decisions=helper.validate_qa(documents['manual_qa'],rows,support['source_input_proof']['sha256'])
    rebuilt=helper.make_selection(rows,source_binding=selection['source_manifest_binding'],decisions=decisions,
        input_binding=selection['source_input_proof_binding'],qa_binding=selection['manual_qa_binding'],frozen=True)
    require(rebuilt==selection,'frozen200 selection does not follow actual source-only QA and deterministic criteria')
    return documents
