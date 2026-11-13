"""Exact accepted atomic count aggregation, with the evaluated YOLOv5 target ID."""
DETECTORS = ('fasterrcnn', 'ssd300', 'yolov5s', 'fcos')
DEPTHS = ('monodepth2', 'dpt_large', 'depth_anything_v1', 'zoedepth')
THRESHOLDS = {'confidence_threshold': 0.3, 'iou_threshold': 0.5,
              'beta_eval': 0.1, 'rho_min': 0.5, 'target_labels': [3, 6, 8]}

def need(condition, message):
    if not condition:
        raise ValueError(message)

def summarize_atomic_jasr(atomic, sample_ids):
    expected={(sid,d,z) for sid in sample_ids for d in DETECTORS for z in DEPTHS}
    keys=[(r['sample_id'],r['detector'],r['depth_model']) for r in atomic]
    need(len(keys)==len(set(keys))==len(expected) and set(keys)==expected,'exact case-by16 atomic coverage')
    output=[]
    for d in DETECTORS:
        for z in DEPTHS:
            valid=[]
            for r in atomic:
                if r['detector']!=d or r['depth_model']!=z:continue
                need(type(r['clean_match']['matched']) is bool and type(r['final_match']['matched']) is bool and
                     type(r['clean_depth_valid']) is bool and type(r['detection_success']) is bool and
                     (type(r['depth_success']) is bool if r['clean_depth_valid'] else r['depth_success'] is None),
                     'original Boolean/null atomic outcomes')
                eligible=r['clean_match']['matched'] and r['clean_depth_valid']
                need(type(r['eligible_clean_pair']) is bool and r['eligible_clean_pair']==eligible and
                     r['detection_success']==(not r['final_match']['matched']) and
                     r['joint_success']==(bool(r['detection_success'] and r['depth_success']) if eligible else None),'samevehicle event and clean-only denominator')
                if eligible:valid.append(r)
            n=len(valid);dh=sum(int(r['detection_success']) for r in valid);zh=sum(int(r['depth_success']) for r in valid);jh=sum(int(r['joint_success']) for r in valid)
            output.append({'detector':d,'depth_model':z,'eligible_count':n,'excluded_count':len(sample_ids)-n,
              'detection_success_count':dh,'depth_success_count':zh,'joint_success_count':jh,
              'DetASR':dh/n if n else None,'DepASR':zh/n if n else None,'JASR':jh/n if n else None})
    return output
