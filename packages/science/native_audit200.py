"""SCI200 native ownership and saved-gradient audit; original arithmetic unchanged.

Native verifier/replay functions below are copied verbatim from reviewed OPS
b1183f56. Only the final two wrappers route explicit old/new scientific owners.
No complete BATCH is invented and no original metadata is rewritten.
"""
from __future__ import annotations
import copy,hashlib,importlib,importlib.util,json,math,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
IDENTITY='ours_complete_eot1_parallel200_v1'
PROBE_STEPS=(1,25,500)
PROFILE={"PYTHONHASHSEED":"20260919","PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "CUBLAS_WORKSPACE_CONFIG":":4096:8","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
def require(value, message):
    if not value:
        raise ValueError(message)

def plain(path):
    p = Path(path)
    require(p.is_absolute() and ".." not in p.parts and
        not any(x.is_symlink() for x in (p, *p.parents)), "absolute nonsymlink path required")
    return p

def sha(path):
    h = hashlib.sha256()
    with plain(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def binding(path):
    p = plain(path)
    require(p.is_file(), "regular evidence file required")
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}

def read(path):
    def finite(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON")
        return value
    return json.loads(plain(path).read_bytes(), parse_float=finite,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON " + x)))

def checked(bound):
    require(set(bound) == {"path", "bytes", "sha256"} and binding(bound["path"]) == bound,
        "retained artifact changed")
    return plain(bound["path"])

def exact(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b

def module(code, name):
    m = importlib.import_module(name)
    require(Path(m.__file__).resolve() == code / (name + ".py"), "scientific module shadowed: " + name)
    return m

def validate_native_case(context, row, run, *, observe=None):
    """Read-only CPU native ownership/phase/patch/controller checks, no GPU calls."""
    config, code = context["config"], context["code"]
    plan = context["plans"][str(row["original_shard_id"])]
    sample = context["contract"].select_sample(plan, row["sample_id"])
    evaluation, native, artifact = (module(code, n) for n in ("sample_evaluate", "periodic_training_audit", "artifact_io"))
    import torch
    observed = {} if observe is None else observe
    def bound(path):
        b = binding(path); old = observed.get(b["path"])
        require(old is None or old == b, "evidence changed during inspection")
        observed[b["path"]] = b
        return b
    run = plain(run)
    summary_path = run / "summary.json"
    summary = evaluation.validate_summary(plan, sample, read(summary_path)); bound(summary_path)
    require(summary.get("resume_enabled") is False and summary.get("fresh_sample_initialization") is True and
        summary.get("no_cross_sample_state") is True and summary.get("switches") == plan["scientific_config"]["switches"],
        "case fresh-state contract differs")
    runtime_path = run / "runtime_provenance.json"; runtime = read(runtime_path); bound(runtime_path)
    require(runtime.get("identity") == IDENTITY and runtime.get("sample_id") == sample["sample_id"] and
        runtime.get("source_sha256") == sample["source_sha256"] and runtime.get("plan_sha256") == plan["_plan_sha256"] and
        runtime.get("executed_updates") == 500 and runtime.get("real_cuda_run") is True and
        runtime.get("dtype") == "torch.bfloat16", "case native runtime scientific owner differs")
    manifest_path = run / "state_evidence/MANIFEST.json"; manifest = read(manifest_path)
    require(bound(manifest_path)["sha256"] == summary["state_evidence_manifest_sha256"], "state manifest changed")
    evaluation.validate_state_manifest(plan, sample, manifest)
    trace_path, cycle_path = run / "attack_trace.jsonl", run / "cycle_trace.jsonl"
    require(bound(trace_path)["sha256"] == summary["trace_sha256"] and
        bound(cycle_path)["sha256"] == summary["cycle_trace_sha256"], "native trace bytes changed")
    trace, cycles = native.read_jsonl(trace_path), native.read_jsonl(cycle_path)
    replay = native.replay_case_trace(trace, cycles, plan["scientific_config"], sample, 500)
    require(summary.get("statistical_cycles") == replay["statistical_cycles"] == 100 and
        summary.get("weight_refreshes_used") == replay["weight_refreshes_used"] == 80 and replay["outer_reset_verified"] is True,
        "full500 periodic replay differs")
    phase_index = {(e["phase"], e["global_step"]): e for e in manifest["entries"]}
    for t in trace:
        require(set(t.get("state_bindings", {})) == {"before_denoise", "pre_backward", "pre_update", "post_update"},
            "trace phase fingerprints missing")
        for phase, digest in t["state_bindings"].items():
            require(digest == phase_index[phase, t["global_step"]]["state_fingerprint"], "trace/phase hash mismatch")
    retained = {}; probes = []
    for entry in manifest["outer_boundary_checkpoints"] + manifest["fixed_update_math_checks"]:
        p = native.checked_file(run, entry["path"], entry); bound(p)
        payload = artifact.load_torch_lossless(p)
        native.validate_payload_metadata(payload, plan, sample)
        require(payload.get("phase") == entry["phase"] and payload.get("global_step") == entry["global_step"] and
            artifact.state_fingerprint(payload.get("state")) == entry["state_fingerprint"], "saved typed state differs")
        if entry["retention_kind"] == "initial_or_final_state":
            retained[entry["phase"]] = payload["state"]
        else:
            probes.append({"sample_id": sample["sample_id"], "case_id": sample["case_id"],
                "global_step": entry["global_step"], "artifact_binding": observed[str(p)],
                "state_fingerprint": entry["state_fingerprint"], "state": payload["state"], "row": trace[entry["global_step"] - 1]})
    patches = {}
    for name, b in {**summary["patch_bindings"], **summary["checkpoint_bindings"]}.items():
        p = native.checked_file(run, name, b); bound(p)
        payload = torch.load(p, map_location="cpu", weights_only=True)
        count = 500 if name in ("final_patch.pt", "checkpoint_0500.pt") else 0
        # initial_vae_reconstruction also carries the original zero-update owner.
        evaluation.validate_patch_metadata(payload, plan, sample, count)
        patches[name] = payload
    require(torch.equal(patches["final_patch.pt"]["canonical_rgb"], patches["checkpoint_0500.pt"]["canonical_rgb"]) and
        torch.equal(patches["final_patch.pt"]["raw_generator_rgb"], patches["checkpoint_0500.pt"]["raw_generator_rgb"]),
        "fixed final/checkpoint500 differ")
    require(torch.equal(patches["zero_step_patch.pt"]["canonical_rgb"], retained["initialization"]["initial_small"]) and
        torch.equal(patches["input_raw512.pt"]["raw_generator_rgb"], retained["initialization"]["initial_patch"]) and
        torch.equal(patches["final_patch.pt"]["raw_generator_rgb"], retained["outer_end"]["raw_patch"]),
        "own initial/final retained patch identity differs")
    require([p["global_step"] for p in probes] == list(PROBE_STEPS), "three fixed500 probes required")
    return {"row": row, "sample": sample, "plan": plan, "run": run, "summary": summary, "runtime": runtime,
        "summary_binding": observed[str(summary_path)], "runtime_binding": observed[str(runtime_path)],
        "observed": observed, "probes": probes, "replay": replay, "native": native, "artifact": artifact}

def native_remaining_math(case, probe, rebuilt, details):
    """Pure CPU native clip/deficit arithmetic after owner-bound joint evidence."""
    import torch
    state, row, plan = probe["state"], probe["row"], case["plan"]
    require((row.get("sample_id"), row.get("case_id"), row.get("global_step")) ==
        (probe["sample_id"], probe["case_id"], probe["global_step"]), "saved probe owner differs")
    gd, gp, saved = (state[k] for k in ("detection_task_gradient", "depth_task_gradient", "joint_gradient"))
    require(gd.device.type == gp.device.type == saved.device.type == "cpu" and
        gd.dtype in (torch.float32, torch.bfloat16) and gp.dtype in (torch.float32, torch.bfloat16) and saved.dtype == torch.float32 and
        gd.shape == gp.shape == saved.shape and gd.numel() and all(bool(torch.isfinite(v).all()) for v in (gd, gp, saved)),
        "finite retained native VJPs required")
    switches = plan["scientific_config"]["switches"]
    keywords = {"task_balance": switches["task_balance"], "conflict_guard": switches["conflict_guard"],
        "norm_match": switches["norm_match"], "n_min": plan["scientific_config"]["controller_config"]["n_min"]}
    require(state.get("switches") == switches and state.get("current_period_weights") == row["current_period_weights"] and
        state.get("requested_task_priorities") == row["task_weights"] and exact(state["task_allocation"], row["task_allocation"]),
        "saved task allocation/trace differs")
    require(torch.equal(rebuilt, saved) and exact(details, state["task_allocation"]),
        "owner-bound joint evidence differs; no tolerance is broadened")
    native = case["native"]; original_combine = native.combine_task_gradients; calls = []
    def owned(left, right, priorities, **kw):
        require(not calls and left.device.type == right.device.type == "cpu" and left.dtype == gd.dtype and right.dtype == gp.dtype and
            torch.equal(left, gd) and torch.equal(right, gp) and exact(priorities, state["requested_task_priorities"]) and exact(kw, keywords),
            "unowned or repeated native combine request")
        calls.append(True)
        return rebuilt.clone(), copy.deepcopy(details)
    try:
        native.combine_task_gradients = owned
        native_check = native.audit_math_state(state, row, plan["scientific_config"])
    finally:
        native.combine_task_gradients = original_combine
    require(len(calls) == 1, "native arithmetic check did not use exactly one owned combine")
    return native_check

def replay_math_probe(context, case, probe):
    import torch
    state, plan = probe["state"], case["plan"]
    gd, gp, saved = (state[k] for k in ("detection_task_gradient", "depth_task_gradient", "joint_gradient"))
    switches = plan["scientific_config"]["switches"]
    keywords = {"task_balance": switches["task_balance"], "conflict_guard": switches["conflict_guard"],
        "norm_match": switches["norm_match"], "n_min": plan["scientific_config"]["controller_config"]["n_min"]}
    combine = module(context["code"], "task_gradient_balance").combine_task_gradients
    with torch.no_grad():
        left, right = gd.to("cuda:0"), gp.to("cuda:0")
        require(torch.equal(left.cpu(), gd) and torch.equal(right.cpu(), gp), "saved VJP transfer was lossy")
        rebuilt, details = combine(left, right, state["requested_task_priorities"], **keywords)
        rebuilt = rebuilt.cpu()
    require(torch.equal(rebuilt, saved) and exact(details, state["task_allocation"]),
        "current-CUDA saved-gradient replay differs; no tolerance is broadened")
    native_check = native_remaining_math(case, probe, rebuilt, details)
    return {k: probe[k] for k in ("global_step", "artifact_binding", "state_fingerprint")} | {
        "current_cuda_joint_torch_equal": True, "current_cuda_allocation_details_exact": True,
        "saved_joint_typed_sha256": case["artifact"].typed_tensor_sha(saved),
        "replayed_joint_typed_sha256": case["artifact"].typed_tensor_sha(rebuilt),
        "native_check": native_check, "cpu_joint_tolerance_gate_executed": False}

def configure_math_runtime(context):
    require(all(os.environ.get(k) == v for k, v in PROFILE.items()), "deterministic OPS environment must precede Python startup")
    import torch
    import torchvision
    import diffusers
    require(tuple(v.split("+")[0] for v in (torch.__version__, torchvision.__version__, diffusers.__version__)) ==
        ("2.8.0", "0.23.0", "0.38.0") and torch.cuda.is_available(), "pinned real CUDA audit runtime required")
    for key in ("r2_source", "v2_source", "clip_source"):
        sys.path.insert(1, str(context["paths"][key]))
    clip_path = context["paths"]["clip_source"] / "clip_training.py"
    spec = importlib.util.spec_from_file_location("_parallel_math_frozen_clip", clip_path)
    clip = importlib.util.module_from_spec(spec); sys.modules[spec.name] = clip; spec.loader.exec_module(clip)
    clip.configure_determinism()
    module(context["code"], "sample_training").configure_dual_backward_runtime()

def audit_case(plan, sample, run, *, observe=None):
    """Pure CPU500 owner/controller/phase/state/patch proof; no GPU or models."""
    from case_owners200 import resolve_case_owner
    import sample_contract
    from selection_contract200 import NATIVE_FIELDS
    owner=resolve_case_owner(plan,sample)
    key=str(owner.get('shard_id',0))
    row={k:sample[k] for k in NATIVE_FIELDS};row.update(original_shard_id=int(key),plan_sha256=owner['_plan_sha256'])
    context={'config':{},'code':ROOT,'plans':{key:owner},'contract':sample_contract}
    case=validate_native_case(context,row,run,observe=observe)
    case['evaluation_plan_sha256']=plan['_plan_sha256']
    return case


def configure_saved_math_runtime(source_paths):
    """One explicit current CUDA setup; no model or source image loads."""
    context={'code':ROOT,'paths':{k:plain(source_paths[k]) for k in ('r2_source','v2_source','clip_source')}}
    configure_math_runtime(context)
    return context


def audit_saved_math_probe(case, probe):
    """Current CUDA exact saved-VJP replay plus unchanged native remaining math."""
    return replay_math_probe({'code':ROOT},case,probe)


def verify_saved_math_probe(case, actual_probe, report):
    """Pure CPU owner/report check; does not pretend CPU joint tolerance passed."""
    saved=actual_probe['state']['joint_gradient']
    require(report.get('global_step')==actual_probe['global_step'] and report.get('artifact_binding')==actual_probe['artifact_binding'] and
        report.get('state_fingerprint')==actual_probe['state_fingerprint'] and
        report.get('current_cuda_joint_torch_equal') is True and report.get('current_cuda_allocation_details_exact') is True and
        report.get('saved_joint_typed_sha256')==report.get('replayed_joint_typed_sha256')==case['artifact'].typed_tensor_sha(saved) and
        report.get('cpu_joint_tolerance_gate_executed') is False,'actual saved current-CUDA probe owner/exactness differs')
    checked(report['artifact_binding'])
    require(exact(report['native_check'],native_remaining_math(case,actual_probe,saved,actual_probe['state']['task_allocation'])),
        'actual native remaining500 probe arithmetic differs')
    return True
