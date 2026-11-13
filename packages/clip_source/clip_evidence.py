"""Independent cap120 follow-up provenance; original six-group evidence stays sealed."""
from __future__ import annotations

from pathlib import Path
import json

from reference_long_training import LeanStateRecorder as OriginalRecorder
from reference_long_training import sha, write_json
from followup_contract import (
    load_plan, require_completed_off, require_historical_reference,
    validate_followup_plan, require_matching_environment, verify_scientific_kernels,
)

REFERENCE_ID = 'r2_joint_depth_long500_20261001-two-detector-foa-plus-fixed-depth'
REFERENCE_PACKAGE = 'b1a9086bc9a2595cc9a32a0692e884047efb12d63ffd18c67a8026a96dd4b70f'
SCHEMA = 'r2-joint-clip500-lean-state-manifest-v1'


class ClipStateRecorder(OriginalRecorder):
    """Reuse storage, while comparing OFF all500 or finite first-forward only.

    Finite clipping deliberately changes the trajectory. Its gate does NOT claim
    prefix100 equivalence. OFF calls the original update and compares all2041
    scientific state fingerprints to the completed historical joint500.
    """
    def __init__(self, output, candidate_id, reference_run, policy):
        self.output = Path(output)
        self.directory = self.output / 'state_evidence'
        self.directory.mkdir(exist_ok=False)
        self.candidate_id = candidate_id
        self.entries, self.recovery_checkpoints = [], []
        self.updates_applied = 0
        self.runtime_bindings = {}
        self.ledger = None
        self.initialization_state = None
        self.ledger_path = self.directory / 'ledger.jsonl'
        self.ledger_path.open('x').close()
        self.policy = dict(policy)
        self.reference_run = Path(reference_run).resolve()
        summary_path = self.reference_run / 'summary.json'
        summary = json.loads(summary_path.read_text())
        if (summary.get('status') != 'train_optimization_complete'
                or summary.get('executed_updates') != 500
                or summary.get('candidate_id') != REFERENCE_ID
                or summary.get('package_manifest_sha256') != REFERENCE_PACKAGE
                or summary.get('depth_weight') != 1.0
                or summary.get('test_data_opened') is not False
                or summary.get('validation_data_opened') is not False):
            raise ValueError('completed historical train-only joint500 required')
        path = self.reference_run / 'state_evidence/MANIFEST.json'
        if sha(path) != summary.get('state_evidence_manifest_sha256'):
            raise ValueError('historical state manifest SHA differs')
        manifest = json.loads(path.read_text())
        entries = manifest.get('entries', [])
        if (manifest.get('status') != 'complete' or manifest.get('executed_updates') != 500
                or manifest.get('candidate_id') != REFERENCE_ID or len(entries) != 2041):
            raise ValueError('complete historical 2041-state trajectory required')
        self.reference_entries = entries
        self.reference_binding = {'summary_sha256': sha(summary_path),
            'state_manifest_sha256': sha(path), 'candidate_id': REFERENCE_ID}
        self.prefix_counts = {'states_checked': 0}
        self.limit = 2041 if policy['max_norm'] is None else 4

    def _compare_prefix(self, phase, step, state, index):
        if index > self.limit:
            return
        from evidence import fingerprint
        expected = self.reference_entries[index - 1]
        if (expected['index'], expected['phase'], expected['global_step'], expected['updates_applied']) != (index, phase, step, self.updates_applied):
            raise ValueError('reference scientific phase/order differs')
        if fingerprint(state) != expected['state_fingerprint']:
            write_json(self.output / 'CLIPPING_COMPARISON_FAILURE.json',
                {'status': 'different', 'index': index, 'phase': phase, 'step': step,
                 'tolerance': 0, 'policy': self.policy, 'reference': self.reference_binding})
            raise RuntimeError('clipping control scientific state differs from historical reference')
        self.prefix_counts['states_checked'] += 1

    def finish(self, status, executed_updates):
        if status == 'complete':
            if (executed_updates != 500 or len(self.entries) != 2041
                    or len(self.recovery_checkpoints) != 20 or self.prefix_counts['states_checked'] != self.limit):
                raise RuntimeError('complete500 and the declared clipping comparison gate required')
            write_json(self.output / 'CLIPPING_COMPARISON_GATE.json', {
                'status': 'exact_full_trajectory' if self.limit == 2041 else 'exact_initialization_and_first_forward',
                'tolerance': 0, 'compared_states': self.limit, 'reference': self.reference_binding,
                'max_norm': self.policy['max_norm'], 'group_id': self.policy['group_id'],
                'grid_sha256': self.policy['grid_sha256'],
                'claim': 'finite groups may diverge starting with the first clipped update; no prefix100 equivalence claim'})
        value = {'schema': SCHEMA, 'candidate_id': self.candidate_id, 'status': status,
            'executed_updates': executed_updates, 'entries': self.entries,
            'initialization_state': self.initialization_state, 'recovery_checkpoints': self.recovery_checkpoints,
            'reference_binding': self.reference_binding, 'comparison_counts': self.prefix_counts,
            'gradient_clipping': self.policy, 'ledger_sha256': sha(self.ledger_path),
            'resume_enabled': False, 'recovery_policy': 'outer-boundary capture only; no tested resume implementation'}
        path = self.directory / 'MANIFEST.json'
        write_json(path, value)
        return sha(path)


def summarize_clipping(trace):
    if len(trace) != 500:
        raise ValueError('500 trace rows required')
    values = [r['clipping_diagnostics'] for r in trace]
    return {'steps': 500, 'triggered_steps': sum(v['clipped'] for v in values),
        'triggered_fraction': sum(v['clipped'] for v in values) / 500,
        'max_norm_before': max(v['norm_before'] for v in values),
        'max_norm_after': max(v['norm_after'] for v in values),
        'mean_realized_update_rms': sum(v['realized_update_rms'] for v in values) / 500,
        'note': 'actual clipped trajectory; BF16 rounding can slightly exceed the ideal norm bound'}
