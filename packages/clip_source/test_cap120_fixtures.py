"""Synthetic CPU fixtures only; never used by a CLI or live plan validator."""
from followup_contract import ENVIRONMENT_FIELDS, scientific_environment, require_matching_environment


def environment_evidence(package_sha):
    runtime = {key: None for key in ENVIRONMENT_FIELDS}
    runtime['gpu_identity'] = 'SYNTHETIC DEVICE, GPU-SYNTHETIC, SYNTHETIC-DRIVER'
    contract = scientific_environment(runtime)
    execution = {'origin': 'independent_off500_same_original_package_full2041_exact',
                 'package_sha256': 'a' * 64, 'files': {'summary.json': 'b' * 64},
                 'scientific_environment': contract}
    off = {'run': '/synthetic/off', 'control_kind': execution['origin'],
           'prior_package_sha256': execution['package_sha256'], 'followup_package_sha256': package_sha,
           'files': execution['files'], 'scientific_environment': contract, 'compared_states': 2041, 'tolerance': 0}
    gate = require_matching_environment(runtime, off)
    runtime['execution_off_environment_gate'] = gate
    return runtime, execution, off, gate
