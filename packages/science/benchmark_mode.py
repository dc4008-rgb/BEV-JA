"""One explicit benchmark mode per new process; no runtime facts or CUDA setup."""
MODES = ('A3', 'B1', 'C1', 'C1_PREFETCH_SERIAL')
_mode = None


def configure(mode):
    global _mode
    if mode not in MODES: raise ValueError('explicit A3/B1/C1 benchmark mode required')
    if _mode is not None and _mode != mode: raise RuntimeError('one immutable benchmark mode per process')
    _mode = mode
    return mode


def current_mode():
    if _mode is None: raise RuntimeError('configure benchmark mode before importing/running the original API')
    return _mode


def random_views(): return 3 if current_mode() == 'A3' else 1
def total_views(): return 1 + random_views()
def uses_prefetch(): return current_mode() in ('C1', 'C1_PREFETCH_SERIAL')


def policy():
    return dict(mode=current_mode(), random_views=random_views(), total_shared_views=total_views(),
                source_derivative=True, old_SCI7832_identity_claim=False,
                scheduling='four_per_model_CUDA_streams_and_post_join_evidence' if current_mode() == 'C1'
                else 'prefetch_serial_shadow' if uses_prefetch() else 'original_serial_control_flow',
                task_autograd_calls_per_update=2, simultaneous_shared_graph_autograd=False,
                original_strict_determinism_and_single_CPU_thread_unchanged=True,
                formal_scientific_acceptance_claim=False)
