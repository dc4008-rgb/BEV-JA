"""C-only same-step live prediction collection, join, then original math/evidence.

No model construction, global policy mutation, autograd invocation, SSH or launch.
The original predictor is called exactly once in the real producer phase. Cached
consumption is a separately named no-count operation, never another model call.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
import copy
import json
import random
import threading
import time


def need(value, message):
    if not value: raise RuntimeError(message)


class ConcurrentLedger:
    """Same callback-attempt/return semantics, protected counters, no fn lock."""
    def __init__(self, original_factory, budget):
        self.template = copy.deepcopy(original_factory(budget).snapshot())
        need(self.template['planned'] == budget, 'original callback plan is explicit')
        self.planned = dict(budget)
        self.attempted = dict(self.template['attempted'])
        self.completed = dict(self.template['completed'])
        need(all(v == 0 for v in self.attempted.values()) and all(v == 0 for v in self.completed.values()), 'new ledger starts empty')
        self.lock = threading.Lock(); self.active = 0

    def call(self, name, fn, *args, **kwargs):
        with self.lock:
            need(name in self.planned, 'declared original callback category')
            self.attempted[name] += 1
            need(self.attempted[name] <= self.planned[name], 'original callback budget exceeded')
            self.active += 1
        try: result = fn(*args, **kwargs)
        except BaseException:
            with self.lock: self.active -= 1
            raise
        with self.lock:
            self.completed[name] += 1; self.active -= 1
        return result

    def snapshot(self):
        with self.lock:
            value = copy.deepcopy(self.template)
            value.update(planned=dict(self.planned), attempted=dict(self.attempted), completed=dict(self.completed))
            return value

    def require_complete(self):
        with self.lock:
            need(self.active == 0 and self.planned == self.attempted == self.completed, 'all true original callbacks returned exactly once')


def _clone_live(value, torch):
    if isinstance(value, torch.Tensor): return value.clone()  # preserves the live autograd edge
    if isinstance(value, dict): return {key: _clone_live(v, torch) for key, v in value.items()}
    if isinstance(value, list): return [_clone_live(v, torch) for v in value]
    if isinstance(value, tuple):
        values = [_clone_live(v, torch) for v in value]
        return type(value)(*values) if hasattr(value, '_fields') else tuple(values)
    return value


def _record(value, stream, torch):
    if isinstance(value, torch.Tensor):
        if value.is_cuda: value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values(): _record(item, stream, torch)
    elif isinstance(value, (list, tuple)):
        for item in value: _record(item, stream, torch)


def _state_key(tensor, view):
    return (tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.device), str(tensor.dtype), bool(tensor.requires_grad), int(tensor._version),
            json.dumps(view, sort_keys=True, separators=(',', ':'), allow_nan=False))


def _input_description(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.device),
            str(tensor.dtype), bool(tensor.requires_grad), int(tensor._version))


class GeometryCache:
    """Only this step's exact inputs/graph; no cross-step model or tensor reuse."""
    def __init__(self): self.values = {}; self.actual_transforms = self.reused_transforms = 0
    def transform(self, native, tensor, view):
        key = _state_key(tensor, view)
        if key in self.values:
            original, output = self.values[key]
            need(_state_key(original, view) == key, 'same-step transform source was mutated')
            self.reused_transforms += 1
            return output
        output = native(tensor, view)
        self.values[key] = (tensor, output); self.actual_transforms += 1
        return output


class _CachedAdapter:
    def __init__(self, name): self.name = name; self.queues = {}; self.consumed = 0; self.last_spatial = {}
    def append(self, category, image, output, spatial_receipt=None, expected_input=None):
        expected_input = _input_description(image) if expected_input is None else expected_input
        need(_input_description(image) == expected_input, 'predictor must not mutate its shared actual input')
        self.queues.setdefault(category, deque()).append((image, output, spatial_receipt, expected_input))
    def predict(self, image): raise RuntimeError('cached prediction must pass explicit no-count consumer')
    def predict_for_training(self, image): raise RuntimeError('cached depth must pass explicit no-count consumer')
    def consume(self, category, image):
        q = self.queues.get(category); need(q, 'no cached result reused or additional predictor request')
        expected, output, spatial, description = q.popleft()
        need(image is expected and _input_description(image) == description,
             'loss/evidence must consume the exact unmodified original live prediction input tensor')
        if spatial is not None: self.last_spatial[category] = copy.deepcopy(spatial)
        self.consumed += 1
        return output


class _ConsumeLedger:
    def __init__(self, original): self.original = original; self.depth_adapters = {}
    def call(self, name, fn, *args, **kwargs):
        adapter = getattr(fn, '__self__', None)
        if isinstance(adapter, _CachedAdapter):
            need(len(args) == 1 and not kwargs and name in ('detector_forwards', 'depth_clean_forwards', 'depth_adv_forwards'), 'exact declared no-count forward consumption')
            return adapter.consume(name, args[0])
        raise RuntimeError('consumer cannot execute a new model or other scientific callback')
    def benchmark_spatial_receipt(self, category, model_id):
        return copy.deepcopy(self.depth_adapters[model_id].last_spatial[category])


class _Parent:
    def __init__(self, original, patch, context, rendered):
        self.original, self.patch, self.context, self.rendered = original, patch, context, rendered
    def __getattr__(self, name): return getattr(self.original, name)
    def RoadRGBView(self, patch):
        need(patch is self.patch, 'only this actual patch may consume the prepared render graph')
        owner = self
        class View:
            def render(self, context):
                need(context is owner.context, 'same actual context for prepared render')
                return owner.rendered
        return View()


class Prepared:
    def __init__(self, *, parent, patch, context, views, adapters, depth_objective, ledger, mode):
        import torch
        import numpy as np
        import geometry_eot
        from depth_spatial import DepthSpatialScope
        need(mode in ('C1', 'C1_PREFETCH_SERIAL') and len(views) == 2, 'only explicit two-view C scheduling')
        need(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled()
             and not torch.is_deterministic_algorithms_warn_only_enabled(), 'original single CPU thread strict policy unchanged')
        need(torch.is_grad_enabled() and not torch.is_autocast_enabled(), 'original explicit precision/live graph; no AMP override')
        need(patch.is_cuda and set(adapters) == {'fasterrcnn', 'ssd300'}
             and set(depth_objective.adapters) == {'monodepth2', 'dpt_large'}, 'four actual distinct frozen model adapters')
        self.mode, self.cache, self.geometry = mode, GeometryCache(), geometry_eot
        self.adapters = {name: _CachedAdapter(name) for name in adapters}
        cached_depth = {name: _CachedAdapter(name) for name in depth_objective.adapters}
        self.depth_objective = copy.copy(depth_objective)
        self.depth_objective.adapters = cached_depth
        self.ledger = _ConsumeLedger(ledger)
        self.ledger.depth_adapters = cached_depth
        started = time.perf_counter()
        # Preparation happens once on the producer/default stream, before any
        # worker sees inputs. The same graph objects are used by original math.
        with geometry_eot.benchmark_transform_cache(self.cache):
            rendered = parent.RoadRGBView(patch).render(context)
            det_images = [geometry_eot.transform_tensor(rendered.image, view) for view in views]
            clean_images = [geometry_eot.transform_tensor(context.benign.detach(), view) for view in views]
            adv_images = [geometry_eot.transform_tensor(rendered.image, view) for view in views]
        self.parent = _Parent(parent, patch, context, rendered)
        self.depth_objective.parent = self.parent
        original_cpu_rng = torch.get_rng_state().clone()
        original_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        original_python_rng = random.getstate()
        original_numpy_rng = np.random.get_state()
        main = torch.cuda.current_stream(patch.device)
        origin = torch.cuda.Event(enable_timing=True); origin.record(main)
        keys = ('fasterrcnn', 'ssd300', 'monodepth2', 'dpt_large')
        streams = {name: torch.cuda.Stream(device=patch.device) for name in keys} if mode == 'C1' else {name: main for name in keys}
        if mode == 'C1': need(len({stream.cuda_stream for stream in streams.values()}) == 4, 'four truly distinct CUDA stream handles')
        events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for name in keys}
        complete = {}; lock = threading.Lock()

        def branch(name):
            stream = streams[name]
            with torch.cuda.stream(stream), torch.enable_grad():
                stream.wait_event(origin); events[name][0].record(stream)
                try:
                    if name in adapters:
                        model, cached = adapters[name], self.adapters[name]
                        for image in det_images:
                            _record(image, stream, torch)
                            description = _input_description(image)
                            value = ledger.call('detector_forwards', model.predict, image)
                            cached.append('detector_forwards', image, _clone_live(value, torch), expected_input=description)
                    else:
                        model, cached = depth_objective.adapters[name], cached_depth[name]
                        for clean, adversarial in zip(clean_images, adv_images):
                            _record(clean, stream, torch); _record(adversarial, stream, torch)
                            clean_description = _input_description(clean)
                            with torch.no_grad(), DepthSpatialScope(name) as clean_scope:
                                value = ledger.call('depth_clean_forwards', model.predict_for_training, clean)
                                cloned = _clone_live(value, torch)
                            cached.append('depth_clean_forwards', clean, cloned, clean_scope.receipt(), clean_description)
                            adv_description = _input_description(adversarial)
                            with DepthSpatialScope(name) as adv_scope:
                                value = ledger.call('depth_adv_forwards', model.predict_for_training, adversarial)
                                cloned = _clone_live(value, torch)
                            cached.append('depth_adv_forwards', adversarial, cloned, adv_scope.receipt(), adv_description)
                finally:
                    events[name][1].record(stream)
                    with lock: complete[name] = True

        try:
            if mode == 'C1':
                # A model has a single worker and stream, so its mutable forward
                # internals are never concurrently re-entered. Constructors remain
                # in the original single-thread API startup.
                with ThreadPoolExecutor(max_workers=4, thread_name_prefix='owned_model_branch') as pool:
                    futures = [pool.submit(branch, name) for name in keys]
                    wait(futures)
                    for future in futures: future.result()
            else:
                for name in keys: branch(name)
        except BaseException as error:
            # Reap all owned host workers above, then drain only already-recorded
            # branch ends before propagating the original failure. Never retry a
            # predictor or pretend the incomplete callback budget succeeded.
            for name in tuple(complete):
                try: events[name][1].synchronize()
                except BaseException as drain_error:
                    if hasattr(error, 'add_note'): error.add_note('owned branch drain failed: ' + repr(drain_error))
            raise
        need(set(complete) == set(keys), 'all four owned producers terminated')
        for name in keys:
            main.wait_event(events[name][1]); events[name][1].synchronize()
        for adapter in list(self.adapters.values()) + list(cached_depth.values()):
            for queue in adapter.queues.values():
                for _, output, _, _ in queue: _record(output, main, torch)
        numpy_after = np.random.get_state()
        need(torch.equal(torch.get_rng_state(), original_cpu_rng)
             and len(torch.cuda.get_rng_state_all()) == len(original_cuda_rng)
             and all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), original_cuda_rng))
             and random.getstate() == original_python_rng
             and numpy_after[0] == original_numpy_rng[0]
             and np.array_equal(numpy_after[1], original_numpy_rng[1])
             and numpy_after[2:] == original_numpy_rng[2:], 'frozen prediction branches unexpectedly consumed global RNG')
        intervals = [{'model': name, 'stream_handle': streams[name].cuda_stream,
                      'start_ms_from_producer_event': origin.elapsed_time(events[name][0]),
                      'end_ms_from_producer_event': origin.elapsed_time(events[name][1])} for name in keys]
        self.metadata = dict(mode=mode, true_predictor_calls={'detector_forwards': 4, 'depth_clean_forwards': 4, 'depth_adv_forwards': 4},
            no_count_consumption_is_not_a_second_model_call=True, producer_wait_and_join=True,
            original_model_construction_not_parallelized=True, output_clone_preserves_autograd_graph=True,
            no_cross_step_prediction_reuse=True, global_RNG_unchanged=True,
            producer_prepare_and_model_join_wall_seconds=time.perf_counter() - started,
            branch_device_intervals=intervals, intervals_are_not_kernel_active_overlap_proof=True,
            task_VJPs_still_run_sequentially_on_shared_graph=True)

    @contextmanager
    def consume_context(self):
        with self.geometry.benchmark_transform_cache(self.cache): yield

    def finish(self):
        for adapter in list(self.adapters.values()) + list(self.depth_objective.adapters.values()):
            need(all(not q for q in adapter.queues.values()), 'every real prediction consumed exactly once')
        self.metadata.update(actual_geometry_transform_calls=self.cache.actual_transforms,
                             same_step_exact_geometry_reuses=self.cache.reused_transforms,
                             total_no_count_consumptions=sum(a.consumed for a in list(self.adapters.values()) + list(self.depth_objective.adapters.values())))
        need(self.metadata['total_no_count_consumptions'] == 12, 'exact two-view original predictor callback count')
        self.cache.values.clear()
        return self.metadata
