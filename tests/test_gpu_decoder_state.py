"""The upstream accept-on-nonblank rule must survive device-resident state."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time

import numpy as np
import pytest

from parakeet_service import tensorrt_backend as trt


class Value:
    def __init__(self, array, device):
        self.array = np.array(array, copy=True) if device == 'cuda' else np.asarray(array)
        self.device = device

    @staticmethod
    def ortvalue_from_numpy(array, device='cpu', device_id=0):
        return Value(array, device)

    @staticmethod
    def ortvalue_from_shape_and_type(shape, dtype, device, device_id):
        return Value(np.empty(shape, dtype), device)

    def numpy(self):
        assert self.device == 'cpu', 'decoder state was copied to the host'
        return self.array

    def update_inplace(self, array):
        self.array[...] = array


class Binding:
    def __init__(self):
        self.inputs = {}
        self.devices = {}
        self.bound_outputs = {}

    def bind_cpu_input(self, name, value):
        if name in {'targets', 'target_length'}:
            assert value.dtype == np.int32
        self.inputs[name] = value.copy()  # ORT may copy host input at bind time.

    def bind_ortvalue_input(self, name, value):
        assert value.device == 'cuda'
        if name == 'targets': assert value.array.dtype == np.int32
        self.inputs[name] = value.array

    def bind_output(self, name, device, device_id=0):
        self.devices[name] = device

    def bind_ortvalue_output(self, name, value):
        self.bound_outputs[name] = value
        self.devices[name] = value.device

    def get_outputs(self):
        return self.outputs


class Decoder:
    def get_inputs(self):
        return [SimpleNamespace(name=f'input_states_{i}', shape=[2, 'batch', 4]) for i in (1, 2)]

    def io_binding(self):
        return Binding()

    def run(self, names, inputs):
        state = inputs['input_states_1']
        blank = int(inputs['encoder_outputs'].flat[0]) == 0
        # A blank produces a deliberately poisonous candidate. Subsequent
        # accepted tokens must still see the previous accepted state.
        token = 0 if blank else (1 if state.flat[0] < 10 else 2)
        logits = np.zeros((1, 1, 5), np.float32)
        logits[0, 0, token] = 1
        logits[0, 0, 4] = 1  # TDT duration = 1 frame.
        return [logits, state + (100 if blank else 1), inputs['input_states_2'] + 1]

    def run_with_iobinding(self, binding):
        time.sleep(.001)
        names = ['outputs', 'output_states_1', 'output_states_2']
        binding.outputs = []
        for name, value in zip(names, self.run(names, binding.inputs)):
            if name in binding.bound_outputs:
                result = binding.bound_outputs[name]
                assert all(not np.shares_memory(result.array, v) for v in binding.inputs.values())
                result.array[...] = value
            else:
                result = Value(value, binding.devices[name])
            binding.outputs.append(result)


@pytest.fixture
def asr(monkeypatch):
    monkeypatch.setattr(trt.ort, 'OrtValue', Value)
    model = trt.TensorRTParakeet.__new__(trt.TensorRTParakeet)
    model._decoder_joint = Decoder()
    model._token_dtypes = {'targets': np.int32, 'target_length': np.int32}
    model._vocab_size = 3
    model._blank_idx = 0
    model.config = {}
    model.use_low_precision = False
    model.gpu_decoder_state = True
    return model


def decode(asr):
    # Emit, blank, emit, blank, blank, emit.
    encoded = np.array([[[1], [0], [1], [0], [0], [1]]], np.float32)
    return list(asr._decoding(encoded, np.array([6], np.int64), need_logprobs=True))


def test_blank_candidates_do_not_overwrite_accepted_state_or_timestamps(asr):
    gpu = decode(asr)
    asr.gpu_decoder_state = False
    host = decode(asr)
    assert gpu == host
    assert gpu[0][0] == [1, 1, 1]
    assert gpu[0][1] == [0, 2, 5]


def test_blank_candidate_is_distinct_and_cannot_mutate_accepted_storage(asr):
    accepted = asr._create_state()
    _, _, candidate = asr._decode([], accepted, np.array([0], np.float32))
    for original, proposed in zip(accepted.request.states[accepted.slot], candidate.request.states[candidate.slot]):
        assert original is not proposed
        np.testing.assert_array_equal(original.array, 0)
        assert not np.shares_memory(original.array, proposed.array)


def test_workers_share_sessions_but_never_decoder_state(asr):
    expected = decode(asr)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: decode(asr), range(12)))
    assert all(result == expected for result in results)
