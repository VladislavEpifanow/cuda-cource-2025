import numpy as np
import tensorrt as trt
import torch

from .config import INPUT_HEIGHT, INPUT_WIDTH


TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}
if hasattr(trt.DataType, "UINT8"):
    TRT_TO_TORCH_DTYPE[trt.DataType.UINT8] = torch.uint8
if hasattr(trt.DataType, "INT64"):
    TRT_TO_TORCH_DTYPE[trt.DataType.INT64] = torch.int64


def trt_dtype_to_torch(dtype):
    if dtype not in TRT_TO_TORCH_DTYPE:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return TRT_TO_TORCH_DTYPE[dtype]


class TrtModel:
    def __init__(self, engine_path):
        self._closed = False
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        # Use a dedicated stream for TRT to avoid depending on global/current PyTorch stream.
        self.trt_stream = torch.cuda.Stream()
        # Event to synchronize TRT completion with the consumer stream.
        self.trt_done_event = torch.cuda.Event(blocking=False)

        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(
            n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self.output_names = [
            n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]

        self.context.set_input_shape(self.input_name, (1, 3, INPUT_HEIGHT, INPUT_WIDTH))

        input_torch_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(self.input_name))
        self.input_tensor = torch.empty(
            (1, 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=input_torch_dtype, device="cuda"
        )
        self.host_input_tensor = torch.empty(
            (1, 3, INPUT_HEIGHT, INPUT_WIDTH),
            dtype=input_torch_dtype,
            device="cpu",
            pin_memory=True,
        )
        self.output_tensors = {}

        self.context.set_tensor_address(self.input_name, int(self.input_tensor.data_ptr()))
        for name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            if any(dim <= 0 for dim in out_shape):
                raise RuntimeError(
                    f"Unexpected dynamic/zero output shape for '{name}': {out_shape}. "
                    "Rebuild engine with build_engine.py."
                )
            out_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            tensor = torch.empty(out_shape, dtype=out_dtype, device="cuda")
            self.output_tensors[name] = tensor
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

    def infer(self, input_data_np):
        if self._closed:
            raise RuntimeError("TrtModel is closed.")
        if input_data_np.dtype != np.float32:
            input_data_np = input_data_np.astype(np.float32)
        if not input_data_np.flags["C_CONTIGUOUS"]:
            input_data_np = np.ascontiguousarray(input_data_np)

        cpu_src = torch.from_numpy(input_data_np)
        self.host_input_tensor.copy_(cpu_src, non_blocking=False)

        consumer_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.trt_stream):
            # Async H2D copy is effective when source is pinned host memory.
            self.input_tensor.copy_(self.host_input_tensor, non_blocking=True)
            ok = self.context.execute_async_v3(stream_handle=self.trt_stream.cuda_stream)
            if ok:
                self.trt_done_event.record(self.trt_stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed (execute_async_v3 returned False).")
        consumer_stream.wait_event(self.trt_done_event)

        return {name: tensor.clone() for name, tensor in self.output_tensors.items()}

    def close(self):
        if self._closed:
            return
        if self.trt_stream is not None:
            try:
                # Ensure queued TRT work is finished before releasing GPU resources.
                self.trt_stream.synchronize()
            except Exception:
                pass
        self.output_tensors.clear()
        self.host_input_tensor = None
        self.input_tensor = None
        self.context = None
        self.engine = None
        self.runtime = None
        self.logger = None
        self.trt_done_event = None
        self.trt_stream = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
