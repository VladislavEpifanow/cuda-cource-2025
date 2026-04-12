import numpy as np
import tensorrt as trt
import torch

from .config import INPUT_HEIGHT, INPUT_WIDTH


def trt_dtype_to_torch(dtype):
    np_dtype = trt.nptype(dtype)
    return torch.from_numpy(np.empty((), dtype=np_dtype)).dtype


class TrtModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

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
        if input_data_np.dtype != np.float32:
            input_data_np = input_data_np.astype(np.float32)
        if not input_data_np.flags["C_CONTIGUOUS"]:
            input_data_np = np.ascontiguousarray(input_data_np)

        self.input_tensor.copy_(torch.from_numpy(input_data_np), non_blocking=True)
        ok = self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed (execute_async_v3 returned False).")

        return self.output_tensors
