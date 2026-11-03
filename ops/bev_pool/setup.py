"""Install BEV pooling as the bevfusion-bev-pool-ops pip package.

中文：pip install --no-build-isolation --no-deps ./ops/bev_pool
Build only / 仅原地编译：python ops/bev_pool/setup.py build_ext --inplace
"""
import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.errors import SetupError
from torch.utils.cpp_extension import BuildExtension

# Some Windows hosts expose an invalid OEM code page to Python. Compiler/CUDA
# version checks need only ASCII digits; preserve diagnostics with replacement.
if os.name == "nt":
    from torch.utils import cpp_extension
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

ROOT = Path(__file__).resolve().parent
SOURCES = ["src/bev_pool_cpu.cpp", "src/bev_pool_cuda.cu"]
EXTENSION = "ops.bev_pool.bev_pool_ext"


class CUDABuildExtension(BuildExtension):
    def run(self):
        if not getattr(self, "_cuda_configured", False):
            import torch
            from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
            if torch.version.cuda is None:
                raise SetupError("CUDA-enabled PyTorch is required / 请使用支持 CUDA 的 PyTorch")
            if CUDA_HOME is None:
                raise SetupError("CUDA Toolkit not found. Install nvcc and set CUDA_HOME/CUDA_PATH "
                                 "/ 未找到 CUDA Toolkit，需安装 nvcc 并设置 CUDA_HOME/CUDA_PATH")
            nvcc = Path(CUDA_HOME) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
            if not nvcc.is_file():
                raise SetupError(f"CUDA compiler not found / 未找到 CUDA 编译器: {nvcc}")
            cuda_extension = CUDAExtension(
                EXTENSION,
                sources=SOURCES,
                define_macros=[("WITH_CUDA", None)],
                extra_compile_args={
                    "cxx": ["/O2", "/std:c++20"] if os.name == "nt" else ["-O3", "-std=c++20"],
                    "nvcc": ["-O3", "-std=c++20", "-D__CUDA_NO_HALF_OPERATORS__",
                             "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"],
                },
            )
            # Retain setuptools' finalized bookkeeping on the same object.
            self.extensions[0].__dict__.update(cuda_extension.__dict__)
            # Honor TORCH_CUDA_ARCH_LIST; otherwise let PyTorch detect the GPU.
            self._cuda_configured = True
        super().run()


if __name__ == "__main__":
    os.chdir(ROOT)
    setup(
        name="bevfusion-bev-pool-ops",
        version="0.1.0",
        description="BEVFusion CUDA segmented BEV pooling operator",
        python_requires=">=3.8",
        install_requires=["torch"],
        packages=["ops.bev_pool"],
        package_dir={"ops.bev_pool": "."},
        package_data={"ops.bev_pool": ["LICENSE", "src/*.cpp", "src/*.cu"]},
        license="Apache-2.0",
        license_files=["LICENSE"],
        ext_modules=[Extension(EXTENSION, sources=SOURCES)],
        cmdclass={"build_ext": CUDABuildExtension},
        zip_safe=False,
    )
