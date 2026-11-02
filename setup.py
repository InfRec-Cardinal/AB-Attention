"""Build the paper kernels with PyTorch's extension tooling."""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = "csrc"
os.environ.setdefault("MAX_JOBS", "4")
flags = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3", "-gencode=arch=compute_90a,code=sm_90a",
        "--expt-extended-lambda", "--expt-relaxed-constexpr", "--use_fast_math",
    ],
}

setup(
    ext_modules=[
        CUDAExtension(
            "ab_attention._C",
            sources=[f"{CSRC}/{name}" for name in (
                "bindings.cpp", "ibcs.cu", "sip.cu",
            )],
            include_dirs=[str(ROOT / CSRC)],
            extra_compile_args=flags,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
