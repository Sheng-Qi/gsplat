# gsplat C++ Bridge for gsplat_viewer

This directory contains the bridge that exposes `gsplat/cuda/csrc` kernels via a pure raw-pointer/struct C++ interface. 
It uses LibTorch internally to wrap the pointers into `at::Tensor` and call the original `gsplat` kernels, avoiding the need for `Python` runtime and preventing `Torch` dependency from leaking into the main `gsplat_viewer` application logic.
