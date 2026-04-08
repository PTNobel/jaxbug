// Minimal FFI handler: validates a float64 constant, does cudaMalloc.
// The cudaMalloc corrupts XLA's constant buffers inside jax.lax.scan.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

static bool g_allocated = false;
static double* g_workspace = nullptr;

ffi::Error Impl(
    cudaStream_t stream, int64_t n, int64_t use_async,
    ffi::Buffer<ffi::F64> const_f64,
    ffi::Buffer<ffi::F64> q, ffi::Buffer<ffi::F64> b,
    ffi::ResultBuffer<ffi::F64> x_out, ffi::ResultBuffer<ffi::F64> info_out
) {
    int64_t batch = q.dimensions().size() == 2 ? q.dimensions()[0] : 1;

    // Validate the closure-captured constant
    if (const_f64.element_count() > 0) {
        double v;
        cudaMemcpy(&v, const_f64.typed_data(), sizeof(double), cudaMemcpyDeviceToHost);
        if (v <= 0.0 || v >= 1.0)
            return ffi::Error::InvalidArgument("const_f64 corrupted");
    }

    // Allocate GPU memory once (THIS CORRUPTS XLA'S CONSTANT BUFFERS)
    if (!g_allocated) {
        size_t bytes = sizeof(double) * n * 16;
        if (use_async) cudaMallocAsync(&g_workspace, bytes, stream);
        else           cudaMalloc(&g_workspace, bytes);
        g_allocated = true;
    }

    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(x_out->typed_data(), q.typed_data(),
                    sizeof(double)*n*batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(info_out->typed_data(), 0, sizeof(double)*batch, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CheckBuf, Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("n").Attr<int64_t>("use_async")
        .Arg<ffi::Buffer<ffi::F64>>()   // const_f64
        .Arg<ffi::Buffer<ffi::F64>>()   // q
        .Arg<ffi::Buffer<ffi::F64>>()   // b
        .Ret<ffi::Buffer<ffi::F64>>()   // x_out
        .Ret<ffi::Buffer<ffi::F64>>()   // info_out
);
