cmake_minimum_required(VERSION 3.18)
project(check_buf LANGUAGES CXX CUDA)

# Find JAX's XLA FFI headers
execute_process(
    COMMAND python3 -c "import jaxlib; print(jaxlib.__path__[0])"
    OUTPUT_VARIABLE JAXLIB_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(check_buf SHARED check_buf.cu)
set(JAXLIB_INC "" CACHE STRING "Path to jaxlib include dir (containing xla/ffi/api/ffi.h)")
target_include_directories(check_buf PRIVATE ${JAXLIB_INC})
set_target_properties(check_buf PROPERTIES
    CUDA_ARCHITECTURES "75"
    PREFIX "lib"
    SUFFIX ".so"
)
