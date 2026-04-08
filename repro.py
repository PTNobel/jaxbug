"""
cudaMalloc inside an XLA FFI handler corrupts closure-captured float64
constant buffers in jax.lax.scan. Requires custom_vmap + custom_vjp and
two distinct FFI-backed solvers in the same JIT scope.

Build:  cd bug_repro/build && cmake .. -DJAXLIB_INC=$(python -c \
        "import jaxlib; print(jaxlib.__path__[0]+'/include')") && make
"""
import os, ctypes, jax, jax.numpy as jnp, numpy as np
from jax import custom_vjp
from jax.custom_batching import custom_vmap
jax.config.update("jax_enable_x64", True)

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "build", "libcheck_buf.so"))
jax.ffi.register_ffi_target("check_buf", jax.ffi.pycapsule(lib.CheckBuf), platform="CUDA")

N = 434

def make_solver(const_val, n_out):
    """Each call creates a NEW closure-captured constant + wrapped FFI."""
    const = jax.device_put(jnp.full(N, const_val, dtype=jnp.float64))

    def raw(q, b):
        bs = q.shape[0]
        return jax.ffi.ffi_call("check_buf",
            (jax.ShapeDtypeStruct((bs, n_out), jnp.float64),
             jax.ShapeDtypeStruct((bs,), jnp.float64))
        )(const, q, b, n=np.int64(n_out), use_async=np.int64(0))

    @custom_vmap
    def vmapped(q, b):
        x, i = raw(q[None], b[None]); return x[0], i[0]
    @vmapped.def_vmap
    def _(sz, bat, q, b):
        eb = lambda a, b: a if b else jnp.broadcast_to(a, (sz,)+a.shape)
        return raw(eb(q, bat[0]), eb(b, bat[1])), (True, True)

    @custom_vjp
    def solve(q, b): return vmapped(q, b)
    def fwd(q, b): return vmapped(q, b), (q, b)
    def bwd(res, g): return (jnp.zeros_like(res[0]), jnp.zeros_like(res[1]))
    solve.defvjp(fwd, bwd)
    return solve

# Two solvers: different constants, different output sizes (like first_layer vs iter_layer)
solve_a = make_solver(0.5, 2 * N + 1)   # no power cones equivalent
solve_b = make_solver(0.667, 4 * N + 1)  # power cones equivalent — THIS ONE CORRUPTS

rng = np.random.default_rng(42)
xs = rng.standard_normal((252, N, 15))
ys = rng.standard_normal((252, N))

@jax.jit
def run(xs, ys):
    # First call uses solve_a (outside scan)
    w = solve_a(jnp.concat([xs[0,:,0], jnp.zeros(N+1)]), jnp.ones(2*N+1))[0][:N]

    def body(w, data):
        x, y = data
        w = w * 0.99 + 0.01 * y
        q = jnp.concat([w, jnp.sqrt(jnp.sum(x**2, axis=1)), jnp.zeros(2*N+1)])
        # Scan uses solve_b (different solver, different captured constant)
        w_new = solve_b(q, jnp.ones(4*N+1))[0][:N]
        return w_new, jnp.sum(w_new)

    _, sums = jax.lax.scan(body, w, (jnp.array(xs[1:]), jnp.array(ys[1:])))
    return sums

try:
    r = run(xs, ys); jax.block_until_ready(r)
    print("PASS")
except Exception as e:
    print("BUG" if "corrupted" in str(e) else f"FAIL: {str(e)[:80]}")
