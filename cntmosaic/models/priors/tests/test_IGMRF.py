import pytest
from collections import namedtuple
import inspect

import numpy as np
from numpy.testing import assert_allclose

import jax
import jax.numpy as jnp
from jax import random, vmap

from numpyro import distributions as dist
from numpyro.distributions.batch_util import vmap_over

from .._IGMRF import IGMRF, diff_matrix, laplacian

# language: python
    
def _identity(x):
  return x
    
class T(namedtuple("TestCase", ["jax_dist", "params"])):
  def __new__(cls, jax_dist, *params):
    return super(cls, T).__new__(cls, jax_dist, params)
  
def _allclose_or_equal(a1, a2):
  if isinstance(a1, np.ndarray):
      return np.allclose(a2, a1)
  elif isinstance(a1, jnp.ndarray):
      return jnp.allclose(a2, a1)
  else:
      return a2 == a1 or a2 is a1

def _tree_equal(t1, t2):
  t = jax.tree.map(_allclose_or_equal, t1, t2)
  return jnp.all(jax.flatten_util.ravel_pytree(t)[0])
 
def _get_vmappable_dist_init_params(jax_dist):
  if jax_dist.__name__ == ("_TruncatedCauchy"):
    return [2, 3]
  elif jax_dist.__name__ == ("_TruncatedNormal"):
    return [2, 3]
  elif issubclass(jax_dist, dist.Distribution):
    init_parameters = list(inspect.signature(jax_dist.__init__).parameters.keys())[
      1:
    ]
    vmap_over_parameters = list(
      inspect.signature(vmap_over.dispatch(jax_dist)).parameters.keys()
    )[1:]
    return list(
      [
        i
        for i, name in enumerate(init_parameters)
        if name in vmap_over_parameters
      ]
    )
  else:
    raise ValueError
  
CASES = [
  T(IGMRF, 5, 2),
  T(IGMRF, 6, 1, 2.0),
  T(IGMRF, 4, 3, 1.0, jnp.array([1.0, 2.0, 3.0, 4.0])),
]

@pytest.mark.parametrize("jax_dist_cls, params", CASES)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
def test_dist_shape(jax_dist_cls, params, prepend_shape):
    jax_dist = jax_dist_cls(*params)
    rng_key = random.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape + jax_dist.event_shape
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)
    
    assert jnp.shape(samples) == expected_shape

@pytest.mark.parametrize("jax_dist, params", CASES)
def test_infer_shapes(jax_dist, params):
    shapes = []
    for param in params:
        if param is None:
            shapes.append(None)
            continue
        shape = getattr(param, "shape", ())
        if callable(shape):
            shape = shape()
        shapes.append(shape)
    jax_dist = jax_dist(*params)
    try:
        expected_batch_shape, expected_event_shape = type(jax_dist).infer_shapes(
            *shapes
        )
    except NotImplementedError:
        pytest.skip(f"{type(jax_dist).__name__}.infer_shapes() is not implemented")
    assert jax_dist.batch_shape == expected_batch_shape
    assert jax_dist.event_shape == expected_event_shape
    
@pytest.mark.parametrize("jax_dist, params", CASES)
def test_sample_gradient(jax_dist, params):
    dist_args = [
      p
      for p in (
        inspect.getfullargspec(jax_dist.__init__)[0][1:]
        if inspect.isclass(jax_dist)
        # account the the case jax_dist is a function
        else inspect.getfullargspec(jax_dist)[0]
      )
    ]
    params_dict = dict(zip(dist_args[: len(params)], params))

    jax_class = type(jax_dist(**params_dict))
    reparametrized_params = [
      p for p in jax_class.reparametrized_params
    ]

    nonrepara_params_dict = {
        k: v for k, v in params_dict.items() if k not in reparametrized_params
    }
    repara_params = tuple(
        v for k, v in params_dict.items() if k in reparametrized_params
    )

    rng_key = random.PRNGKey(0)

    def fn(args):
        args_dict = dict(zip(reparametrized_params, args))
        return jnp.sum(
            jax_dist(**args_dict, **nonrepara_params_dict).sample(key=rng_key)
        )

    actual_grad = jax.grad(fn)(repara_params)
    assert len(actual_grad) == len(repara_params)

    eps = 1e-3
    for i in range(len(repara_params)):
        if repara_params[i] is None:
            continue
        args_lhs = [p if j != i else p - eps for j, p in enumerate(repara_params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(repara_params)]
        fn_lhs = fn(args_lhs)
        fn_rhs = fn(args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2.0 * eps)
        assert jnp.shape(actual_grad[i]) == jnp.shape(repara_params[i])
        assert_allclose(jnp.sum(actual_grad[i]), expected_grad, rtol=0.02, atol=0.03)

@pytest.mark.parametrize("jax_dist, params", CASES)
def test_jit_log_likelihood(jax_dist, params):
  # if jax_dist.__name__ in (
  #   "IGMRF",
  # ):
  #   pytest.xfail(reason="non-jittable params")
    
  rng_key = random.PRNGKey(0)
  samples = jax_dist(*params).sample(key=rng_key, sample_shape=(2, 3))

  def log_likelihood(*params):
    return jax_dist(*params).log_prob(samples)

  expected = log_likelihood(*params)
  # Use static_argnums=(0,1) to treat num_nodes and order as static arguments
  actual = jax.jit(log_likelihood, static_argnums=(0, 1))(*params)
  assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

@pytest.mark.parametrize("jax_dist, params", CASES)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("jit", [False, True])
def test_log_prob(jax_dist, params, prepend_shape, jit):
    jax_dist = jax_dist(*params)

    rng_key = random.PRNGKey(0)
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)
    assert jax_dist.log_prob(samples).shape == prepend_shape + jax_dist.batch_shape
    
@pytest.mark.parametrize("jax_dist, params", CASES)
def test_dist_pytree(jax_dist, params):
  def f(x):
    return jax_dist(*params)
  jax.jit(f)(0)  # this test for flatten/unflatten
  jax.lax.map(f, np.ones(3))  # this test for compatibility w.r.t. scan
  
  # Test that parameters do not change after flattening.
  expected_dist = f(0)
  actual_dist = jax.jit(f)(0)
  
  for name in expected_dist.arg_constraints:
      expected_arg = getattr(expected_dist, name)
      actual_arg = getattr(actual_dist, name)
      assert actual_arg is not None, f"arg {name} is None"
      if np.issubdtype(np.asarray(expected_arg).dtype, np.number):
          assert_allclose(actual_arg, expected_arg, atol=1e-7)
      else:
          assert (
              actual_arg.shape == expected_arg.shape
              and actual_arg.dtype == expected_arg.dtype
          )
          
  expected_sample = expected_dist.sample(random.PRNGKey(0))
  actual_sample = actual_dist.sample(random.PRNGKey(0))
  
  expected_log_prob = expected_dist.log_prob(expected_sample)
  actual_log_prob = actual_dist.log_prob(actual_sample)
  
  assert_allclose(actual_sample, expected_sample, rtol=3e-6)
  assert_allclose(actual_log_prob, expected_log_prob, rtol=3e-5)

@pytest.mark.parametrize("jax_dist, params", CASES)
@pytest.mark.parametrize("prepend_shape", [(), (2, 3)])
@pytest.mark.parametrize("sample_shape", [(), (4,)])
def test_expand(jax_dist, params, prepend_shape, sample_shape):
  jax_dist = jax_dist(*params)
  new_batch_shape = prepend_shape + jax_dist.batch_shape
  expanded_dist = jax_dist.expand(new_batch_shape)
  rng_key = random.PRNGKey(0)
  samples = expanded_dist.sample(rng_key, sample_shape)
  assert expanded_dist.batch_shape == new_batch_shape
  assert jnp.shape(samples) == sample_shape + new_batch_shape + jax_dist.event_shape
  assert expanded_dist.log_prob(samples).shape == sample_shape + new_batch_shape
  # test expand of expand
  assert (
    expanded_dist.expand((3,) + new_batch_shape).batch_shape
    == (3,) + new_batch_shape
  )
  # test expand error
  if prepend_shape:
    with pytest.raises(ValueError, match="Cannot broadcast distribution of shape"):
      assert expanded_dist.expand((3,) + jax_dist.batch_shape)
          
@pytest.mark.parametrize("jax_dist, params", CASES)
def test_vmap_dist(jax_dist, params):
    param_names = list(inspect.signature(jax_dist).parameters.keys())
    vmappable_param_idxs = _get_vmappable_dist_init_params(jax_dist)
    vmappable_param_idxs = vmappable_param_idxs[: len(params)]

    if len(vmappable_param_idxs) == 0:
      return

    def make_jax_dist(*params):
      return jax_dist(*params)

    def sample(d: dist.Distribution):
      return d.sample(random.PRNGKey(0))

    d = make_jax_dist(*params)

    in_out_axes_cases = [
        # vmap over all args
        (
          tuple(0 if i in vmappable_param_idxs else None for i in range(len(params))),
          0,
        ),
        # vmap over a single arg, out over all attributes of a distribution
        *(
          ([0 if i == idx else None for i in range(len(params))], 0)
          for idx in vmappable_param_idxs
          if params[idx] is not None
        ),
        # vmap over a single arg, out over the associated attribute of the distribution
        *(
            (
              [0 if i == idx else None for i in range(len(params))],
              vmap_over(d, **{param_names[idx]: 0}),
            )
            for idx in vmappable_param_idxs
            if params[idx] is not None
        ),
        # vmap over a single arg, axis=1, (out single attribute, axis=1)
        *(
            (
              [1 if i == idx else None for i in range(len(params))],
              vmap_over(d, **{param_names[idx]: 1}),
            )
            for idx in vmappable_param_idxs
            if isinstance(params[idx], jnp.ndarray)
            and jnp.array(params[idx]).ndim > 0
        ),
    ]

    for in_axes, out_axes in in_out_axes_cases:
      batched_params = [
          (
              jax.jax.tree.map(lambda x: jnp.expand_dims(x, ax), arg)
              if isinstance(ax, int)
              else arg
          )
          for arg, ax in zip(params, in_axes)
      ]
      # Recreate the jax_dist to avoid side effects coming from `d.sample`
      # triggering lazy_property computations, which, in a few cases, break
      # vmap_over's expectations regarding existing attributes to be vmapped.
      d = make_jax_dist(*params)
      batched_d = jax.vmap(make_jax_dist, in_axes=in_axes, out_axes=out_axes)(
          *batched_params
      )
      eq = vmap(lambda x, y: _tree_equal(x, y), in_axes=(out_axes, None))(
          batched_d, d
      )
      assert eq == jnp.array([True])

      samples_dist = sample(d)
      samples_batched_dist = jax.vmap(sample, in_axes=(out_axes,))(batched_d)
      assert samples_batched_dist.shape == (1, *samples_dist.shape)