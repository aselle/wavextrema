# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

backend = "numpy"

# backend="jax"
# backend="cupy"


# default jit decorator is nothing stuff
def jit(x):
    return x


if backend == "numpy":
    import numpy
    np = numpy
if backend == "tf":
    import tensorflow as tf
    np = tf.experimental.numpy
if backend == "jax":
    import jax
    np = jax.numpy
    jit = jax.jit
if backend == "cupy":
    import cupy
    np = cupy


# Work around lack of mutable operations in TF and JAX
class Index:
    def __getitem__(self, index):
        return index


def CopyOrMutate(array, index, value):
    """
        Backend agnostic way of doing sliced updates.

        e.g.
            a = np.array([1,2,3])
            backend.CopyOrMutate(a, backend.index[2:], 10)
    """
    if backend == "jax":
        return jax.ops.index_update(array, index, value)
    else:
        array[index] = value
        return array


index = Index()
