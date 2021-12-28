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
import time
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.collections import CircleCollection
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import quadpy
import unittest

import backend
from backend import np as np

# Constants for type arguments. TODO(aselle): enum.
INTERIOR = "interior"
EXTERIOR = "exterior"
# Constants for method arguments. TODO(aselle): enum.
BARTON_MILLER = "barton_miller"
DUAL_SURFACE = "dual_surface"
# Quadrature order.
QUADRATURE_ORDER = 7  # 21
# Number of matrix rows to do in each vectorized call.
BATCH_SIZE = 100

# Setup quadrature.
# 4, 17, 21 are examples that work here
quad_scheme = quadpy.t2.get_good_scheme(QUADRATURE_ORDER)
# print(quad_scheme)
quad_alpha = np.array(quad_scheme.points[0, :])
quad_beta = np.array(quad_scheme.points[1, :])
quad_weights = np.array(quad_scheme.weights)
# Make sure the quadrature scheme does not have the triangle center!
if np.count_nonzero(
        np.square(quad_alpha - .3333) +
        np.square(quad_beta - .3333) < 1e-3) != 0:
    raise ValueError(
        "Quadrature scheme has triangle center, this causes singularities!")


def WaveNumber(frequency, speed_of_sound):
    """Compute a wave number given frequency and speed of sound."""
    return 2 * np.pi * frequency / speed_of_sound


def norm(x, axis):
    """Workaround limitations on linalg.norm() in numba."""
    return np.sqrt(np.sum(np.square(x), axis=axis))


def geom_v(x1, x2, x3):
    """Compute normal and area of triangle element.

    Args:
        x1: Point 1 of triangle. (e, 3)
        x2: Point 2 of triangle. (e, 3)
        x3: Point 3 of triangle. (e, 3)

    Returns:
        n: normalized normal of triangle (e, 3)
        A: area of triangle (e,)
    """
    u = x2 - x1
    v = x3 - x1
    n = np.cross(u, v)
    l = norm(n, axis=-1)
    n /= l[:, None]
    A = l * .5
    return n, A


def get_r_v(x1, x2, x3, p):
    """
    Computes r vector from triangle (x1, x2, x3) to point p.
        x1 in (e,3)
        x2 in (e,3)
        x3 in (e,3)
        p in (p,3)
        alpha in (q,)
        beta in (q,)
    Returns:
        r, rl where r is radius and rl is its length
    """
    a = quad_alpha[None, None, :, None]
    b = quad_beta[None, None, :, None]
    q = a * x1[None, :, None, :] + b * x2[None, :, None, :] + (
        1 - a - b) * x3[None, :, None, :]
    r = p[:, None, None, :] - q
    rl = norm(r, axis=-1)
    return r, rl


def get_angles(x1, x2, x3, p):
    """Computes angles needed for hypersingular integration of a triangle
    defined by x1,x2,x3 with p inside it.
    """
    x = np.array([
        p[:, None, :] - x1[None, :, :], p[:, None, :] - x2[None, :, :],
        p[:, None, :] - x3[None, :, :]
    ])
    #print("x",x.shape)
    outeredge = np.array([x1 - x2, x2 - x3, x3 - x1])
    outerdist = norm(outeredge, axis=-1)[:, None, :]
    d = norm(x, axis=-1)
    #print("d.shape", d.shape, "outer.shape", outerdist.shape)
    doff = np.roll(d, shift=-1, axis=0)
    r0 = np.where(d < doff, doff, d)
    ra = np.where(d >= doff, doff, d)
    # TODO(rmlarsen): catastrophic cancel
    A = np.arccos((r0 * r0 + ra * ra - outerdist * outerdist) / (2 * r0 * ra))
    B = np.arctan(ra * np.sin(A) / (r0 - ra * np.cos(A)))
    #print("d", d[:,1,0],"r0",r0[:,1,0], "ra", ra[:,1,0],"A",A,"B",B)
    return r0, A, B


class Timer(object):
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print("%30s %f s" % (self.label, time.time() - self.start))


@backend.jit
def ComputeIntegralMatrices(x1, x2, x3, p, n_p, k, p_on_q):
    """
    """
    n, area = geom_v(x1, x2, x3)  # (e,3), (e)
    rvec, rl = get_r_v(x1, x2, x3, p)  # (p,e,q,3), (p,e,q)

    g_0 = 1 / (4 * np.pi * rl)  # (p,e,q) eq(3.8)
    g_k = g_0 * np.exp(1.j * k * rl)  # (p,e,q,3) eq(3.6)
    g_k = np.array(g_0, np.complex128) * np.exp(
        1.j * np.array(k * rl, np.complex128))  # (p,e,q,3) eq(3.6)
    terms = area[None, :, None] * quad_weights[None, None, :] * (g_k)
    Lk_ = np.sum(terms, axis=2)
    exp = np.exp(1.j * k * rl)
    dr_dnq = -np.einsum("jl,ijkl->ijk", n, rvec) / rl  # shape (p,e,q)
    dGk_dr = 1. / (4 * np.pi * rl * rl) * exp * (1.j * k * rl - 1
                                                 )  # shape (p,e,q)
    terms = area[None, :, None] * quad_weights[None,
                                               None, :] * (dGk_dr * dr_dnq)
    Mk_ = np.sum(terms, axis=2)
    return Lk_, Mk_, None, None


@backend.jit
def ComputeIntegralMatricesExtended(x1, x2, x3, p, n_p, k, p_on_q):
    """
    Compute matrices required for Barton-Miller (including hypersingular Lk and Nk forms).

    Returns:
        Lk, Mk, Mkt, Nk
    """
    # Lookup geometry
    n, area = geom_v(x1, x2, x3)  # (e,3), (e)
    rvec, rl = get_r_v(x1, x2, x3, p)  # (p,e,q,3), (p,e,q)
    r0, A, B = get_angles(x1, x2, x3, p)
    # Compute Lk
    L0e = np.sum(1. / (4. * np.pi) * r0 * np.sin(B) *
                 (np.log(np.tan((B + A) / 2)) - np.log(np.tan(B / 2))),
                 axis=0)  #p,e
    L0e = np.where(p_on_q, L0e, 0.)  #p,e
    g_0 = 1 / (4 * np.pi * rl)  # (p,e,q) eq(3.8)
    g_k = g_0 * np.exp(1.j * k * rl)  # (p,e,q,3) eq(3.6)
    g_k = np.array(g_0, np.complex128) * np.exp(
        1.j * np.array(k * rl, np.complex128))  # (p,e,q,3) eq(3.6)
    terms = area[None, :, None] * quad_weights[None, None, :] * (
        g_k - np.where(p_on_q[:, :, None], g_0, 0.))
    Lk_ = np.sum(terms, axis=2) + L0e
    # Compute Mk
    exp = np.exp(1.j * k * rl)
    dr_dnq = -np.einsum("jl,ijkl->ijk", n, rvec) / rl  # shape (p,e,q)
    dGk_dr = 1. / (4 * np.pi * rl * rl) * exp * (1.j * k * rl - 1
                                                 )  # shape (p,e,q)
    terms = area[None, :, None] * quad_weights[None,
                                               None, :] * (dGk_dr * dr_dnq)
    Mk_ = np.sum(terms, axis=2)
    # Compute Mkt
    dr_dup = np.einsum("il,ijkl->ijk", n_p, rvec) / rl  # (p,e,q)
    terms = area[None, :, None] * quad_weights[None,
                                               None, :] * (dGk_dr * dr_dup)
    Mkt_ = np.sum(terms, axis=-1)
    # Compute Nk
    d2r_dupdnq = -1 / rl * (np.einsum("jl,il->ij", n, n_p)[:, :, None] +
                            dr_dup * dr_dnq)
    d2Gk_dr2 = 1. / (4 * np.pi * rl * rl * rl) * exp * (2 - 2.j * k * rl -
                                                        k * k * rl * rl)
    dG0_dr = np.where(p_on_q[:, :, None], -1. / (4 * np.pi * rl * rl), 0.)
    d2G0_dr2 = np.where(p_on_q[:, :, None], 1. / (2 * np.pi * rl * rl * rl),
                        0.)
    g_0 = 1. / (4 * np.pi * rl)
    G0_term = np.where(p_on_q[:, :, None], .5 * k * k * g_0, 0.)
    d2Gk_dupdnq = (dGk_dr - dG0_dr) * d2r_dupdnq + (d2Gk_dr2 -
                                                    d2G0_dr2) * dr_dup * dr_dnq
    N0e = np.sum(1. / (4 * np.pi) * (np.cos(B + A) - np.cos(B)) /
                 (r0 * np.sin(B)),
                 axis=0)  #p,e
    N0e = np.where(p_on_q, N0e, 0.)  #p,e
    terms = area[None, :,
                 None] * quad_weights[None, None, :] * (d2Gk_dupdnq + G0_term)
    Nk_ = np.sum(terms, axis=-1) + N0e - .5 * k * k * L0e

    return Lk_, Mk_, Mkt_, Nk_


@backend.jit
def ComputeIntegralMatricesForEval(x1, x2, x3, p, n_p, k, p_on_q):
    """
    Evaluate forms used for evaluation. 

    TODO(aselle): This can probably be removed 
    """
    n, area = geom_v(x1, x2, x3)  # (e,3), (e)
    rvec, rl = get_r_v(x1, x2, x3, p)  # (p,e,q,3), (p,e,q)
    r0, A, B = get_angles(x1, x2, x3, p)
    # Compute Lk (and compute Hypersingular fix)
    L0e = np.sum(1. / (4. * np.pi) * r0 * np.sin(B) *
                 (np.log(np.tan((B + A) / 2)) - np.log(np.tan(B / 2))),
                 axis=0)  #p,e
    L0e = np.where(p_on_q, L0e, 0.)  #p,e
    g_0 = 1 / (4 * np.pi * rl)  # (p,e,q) eq(3.8)
    g_k = g_0 * np.exp(1.j * k * rl)  # (p,e,q,3) eq(3.6)
    g_k = np.array(g_0, np.complex128) * np.exp(
        1.j * np.array(k * rl, np.complex128))  # (p,e,q,3) eq(3.6)
    terms = area[None, :, None] * quad_weights[None, None, :] * (
        g_k - np.where(p_on_q[:, :, None], g_0, 0.))
    Lk_ = np.sum(terms, axis=2) + L0e
    # Compute Mk
    dr_dnq = -np.einsum("jl,ijkl->ijk", n, rvec) / rl  # shape (p,e,q)
    dGk_dr = 1. / (4 * np.pi * rl * rl) * np.exp(1.j * k * rl) * (
        1.j * k * rl - 1)  # shape (p,e,q)
    terms = area[None, :, None] * quad_weights[None,
                                               None, :] * (dGk_dr * dr_dnq)
    Mk_ = np.sum(terms, axis=2)

    return Lk_, Mk_


def BuildSystem_v(k, elems_x1, elems_x2, elems_x3, type, method):
    """equation 4.17 & 4.18"""
    rows = elems_x1.shape[0]
    cols = elems_x1.shape[0]
    p = (1. / 3.) * (elems_x1 + elems_x2 + elems_x3)
    n_p, l_elem = geom_v(elems_x1, elems_x2, elems_x3)
    p_on_q = np.eye(rows, dtype=np.dtype('bool'))

    batch = BATCH_SIZE
    if method == BARTON_MILLER:
        Lk_ = np.eye(rows, dtype=np.complex128)
        Mk_ = np.eye(rows, dtype=np.complex128)
        Mkt_ = np.eye(rows, dtype=np.complex128)
        Nk_ = np.eye(rows, dtype=np.complex128)
        for i in range(0, rows, batch):
            print(i, )
            Lk_batch, Mk_batch, Mkt_batch, Nk_batch = ComputeIntegralMatricesExtended(
                elems_x1, elems_x2, elems_x3, p[i:i + batch], n_p[i:i + batch],
                k, p_on_q[i:i + batch])
            Lk_ = backend.CopyOrMutate(Lk_, backend.index[i:i + batch],
                                       Lk_batch)
            Mk_ = backend.CopyOrMutate(Mk_, backend.index[i:i + batch],
                                       Mk_batch)
            Nkt_ = backend.CopyOrMutate(Mkt_, backend.index[i:i + batch],
                                        Mkt_batch)
            Nk_ = backend.CopyOrMutate(Nk_, backend.index[i:i + batch],
                                       Nk_batch)
        # Burton-Miller
        # TODO mu setting may not be right. see paper on burton miller not sure it should
        # be same for both
        if type == INTERIOR:
            mu = 1.j / (k + 1)
            A = Mk_ + mu * Nk_ + np.eye(rows) * .5
            B = Lk_ + mu * (Mkt_) - np.eye(rows) * .5 * mu
        elif type == EXTERIOR:
            mu = 1.j / (k + 1)
            A = Mk_ + mu * Nk_ - np.eye(rows) * .5
            B = Lk_ + mu * (Mkt_) + np.eye(rows) * .5 * mu
        else:
            assert False
        return A, B
    elif method == DUAL_SURFACE:
        # delta = .5 * np.pi / k  #  k delta  < pi ==>  delta < pi / k
        # This is what the paper uses for one of its examplesdelta = 2.1 / k
        delta = .5 / k
        alpha = -1.j  # purely iamginary weighting

        # TODO(aselle): obviously this surface isn't water tight anymore
        # or even remotely the right sized elements, but it probably doesn't matter
        # according to lit.
        # I first thought you would apply the operator to a different surface,
        # but it seems you only use the points of the field on the surface.
        # accdording to Mohsen description of eq 8.
        #elems2_x1 = elems_x1 - delta * n_p
        #elems2_x2 = elems_x2 - delta * n_p
        #elems2_x3 = elems_x3 - delta * n_p
        p2 = p - delta * n_p

        Lk = np.eye(rows, dtype=np.complex128)
        Mk = np.eye(rows, dtype=np.complex128)
        Lk_bar = np.eye(rows, dtype=np.complex128)
        Mk_bar = np.eye(rows, dtype=np.complex128)
        for i in range(0, rows, batch):
            print(i)
            Lk_batch, Mk_batch, Mkt_batch, Nk_batch = ComputeIntegralMatrices(
                elems_x1, elems_x2, elems_x3, p[i:i + batch], n_p[i:i + batch],
                k, p_on_q[i:i + batch])
            Lk2_batch, Mk2_batch, Mkt_batch, Nk_batch = ComputeIntegralMatrices(
                elems_x1, elems_x2, elems_x3, p2[i:i + batch],
                n_p[i:i + batch], k, p_on_q[i:i + batch])
            Lk = backend.CopyOrMutate(Lk, backend.index[i:i + batch], Lk_batch)
            Mk = backend.CopyOrMutate(Mk, backend.index[i:i + batch], Mk_batch)
            Lk_bar = backend.CopyOrMutate(Lk_bar, backend.index[i:i + batch],
                                          Lk2_batch)
            Mk_bar = backend.CopyOrMutate(Mk_bar, backend.index[i:i + batch],
                                          Mk2_batch)
        if type == INTERIOR:
            raise RuntimeErrror(
                "Dual Surface Interior mode is not implemented yet.")
        elif type == EXTERIOR:
            A = Mk + alpha * Mk_bar - np.eye(rows) * .5
            B = Lk + alpha * Lk_bar
            return A, B
    else:
        raise ValueError(f"Invalid solution method '{method}'")


def SolveHelmholtz_v(k,
                     elems_x1,
                     elems_x2,
                     elems_x3,
                     phi_w,
                     phi,
                     v_w,
                     v,
                     type,
                     method=DUAL_SURFACE):
    """Solves a Helmholtz Accoustic problem on boundary defined by 
    triangle mesh subject to boundary conditions.

    Args:
        k: wave number.
        elems_x1: point 1 of each triangle element (elements, 3)
        elems_x2: point 2 of each triangle element (elements, 3)
        elems_x3: point 3 of each triangle element (elements, 3)
        phi_w: weight of dirichlet (velocity potential) (elements,)
        phi: velocity potential on each element center. (elements,)
        v_w: weight of neumann (normal velocity) (elements,)
        v: velocity on boundary. (elements,)
        type: bem_3d.INTERIOR for interior problems, bem_3d.EXTERIOR for exterior problems.
        method: bem_3d.DUAL_SURFACE or bem_3dBARTON_MILLER 

    Returns:
        phi, v
    """
    if type not in [INTERIOR, EXTERIOR]:
        raise ValueError(
            "Invalid type of solve INTERIOR and EXTERIOR are only allowed.")
    if method not in [DUAL_SURFACE, BARTON_MILLER]:
        raise ValueError(
            "Invalid method of solve DUAL_SURFACE and BARTON_MILLER are only allowed."
        )
    with Timer("build system"):
        A, B = BuildSystem_v(k,
                             elems_x1,
                             elems_x2,
                             elems_x3,
                             type=type,
                             method=method)
        C = np.concatenate([
            np.concatenate([A, -B], axis=1),
            np.concatenate([np.diag(phi_w), np.diag(v_w)], axis=1)
        ],
                           axis=0)
        N = elems_x1.shape[0]
        F = np.zeros(N * 2, dtype=np.complex128)
        F = backend.CopyOrMutate(F, backend.index[N:], phi_w * phi + v_w * v)
    with Timer("linear solve"):
        z = np.linalg.solve(C, F)
        phi, v = z[:N], z[N:]
    return phi, v


def EvaluatePosition_v(k, elems_x1, elems_x2, elems_x3, phi, v, p, type):
    """
    Evaluates points in solution domain  equation 4.19. (corrected, swapping phi and v).
    
    Args:
        k: wave number.
        elems_x1: point 1 of each triangle element (elements, 3)
        elems_x2: point 2 of each triangle element (elements, 3)
        elems_x3: point 3 of each triangle element (elements, 3)
        phi: velocity potential on each element center. (elements,)
        v: velocity on boundary. (elements,)
        p: points to evaluate solution on (points, 3)
        type: bem_3d.INTERIOR for interior problems, bem_3d.EXTERIOR for exterior problems.        
    """
    cols = elems_x1.shape[0]
    result = 0. + 0.j
    n_p = float('nan')  # make sure not used
    p_on_q = np.zeros((p.shape[0], elems_x1.shape[0]))
    Lk, Mk = ComputeIntegralMatricesForEval(elems_x1, elems_x2, elems_x3, p,
                                            n_p, k, p_on_q)
    Lk_times_v = np.dot(Lk, v)
    Mk_times_phi = np.dot(Mk, phi)
    if type == INTERIOR:
        return Lk_times_v - Mk_times_phi
    elif type == EXTERIOR:
        return Mk_times_phi - Lk_times_v
    else:
        raise ValueError("Invalid solve type")


# TODO(aselle): Make these into real test cases.


def testLk():
    x1 = np.array([[1., 3., 1.], [1., 1., 3.]])
    x2 = np.array([[2., 3., 4.], [2., 1., 3.]])
    x3 = np.array([[1., 3., 6.], [2., 2., 3.]])
    p = np.array([[.5, .5, .5], [1.5, 3., 4], [1.9, 1.3, 3.]])
    results = Lk_v(x1, x2, x3, p, None, 1.,
                   np.array([[False, False], [True, False], [False, True]]))
    print("Lk", results.dtype, results)
    diff = np.abs(results[0][0] - (-.0230714 - .0279797j))
    assert diff < 1e-2
    diff = np.abs(results[1][0] - (.308841 + .163383j))
    assert diff < 1e-2
    diff = np.abs(results[2][0] - (-.0443014 + .0779867j))
    assert diff < 1e-2
    diff = np.abs(results[0][1] - (-.0132009 + .00338288j))
    assert diff < 1e-2
    diff = np.abs(results[1][1] - (-.007425 + .0185884j))
    assert diff < 1e-2
    diff = np.abs(results[2][1] - (.165473 + .03870066j))
    assert diff < 1e-2


def testMk():
    x1 = np.array([[1., 3., 1.], [1., 1., 3.]])
    x2 = np.array([[2., 3., 4.], [2., 1., 3.]])
    x3 = np.array([[1., 3., 6.], [2., 2., 3.]])
    p = np.array([[.5, .5, .5], [1.5, 3., 4], [1.9, 1.3, 3.]])
    results = Mk_v(x1, x2, x3, p, None, 1.)
    print(results)


def testMkt():
    x1 = np.array([[1., 3., 1.], [1., 1., 3.]])
    x2 = np.array([[2., 3., 4.], [2., 1., 3.]])
    x3 = np.array([[1., 3., 6.], [2., 2., 3.]])
    p = np.array([[.5, .5, .5], [1.5, 3., 4], [1.9, 1.3, 3.]])
    n = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    results = Mkt_v(x1, x2, x3, p, n, 1.)
    print(results)


# def testNk():
#     x1=np.array([[1.,3.,1.],[1.,1.,3.]])
#     x2=np.array([[2.,3.,4.],[2.,1.,3.]])
#     x3=np.array([[1.,3.,6.],[2.,2.,3.]])
#     p=np.array([[.5,.5,.5],[1.5,3.,4],[1.9,1.3,3.]])
#     n=np.array([[1,0,0],[0,-1,0],[0,0,1]])
#     results = Nk_v(x1,x2,x3,p,n, 1., np.array([[False,False],[True,False],[False,True]]))
#     print(results)

# testLk()
# testMk()
# testMkt()
# testNk()
