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
import scipy
import scipy.integrate
import time
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.collections import CircleCollection
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd

from backend import np as np

Ndefault = 10


def WaveNumber(frequency, speed_of_sound):
    return 2 * np.pi * frequency / speed_of_sound


def draw(elems_x1, elems_x2, points, links):
    fig, ax = plt.subplots()
    ax.set_xlim(-.02, .12)
    ax.set_ylim(-.02, .12)
    lines = np.stack([points[links[:, 0]], points[links[:, 1]]], axis=1)
    lc = LineCollection(lines, transOffset=ax.transData)
    normals = []
    centers = []
    for i in range(elems_x1.shape[0]):
        n, l = geom(elems_x1[i], elems_x2[i])
        center = .5 * (elems_x1[i] + elems_x2[i])
        normals.append([center, .5 * l * n + center])
        centers.append(center)
    lc2 = LineCollection(normals, transOffset=ax.transData)

    facecolors = [
        '%.1f' % c for c in np.linspace(0.25, 0.75, elems_x1.shape[0])
    ]
    cc = CircleCollection(np.ones(elems_x1.shape[0]) * 10.,
                          facecolors=facecolors,
                          offsets=centers,
                          transOffset=ax.transData)
    ax.add_collection(lc)
    ax.add_collection(lc2)
    ax.add_collection(cc)

    plt.show()


def get_r_v(x1, x2, p, alphas):
    """given x1 and x2 and a set of points in [0,1] given in np array alphas
    produce a set of points interpolated along x1, x2 and return the radius.
    x1: (nelems, 2)
    x2: (nelems, 2)
    p: (npts, 2,)
    alphas: (nquad,)
    
    returns: 
        r (npts, nelems, nquad, 2)
        rl (npts, nelems, nquad)
    """
    w = alphas[None, None, :, None]
    qpts = w * x1[None, :, None, :] + (
        1 - w) * x2[None, :, None, :]  #+ np.outer(x2,1-alphas)
    r = p[:, None, None, :] - qpts
    rl = np.linalg.norm(r, axis=-1)
    return r, rl


def geom_v(x1, x2):
    assert len(x1.shape) == 2
    """
    args:
        x1: (nelems, 2)
        x2: (nelems, 2)

    returns:
        normal (nelems, 2)
        length (nelems)
    """
    t = x2 - x1
    n = t[..., ::-1]
    n[..., 1] = -n[..., 1]
    l = np.linalg.norm(n, axis=1)
    return n / l[:, None], l


def Lk_v(x1, x2, p, n_p, k, p_on_q, N=Ndefault):
    """Eq 3.36
    
    Args: 
        x1, x2 (nelems, 2)
        p (npts, 2)
        n_p (npts, 2)
        k ()
        p_on_q (npts, nelems)
    Returns:
        (npts, nelems)
    """
    assert (k != 0)
    n, l = geom_v(x1, x2)

    def integrand(alphas):
        r, rl = get_r_v(x1, x2, p, alphas)
        r = None
        g_k = 1.j / 4 * scipy.special.hankel1(0, rl * k)
        g_0 = np.where(p_on_q[:, :, None], -1. / (2 * np.pi) * np.log(rl), 0.)
        return l[None, :, None] * (
            g_k - g_0)  # l because we are integrating over change of var

    # TODO(aselle): L0 is not included in the Kirkup's fortran code.
    # Eq 3.41.
    a = np.einsum('jk,ijk->ij', x2 - x1, p[:, None, :] - x1[None, :, :]) / l
    b = l - a
    L0 = 1. / (2 * np.pi) * (a + b - a * np.log(a) - b * np.log(b))
    return scipy.integrate.fixed_quad(integrand, 0, 1, n=N)[0] + np.where(
        p_on_q, L0, 0.)


def Mk_v(x1, x2, p, n_p, k, N=Ndefault, verbose=False):
    """Eq 3.31"""
    n, l = geom_v(x1, x2)
    assert k != 0

    def integrand(alphas):
        r = None
        rvec, rl = get_r_v(x1, x2, p, alphas)
        dr_dnq = -np.einsum("jl,ijkl->ijk", n, rvec) / rl  # shape (p,e,q)
        dGk_dr = -1.j / 4 * k * scipy.special.hankel1(1,
                                                      rl * k)  # shape (p,e,q)
        integrand = l[None, :,
                      None] * dGk_dr * dr_dnq  # l is (e,) output (p,e,q)
        if verbose: print(integrand)
        return integrand

    res = scipy.integrate.fixed_quad(integrand, 0, 1, n=N)[0]
    return res


def Mkt_v(x1, x2, p, n_p, k, N=Ndefault):
    """Eq 3.32"""
    n, l = geom_v(x1, x2)
    assert k != 0

    def integrand(alphas):
        r = None
        rvec, rl = get_r_v(x1, x2, p, alphas)  # (p,e,q,k),(p,e,q)
        dr_dup = np.einsum("il,ijkl->ijk", n_p, rvec) / rl  # (p,e,q)
        dGk_dr = -1.j / 4 * k * scipy.special.hankel1(1, rl * k)  # (p,e,q)
        return l[None, :, None] * dGk_dr * dr_dup

    return scipy.integrate.fixed_quad(integrand, 0, 1, n=N)[0]


def Nk_v(x1, x2, p, n_p, k, p_on_q, N=Ndefault, verbose=False):
    """Eq 3.37"""
    n, l = geom_v(x1, x2)
    assert k != 0

    def integrand(alphas):
        rvec, rl = get_r_v(x1, x2, p, alphas)
        r = None
        dr_dup = np.einsum("il,ijkl->ijk", n_p, rvec) / rl
        dr_dnq = -np.einsum("jl,ijkl->ijk", n, rvec) / rl
        d2r_dupdnq = -1 / rl * (np.einsum("jl,il->ij", n, n_p)[:, :, None] +
                                dr_dup * dr_dnq)
        d2Gk_dr2 = 1.j / 4 * k * k * (scipy.special.hankel1(1, rl * k) /
                                      (k * rl) -
                                      scipy.special.hankel1(0, rl * k))
        dGk_dr = -1.j / 4 * k * scipy.special.hankel1(1, rl * k)
        dG0_dr = np.where(p_on_q[:, :, None], -1. / (2 * np.pi * rl), 0.)
        d2G0_dr2 = np.where(p_on_q[:, :, None], 1. / (2 * np.pi * rl * rl), 0.)
        G0 = -1. / (2 * np.pi) * np.log(rl)
        G0_term = np.where(p_on_q[:, :, None], .5 * k * k * G0, 0.)
        d2Gk_dupdnq = (dGk_dr - dG0_dr) * d2r_dupdnq + (
            d2Gk_dr2 - d2G0_dr2) * dr_dup * dr_dnq
        ret = l[None, :, None] * (d2Gk_dupdnq + G0_term)
        if (verbose): print(ret)
        return ret

    integral = scipy.integrate.fixed_quad(integrand, 0, 1, n=N)[0]
    """Eq 3.41"""
    a = np.einsum('ij,ij->i', x2 - x1, p - x1) / l
    b = l - a
    L0 = np.where(p_on_q,
                  1. / (2 * np.pi) * (a + b - a * np.log(a) - b * np.log(b)),
                  0.)
    N0 = -1. / (2 * np.pi) * (1. / a + 1. / b)
    return np.where(p_on_q, N0 - .5 * k * k * L0, 0.) + integral


def BuildSystem_v(k, elems_x1, elems_x2, type, check_against_scalar=False):
    """equation 4.17 & 4.18"""
    rows = elems_x1.shape[0]
    cols = elems_x1.shape[0]
    p = .5 * (elems_x1 + elems_x2)
    n_p, l_elem = geom_v(elems_x1, elems_x2)
    p_on_q = np.eye(rows, dtype=np.bool)
    Lk_ = Lk_v(elems_x1, elems_x2, p, n_p, k, p_on_q)
    Mk_ = Mk_v(elems_x1, elems_x2, p, n_p, k)
    Mkt_ = Mkt_v(elems_x1, elems_x2, p, n_p, k)
    Nk_ = Nk_v(elems_x1, elems_x2, p, n_p, k, p_on_q)

    # Burton-Miller
    # TODO mu setting may not be right. see paper on burton miller not sure it should
    # be same for both
    if type == "interior":
        mu = 1.j / (k + 1)
        A = Mk_ + mu * Nk_ + np.eye(rows) * .5
        B = Lk_ + mu * (Mkt_) - np.eye(rows) * .5 * mu
    elif type == "exterior":
        mu = 1.j / (k + 1)
        A = Mk_ + mu * Nk_ - np.eye(rows) * .5
        B = Lk_ + mu * (Mkt_) + np.eye(rows) * .5 * mu
    # Naive method
    #A = Mk_ + np.eye(rows)*.5
    #B = Lk_
    return A, B


def SolveHelmholtz_v(k, elems_x1, elems_x2, phi_w, phi, v_w, v, type):
    A, B = BuildSystem_v(k, elems_x1, elems_x2, type=type)
    C = np.concatenate([
        np.concatenate([A, -B], axis=1),
        np.concatenate([np.diag(phi_w), np.diag(v_w)], axis=1)
    ],
                       axis=0)
    N = elems_x1.shape[0]
    F = np.zeros(N * 2, dtype=np.complex64)
    F[N:] = phi_w * phi + v_w * v
    z = scipy.linalg.solve(C, F)
    phi, v = z[:N], z[N:]
    return phi, v


def EvaluatePosition_v(k, elems_x1, elems_x2, phi, v, p, type):
    """equation 4.19. (corrected, swapping phi and v)"""
    cols = elems_x1.shape[0]
    result = 0. + 0.j
    #x1, x2 = elems_x1[j], elems_x2[j]
    n_p = float('nan')  # make sure not used
    p_on_q = np.zeros((p.shape[0], elems_x1.shape[0]))
    Lk_times_v = np.dot(Lk_v(elems_x1, elems_x2, p, n_p, k, p_on_q), v)
    Mk_times_phi = np.dot(Mk_v(elems_x1, elems_x2, p, n_p, k, verbose=False),
                          phi)
    if type == "interior":
        return Lk_times_v - Mk_times_phi
    elif type == "exterior":
        return Mk_times_phi - Lk_times_v
    raise ValueError("Invalid solve type")
