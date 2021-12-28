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
import os
import bem_2d
import time
import gmsh_parser

import numpy as np
import numpy as nnp
import pandas as pd
import scipy.special
import math
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import CircleCollection
import bem_3d
import backend
import matplotlib
from backend import np as np
import sys

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def draw_surface(points, triangles, n, scalar, sample_points, frame, freq):
    fig = plt.figure()
    ax = Axes3D(fig=fig)
    min = nnp.min(scalar)
    max = nnp.max(scalar)
    scaled = nnp.minimum(nnp.maximum(0., (scalar - min) / (max - min)), 1.)
    colors = matplotlib.cm.jet(scaled)
    lighting = (nnp.abs(n[:, 1]))[..., None] * nnp.array(
        [1, 1, 1, 1])[None, :]  # n.l ish (directional light)
    lighting[:, 3] = 1  # set alpha to 1
    colors *= lighting

    raw = []
    lines = []
    for i in range(triangles.shape[0]):
        ps = [
            points[triangles[i, 0]], points[triangles[i, 1]],
            points[triangles[i, 2]]
        ]
        raw.append(nnp.stack(ps))
        c = .3333 * (ps[0] + ps[1] + ps[2])
    for sp in sample_points:
        for k in range(3):
            off = nnp.zeros(3)
            off[k] = .05
            lines.append((sp - off, sp + off))
    ax.view_init(elev=10., azim=80.)
    polys = Poly3DCollection(raw)
    edges = Line3DCollection(lines)
    edges.set_edgecolor((.5, 1, 1, 1.0))
    colors = nnp.maximum(0., nnp.minimum(colors, 1.))
    polys.set_facecolor(colors)
    polys.set_edgecolor((1, 1, 1, .3))
    ax.add_collection(polys)
    ax.add_collection(edges)
    ax.set_xlim(-.05, .05)
    ax.set_ylim(-.05, .05)
    ax.set_zlim(-.05, .05)
    if frame is not None:
        pylab.title('Freq %f Hz' % freq)
        pylab.savefig("frame.%d.png" % frame)

    else:
        pylab.show()


def run(input_mesh_filename):
    method = bem_3d.DUAL_SURFACE
    stuff = gmsh_parser.GmshParser(input_mesh_filename)
    points = stuff.nodes
    edges = stuff.triangles
    masks = stuff.triangle_physical_masks
    points /= 1000.  # convert from mm to meters
    emit = [x for x in masks.keys() if x.find("emit") != -1][0]
    mask = masks[emit]

    angles = np.linspace(-math.pi / 2, math.pi / 2, 65, dtype=np.float32)
    #print(angles)
    half = angles.shape[0] // 2
    num = angles.shape[0] // 2 // 4
    sub_angles = np.arange(half, angles.shape[0], num)

    mic_positions = np.array([
        np.sin(angles),
        np.cos(angles),
        np.zeros(angles.shape[0]),
    ])
    mic_positions_sub = nnp.array([
        np.sin(angles[sub_angles]),
        np.cos(angles[sub_angles]),
        np.zeros(sub_angles.shape[0])
    ])
    mic_positions_sub = np.transpose(mic_positions_sub)

    elems_x1 = points[edges[:, 0]]  # first point in element
    elems_x2 = points[edges[:, 1]]  # second point in element
    elems_x3 = points[edges[:, 2]]  # second point in element
    elem_centers = (.3333333333 * (elems_x1 + elems_x2 + elems_x3)).astype(
        np.float32)
    n, area = bem_3d.geom_v(elems_x1, elems_x2, elems_x3)
    N_elements = elems_x1.shape[0]
    if True:
        scalar = mask * 1.
        draw_surface(points, stuff.triangles, n, scalar, mic_positions_sub,
                     None, 0.)

    bins = np.logspace(np.log10(800), np.log10(20000), 128)

    solve_type = bem_3d.EXTERIOR
    datas = []
    for idx, frequency in enumerate(bins):
        with bem_3d.Timer("Whole solve"):
            print("method %s %d/%d freq %f" %
                  (method, idx, bins.shape[0], frequency))
            density = 1.29  # kg/m^3
            omega = 2 * math.pi * frequency  # Hz
            c = 344  # m/s
            k = bem_3d.WaveNumber(frequency, c)

            phi_w = np.zeros(N_elements)
            v_w = np.ones(N_elements)
            phi = np.zeros(N_elements, np.complex64)

            norm = n[:, 1]
            v = np.where(backend.index[mask],
                         norm * np.array(1. / (1.j * omega), np.complex64),
                         np.zeros(N_elements, np.complex64))
            phi, v = bem_3d.SolveHelmholtz_v(k,
                                             elems_x1,
                                             elems_x2,
                                             elems_x3,
                                             phi_w,
                                             phi,
                                             v_w,
                                             v,
                                             type=solve_type)
            # draw_surface(points, stuff.triangles, n, nnp.array(np.abs(phi)[:,0]), mic_positions_sub, frame=idx, freq=frequency)
            with bem_3d.Timer("Evaluate microphones"):
                vals = (bem_3d.EvaluatePosition_v(
                    k, elems_x1, elems_x2, elems_x3, phi, v,
                    np.transpose(mic_positions), solve_type) * omega * density)
                datas.append(vals)
    data = np.stack(datas, axis=1)

    stuff = []
    stuffnorm = []
    dbnorm = 20 * np.log10(np.abs(data / data[sub_angles[0], :][None, :]))
    db = 20 * np.log10(np.abs(data) / np.max(np.abs(data)))
    labels = []
    for idx in sub_angles:
        stuff.extend([bins, db[idx, :], '-'])
        stuffnorm.extend([bins, dbnorm[idx, :], '-'])
        labels.append(str(float(angles[idx] / math.pi * 180)))

    pylab.figure(figsize=(8, 16))
    pylab.subplot(3, 1, 1)
    pylab.set_cmap('jet')
    pylab.pcolor(db)
    pylab.colorbar()
    pylab.xlabel('frequency index')
    pylab.ylabel('angle index')
    pylab.subplot(3, 1, 2)
    pylab.plot(*stuff)
    pylab.title('Frequency Response')
    pylab.semilogx()
    pylab.xlabel('Freq Hz')
    pylab.ylabel('dB')
    pylab.legend(labels)

    pylab.subplot(3, 1, 3)

    pylab.plot(*stuffnorm)
    pylab.title('Frequency Response normalized to on-axis')
    pylab.semilogx()
    pylab.legend(labels)
    pylab.xlabel('Freq Hz')
    pylab.ylabel('dB')
    pylab.ylim([-30, 10])
    pylab.savefig('test.png')


# Check arguments
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <mesh file>")
    print(" Runs directivity simulation across frequency spectrum for mesh.")
    sys.exit(1)

# Run example
filename = sys.argv[1]
run(filename)
