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
import gmshparser
import os
import bem_2d
import time
import numpy as np
import pandas as pd
import scipy.special
import math
import matplotlib.pylab as pylab
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import CircleCollection
import bem_3d
import backend
from backend import np as np
import sys


def read(filename):
    vertices = []
    faces = []
    with open(filename) as fp:
        while 1:
            line = fp.readline()
            if line == "": break
            parts = line.split(" ")
            if parts[0] == ("v"):
                vertices.append([float(s) for s in parts[1:]])
            if line[0] == ("f"):
                faces.append([int(s) - 1 for s in parts[1:]])
    return np.array(vertices), np.array(faces)


def draw(ax, elems_x1, elems_x2, points, links):
    #ax.set_xlim(-.02, .12)
    #ax.set_ylim(-.02, .12)
    lines = np.stack([points[links[:, 0]], points[links[:, 1]]], axis=1)
    print(lines.shape)
    lc = LineCollection(lines, transOffset=ax.transData)
    #print(elem_centers.shape)
    normals = []
    centers = []
    n, l = bem_2d.geom_v(elems_x1, elems_x2)
    centers = .5 * (elems_x1 + elems_x2)
    normals = .5 * l[:, None] * n + centers
    normal_lines = np.stack([centers, normals], axis=1)
    lc2 = LineCollection(normal_lines, transOffset=ax.transData)

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


def example1():
    """Simple simulation of /interior/ domain with square boundary in 2D 
    compared against analytical."""
    # geometry
    Nside = 16
    for Nside in [4]:
        start = time.time()
        # Setup square domain
        domain = np.linspace(0, .1, Nside + 1)
        pointsx = np.concatenate(
            [np.zeros(Nside), domain[:-1],
             np.ones(Nside) * .1, domain[:0:-1]])
        pointsy = np.concatenate(
            [domain[:-1],
             np.ones(Nside) * .1, domain[:0:-1],
             np.zeros(Nside)])
        points = np.stack([pointsx, pointsy], axis=1)
        links = np.stack([
            np.mod(np.arange(Nside * 4) + 1, Nside * 4),
            np.arange(Nside * 4)
        ],
                         axis=1)
        elems_x1 = points[links[:, 0]]  # first point in element
        elems_x2 = points[links[:, 1]]  # second point in element
        n, l = bem_2d.geom_v(elems_x1, elems_x2)
        Vn = np.zeros(points.shape)
        Vn[links[:, 0], :] += .5 * n[..., :]
        Vn[links[:, 1], :] += .5 * n[..., :]
        #Vn /= scipy.linalg.norm(Vn, axis=-1)[:,None]
        points2 = points - .01 * Vn
        elems2_x1 = points2[links[:, 0]]  # first point in element
        elems2_x2 = points2[links[:, 1]]  # second point in element
        elem_centers = .5 * (elems_x1 + elems_x2)
        N_elements = elems_x1.shape[0]

        # 400 hz at 344 m/s
        frequency = 400
        density = 1.29
        omega = 2 * math.pi * frequency
        c = 334
        k = bem_2d.WaveNumber(frequency, c)

        # Analytical solution to Helmholtz
        def analytical(x, y):
            return np.sin(k * x / np.sqrt(2)) * np.sin(k * y / np.sqrt(2))

        # Solve all dirichlet
        phi_w = np.ones(N_elements)
        phi_analytic = analytical(elem_centers[:, 0], elem_centers[:, 1])
        v_w = np.zeros(N_elements)
        v = np.zeros(N_elements)
        phi, v = bem_2d.SolveHelmholtz_v(k,
                                         elems_x1,
                                         elems_x2,
                                         phi_w,
                                         phi_analytic,
                                         v_w,
                                         v,
                                         type="interior")
        print("Dirichlet solve error: ", np.linalg.norm(phi_analytic - phi),
              np.linalg.norm(v - v))
        # Solve all Neumann
        phi_w = np.zeros(N_elements)
        v_w = np.ones(N_elements)
        phi2, v2 = bem_2d.SolveHelmholtz_v(k,
                                           elems_x1,
                                           elems_x2,
                                           phi_w,
                                           phi,
                                           v_w,
                                           v,
                                           type="interior")
        print("Neumann solve error: ", np.linalg.norm(phi2 - phi),
              np.linalg.norm(v2 - v))
        # Solve mixture
        phi_w = np.concatenate(
            [np.zeros(N_elements // 2),
             np.ones(N_elements // 2)])
        v_w = np.concatenate(
            [np.ones(N_elements // 2),
             np.zeros(N_elements // 2)])
        phi3, v3 = bem_2d.SolveHelmholtz_v(k,
                                           elems_x1,
                                           elems_x2,
                                           phi_w,
                                           phi,
                                           v_w,
                                           v,
                                           type="interior")
        print("Mixture solve error: ", np.linalg.norm(phi3 - phi),
              np.linalg.norm(v3 - v))

        # Produce pseudocolor plot
        Xs = np.linspace(-.1, .2, 50)
        Ys = np.linspace(-.1, .2, 50)
        ptsx, ptsy = np.meshgrid(Xs, Ys)
        pts = np.zeros((ptsx.shape[0], ptsx.shape[1], 2), np.complex64)
        pts[:, :, 0] = ptsx
        pts[:, :, 1] = ptsy

        pts = pts.reshape(pts.shape[0] * pts.shape[1], -1)
        phi_pts = (bem_2d.EvaluatePosition_v(k,
                                             elems_x1,
                                             elems_x2,
                                             phi,
                                             v,
                                             np.array(pts),
                                             type="exterior"))

        def analytical_bounded(x, y):
            "Only compute analytic in the solution domain"
            return (x >= 0) * (x <= .1) * (y <= .1) * (y >= 0) * analytical(
                x, y)

        phi_analytical = analytical_bounded(pts[..., 0], pts[..., 1])
        pressure_analytic_pts = np.abs(phi_analytical *
                                       (1.j * density * omega))
        pressure_computed_pts = np.abs(phi_pts * (1.j * density * omega))

        fig, ((ax0, ax1), (ax2, ax3)) = pyplot.subplots(2, 2)

        ax0.pcolor(ptsx, ptsy,
                   pressure_analytic_pts.reshape(ptsx.shape[0], ptsx.shape[1]))
        ax0.set_title("Analytic pressure")
        ax1.pcolor(ptsx, ptsy,
                   pressure_computed_pts.reshape(ptsx.shape[0], ptsx.shape[1]))
        ax1.set_title("Compujted pressure")
        draw(ax0, elems_x1, elems_x2, points, links)
        draw(ax1, elems_x1, elems_x2, points, links)
        draw(ax2, elems_x1, elems_x2, points, links)
        for ax in [ax0, ax1, ax2]:
            ax.set_xlim([-.05, .15])
            ax.set_ylim([-.05, .15])

        # Evaluate Microphone points
        pts = [[.025, .025], [.075, .025], [.025, .075], [.075, .075],
               [.050, .050]]
        phi_pts = bem_2d.EvaluatePosition_v(k,
                                            elems_x1,
                                            elems_x2,
                                            phi,
                                            v,
                                            np.array(pts),
                                            type="interior")
        ax2.set_title("Discrete point rel error comparison")
        ax2.plot(np.array(pts)[:, 0], np.array(pts)[:, 1], 'o')
        results = []
        for pt, sim_phi in zip(pts, phi_pts):
            val = analytical(pt[0], pt[1])
            rel_error = np.abs(val - sim_phi) / np.abs(val)
            results.append([pt, rel_error])
            ax2.text(pt[0], pt[1], "%.4f" % (rel_error))
        fig.suptitle("2d Interior Boundary Problem")
        pyplot.show()


def example2():
    """Simple simulation of /exterior/ domain with square boundary in 2D 
    compared against analytical."""
    # geometry
    Nside = 8
    for Nside in [16]:
        start = time.time()
        # Setup square domain
        domain = np.linspace(0, .1, Nside + 1)
        pointsx = np.concatenate(
            [np.zeros(Nside), domain[:-1],
             np.ones(Nside) * .1, domain[:0:-1]])
        pointsy = np.concatenate(
            [domain[:-1],
             np.ones(Nside) * .1, domain[:0:-1],
             np.zeros(Nside)])
        points = np.stack([pointsx, pointsy], axis=1)
        links = np.stack([
            np.mod(np.arange(Nside * 4) + 1, Nside * 4),
            np.arange(Nside * 4)
        ],
                         axis=1)
        elems_x1 = points[links[:, 0]]  # first point in element
        elems_x2 = points[links[:, 1]]  # second point in element
        elem_centers = .5 * (elems_x1 + elems_x2)
        N_elements = elems_x1.shape[0]
        print("N_elements %d" % N_elements)

        # 400 hz at 344 m/s
        frequency = 400  # Hz
        density = 1.29  # kg/m^3
        omega = 2 * math.pi * frequency  # Hz
        c = 334  # m/s
        k = bem_2d.WaveNumber(frequency, c)

        # Analytical solution to Helmholtz
        def analytical(x, y):
            dist = np.sqrt((x - .05)**2 + (y - .05)**2)
            return 1.j / 4. * scipy.special.hankel1(0, k * dist)

        # Solve all dirichlet
        phi_w = np.ones(N_elements)
        phi_analytical = analytical(elem_centers[:, 0], elem_centers[:, 1])
        v_w = np.zeros(N_elements)
        v = np.zeros(N_elements)
        phi, v = bem_2d.SolveHelmholtz_v(k,
                                         elems_x1,
                                         elems_x2,
                                         phi_w,
                                         phi_analytical,
                                         v_w,
                                         v,
                                         type="exterior")
        # Solve all Neumann
        phi_w = np.zeros(N_elements)
        v_w = np.ones(N_elements)
        phi2, v2 = bem_2d.SolveHelmholtz_v(k,
                                           elems_x1,
                                           elems_x2,
                                           phi_w,
                                           phi_analytical,
                                           v_w,
                                           v,
                                           type="exterior")
        print(np.linalg.norm(phi2 - phi_analytical), np.linalg.norm(v2 - v))
        # Solve mixture
        phi_w = np.concatenate(
            [np.zeros(N_elements // 2),
             np.ones(N_elements // 2)])
        v_w = np.concatenate(
            [np.ones(N_elements // 2),
             np.zeros(N_elements // 2)])
        phi3, v3 = bem_2d.SolveHelmholtz_v(k,
                                           elems_x1,
                                           elems_x2,
                                           phi_w,
                                           phi_analytical,
                                           v_w,
                                           v,
                                           type="exterior")
        print(np.linalg.norm(phi3 - phi_analytical), np.linalg.norm(v3 - v))

        # Evaluate Microphone points

        Xs = np.linspace(-.1, .2, 50)
        Ys = np.linspace(-.1, .2, 50)
        ptsx, ptsy = np.meshgrid(Xs, Ys)
        pts = np.zeros((ptsx.shape[0], ptsx.shape[1], 2), np.complex64)
        pts[:, :, 0] = ptsx
        pts[:, :, 1] = ptsy

        #np.concatenate([ptsx,ptsy])
        #print("pts",pts.shape)
        pts = pts.reshape(pts.shape[0] * pts.shape[1], -1)
        phi_pts = (bem_2d.EvaluatePosition_v(k,
                                             elems_x1,
                                             elems_x2,
                                             phi,
                                             v,
                                             np.array(pts),
                                             type="exterior"))

        def analytical_bounded(x, y):
            "Only compute analytic in the solution domain"
            return (1 - (x >= 0) * (x <= .1) * (y <= .1) *
                    (y >= 0)) * analytical(x, y)

        phi_analytical = analytical_bounded(pts[..., 0], pts[..., 1])
        pressure_analytic_pts = np.abs(phi_analytical *
                                       (1.j * density * omega))
        pressure_computed_pts = np.abs(phi_pts * (1.j * density * omega))

        fig, ((ax0, ax1), (ax2, ax3)) = pyplot.subplots(2, 2)

        ax0.pcolor(ptsx, ptsy,
                   pressure_analytic_pts.reshape(ptsx.shape[0], ptsx.shape[1]))
        ax0.set_title("Analytic pressure")
        ax1.pcolor(ptsx, ptsy,
                   pressure_computed_pts.reshape(ptsx.shape[0], ptsx.shape[1]))
        ax1.set_title("Computed pressure")
        draw(ax0, elems_x1, elems_x2, points, links)
        draw(ax1, elems_x1, elems_x2, points, links)
        draw(ax2, elems_x1, elems_x2, points, links)
        for ax in [ax0, ax1, ax2]:
            ax.set_xlim([-.1, .2])
            ax.set_ylim([-.1, .2])

        pts = [[.0, .15], [.05, .15], [.1, .15], [.050, -.1]]
        phi_pts = bem_2d.EvaluatePosition_v(k,
                                            elems_x1,
                                            elems_x2,
                                            phi,
                                            v,
                                            np.array(pts),
                                            type="exterior")
        ax2.set_title("Discrete point rel error comparison")
        ax2.plot(np.array(pts)[:, 0], np.array(pts)[:, 1], 'o')
        results = []
        for pt, sim_phi in zip(pts, phi_pts):
            val = analytical(pt[0], pt[1])
            rel_error = np.abs(val - sim_phi) / np.abs(val)
            results.append([pt, rel_error])
            ax2.text(pt[0], pt[1], "%.4f" % (rel_error))
        fig.suptitle("2d Exterior Boundary Problem")
        pyplot.show()


def example3():
    start = time.time()
    # geometry
    Nside = 1
    for Nside in [16]:
        start = time.time()
        # Setup square domain
        domain = np.linspace(0, .1, Nside + 1)
        pointsx = np.concatenate(
            [np.zeros(Nside), domain[:-1],
             np.ones(Nside) * .1, domain[:0:-1]])
        pointsy = np.concatenate(
            [domain[:-1],
             np.ones(Nside) * .1, domain[:0:-1],
             np.zeros(Nside)])
        points = np.stack([pointsx, pointsy], axis=1)
        links = np.stack([
            np.mod(np.arange(Nside * 4) + 1, Nside * 4),
            np.arange(Nside * 4)
        ],
                         axis=1)
        elems_x1 = points[links[:, 0]]  # first point in element
        elems_x2 = points[links[:, 1]]  # second point in element
        elem_centers = .5 * (elems_x1 + elems_x2)
        N_elements = elems_x1.shape[0]
        print("N_elements %d" % N_elements)

        #draw(elems_x1, elems_x2, points, links)
        # 400 hz at 344 m/s
        frequency = 10000  # Hz
        density = 1.29  # kg/m^3
        omega = 2 * math.pi * frequency  # Hz
        c = 334  # m/s
        k = bem_2d.WaveNumber(frequency, c)

        # Analytical solution to Helmholtz
        def analytical(x, y):
            dist = np.sqrt((x - .05)**2 + (y - .05)**2)
            return 1.j / 4. * scipy.special.hankel1(0, k * dist)

        phi_w = np.zeros(N_elements)
        v_w = np.ones(N_elements)
        phi = np.zeros(N_elements)
        v = np.zeros(N_elements)
        v[(elem_centers[:, 1] > .09) * (elem_centers[:, 0] >= 0.) *
          (elem_centers[:, 0] < .1)] = 1.
        #v[(elem_centers[:,1] < .01) * (elem_centers[:,0] >= 0.) * (elem_centers[:,0] < .1)]  = 1.
        phi, v = bem_2d.SolveHelmholtz_v(k,
                                         elems_x1,
                                         elems_x2,
                                         phi_w,
                                         phi,
                                         v_w,
                                         v,
                                         type="exterior")
        #print(np.linalg.norm(phi2-phi), np.linalg.norm(v2-v))

        # Evaluate Microphone points

        Xs = np.linspace(-.1, .2, 50)
        Ys = np.linspace(-.1, .2, 50)
        ptsx, ptsy = np.meshgrid(Xs, Ys)
        pts = np.zeros((ptsx.shape[0], ptsx.shape[1], 2), np.complex64)
        pts[:, :, 0] = ptsx
        pts[:, :, 1] = ptsy

        #np.concatenate([ptsx,ptsy])
        #print("pts",pts.shape)
        pts = pts.reshape(pts.shape[0] * pts.shape[1], -1)
        phi_pts = (bem_2d.EvaluatePosition_v(k,
                                             elems_x1,
                                             elems_x2,
                                             phi,
                                             v,
                                             np.array(pts),
                                             type="exterior"))
        pressure_pts = np.abs(phi_pts / (1.j * density * omega))
        pylab.pcolor(pressure_pts.reshape(ptsx.shape[0], ptsx.shape[1]))
        pylab.colorbar()
        pylab.savefig("test.png")

        #pts = [[.0,.15], [.05,.15], [.1, .15], [.050, -.1]]
        #phi_pts = bem_2d.EvaluatePosition_v(k, elems_x1, elems_x2, phi, v, np.array(pts), type="exterior")

        #results = []
        #for pt, sim_phi in zip(pts, phi_pts):
        #    #sim =  bem_2d.EvaluatePosition(k, elems_x1, elems_x2, phi, v, np.array(pt) )
        #    val =  analytical(pt[0], pt[1])
        #    results.append([pt, sim_phi, val, np.abs(val-sim_phi)/val])
        #print(pd.DataFrame(results, columns=["Point","Sim", "Analytic","Rel.Error"]))
        print()

    print("total is ", time.time() - start)


def icosohedron():
    points = np.array([[0, 0, 1], [0, .745, .667], [.645, .372, .667],
                       [.645, -.372, .667], [0, -.745, .667],
                       [-.645, -.372, .667], [-.645, .372, .667],
                       [.5, .866, 0], [1.0, 0, 0], [.5, -.866, 0],
                       [-.5, -.866, 0], [-1., 0, 0], [-.5, .866, 0],
                       [0, .745, -.667], [0.645, .372, -.667],
                       [0.645, -.372, -.667], [0., -.745, -.667],
                       [-.645, -.372, -.667], [-.645, .372, -.667], [0, 0,
                                                                     -1]])
    edges = np.array([[1, 3, 2], [1, 4, 3], [1, 5, 4], [1, 6, 5], [1, 7, 6],
                      [1, 2, 7], [2, 3, 8], [3, 9, 8], [3, 4, 9], [4, 10, 9],
                      [4, 5, 10], [5, 11, 10], [5, 6, 11], [6, 12, 11],
                      [6, 7, 12], [7, 13, 12], [7, 2, 13], [2, 8, 13],
                      [8, 15, 14], [8, 9, 15], [9, 16, 15], [9, 10, 16],
                      [10, 17, 16], [10, 11, 17], [11, 18, 17], [11, 12, 18],
                      [12, 19, 18], [12, 13, 19], [13, 14, 19], [13, 8, 14],
                      [14, 15, 20], [15, 16, 20], [16, 17, 20], [17, 18, 20],
                      [18, 19, 20], [19, 14, 20]])
    # with open("test.obj","w") as fp:
    #     for v in points:
    #         fp.write("v %f %f %f\n" % (v[0],v[1],v[2]))
    #     for f in edges:
    #         fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))

    edges -= 1
    return points, edges


def example4():
    points, edges = read("spheresm.obj")
    points, edges = icosohedron()
    #print(edges -1)

    method = bem_3d.BARTON_MILLER
    #method = bem_3d.DUAL_SURFACE

    elems_x1 = points[edges[:, 0]]  # first point in element
    elems_x2 = points[edges[:, 1]]  # second point in element
    elems_x3 = points[edges[:, 2]]  # second point in element
    elem_centers = .3333333333 * (elems_x1 + elems_x2 + elems_x3)
    n, area = bem_3d.geom_v(elems_x1, elems_x2, elems_x3)
    N_elements = elems_x1.shape[0]

    frequency = 20  # Hz
    density = 1.29  # kg/m^3
    omega = 2 * math.pi * frequency  # Hz
    c = 344  # m/s
    k = bem_2d.WaveNumber(frequency, c)

    def analytic_phi(X):
        return np.sin(k * X[:, 2])

    phi_w = np.ones(N_elements)
    v_w = np.zeros(N_elements)
    phi = analytic_phi(elem_centers)
    v = np.zeros(N_elements)
    v[(elem_centers[:, 1] > .09) * (elem_centers[:, 0] >= 0.) *
      (elem_centers[:, 0] < .1)] = 1.
    v[(elem_centers[:, 1] < .01) * (elem_centers[:, 0] >= 0.) *
      (elem_centers[:, 0] < .1)] = 1.
    phi1, v1 = bem_3d.SolveHelmholtz_v(k,
                                       elems_x1,
                                       elems_x2,
                                       elems_x3,
                                       phi_w,
                                       phi,
                                       v_w,
                                       v,
                                       type=bem_3d.INTERIOR,
                                       method=method)

    phi_w = np.zeros(N_elements)
    v_w = np.ones(N_elements)
    phi = np.zeros(N_elements)
    v = k * np.cos(k * elem_centers[:, 2]) * n[:, 2]
    phi2, v2 = bem_3d.SolveHelmholtz_v(k,
                                       elems_x1,
                                       elems_x2,
                                       elems_x3,
                                       phi_w,
                                       phi,
                                       v_w,
                                       v,
                                       type=bem_3d.INTERIOR,
                                       method=method)

    print("Problem Exterior 3D Sphere")
    for name, phi, v in [["dirichlet", phi1, v1], ["neumann", phi2, v2]]:
        print(name)
        p = np.array([[.5, 0., 0.], [0., 0., .25], [0., 0., .5], [.0, 0.,
                                                                  .75]])
        anal = (analytic_phi(p))
        sim = (bem_3d.EvaluatePosition_v(k,
                                         elems_x1,
                                         elems_x2,
                                         elems_x3,
                                         phi,
                                         v,
                                         p,
                                         type=bem_3d.INTERIOR))
        results = zip(p, sim, anal, np.abs(sim - anal),
                      np.abs((sim - anal) / anal))
        print(
            pd.DataFrame(
                results,
                columns=["Point", "Sim", "Analytic", "Abs.Err", "Rel.Err"]))


def example5():
    points, edges = icosohedron()
    #points,edges = parse_mesh("/Users/aselle/Desktop/sphere.msh")

    method = bem_3d.DUAL_SURFACE
    type = bem_3d.EXTERIOR
    #points, edges = read("sphere.obj")

    #print(edges -1)

    elems_x1 = points[edges[:, 0]]  # first point in element
    elems_x2 = points[edges[:, 1]]  # second point in element
    elems_x3 = points[edges[:, 2]]  # second point in element
    elem_centers = .3333333333 * (elems_x1 + elems_x2 + elems_x3)
    n, area = bem_3d.geom_v(elems_x1, elems_x2, elems_x3)
    N_elements = elems_x1.shape[0]

    frequency = 100  # Hz
    density = 1.29  # kg/m^3
    omega = 2 * math.pi * frequency  # Hz
    c = 344  # m/s
    k = bem_2d.WaveNumber(frequency, c)

    def analytic_phi(X):
        return np.exp(1.j * k * np.linalg.norm(X, axis=-1)) / np.linalg.norm(
            X, axis=-1)

    def analytic_v(X):
        r = np.linalg.norm(X, axis=-1)
        return (1.j * k * r * np.exp(1.j * k * r) - np.exp(1.j * k * r)) / (r *
                                                                            r)

    phi_w = np.ones(N_elements)
    v_w = np.zeros(N_elements)
    phi = analytic_phi(elem_centers)
    v = np.zeros(N_elements)
    phi1, v1 = bem_3d.SolveHelmholtz_v(k,
                                       elems_x1,
                                       elems_x2,
                                       elems_x3,
                                       phi_w,
                                       phi,
                                       v_w,
                                       v,
                                       type=type,
                                       method=method)

    phi_w = np.zeros(N_elements)
    v_w = np.ones(N_elements)
    phi = np.zeros(N_elements)
    v = analytic_v(elem_centers)
    #phi2, v2 = bem_3d.SolveHelmholtz_v(k, elems_x1, elems_x2, elems_x3, phi_w, phi, v_w, v, type=type, method=method)
    phi2, v2 = phi1, v1

    print("Problem Exterior 3D Sphere")
    tests = [["dirichlet", phi1, v1], ["neumann", phi2, v2]]
    tests = [["dirichlet", phi1, v1]]  # , ["neumann",phi2,v2]]
    for name, phi, v in tests:
        print(name + " " + method)
        p = np.array([[0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, -2]])
        type = "exterior"
        anal = (analytic_phi(p))
        sim = (bem_3d.EvaluatePosition_v(k,
                                         elems_x1,
                                         elems_x2,
                                         elems_x3,
                                         phi,
                                         v,
                                         p,
                                         type=type))
        results = zip(p, sim, anal, np.abs(sim - anal),
                      np.abs((sim - anal) / anal))
        print(
            pd.DataFrame(
                results,
                columns=["Point", "Sim", "Analytic", "Abs.Err", "Rel.Err"]))


def parse_mesh(filename, scale=1. / 1000):
    mesh = gmshparser.parse(filename)
    elements = []
    points = []
    for entity in mesh.get_node_entities():
        for node in entity.get_nodes():
            nid = node.get_tag()
            points.append(node.get_coordinates())
            #print("Node id = %s, node coordinates = %s" % (nid, ncoords))
    for entity in mesh.get_element_entities():
        eltype = entity.get_element_type()
        #print("Element type: %s" % eltype)
        if eltype != 2: continue
        for element in entity.get_elements():
            elid = element.get_tag()
            elcon = element.get_connectivity()
            elements.append(elcon)
            #print("Element id = %s, connectivity = %s" % (elid, elcon))
    points = scale * np.array(points, np.float64)  # to meters

    print(np.min(points[:, 0]))
    print(np.max(points[:, 0]))
    edges = np.array(elements) - 1
    #print(f"Read {filename} wiht {points.shape[0]} pointsand {edges.shape[0]} elements")
    return points, edges


def example6():
    """Simple waveguide example simulated with an array of microphones to get frequency response."""
    method = bem_3d.DUAL_SURFACE
    #mesh = gmshparser.parse("/Users/aselle/Desktop/M212-Gen5-v33 As Solid For AndySim v2-Adapt.msh")
    #mesh = gmshparser.parse("/Users/aselle/Desktop/sphere.5.msh")
    #points,edges = parse_mesh("/Users/aselle/Desktop/sphere.5.msh")
    #points, edges = parse_mesh(os.environ["HOME"] + "/Desktop/M212-Gen5-v33-wider-adaptive-.75.msh")
    points, edges = parse_mesh(os.environ["HOME"] +
                               "/M212-Gen5-v33-wider-adaptive.msh")
    #points, edges = parse_mesh("/Users/aselle/Desktop/sphere.5.msh")

    elems_x1 = points[edges[:, 0]]  # first point in element
    elems_x2 = points[edges[:, 1]]  # second point in element
    elems_x3 = points[edges[:, 2]]  # second point in element
    elem_centers = (.3333333333 * (elems_x1 + elems_x2 + elems_x3)).astype(
        np.float64)
    n, area = bem_3d.geom_v(elems_x1, elems_x2, elems_x3)
    N_elements = elems_x1.shape[0]
    print("ELEMENTS", elem_centers.shape)
    #sys.[exit(1)

    bins = np.logspace(np.log10(600), np.log10(20000), 60)
    angles = np.linspace(-math.pi / 2, math.pi / 2, 65)
    print(angles)
    sub_angles = np.array([
        0, angles.shape[0] // 8, angles.shape[0] // 8 * 2,
        angles.shape[0] // 8 * 3, angles.shape[0] // 2
    ])
    #print(angles[sub_angles] / math.pi * 180)
    #sys.exit(1)
    solve_type = bem_3d.EXTERIOR
    data = np.zeros((angles.shape[0], bins.shape[0]))
    for idx, frequency in enumerate(bins):
        print("method %s %d/%d freq %f" %
              (method, idx, bins.shape[0], frequency))
        #frequency = 5000 # Hz
        density = 1.29  # kg/m^3
        omega = 2 * math.pi * frequency  # Hz
        c = 344  # m/s
        k = bem_3d.WaveNumber(frequency, c)

        #mask = (elem_centers[:,1] < -.02) * (n[:,1] > .99)
        mask = (elem_centers[:, 2] < -.02) * (n[:, 2] > .99)
        #mask = np.ones(elem_centers.shape[0], dtype=bool)

        #mask = n[:,2] > .99
        print("match", np.count_nonzero(mask))
        #print("notmatch", elem_centers.shape)

        phi_w = np.zeros(N_elements)
        v_w = np.ones(N_elements)
        phi = np.zeros(N_elements, np.complex64)
        v = np.zeros(N_elements, np.complex64)
        v = backend.CopyOrMutate(v, backend.index[mask], 1. / (1.j * omega))
        phi, v = bem_3d.SolveHelmholtz_v(k,
                                         elems_x1,
                                         elems_x2,
                                         elems_x3,
                                         phi_w,
                                         phi,
                                         v_w,
                                         v,
                                         type=solve_type)
        p = np.array(
            [np.sin(angles),
             np.zeros(angles.shape[0]),
             np.cos(angles)])
        #p = np.array([np.sin(angles), np.cos(angles), np.zeros(angles.shape[0])])
        p = np.transpose(p)

        vals = np.abs(
            bem_3d.EvaluatePosition_v(k, elems_x1, elems_x2, elems_x3, phi, v,
                                      p, solve_type) * omega * density)
        data = backend.CopyOrMutate(data, backend.index[:, idx], vals)
        #print(vals)

    stuff = []
    stuffnorm = []
    #print(data/np.max(data, axis=0))
    dbnorm = 20 * np.log10(data / data[-1, :][None, :])
    db = 20 * np.log10(data / np.max(data))
    for idx in sub_angles:
        print(bins, data[idx, :])
        stuff.extend([bins, db[idx, :], '-'])
        stuffnorm.extend([bins, dbnorm[idx, :], '-'])

    pylab.set_cmap('jet')
    pylab.pcolor(db)
    pylab.colorbar()
    pylab.xlabel('frequency index')
    pylab.ylabel('angle index')
    pylab.savefig("pcolor.png")
    pylab.clf()
    pylab.plot(*stuff)
    pylab.semilogx()
    pylab.xlabel('Freq Hz')
    pylab.ylabel('dB')
    pylab.savefig("test.png")
    pylab.clf()
    pylab.plot(*stuffnorm)
    pylab.semilogx()
    pylab.xlabel('Freq Hz')
    pylab.ylabel('dB')
    pylab.savefig("testnorm.png")


valid = [x for x in locals().keys() if x.find("example") == 0]
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <example>")
    print("<examples> can be any of the following:")
    for example in valid:
        print(example)
        doc = locals()[example].__doc__
        if doc:
            print("    " + doc)
    sys.exit(1)
# Run example
if sys.argv[1] in valid:
    locals()[sys.argv[1]]()
