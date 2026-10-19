"""A prototype tetrahedral mesher for a solids recipe, by signed
distance and force equilibrium (Persson and Strang, 2004): points
relax to even edge lengths, points that leave the solid are projected
back onto its surface, scipy's Delaunay connects them, and tets whose
centroids fall outside are dropped. Sharp edges are held by fixed
points. Not part of the package: it reports quality and draws a
picture, and is the measure of what a tet mesher would take.

Run: python tools/prototypes/tetmesh_proto.py PROJECT.vdyn H OUT.png (H in inches; the project's one geometry, its solids meshed as one part)
"""
import sys
import time

import matplotlib
import numpy as np
from scipy.spatial import Delaunay, cKDTree

import visualdynamics
from visualdynamics.core import solids

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

IN = 0.0254
h = float(sys.argv[2])
out = sys.argv[3]
project = visualdynamics.Project.open(sys.argv[1])
g = project['Geometry']
PART = solids.top_level(g)[0].id
LEAVES = solids._leaves(g, PART)


# ---------------------------------------------------------------- SDF
def sdf(solid_id, p):
    """Signed distance (negative inside), in inches, of the recipe."""
    item = solids.solid(g, solid_id)
    if item.kind == 'block':
        c = np.asarray(item.center) / IN
        half = np.asarray(item.widths) / IN / 2
        local = (p - c) @ solids._frame(item).T
        q = np.abs(local) - half
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    if item.kind == 'cylinder':
        c = np.asarray(item.center) / IN
        local = (p - c) @ solids._frame(item).T
        r = np.hypot(local[:, 0], local[:, 1]) - item.radius / IN
        z = np.abs(local[:, 2]) - item.length / IN / 2
        q = np.stack([r, z], axis=1)
        return np.minimum(np.maximum(r, z), 0) + np.linalg.norm(np.maximum(q, 0), axis=1)
    ds = [sdf(o, p) for o in item.operands]
    if item.kind == 'union':
        return np.min(ds, axis=0)
    if item.kind == 'intersect':
        return np.max(ds, axis=0)
    return np.maximum(ds[0], -np.min(ds[1:], axis=0))


def fd(p):
    return sdf(PART, p)


# -------------------------------------------------- fixed points (sharp edges)
# The sharp edges are the curves where two of the recipe's surfaces
# cross on the solid's boundary, and the corners where three do. Each
# block gives six planes and each cylinder a wall and two caps, all
# unbounded; a point is on an edge when it is on two of them and on
# the solid (its signed distance zero). Seeds on one surface are
# projected onto both in turn until they sit on the crossing, kept
# where the solid's boundary runs through them, and thinned to the
# mesh size. Coincident planes (a hole's cap in a face) are skipped.
surfaces = []          # (name, g(p) -> signed distance to the unbounded surface, seeds)
extent = np.abs(np.concatenate([np.asarray(leaf.center) / IN
                                + (max(leaf.widths) + leaf.length) / IN
                                for leaf in LEAVES])).max() + 1.0
fine = h / 3.0
rail = np.arange(-extent, extent + fine, fine)


def plane(point, normal):
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    point = np.asarray(point, float)
    a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(normal, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    gu, gv = np.meshgrid(rail, rail, indexing='ij')
    seeds = point + gu.ravel()[:, None] * e1 + gv.ravel()[:, None] * e2
    return (lambda q: (q - point) @ normal), seeds


def wall(center, axis, radius):
    center = np.asarray(center, float); axis = np.asarray(axis, float)
    a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    t = np.arange(0, 2 * np.pi, fine / radius)
    gt, gs = np.meshgrid(t, rail, indexing='ij')
    seeds = (center + radius * (np.cos(gt.ravel())[:, None] * e1 + np.sin(gt.ravel())[:, None] * e2)
             + gs.ravel()[:, None] * axis)

    def g(q):
        rel = q - center
        along = rel @ axis
        return np.linalg.norm(rel - along[:, None] * axis, axis=1) - radius
    return g, seeds


planes = []
for leaf in LEAVES:
    c = np.asarray(leaf.center) / IN
    frame = solids._frame(leaf)
    if leaf.kind == 'block':
        for k in range(3):
            for sign in (-1, 1):
                n = sign * frame[k]
                planes.append((c + n * leaf.widths[k] / IN / 2, n))
    elif leaf.kind == 'cylinder':
        axis = frame[2]
        surfaces.append((f'wall {leaf.id}', *wall(c, axis, leaf.radius / IN)))
        for sign in (-1, 1):
            planes.append((c + sign * axis * leaf.length / IN / 2, sign * axis))
kept_planes = []
for point, normal in planes:
    if any(abs(abs(normal @ n2) - 1) < 1e-9 and abs((point - p2) @ n2) < 1e-9
           for p2, n2 in kept_planes):
        continue
    kept_planes.append((point, normal))
    surfaces.append((f'plane {len(kept_planes)}', *plane(point, normal)))


def gradient(g, q, step=1e-6):
    return np.stack([(g(q + np.eye(3)[k] * step) - g(q)) / step for k in range(3)], axis=1)


def onto(g, q):
    val = g(q)
    grad = gradient(g, q)
    return q - (val / np.maximum((grad ** 2).sum(axis=1), 1e-30))[:, None] * grad


def crossing(gs, seeds, rounds=25):
    q = seeds.copy()
    for _ in range(rounds):
        for g in gs:
            q = onto(g, q)
    ok = np.all([np.abs(g(q)) < 1e-7 for g in gs], axis=0) & (np.abs(fd(q)) < 1e-6)
    ok &= np.all(np.abs(q) <= extent, axis=1)
    return q[ok]


corners, curves = [], []
for i in range(len(surfaces)):
    for j in range(i + 1, len(surfaces)):
        gi, gj = surfaces[i][1], surfaces[j][1]
        seeds = surfaces[i][2]
        seeds = seeds[np.abs(gj(seeds)) < 2 * h]
        if not len(seeds):
            continue
        found = crossing([gi, gj], seeds)
        if not len(found):
            continue
        curves.append(found)
        for k in range(len(surfaces)):
            if k in (i, j):
                continue
            gk = surfaces[k][1]
            near = found[np.abs(gk(found)) < h]
            if len(near):
                corners.append(crossing([gi, gj, gk], near))
_Tree = cKDTree
pfix = np.zeros((0, 3))


def thin(points, spacing):
    """Points at least `spacing` apart, taken in order."""
    global pfix
    for q in points:
        if not len(pfix) or _Tree(pfix).query(q)[0] >= spacing:
            pfix = np.vstack([pfix, q])


corner_points = np.concatenate(corners) if corners else np.zeros((0, 3))
thin(corner_points, 0.25 * h)
ncorners = len(pfix)
for curve in curves:
    # along the curve in order: each next point the nearest unvisited
    order, left = [0], set(range(1, len(curve)))
    tree = _Tree(curve)
    while left:
        dists, idx = tree.query(curve[order[-1]], k=min(len(curve), 12))
        nxt = next((k for k in np.atleast_1d(idx) if k in left), None)
        if nxt is None:
            nxt = min(left, key=lambda k: np.linalg.norm(curve[k] - curve[order[-1]]))
        order.append(nxt); left.discard(nxt)
    thin(curve[order], 0.9 * h)
print(f'{len(surfaces)} surfaces, {len(curves)} edge curves, {ncorners} corners, {len(pfix)} fixed points')

# ------------------------------------------------------ initial points
# the solid's extent: the blocks' corners
boxes = np.array([[(np.asarray(leaf.center) / IN + sign * np.asarray(leaf.widths) / IN / 2)
                   for sign in (-1, 1)] for leaf in LEAVES if leaf.kind == 'block'])
lo, hi = boxes[:, 0].min(axis=0) - h, boxes[:, 1].max(axis=0) + h
grid = np.mgrid[lo[0]:hi[0]:h, lo[1]:hi[1]:h, lo[2]:hi[2]:h].reshape(3, -1).T
# a body-centered lattice packs tets better than a cubic one
grid = np.concatenate([grid, grid + h / 2])
geps = 0.001 * h
p = grid[fd(grid) < -geps]
# clear a hole around each fixed point
near = cKDTree(pfix).query(p, distance_upper_bound=0.6 * h)[0] < np.inf
p = np.concatenate([pfix, p[~near]])
nfix = len(pfix)
print('points', len(p))

# -------------------------------------------------------- relaxation
Fscale, dt, ttol, dptol = 1.1, 0.1, 0.1, 0.002
deps = np.sqrt(np.finfo(float).eps) * h
pold = np.full_like(p, np.inf)
t0 = time.perf_counter()
retri = 0
for it in range(300):
    if np.sqrt(((p - pold) ** 2).sum(axis=1)).max() > ttol * h:
        pold = p.copy()
        tri = Delaunay(p)
        tets = tri.simplices
        cent = p[tets].mean(axis=1)
        tets = tets[fd(cent) < -geps]
        bars = np.unique(np.sort(np.concatenate([tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
                                                 tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]]]), axis=1), axis=0)
        retri += 1
    vec = p[bars[:, 0]] - p[bars[:, 1]]
    L = np.linalg.norm(vec, axis=1)
    L0 = Fscale * (np.mean(L ** 3)) ** (1 / 3)
    F = np.maximum(L0 - L, 0)
    Fvec = (F / L)[:, None] * vec
    Ftot = np.zeros_like(p)
    np.add.at(Ftot, bars[:, 0], Fvec)
    np.add.at(Ftot, bars[:, 1], -Fvec)
    Ftot[:nfix] = 0
    p = p + dt * Ftot
    d = fd(p)
    ix = d > 0
    if ix.any():
        grad = np.stack([(fd(p[ix] + np.eye(3)[k] * deps) - d[ix]) / deps for k in range(3)], axis=1)
        p[ix] -= (d[ix] / (grad ** 2).sum(axis=1))[:, None] * grad
    interior = d < -geps
    move = np.sqrt((dt * Ftot[interior] ** 2).sum(axis=1)).max() / h if interior.any() else 0
    if move < dptol:
        break
print(f'{it + 1} iterations, {retri} triangulations, {time.perf_counter() - t0:.1f} s')
tri = Delaunay(p)
tets = tri.simplices
tets = tets[fd(p[tets].mean(axis=1)) < -geps]

# ------------------------------------------------------------ quality
a, b, c, d4 = (p[tets[:, k]] for k in range(4))
vol = np.einsum('ij,ij->i', b - a, np.cross(c - a, d4 - a)) / 6
vol = np.abs(vol)
def tri_area(x, y, z):
    return 0.5 * np.linalg.norm(np.cross(y - x, z - x), axis=1)
area = tri_area(a, b, c) + tri_area(a, b, d4) + tri_area(a, c, d4) + tri_area(b, c, d4)
r_in = 3 * vol / area
edges = [b - a, c - a, d4 - a, c - b, d4 - b, d4 - c]
# circumradius from the Cayley-Menger-free formula: R = |AB x CD ... | use the standard product formula
pa, pb, pc = np.linalg.norm(edges[0], axis=1) * np.linalg.norm(edges[5], axis=1), \
    np.linalg.norm(edges[1], axis=1) * np.linalg.norm(edges[4], axis=1), \
    np.linalg.norm(edges[2], axis=1) * np.linalg.norm(edges[3], axis=1)
s = (pa + pb + pc) / 2
r_circ = np.sqrt(np.maximum(s * (s - pa) * (s - pb) * (s - pc), 0)) / (6 * vol)
q = 3 * r_in / r_circ
print(f'tets {len(tets)}, nodes {len(p)}, volume {vol.sum():.1f} in^3')
print(f'radius ratio: min {q.min():.3f}, median {np.median(q):.3f}, under 0.3: {(q < 0.3).sum()} ({100 * (q < 0.3).mean():.2f}%), under 0.1: {(q < 0.1).sum()}')
Lall = np.linalg.norm(np.concatenate(edges), axis=1)
print(f'edge length: min {Lall.min():.3f}, median {np.median(Lall):.3f}, max {Lall.max():.3f} (h = {h})')
long = np.flatnonzero(Lall > 1.6 * h)
if len(long):
    mids = (np.concatenate([a, a, a, b, b, c])[long] + np.concatenate([b, c, d4, c, d4, d4])[long]) / 2
    print(f'{len(long)} edges over 1.6 h, around', np.round(mids.mean(axis=0), 2), 'spread', np.round(mids.std(axis=0), 2))
# the volume the recipe says, by sampling
samp = np.random.default_rng(0).uniform(lo, hi, (400000, 3))
print(f'recipe volume ~ {(fd(samp) < 0).mean() * np.prod(hi - lo):.1f} in^3')

# ------------------------------------------------------------- picture
faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
key = np.sort(faces, axis=1)
_, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
boundary = faces[idx[counts == 1]]
fig = plt.figure(figsize=(18, 8))
for k, (elev, azim, title) in enumerate([(35, -60, 'from above'), (8, 100, 'from the side (the hole along Y)')]):
    ax = fig.add_subplot(1, 2, k + 1, projection='3d')
    polys = p[boundary]
    normals = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None] + 1e-12
    light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.abs(normals @ light)
    coll = Poly3DCollection(polys, facecolors=np.c_[0.75 * shade, 0.78 * shade, 0.9 * shade], edgecolor='#223', linewidth=0.15)
    ax.add_collection3d(coll)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-5, 5)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_title(f'{title}: {len(tets)} tets at h = {h} in')
fig.savefig(out, dpi=130, bbox_inches='tight')
