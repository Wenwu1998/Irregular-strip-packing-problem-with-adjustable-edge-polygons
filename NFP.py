import numpy as np
import copy
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import matplotlib.pyplot as plt
from shapely import set_precision
from shapely import affinity
class Convex:
    def __init__(self, poly1, poly2):
        self.stationary = copy.deepcopy(poly1)
        self.sliding = copy.deepcopy(poly2)
        self.A = np.array(self.stationary, dtype=float)
        self.B = np.array(self.sliding, dtype=float)
        self.nfp = []
        self.compute_nfp()   
 
    def compute_nfp(self):
        self.A = self._ensure_ccw(self.A)
        self.B = self._ensure_ccw(self.B)

        bottom_idx = self._find_bottom_first(self.A)
        top_idx = self._find_top_first(self.B)
        top_pt = self.B[top_idx]
        self.ref = top_pt
        C = top_pt - self.B
        C = self._ensure_ccw(C)

        bottom_idx_C = self._find_bottom_first(C)
        start_pt = self.A[bottom_idx] + C[bottom_idx_C]

        nfp_vertices = self._minkowski_sum_with_start(
            self.A, C, bottom_idx, bottom_idx_C, start_pt
        )
        self.nfp = nfp_vertices.tolist()
    
    

    def _find_top_first(self, points):
        pts = np.array(points)
        order = np.lexsort((pts[:, 0], -pts[:, 1]))
        return order[0]
    
    def _find_bottom_first(self, points):
        pts = np.array(points)
        order = np.lexsort((pts[:, 0], pts[:, 1]))
        return order[0]
    



    def _ensure_ccw(self, poly):
        x = poly[:, 0]
        y = poly[:, 1]
        area = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + 0.5*(x[-1]*y[0] - x[0]*y[-1])
        if area < 0:
            return poly[::-1]
        return poly

    def _minkowski_sum_with_start(self, P, Q, start_idx_P, start_idx_Q, start_pt):
        P = self._ensure_ccw(P)
        Q = self._ensure_ccw(Q)
        nP, nQ = len(P), len(Q)
        raw_edges = []  
        for i in range(nP):
            vec = P[(i+1)%nP] - P[i]
            angle = np.arctan2(vec[1], vec[0])
            raw_edges.append((angle, vec))
        for j in range(nQ):
            vec = Q[(j+1)%nQ] - Q[j]
            angle = np.arctan2(vec[1], vec[0])
            raw_edges.append((angle, vec))

        raw_edges.sort(key=lambda x: x[0])

        merged = []
        cur_angle = raw_edges[0][0]
        cur_vec = raw_edges[0][1].copy()
        for angle, vec in raw_edges[1:]:
            if abs(angle - cur_angle) < 1e-12:
                cur_vec += vec
            else:
                merged.append((cur_angle, cur_vec))
                cur_angle = angle
                cur_vec = vec.copy()
        merged.append((cur_angle, cur_vec))

        vec_start_P = P[(start_idx_P+1)%nP] - P[start_idx_P]
        angle_start_P = np.arctan2(vec_start_P[1], vec_start_P[0])
        vec_start_Q = Q[(start_idx_Q+1)%nQ] - Q[start_idx_Q]
        angle_start_Q = np.arctan2(vec_start_Q[1], vec_start_Q[0])

        tol = 1e-12
        if abs(angle_start_P - angle_start_Q) < tol:
            start_angle = angle_start_P
        else:
            start_angle = min(angle_start_P, angle_start_Q)

        start_idx = None
        for i, (angle, _) in enumerate(merged):
            if abs(angle - start_angle) < tol:
                start_idx = i
                break
        if start_idx is None:
            raise ValueError("error")

        vertices = [start_pt]
        current = start_pt
        m = len(merged)
        for k in range(m):
            idx = (start_idx + k) % m
            vec = merged[idx][1]
            current = current + vec
            vertices.append(current)

        vertices = np.array(vertices[:-1])
        vertices = self._remove_collinear_points(vertices)

        return vertices


    def _remove_collinear_points(self, poly, tol=1e-9):
        if len(poly) < 3:
            return poly
        pts = poly.tolist()
        new_pts = []
        n = len(pts)
        for i in range(n):
            a = np.array(pts[i])
            b = np.array(pts[(i+1) % n])
            c = np.array(pts[(i+2) % n])
            cross_product = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
            if abs(cross_product) > tol:
                new_pts.append(pts[(i+1) % n])
        if len(new_pts) < 3:
            return poly
        return np.array(new_pts)


class NonConvex:
    def __init__(self, poly1, poly2):
        self.poly1 = copy.deepcopy(poly1)
        self.poly2 = copy.deepcopy(poly2)
        self.nfp = None
        self.compute_nfp()
        # self.plot()

    def compute_nfp(self):
        outer_polygons = []   
        points = [[]for i in range(len(self.poly2))]
        for part1 in self.poly1:
            for i,part2 in enumerate(self.poly2):
                nfp_convex = Convex(part1, part2)
                if len(points[i]) == 0:
                    points[i] = nfp_convex.ref  
                tx = self.poly2[0][0][0] - points[i][0] 
                ty = self.poly2[0][0][1] - points[i][1] 
                poly1 = Polygon(nfp_convex.nfp)
                poly = affinity.translate(poly1, tx, ty)
                if not poly.is_valid:
                    poly = make_valid(poly)
                outer_polygons.append(poly)

        outer_union = unary_union(outer_polygons)
        outer_union = set_precision(outer_union, 0.01)
        self.nfp = outer_union
        
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        for part in self.poly1:
            poly = Polygon(part)
            if not poly.is_valid:
                poly = make_valid(poly)
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.3, fc='r', ec='black', label='Polygon A' if part is self.poly1[0] else "")
        for part in self.poly2:
            poly = Polygon(part)
            if not poly.is_valid:
                poly = make_valid(poly)
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.3, fc='b', ec='black', label='Polygon B' if part is self.poly2[0] else "")

        if self.nfp is not None:
            if self.nfp.geom_type == 'Polygon':
                x, y = self.nfp.exterior.xy
                ax.plot(x, y, 'g-', linewidth=2, label='NFP')
                ax.fill(x, y, alpha=0.2, fc='g', ec='none')
            elif self.nfp.geom_type == 'MultiPolygon':
                for poly in self.nfp.geoms:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, 'g-', linewidth=2, label='NFP' if poly is self.nfp.geoms[0] else "")
                    ax.fill(x, y, alpha=0.2, fc='g', ec='none')

        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('No-Fit Polygon (NFP)')
        return ax

