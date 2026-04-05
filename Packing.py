import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import NFP as NFPfun
from shapely import affinity
from shapely import set_precision
import matplotlib.pyplot as plt
import time


class TOPOS:
    def __init__(self, original_polys, listply, width, NFP_history, step, pid):
        self.original =copy.deepcopy(original_polys)         
        self.polys = copy.deepcopy(original_polys)
        self.order = listply
        self.width = width
        self.NFP = NFP_history
        self.step = step
        self.bbox = [] 
        self.allscores = 0         
        self.ref_points = []  
        self.pid = pid      
        self.poly_shapely = [] 
        self.pstart = [[]for i in range(len(original_polys))] 
        self.pend   = [[]for i in range(len(original_polys))] 
        for p in self.order:
            self.pstart[p] = original_polys[p][0][0]           
        for poly in self.polys:
            all_x = np.concatenate([np.array(ring)[:,0] for ring in poly])
            all_y = np.concatenate([np.array(ring)[:,1] for ring in poly])
            minx, maxx = all_x.min(), all_x.max()
            miny, maxy = all_y.min(), all_y.max()
            self.bbox.append([minx, maxx, miny, maxy])
            self.ref_points.append(poly[0][0])
            parts = []
            for ring in poly:
                if ring[0] != ring[-1]:
                    ring = ring + [ring[0]]
                parts.append(Polygon(ring))
            if len(parts) == 1:
                self.poly_shapely.append(parts[0])
            else:
                try:
                    self.poly_shapely.append(unary_union(parts))
                except:                    
                    for i in range(len(parts)):
                        parts[i] = make_valid(parts[i])                       
                    self.poly_shapely.append(unary_union(parts))
                    

        first_idx = self.order[0]
        self.border_left   = self.bbox[first_idx][0]
        self.border_right  = self.bbox[first_idx][1]
        self.border_bottom = self.bbox[first_idx][2]
        self.border_top    = self.bbox[first_idx][3]
        self.border_height = self.border_top - self.border_bottom
        self.border_width  = self.border_right - self.border_left
        self.outpolygon = [copy.deepcopy(self.polys[first_idx])]
        self.score = 0.0
        self.best_score = 20
        

        self.run()
        # self.show_result()


    def _poly_outer_shapely(self, poly_rings):
        outer = poly_rings[0]
        if outer[0] != outer[-1]:
            outer = outer + [outer[0]]
        return Polygon(outer)

    def _translate_rings(self, rings, dx, dy):
        for ring in rings:
            for pt in ring:
                pt[0] += dx
                pt[1] += dy

    def _update_bbox(self, global_idx):
        poly = self.polys[global_idx]
        all_x = np.concatenate([np.array(ring)[:,0] for ring in poly])
        all_y = np.concatenate([np.array(ring)[:,1] for ring in poly])
        self.bbox[global_idx] = [all_x.min(), all_x.max(), all_y.min(), all_y.max()]


    def _compute_nfp(self, polyA_rings, polyB_rings):
        result = NFPfun.NonConvex(polyA_rings, polyB_rings)
        nfp_geom = result.nfp
        return nfp_geom


    def _translate_shapely_poly(self, poly, dx, dy):
        if poly.is_empty:
            return poly
        if poly.geom_type == 'Polygon':
            new_coords = np.array(poly.exterior.coords) + [dx, dy]
            interiors = [np.array(ring.coords) + [dx, dy] for ring in poly.interiors]
            A = Polygon(new_coords, interiors)
            A = set_precision(A, 0.01)
            return A
        elif poly.geom_type == 'MultiPolygon':
            new_polys = [self._translate_shapely_poly(p, dx, dy) for p in poly.geoms]
            A = MultiPolygon(new_polys)
            A = set_precision(A, 0.01)
            return A
        else:
            
            return poly


    def _get_feasible_points(self, feasible_region, new_global_idx): 
        new_height = self.bbox[new_global_idx][3] - self.bbox[new_global_idx][2]
        h1 = self.border_top + (self.width - self.border_height)
        h2 = self.border_bottom - (self.width - self.border_height) + new_height
        points = []  
        seen = set()

        def add_point(x, y, is_interior):
            dy = y - self.ref_points[new_global_idx][1]
            new_bottom = self.bbox[new_global_idx][2] + dy
            new_top    = self.bbox[new_global_idx][3] + dy
            overall_min_y = min(self.border_bottom, new_bottom)
            overall_max_y = max(self.border_top, new_top)
            if overall_max_y - overall_min_y <= self.width + 1e-6:
                key = f"{round(x,3)},{round(y,3)}"
                if key not in seen:
                    seen.add(key)
                    points.append(([round(x,2), round(y,2)], is_interior))
    

        def process_ring(ring, is_interior):
            coords = np.array(ring.coords)
            n = len(coords)
            for i in range(n-1):
                x1, y1 = coords[i]
                x2, y2 = coords[i+1]
                add_point(x1, y1, is_interior)

                dx = x2 - x1
                dy = y2 - y1
                length = np.hypot(dx, dy)
                if length > self.step:
                    num = int(length / self.step)
                    for k in range(1, num):
                        t = k / num
                        x = x1 + t * dx
                        y = y1 + t * dy
                        add_point(x, y, is_interior)

                if (y1 - h1) * (y2 - h1) <= 0 and abs(y2 - y1) > 1e-9:
                    t = (h1 - y1) / (y2 - y1)
                    x = x1 + t * (x2 - x1)
                    add_point(x, h1, is_interior)
                if (y1 - h2) * (y2 - h2) <= 0 and abs(y2 - y1) > 1e-9:
                    t = (h2 - y1) / (y2 - y1)
                    x = x1 + t * (x2 - x1)
                    add_point(x, h2, is_interior)

        if feasible_region.geom_type == 'Polygon':
            process_ring(feasible_region.exterior, is_interior=False)
            for interior in feasible_region.interiors:
                process_ring(interior, is_interior=True)
        elif feasible_region.geom_type == 'MultiPolygon':
            for poly in feasible_region.geoms:
                process_ring(poly.exterior, is_interior=False)
                for interior in poly.interiors:
                    process_ring(interior, is_interior=True)
        return points
    

    def _evaluate_point(self, pt, new_global_idx, placed_indices):
        ref_x, ref_y = self.ref_points[new_global_idx]
        dx = pt[0] - ref_x
        dy = pt[1] - ref_y

        moved_new = self._translate_shapely_poly(self.poly_shapely[new_global_idx], dx, dy)


        for idx in placed_indices:
            try :
                if moved_new.intersects(self.poly_shapely[idx]):
                    inter = moved_new.intersection(self.poly_shapely[idx])
                    if inter.area > 0: 
                        return float('inf')
            except:
                moved_new = set_precision(moved_new, 0.01).buffer(0)
                self.poly_shapely[idx] = set_precision(self.poly_shapely[idx], 0.01).buffer(0)
                if moved_new.intersects(self.poly_shapely[idx]):
                    inter = moved_new.intersection(self.poly_shapely[idx])
                    if inter.area > 0:  
                        return float('inf')

            
            
        px = self.bbox[new_global_idx][1] - self.bbox[new_global_idx][0]
        py = self.bbox[new_global_idx][3] - self.bbox[new_global_idx][2]
        if px == 0 or py == 0:
            return float('inf')

        new_left   = self.bbox[new_global_idx][0] + dx
        new_right  = self.bbox[new_global_idx][1] + dx
        new_bottom = self.bbox[new_global_idx][2] + dy
        new_top    = self.bbox[new_global_idx][3] + dy

        Lnew = max(0, self.border_left - new_left) + max(0, new_right - self.border_right)
        Wnew = max(0, self.border_bottom - new_bottom) + max(0, new_top - self.border_top)

        overlap = 0.0
        for idx in placed_indices:
            b = self.bbox[idx]
            ol = max(new_left, b[0])
            o_r = min(new_right, b[1])
            ob = max(new_bottom, b[2])
            o_t = min(new_top, b[3])
            if ol < o_r and ob < o_t:
                overlap += (o_r - ol) * (o_t - ob)

        fun1 = Lnew / px
        fun2 = overlap / (px * py)
        fun3 = Lnew * Wnew / (px * py)
        score =  0.2*fun1 -  0.6*fun2 +  0.2*fun3
        return score

    def _place_polygon(self, global_idx, target_point):
        ref_x, ref_y = self.ref_points[global_idx]
        dx = target_point[0] - ref_x
        dy = target_point[1] - ref_y

        self._translate_rings(self.polys[global_idx], dx, dy)
        self.ref_points[global_idx] = target_point
        self._update_bbox(global_idx)
        b = self.bbox[global_idx]
        self.border_left   = min(self.border_left, b[0])
        self.border_right  = max(self.border_right, b[1])
        self.border_bottom = min(self.border_bottom, b[2])
        self.border_top    = max(self.border_top, b[3])
        self.border_height = self.border_top - self.border_bottom
        self.border_width  = self.border_right - self.border_left

        parts = []
        for ring in self.polys[global_idx]:
            if ring[0] != ring[-1]:
                ring = ring + [ring[0]]
            parts.append(Polygon(ring))
        if len(parts) == 1:
            self.poly_shapely[global_idx] = parts[0]
        else:
            try:
                self.poly_shapely[global_idx] = unary_union(parts)
            except:                
                for i in range(len(parts)):
                    try:
                        parts[i] = make_valid(parts[i]).buffer(0)
                    except:
                        parts[i] = set_precision(parts[i],0.01)
                    
                self.poly_shapely[global_idx] = unary_union(parts)
                
        return dx, dy


    def run(self):
        n = len(self.order)
        dx1 = [0]
        dy1 = [0]
        for new_idx in range(1, n):
            new_global_idx = self.order[new_idx]
            placed_indices = self.order[:new_idx]   
            first_outer = self._poly_outer_shapely(self.polys[self.order[0]])
            feasible = first_outer
           
            for ii, placed_idx in enumerate(placed_indices):
                i, j = placed_idx, new_global_idx
                nfp_poly = self.NFP[self.pid[i]][self.pid[j]]
                
                if not nfp_poly: 
                    nfp_poly = self._compute_nfp(self.original[i], self.original[j])
                    self.NFP[self.pid[i]][self.pid[j]] = nfp_poly                      
                    
                nfp_poly = affinity.translate(nfp_poly,dx1[ii], dy1[ii])
                nfp_poly = make_valid(nfp_poly)   
                feasible = make_valid(feasible)
                feasible = feasible.union(nfp_poly)
                
            
            candidates = self._get_feasible_points(feasible, new_global_idx)  
            if not candidates:   
                self.allscores = 99999
                break 

            best_pt = None
            self.best_score = float('inf')
            AAA = []
            for pt, is_interior in candidates:
               
                AAA.append(pt)
                score = self._evaluate_point(pt, new_global_idx, placed_indices)
                adjusted_score = score - 100 if is_interior else score
                if adjusted_score < self.best_score:
                    self.best_score = adjusted_score
                    best_pt = pt
                    best_raw_score = score 
     
            
            self.allscores += self.best_score
            if best_pt is None:
                self.allscores = 99999
                break                       
            dx2, dy2 = self._place_polygon(new_global_idx, best_pt)    
            dx1.append(dx2)
            dy1.append(dy2)
            self.score += best_raw_score                       
            self.outpolygon.append(copy.deepcopy(self.polys[new_global_idx]))
                
            
        for i in range(len(self.polys)):
            if i in self.order:   
                dx = self.ref_points[i][0] - self.pstart[i][0]
                dy = self.ref_points[i][1] - self.pstart[i][1]                
                self.pend[i] = [dx, dy]          
    
    def show_result(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))            
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.outpolygon)))      
        for idx, rings in enumerate(self.outpolygon):
            color = colors[idx]
            for ring in rings:
                if ring[0] != ring[-1]:
                    closed_ring = ring + [ring[0]]
                else:
                    closed_ring = ring
                xs, ys = zip(*closed_ring)
                plt.plot(xs, ys, color=color, linewidth=1.5)
        
        plt.axis('equal')
        plt.show()


class TOPOSONE:
    def __init__(self, original_polys, placed_order, placed_translations,
                 new_idx, NFP_history, container_width, step=2):

        self.original = copy.deepcopy(original_polys)
        self.placed_order = placed_order
        self.placed_trans = {idx: vec for idx, vec in zip(placed_order, placed_translations)}
        self.new_idx = new_idx
        self.NFP = NFP_history
        self.width = container_width
        self.step = step

        self.curr_left = float('inf')
        self.curr_right = -float('inf')
        self.curr_bottom = float('inf')
        self.curr_top = -float('inf')
        self.placed_shapely = []
        self.placed_bbox = []
        self.placed_ref_orig = []

        for idx in placed_order:
            dx, dy = self.placed_trans[idx]
            poly_rings = original_polys[idx]
            self.placed_ref_orig.append(poly_rings[0][0])

            parts = []
            for ring in poly_rings:
                closed = ring + [ring[0]] if ring[0] != ring[-1] else ring
                trans_ring = [[p[0] + dx, p[1] + dy] for p in closed]
                parts.append(Polygon(trans_ring))
            if len(parts) == 1:
                shp = parts[0]
            else:
                shp = unary_union(parts)
            self.placed_shapely.append(shp)

            bounds = shp.bounds
            self.curr_left = min(self.curr_left, bounds[0])
            self.curr_right = max(self.curr_right, bounds[2])
            self.curr_bottom = min(self.curr_bottom, bounds[1])
            self.curr_top = max(self.curr_top, bounds[3])
            self.placed_bbox.append(bounds)

        new_poly = original_polys[new_idx]
        all_x = np.concatenate([np.array(ring)[:, 0] for ring in new_poly])
        all_y = np.concatenate([np.array(ring)[:, 1] for ring in new_poly])
        self.new_bbox = [all_x.min(), all_x.max(), all_y.min(), all_y.max()]
        self.new_ref_orig = new_poly[0][0]
        self.new_width = self.new_bbox[1] - self.new_bbox[0]
        self.new_height = self.new_bbox[3] - self.new_bbox[2]

        parts_new = []
        for ring in new_poly:
            closed = ring + [ring[0]] if ring[0] != ring[-1] else ring
            parts_new.append(Polygon(closed))
        if len(parts_new) == 1:
            self.new_shapely_orig = parts_new[0]
        else:
            self.new_shapely_orig = unary_union(parts_new)

        self.success = False
        self.best_point = None
        self.best_raw_score = None
        self.updated_polys = None
        self.updated_translations = None
        self.updated_NFP = None
        self.final_width = None
        self.final_height = None

    def run(self):

        first_idx = self.placed_order[0]
        first_shp = self.placed_shapely[0]
        forbidden = first_shp

        for i, placed_idx in enumerate(self.placed_order):
            nfp_orig = self.NFP[placed_idx][self.new_idx]
            if not nfp_orig:
                nfp_orig = self._compute_nfp(self.original[placed_idx], self.original[self.new_idx])
                self.NFP[placed_idx][self.new_idx] = nfp_orig

            dx, dy = self.placed_trans[placed_idx]
            nfp_trans = affinity.translate(nfp_orig, dx, dy)
            nfp_trans = self._fix_geometry(nfp_trans)
            forbidden = forbidden.union(nfp_trans)
            forbidden = self._fix_geometry(forbidden)

 
        candidates = self._get_boundary_points(forbidden)
        if not candidates:
            self._set_failure()
            return False


        best_pt = None
        best_score = float('inf')
        best_raw = None

        for pt, is_interior in candidates:
            dx = pt[0] - self.new_ref_orig[0]
            dy = pt[1] - self.new_ref_orig[1]

            new_left = self.new_bbox[0] + dx
            new_right = self.new_bbox[1] + dx
            new_bottom = self.new_bbox[2] + dy
            new_top = self.new_bbox[3] + dy

            overall_left = min(self.curr_left, new_left)
            overall_right = max(self.curr_right, new_right)
            overall_bottom = min(self.curr_bottom, new_bottom)
            overall_top = max(self.curr_top, new_top)


            if overall_top - overall_bottom > self.width + 1e-6:
                continue

            moved_new = self._translate_shapely_poly(self.new_shapely_orig, dx, dy)
            moved_new = self._fix_geometry(moved_new)
            overlap_geom = False
            for shp in self.placed_shapely:
                try:
                    if moved_new.intersects(shp):
                        inter = moved_new.intersection(shp)
                        if inter.area > 0:
                            overlap_geom = True
                            break              
                except:
                    moved_new = set_precision(moved_new, 0.01)
                    shp = set_precision(shp, 0.01)
                    if moved_new.intersects(shp):
                        inter = moved_new.intersection(shp)
                        if inter.area > 0:  
                            overlap_geom = True
                            break

            if overlap_geom:
                continue

            overlap_sum = 0.0
            for placed_bbox in self.placed_bbox:
                ol = max(new_left, placed_bbox[0])
                o_r = min(new_right, placed_bbox[2])
                ob = max(new_bottom, placed_bbox[1])
                o_t = min(new_top, placed_bbox[3])
                if ol < o_r and ob < o_t:
                    overlap_sum += (o_r - ol) * (o_t - ob)

            Lnew = max(0, self.curr_left - new_left) + max(0, new_right - self.curr_right)
            Wnew = max(0, self.curr_bottom - new_bottom) + max(0, new_top - self.curr_top)
            area_new = self.new_width * self.new_height
            if area_new <= 0:
                area_new = 1e-6

            fun1 = Lnew / self.new_width if self.new_width > 0 else 0
            fun2 = overlap_sum / area_new
            fun3 = Lnew * Wnew / area_new
            raw_score = 0.2 * fun1 - 0.6 * fun2 + 0.2 * fun3
            adjusted_score = raw_score - 100 if is_interior else raw_score

            if adjusted_score < best_score:
                best_score = adjusted_score
                best_pt = pt
                best_raw = raw_score

        if best_pt is None:
            self._set_failure()
            return False


        self.best_point = best_pt
        self.best_raw_score = best_raw
        dx, dy = best_pt[0] - self.new_ref_orig[0], best_pt[1] - self.new_ref_orig[1]
        self.new_trans = [dx, dy]

        new_left = self.new_bbox[0] + dx
        new_right = self.new_bbox[1] + dx
        new_bottom = self.new_bbox[2] + dy
        new_top = self.new_bbox[3] + dy
        self.final_left = min(self.curr_left, new_left)
        self.final_right = max(self.curr_right, new_right)
        self.final_bottom = min(self.curr_bottom, new_bottom)
        self.final_top = max(self.curr_top, new_top)
        self.final_width = self.final_right - self.final_left   
        self.final_height = self.final_top - self.final_bottom

        self.updated_translations = dict(self.placed_trans)
        self.updated_translations[self.new_idx] = [dx, dy]

        all_indices = self.placed_order + [self.new_idx]
        self.updated_polys = []
        for idx in all_indices:
            poly_rings = self.original[idx]
            dx_i, dy_i = self.updated_translations[idx]
            translated_rings = []
            for ring in poly_rings:
                new_ring = [[p[0] + dx_i, p[1] + dy_i] for p in ring]
                translated_rings.append(new_ring)
            self.updated_polys.append(translated_rings)

        self.updated_NFP = self.NFP
        self.success = True
        return True

    def _compute_nfp(self, polyA_rings, polyB_rings):
        result = NFPfun.NonConvex(polyA_rings, polyB_rings)
        return result.nfp

    def _translate_shapely_poly(self, poly, dx, dy):
        if poly.is_empty:
            return poly
        if poly.geom_type == 'Polygon':
            new_coords = np.array(poly.exterior.coords) + [dx, dy]
            interiors = [np.array(ring.coords) + [dx, dy] for ring in poly.interiors]
            return Polygon(new_coords, interiors)
        elif poly.geom_type == 'MultiPolygon':
            new_polys = [self._translate_shapely_poly(p, dx, dy) for p in poly.geoms]
            return MultiPolygon(new_polys)
        else:
            return poly

    def _fix_geometry(self, geom):
        try:
            return set_precision(geom, 0.01)
        except:
            return geom.buffer(0)

    def _get_boundary_points(self, geom):
        points = []
        seen = set()

        def add_point(x, y, interior_flag):
            key = f"{round(x,3)},{round(y,3)}"
            if key not in seen:
                seen.add(key)
                points.append(([round(x,2), round(y,2)], interior_flag))

        def process_ring(ring, interior_flag):
            coords = np.array(ring.coords)
            n = len(coords)
            for i in range(n-1):
                x1, y1 = coords[i]
                x2, y2 = coords[i+1]
                add_point(x1, y1, interior_flag)

                dx = x2 - x1
                dy = y2 - y1
                length = np.hypot(dx, dy)
                if length > self.step:
                    num = int(length / self.step)
                    for k in range(1, num):
                        t = k / num
                        x = x1 + t * dx
                        y = y1 + t * dy
                        add_point(x, y, interior_flag)

        if geom.geom_type == 'Polygon':
            process_ring(geom.exterior, interior_flag=False)
            for interior in geom.interiors:
                process_ring(interior, interior_flag=True)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                process_ring(poly.exterior, interior_flag=False)
                for interior in poly.interiors:
                    process_ring(interior, interior_flag=True)
        return points

    def _set_failure(self):
        self.success = False
        self.best_raw_score = 9999
        self.final_width = 9999
        self.final_height = 9999


def plot_polygons(polys_rings, title="Placed Polygons"):
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(polys_rings)))
    for idx, rings in enumerate(polys_rings):
        color = colors[idx]
        for ring in rings:
            if ring[0] != ring[-1]:
                closed_ring = ring + [ring[0]]
            else:
                closed_ring = ring
            xs, ys = zip(*closed_ring)
            plt.fill(xs, ys, color=color, alpha=0.5, edgecolor='black', linewidth=1)
            plt.plot(xs, ys, color='black', linewidth=0.8)
    plt.axis('equal')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

class LowerLeftpart:
    def __init__(self, original_polys, listply, width, NFP_history, step, pid, position=None, length=None, cached_state=None):
        self.original_polys = original_polys
        self.listply = listply
        self.width = width
        self.NFP_history = NFP_history          
        self.pid = pid
        
        if cached_state is not None:
            self.position = copy.deepcopy(cached_state['position'])
            self.current_length = cached_state['current_length']
            self.topospos = copy.deepcopy(cached_state['topospos'])
            self.placed = [idx for comb in listply[:-1] for idx in comb]
            last_idx = len(listply) - 1
            comb_indices = listply[last_idx]
            topos = TOPOS(original_polys, comb_indices, width, NFP_history, step, pid)
            if topos.allscores > 9999:
                self.current_length = 9999
                self.outposition = copy.deepcopy(self.position)
                self.outlength = self.current_length
                return
            
            for pp in comb_indices:
                self.topospos[pp] = copy.deepcopy(topos.pend[pp])
            placed_parts = topos.outpolygon
            ref_pt = original_polys[comb_indices[0]][0][0]
            offsets = []
            for part in placed_parts:
                max_y_part = -1e9
                max_pt = None
                for ring in part:
                    for pt in ring:
                        if pt[1] > max_y_part:
                            max_y_part = pt[1]
                            max_pt = pt
                offsets.append([max_pt[0]-ref_pt[0], max_pt[1]-ref_pt[1]] if max_pt else [0.0, 0.0])
            all_x = [pt[0] for part in placed_parts for ring in part for pt in ring]
            all_y = [pt[1] for part in placed_parts for ring in part for pt in ring]
            if not all_x:
                bbox = [0.0, 0.0, 0.0, 0.0]
            else:
                bbox = [min(all_x)-ref_pt[0], max(all_x)-ref_pt[0],
                        min(all_y)-ref_pt[1], max(all_y)-ref_pt[1]]
            
            last_comb = {
                'indices': comb_indices,
                'parts': placed_parts,
                'offsets': offsets,
                'ref_pt': ref_pt,
                'bbox': bbox
            }
            res = self._place_combination(last_comb, last_idx)
            self.outposition = copy.deepcopy(self.position)
            self.outlength = self.current_length
            
        else:
            if not position:
                self.position = [[] for _ in range(len(original_polys))]
                self.placed = []
            else:
                self.position = copy.deepcopy(position)
                self.placed = [idx for comb in listply[:-1] for idx in comb] if listply else []
            
            self.current_length = 0.0
            self.topospos = [[] for _ in range(len(original_polys))]
            self.comb_data = []
            self.toposscore = []
            
            for comb_idx, comb_indices in enumerate(listply):
                topos = TOPOS(original_polys, comb_indices, width, NFP_history, step, pid)
                if topos.allscores > 9999:
                    self.current_length = 9999
                    return
                self.toposscore.append(topos.allscores)
                placed_parts = topos.outpolygon
                toposres1 = topos.pend
                for pp in comb_indices:
                    self.topospos[pp] = copy.deepcopy(toposres1[pp])
                
                ref_pt = self.original_polys[comb_indices[0]][0][0]
                offsets = []
                for part in placed_parts:
                    max_y_part = -1e9
                    max_pt = None
                    for ring in part:
                        for pt in ring:
                            if pt[1] > max_y_part:
                                max_y_part = pt[1]
                                max_pt = pt
                    offsets.append([max_pt[0]-ref_pt[0], max_pt[1]-ref_pt[1]] if max_pt else [0.0, 0.0])
                
                all_x = [pt[0] for part in placed_parts for ring in part for pt in ring]
                all_y = [pt[1] for part in placed_parts for ring in part for pt in ring]
                if not all_x:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                else:
                    bbox = [min(all_x)-ref_pt[0], max(all_x)-ref_pt[0],
                            min(all_y)-ref_pt[1], max(all_y)-ref_pt[1]]
                
                comb_data = {
                    'indices': comb_indices,
                    'parts': placed_parts,
                    'offsets': offsets,
                    'ref_pt': ref_pt,
                    'bbox': bbox
                }
                self.comb_data.append(comb_data)
                res = self._place_combination(comb_data, comb_idx)
                if comb_idx == len(listply) - 2:
                    self.cached_state = {
                        'position': copy.deepcopy(self.position),
                        'current_length': self.current_length,
                        'topospos': copy.deepcopy(self.topospos)
                    }
                    self.outposition = copy.deepcopy(self.position)
                    self.outlength = self.current_length
                
                if not res:
                    break

    def _place_combination(self, comb, comb_idx):
        left, right, bottom, top = comb['bbox']
        if not self.placed:
            target_pt = [-left, -bottom]
        else:
            INF = 1e3
            x_min_allowed = -left
            y_min_allowed = -bottom
            y_max_allowed = self.width - top
            if y_max_allowed < y_min_allowed:
                self.current_length = 99999
                return False
            init_rect = Polygon([
                [x_min_allowed, y_min_allowed],
                [INF, y_min_allowed],
                [INF, y_max_allowed],
                [x_min_allowed, y_max_allowed]
            ])
            feasible = init_rect
            nfp_poly = self._combination_nfp(comb)
            feasible1 = feasible
            feasible = feasible.difference(nfp_poly)
            if feasible.is_empty:
                minx, miny, maxx, maxy = feasible1.bounds
                feasible = Polygon([
                    [minx, miny],
                    [maxx, miny],
                    [maxx, miny + 0.01],
                    [minx, miny + 0.01]
                ])
                feasible = feasible.difference(nfp_poly)
            pts = []
            if feasible.geom_type == 'Polygon':
                for coord in feasible.exterior.coords[:-1]:
                    pts.append([coord[0], coord[1]])
                for interior in feasible.interiors:
                    for coord in interior.coords[:-1]:
                        pts.append([coord[0], coord[1]])
            elif feasible.geom_type == 'MultiPolygon':
                for poly in feasible.geoms:
                    for coord in poly.exterior.coords[:-1]:
                        pts.append([coord[0], coord[1]])
                    for interior in poly.interiors:
                        for coord in interior.coords[:-1]:
                            pts.append([coord[0], coord[1]])
            elif feasible.geom_type == 'LineString':
                for coord in feasible.coords[:-1]:
                    pts.append([coord[0], coord[1]])
            elif feasible.geom_type == 'MultiLineString':
                for line in feasible.geoms:
                    for coord in line.coords[:-1]:
                        pts.append([coord[0], coord[1]])
            
            if not pts:
                raise RuntimeError(f" {comb['indices']}")
            pts.sort(key=lambda p: (p[0], p[1]))
            target_pt = pts[0]
        
        dx = target_pt[0] - comb['ref_pt'][0]
        dy = target_pt[1] - comb['ref_pt'][1]
        
        placed_parts = []
        for part in comb['parts']:
            new_part = []
            for ring in part:
                new_ring = [[pt[0] + dx, pt[1] + dy] for pt in ring]
                new_part.append(new_ring)
            placed_parts.append(new_part)
        
        self.placed += self.listply[comb_idx]
        for i in self.listply[comb_idx]:
            ddx = dx + self.topospos[i][0]
            ddy = dy + self.topospos[i][1]
            self.position[i] = [ddx, ddy]
        
        for part in placed_parts:
            for ring in part:
                for pt in ring:
                    if pt[0] > self.current_length:
                        self.current_length = pt[0]
        return True

    def _combination_nfp(self, move_comb):
        moveindice = move_comb['indices']
        nfp_union = None
        MM = moveindice[0]
        for i in self.placed:
            for j in moveindice:
                nfp_orig = self.NFP_history[self.pid[i]][self.pid[j]]
                if not nfp_orig:
                    nfp_orig = self._compute_part_nfp(i, j)
                    self.NFP_history[self.pid[i]][self.pid[j]] = nfp_orig
                
                shift_x = self.position[i][0] - (self.topospos[j][0] + self.original_polys[j][0][0][0] - self.original_polys[MM][0][0][0])
                shift_y = self.position[i][1] - (self.topospos[j][1] + self.original_polys[j][0][0][1] - self.original_polys[MM][0][0][1])
                
                nfp_trans = affinity.translate(nfp_orig, shift_x, shift_y)
                
                if nfp_union is None:
                    nfp_union = nfp_trans
                else:
                    try:
                        nfp_union = nfp_union.union(nfp_trans)
                    except:
                        nfp_union = set_precision(nfp_union, 0.01)
                        nfp_trans = set_precision(nfp_trans, 0.01)
                        nfp_union = nfp_union.union(nfp_trans)
        return nfp_union if nfp_union is not None else Polygon()

    def _compute_part_nfp(self, idx_a, idx_b):
        polyA = self.original_polys[idx_a]
        polyB = self.original_polys[idx_b]
        nfp_union = NFPfun.NonConvex(polyA, polyB)
        return nfp_union.nfp






class LowerLeft1:
    def __init__(self, original_polys, listply, width, NFP_history,step, pid, showplt = False):
        self.original_polys = original_polys
        self.listply = listply
        self.width = width
        self.NFP_history = NFP_history
        self.placed = []  
        self.pid = pid       
        self.position = [[]for i in range(len(original_polys))]              
        self.placed_polys_full = []     
        self.current_length = 0.0
        self.topospos = [[]for i in range(len(original_polys))] 
        self.comb_data = []
        self.toposscore = []
        for comb_indices in listply:  
            
            topos = TOPOS(original_polys, comb_indices, width, NFP_history, step, pid)
            if topos.allscores > 9999:
                self.current_length = 9999                
                return 
            self.toposscore.append(topos.allscores)          
            placed_parts = topos.outpolygon  
            toposres1 = topos.pend
            for pp in comb_indices:
                self.topospos[pp] = copy.deepcopy(toposres1[pp])
            ref_pt = self.original_polys[comb_indices[0]][0][0]
            offsets = []
            for part in placed_parts:
                max_y_part = -1e9
                max_pt = None
                for ring in part:
                    for pt in ring:
                        if pt[1] > max_y_part:
                            max_y_part = pt[1]
                            max_pt = pt
                offsets.append([max_pt[0]-ref_pt[0], max_pt[1]-ref_pt[1]] if max_pt else [0.0, 0.0])
            all_x = [pt[0] for part in placed_parts for ring in part for pt in ring]
            all_y = [pt[1] for part in placed_parts for ring in part for pt in ring]
            if not all_x:
                bbox = [0.0, 0.0, 0.0, 0.0]
            else:
                bbox = [min(all_x)-ref_pt[0], max(all_x)-ref_pt[0],
                        min(all_y)-ref_pt[1], max(all_y)-ref_pt[1]]
            self.comb_data.append({
                'indices': comb_indices,
                'parts': placed_parts,        
                'offsets': offsets,             
                'ref_pt': ref_pt,                  
                'bbox': bbox                      
            })
        for i in range(len(self.comb_data)):         
            if not self._place_combination(i):
                break 
        if showplt:
            self.show_result()
         
    def _place_combination(self, comb_idx):
        comb = self.comb_data[comb_idx]
        left, right, bottom, top = comb['bbox']
        if not self.placed:
            target_pt = [-left, -bottom]
        else:
            INF = 1e3
            x_min_allowed = -left                  
            y_min_allowed = -bottom                    
            y_max_allowed = self.width - top          
            if y_max_allowed < y_min_allowed:
                self.current_length = 99999
                return False
            init_rect = Polygon([
                [x_min_allowed, y_min_allowed],
                [INF, y_min_allowed],
                [INF, y_max_allowed],
                [x_min_allowed, y_max_allowed]
            ])
            feasible = init_rect            
            nfp_poly = self._combination_nfp(comb)
            feasible1 = feasible
            feasible = feasible.difference(nfp_poly)
            if feasible.is_empty:
                minx, miny, maxx, maxy = feasible1.bounds
                feasible = Polygon([
                    [minx, miny],
                    [maxx, miny],
                    [maxx, miny + 0.01],
                    [minx, miny + 0.01]
                ])
                feasible = feasible.difference(nfp_poly)    
            pts = []
            if feasible.geom_type == 'Polygon':
                for coord in feasible.exterior.coords[:-1]:
                    pts.append([coord[0], coord[1]])
                for interior in feasible.interiors:
                    for coord in interior.coords[:-1]:
                        pts.append([coord[0], coord[1]])
            elif feasible.geom_type == 'MultiPolygon':
                for poly in feasible.geoms:
                    for coord in poly.exterior.coords[:-1]:
                        pts.append([coord[0], coord[1]])
                    for interior in poly.interiors:
                        for coord in interior.coords[:-1]:
                            pts.append([coord[0], coord[1]])
            elif feasible.geom_type == 'LineString':
                for coord in feasible.coords[:-1]:
                    pts.append([coord[0], coord[1]])
            elif feasible.geom_type == 'MultiLineString':
                for line in feasible.geoms:
                    for coord in line.coords[:-1]:     
                        pts.append([coord[0], coord[1]])                
                
            if not pts:
                raise RuntimeError(f"{comb['indices']}")
            pts.sort(key=lambda p: (p[0], p[1]))
            target_pt = pts[0]
            
        dx = target_pt[0] - comb['ref_pt'][0]
        dy = target_pt[1] - comb['ref_pt'][1]
        
        
        
        placed_parts = []
        for part in comb['parts']:
            new_part = []
            for ring in part:
                new_ring = [[pt[0]+dx, pt[1]+dy] for pt in ring]
                new_part.append(new_ring)
            placed_parts.append(new_part)

        self.placed += self.listply[comb_idx]
        for i in self.listply[comb_idx]:
            ddx = dx + self.topospos[i][0]
            ddy = dy + self.topospos[i][1]     
            self.position[i] = [ddx, ddy]


        self.placed_polys_full.extend(placed_parts)

        for part in placed_parts:
            for ring in part:
                for pt in ring:
                    if pt[0] > self.current_length:
                        self.current_length = pt[0]
        return True
    def _combination_nfp(self, move_comb):
        moveindice = move_comb['indices']
        nfp_union = None
        MM = moveindice[0]
        for i in self.placed:            
            for j in moveindice:  
                nfp_orig = self.NFP_history[self.pid[i]][self.pid[j]]

                if not nfp_orig :
                    nfp_orig = self._compute_part_nfp(i, j)
                    self.NFP_history[self.pid[i]][self.pid[j]] = nfp_orig
                    
                
                shift_x = self.position[i][0] - (self.topospos[j][0] + self.original_polys[j][0][0][0]  - self.original_polys[MM][0][0][0])
                shift_y = self.position[i][1] - (self.topospos[j][1] + self.original_polys[j][0][0][1]  - self.original_polys[MM][0][0][1])                         
                             
                nfp_trans = affinity.translate(nfp_orig, shift_x, shift_y)

                if nfp_union is None:
                    nfp_union = nfp_trans
                else:
                    try:
                        nfp_union = nfp_union.union(nfp_trans) 
                    except:
                        nfp_union = set_precision(nfp_union, 0.01)
                        nfp_trans = set_precision(nfp_trans, 0.01)
                        nfp_union = nfp_union.union(nfp_trans) 
                   
        return nfp_union if nfp_union is not None else Polygon()

    def _compute_part_nfp(self, idx_a, idx_b):
        polyA = self.original_polys[idx_a]
        polyB = self.original_polys[idx_b]
        nfp_union = NFPfun.NonConvex(polyA, polyB)
        return nfp_union.nfp

    def get_result(self):
        return self.placed_polys_full, self.current_length
    
    def show_result(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.placed_polys_full)))

        for idx, rings in enumerate(self.placed_polys_full):
            color = colors[idx]
            for ring in rings:
                if ring[0] != ring[-1]:
                    closed_ring = ring + [ring[0]]
                else:
                    closed_ring = ring
                xs, ys = zip(*closed_ring)
                plt.plot(xs, ys, color=color, linewidth=1.5)   
        plt.show()




class LowerLeft:
    def __init__(self, original_polys, listply, width, NFP_history, step, pid, showplt=False,
                 cached_state=None, cached_index=None):
        self.original_polys = original_polys
        self.listply = listply
        self.width = width
        self.NFP_history = NFP_history
        self.step = step
        self.pid = pid
        self.showplt = showplt

        num_polys = len(original_polys)

        if cached_index is not None and cached_state is not None:
            num_parts = sum(len(comb) for comb in listply[:cached_index+1])
            placed_full = copy.deepcopy(cached_state['placed'])
            self.placed = placed_full[:num_parts]
            self.position = copy.deepcopy(cached_state['position'])
            self.topospos = copy.deepcopy(cached_state['topospos'])
            placed_polys_full_full = copy.deepcopy(cached_state['placed_polys_full'])
            self.placed_polys_full = placed_polys_full_full[:num_parts]

            self.current_length = 0.0
            for part in self.placed_polys_full:
                for ring in part:
                    for pt in ring:
                        if pt[0] > self.current_length:
                            self.current_length = pt[0]

            if len(self.position) < num_polys:
                self.position.extend([[] for _ in range(num_polys - len(self.position))])
            if len(self.topospos) < num_polys:
                self.topospos.extend([[] for _ in range(num_polys - len(self.topospos))])

            if len(cached_state['comb_data']) <= cached_index:
                raise ValueError(" error")
            self.comb_data = copy.deepcopy(cached_state['comb_data'][:cached_index+1])
            self.toposscore = copy.deepcopy(cached_state.get('toposscore', []))[:cached_index+1]

            remaining_comb_indices = listply[cached_index+1:]
            for comb_indices in remaining_comb_indices:
                topos = TOPOS(original_polys, comb_indices, width, NFP_history, step, pid)
                if topos.allscores > 9999:
                    self.current_length = 9999
                    self.outstate = self._build_outstate()
                    return
                self.toposscore.append(topos.allscores)
                placed_parts = topos.outpolygon
                toposres1 = topos.pend
                for pp in comb_indices:
                    self.topospos[pp] = copy.deepcopy(toposres1[pp])

                ref_pt = original_polys[comb_indices[0]][0][0]
                offsets = []
                for part in placed_parts:
                    max_y_part = -1e9
                    max_pt = None
                    for ring in part:
                        for pt in ring:
                            if pt[1] > max_y_part:
                                max_y_part = pt[1]
                                max_pt = pt
                    offsets.append([max_pt[0]-ref_pt[0], max_pt[1]-ref_pt[1]] if max_pt else [0.0, 0.0])

                all_x = [pt[0] for part in placed_parts for ring in part for pt in ring]
                all_y = [pt[1] for part in placed_parts for ring in part for pt in ring]
                if not all_x:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                else:
                    bbox = [min(all_x)-ref_pt[0], max(all_x)-ref_pt[0],
                            min(all_y)-ref_pt[1], max(all_y)-ref_pt[1]]

                comb_data_item = {
                    'indices': comb_indices,
                    'parts': placed_parts,
                    'offsets': offsets,
                    'ref_pt': ref_pt,
                    'bbox': bbox
                }
                self.comb_data.append(comb_data_item)

            start_idx = cached_index
        else:
            self.placed = []
            self.position = [[] for _ in range(num_polys)]
            self.topospos = [[] for _ in range(num_polys)]
            self.placed_polys_full = []
            self.current_length = 0.0
            self.comb_data = []
            self.toposscore = []
            start_idx = -1

            for comb_indices in listply:
                topos = TOPOS(original_polys, comb_indices, width, NFP_history, step, pid)
                if topos.allscores > 9999:
                    self.current_length = 9999
                    self.outstate = self._build_outstate()
                    return
                self.toposscore.append(topos.allscores)
                placed_parts = topos.outpolygon
                toposres1 = topos.pend
                for pp in comb_indices:
                    self.topospos[pp] = copy.deepcopy(toposres1[pp])

                ref_pt = original_polys[comb_indices[0]][0][0]
                offsets = []
                for part in placed_parts:
                    max_y_part = -1e9
                    max_pt = None
                    for ring in part:
                        for pt in ring:
                            if pt[1] > max_y_part:
                                max_y_part = pt[1]
                                max_pt = pt
                    offsets.append([max_pt[0]-ref_pt[0], max_pt[1]-ref_pt[1]] if max_pt else [0.0, 0.0])

                all_x = [pt[0] for part in placed_parts for ring in part for pt in ring]
                all_y = [pt[1] for part in placed_parts for ring in part for pt in ring]
                if not all_x:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                else:
                    bbox = [min(all_x)-ref_pt[0], max(all_x)-ref_pt[0],
                            min(all_y)-ref_pt[1], max(all_y)-ref_pt[1]]

                comb_data_item = {
                    'indices': comb_indices,
                    'parts': placed_parts,
                    'offsets': offsets,
                    'ref_pt': ref_pt,
                    'bbox': bbox
                }
                self.comb_data.append(comb_data_item)

        total_combs = len(self.comb_data)
        for i in range(start_idx+1, total_combs):
            if not self._place_combination(i):
                break

        self.outstate = self._build_outstate()

        if showplt:
            self.show_result()

    def _build_outstate(self):
        return {
            'placed': copy.deepcopy(self.placed),
            'position': copy.deepcopy(self.position),
            'topospos': copy.deepcopy(self.topospos),
            'placed_polys_full': copy.deepcopy(self.placed_polys_full),
            'current_length': self.current_length,
            'comb_data': copy.deepcopy(self.comb_data),
            'toposscore': copy.deepcopy(self.toposscore)
        }
    

    def _place_combination(self, comb_idx):
        comb = self.comb_data[comb_idx]
        current_indices = comb['indices']
        left, right, bottom, top = comb['bbox']
        if not self.placed:
            target_pt = [-left, -bottom]
        else:
            INF = 1e3
            x_min_allowed = -left
            y_min_allowed = -bottom
            y_max_allowed = self.width - top
            if y_max_allowed < y_min_allowed:
                self.current_length = 99999
                return False
            init_rect = Polygon([
                [x_min_allowed, y_min_allowed],
                [INF, y_min_allowed],
                [INF, y_max_allowed],
                [x_min_allowed, y_max_allowed]
            ])
            feasible = init_rect
            nfp_poly = self._combination_nfp(comb)
            feasible1 = feasible
            feasible = feasible.difference(nfp_poly)
            if feasible.is_empty:
                minx, miny, maxx, maxy = feasible1.bounds
                feasible = Polygon([
                    [minx, miny],
                    [maxx, miny],
                    [maxx, miny + 0.01],
                    [minx, miny + 0.01]
                ])
                feasible = feasible.difference(nfp_poly)
            pts = []
            if feasible.geom_type == 'Polygon':
                for coord in feasible.exterior.coords[:-1]:
                    pts.append([coord[0], coord[1]])
                for interior in feasible.interiors:
                    for coord in interior.coords[:-1]:
                        pts.append([coord[0], coord[1]])
            elif feasible.geom_type == 'MultiPolygon':
                for poly in feasible.geoms:
                    for coord in poly.exterior.coords[:-1]:
                        pts.append([coord[0], coord[1]])
                    for interior in poly.interiors:
                        for coord in interior.coords[:-1]:
                            pts.append([coord[0], coord[1]])
            elif feasible.geom_type == 'LineString':
                for coord in feasible.coords[:-1]:
                    pts.append([coord[0], coord[1]])
            elif feasible.geom_type == 'MultiLineString':
                for line in feasible.geoms:
                    for coord in line.coords[:-1]:
                        pts.append([coord[0], coord[1]])

            if not pts:
                raise RuntimeError(f" {comb['indices']} ")
            pts.sort(key=lambda p: (p[0], p[1]))
            target_pt = pts[0]

        dx = target_pt[0] - comb['ref_pt'][0]
        dy = target_pt[1] - comb['ref_pt'][1]

        placed_parts = []
        for part in comb['parts']:
            new_part = []
            for ring in part:
                new_ring = [[pt[0]+dx, pt[1]+dy] for pt in ring]
                new_part.append(new_ring)
            placed_parts.append(new_part)

        # 使用current_indices更新状态
        self.placed.extend(current_indices)
        for i in current_indices:
            ddx = dx + self.topospos[i][0]
            ddy = dy + self.topospos[i][1]
            self.position[i] = [ddx, ddy]

        self.placed_polys_full.extend(placed_parts)

        for part in placed_parts:
            for ring in part:
                for pt in ring:
                    if pt[0] > self.current_length:
                        self.current_length = pt[0]
        return True

    # 以下方法保持不变
    def _combination_nfp(self, move_comb):
        moveindice = move_comb['indices']
        nfp_union = None
        MM = moveindice[0]
        for i in self.placed:
            for j in moveindice:
                nfp_orig = self.NFP_history[self.pid[i]][self.pid[j]]
                if not nfp_orig:
                    nfp_orig = self._compute_part_nfp(i, j)
                    self.NFP_history[self.pid[i]][self.pid[j]] = nfp_orig

                shift_x = self.position[i][0] - (self.topospos[j][0] + self.original_polys[j][0][0][0] - self.original_polys[MM][0][0][0])
                shift_y = self.position[i][1] - (self.topospos[j][1] + self.original_polys[j][0][0][1] - self.original_polys[MM][0][0][1])

                nfp_trans = affinity.translate(nfp_orig, shift_x, shift_y)

                if nfp_union is None:
                    nfp_union = nfp_trans
                else:
                    try:
                        nfp_union = nfp_union.union(nfp_trans)
                    except:
                        nfp_union = set_precision(nfp_union, 0.01)
                        nfp_trans = set_precision(nfp_trans, 0.01)
                        nfp_union = nfp_union.union(nfp_trans)

        return nfp_union if nfp_union is not None else Polygon()

    def _compute_part_nfp(self, idx_a, idx_b):
        polyA = self.original_polys[idx_a]
        polyB = self.original_polys[idx_b]
        nfp_union = NFPfun.NonConvex(polyA, polyB)
        return nfp_union.nfp
    def get_result(self):
        return self.placed_polys_full, self.current_length

    def show_result(self):
        plt.figure(figsize=(8, 6))
        max_y = 0
        for rings in self.placed_polys_full:
            for ring in rings:
                for (x, y) in ring:
                    if y > max_y:
                        max_y = y
        container_height = max_y
        container_width = self.current_length
        for rings in self.placed_polys_full:
            for ring in rings:
                if ring[0] != ring[-1]:
                    closed_ring = ring + [ring[0]]
                else:
                    closed_ring = ring
                xs, ys = zip(*closed_ring)
                plt.fill(xs, ys, color='#FFD8B5', edgecolor='darkorange',
                         linewidth=1.5, alpha=0.7)
        rect_x = [0, container_width, container_width, 0, 0]
        rect_y = [0, 0, container_height, container_height, 0]
        plt.plot(rect_x, rect_y, 'k-', linewidth=2.5)
        plt.show()
    


