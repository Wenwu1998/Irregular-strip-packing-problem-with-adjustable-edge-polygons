import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import Packing
import Tool
import time
class BS_DA:
    def __init__(self, original_polys, repoly, NumR, width, showplt=False):
        polys_rot, KX, KY = Tool.rotate_all(original_polys, repoly, NumR)
        upbound = Tool.getup(repoly)
        moves = Tool.Different_area(original_polys, repoly, KX, KY,upbound)
        for i in range(NumR):
            p = [copy.deepcopy(polys_rot[k][i]) for k in range(len(polys_rot))]
            X = [[i for k in range(len(polys_rot))], moves, []]
            p = Tool.generatepoly(p, X, repoly, KX, KY)
            for k in range(len(polys_rot)):
                polys_rot[k][i] = copy.deepcopy(p[k])
        self.Time = time.time()
        self.parts_rot = polys_rot                    
        self.num_parts = len(self.parts_rot)          
        self.num_rots = NumR
        self.width = width
        self.polys_flat = [self.parts_rot[part][rot] for part in range(self.num_parts) for rot in range(self.num_rots)]
        self.num_flat = self.num_parts * self.num_rots
        self.part_areas = [self._compute_area(self.parts_rot[p][0]) for p in range(self.num_parts)]
        self.NFP_history = [[[] for _ in range(self.num_flat)] for _ in range(self.num_flat)]
        self.beam_width = 2
        self.filter_width = 3
        self.step  = 1
        self.best_solution = None         
        self.search()
        self.Time = time.time() - self.Time
        if showplt:
            self.draw_result()

    class Node:
        def __init__(self, placed_parts: List[int], placed_rots: List[int], offsets: List[Tuple[float, float]]):
            self.placed_parts = placed_parts
            self.placed_rots = placed_rots
            self.offsets = offsets

        def copy(self):
            return BeamSearch.Node(
                copy.deepcopy(self.placed_parts),
                copy.deepcopy(self.placed_rots),
                copy.deepcopy(self.offsets)
            )

        def get_flat_placed(self, num_rots):
            return [p * num_rots + r for p, r in zip(self.placed_parts, self.placed_rots)]


    def _compute_area(self, poly_rings) -> float:
        parts = []
        for ring in poly_rings:
            closed = ring + [ring[0]] if ring[0] != ring[-1] else ring
            parts.append(Polygon(closed))
        if len(parts) == 1:
            poly = parts[0]
        else:
            poly = unary_union(parts)
        return poly.area

    def _flat_idx(self, part, rot):
        return part * self.num_rots + rot


    def _local_evaluation(self, node: Node, next_part: int, next_rot: int) -> Optional[Tuple[float, Tuple[float, float]]]:
        flat_placed = node.get_flat_placed(self.num_rots)
        placed_trans = node.offsets
        cand_flat = self._flat_idx(next_part, next_rot)


        placer = Packing.TOPOSONE(
            original_polys=self.polys_flat,
            placed_order=flat_placed,
            placed_translations=placed_trans,
            new_idx=cand_flat,
            NFP_history=self.NFP_history,
            container_width=self.width,
            step=self.step 
        )
        success = placer.run()
        if not success:
            return None
        return placer.best_raw_score, placer.new_trans


    def _global_evaluation(self, node: Node, next_part: int, next_rot: int, offset: Tuple[float, float]) -> Optional[Tuple[float, List[Tuple[float, float]]]]:

        placed_parts = node.placed_parts + [next_part]
        placed_rots = node.placed_rots + [next_rot]
        placed_trans = node.offsets + [offset]
        nfp_history = copy.deepcopy(self.NFP_history)
        placed_set = set(placed_parts)
        remaining_parts = [p for p in range(self.num_parts) if p not in placed_set]
        remaining_parts.sort(key=lambda p: self.part_areas[p])
        for part in remaining_parts:
            best_score = float('inf')
            best_offset = None
            best_rot = None
            for rot in range(self.num_rots):
                cand_flat = self._flat_idx(part, rot)
                flat_placed = [self._flat_idx(p, r) for p, r in zip(placed_parts, placed_rots)]
                placer = Packing.TOPOSONE(
                    original_polys=self.polys_flat,
                    placed_order=flat_placed,
                    placed_translations=placed_trans,
                    new_idx=cand_flat,
                    NFP_history=nfp_history,
                    container_width=self.width,
                    step=self.step 
                )
                if placer.run():
                    if placer.best_raw_score < best_score:
                        best_score = placer.best_raw_score
                        best_offset = placer.new_trans
                        best_rot = rot
            if best_offset is None:
                return None   
            placed_parts.append(part)
            placed_rots.append(best_rot)
            placed_trans.append(best_offset)

        all_left = float('inf')
        all_right = -float('inf')
        for (part, rot, (dx, dy)) in zip(placed_parts, placed_rots, placed_trans):
            rings = self.parts_rot[part][rot]
            for ring in rings:
                closed = ring + [ring[0]] if ring[0] != ring[-1] else ring
                x_coords = [p[0] + dx for p in closed]
                all_left = min(all_left, min(x_coords))
                all_right = max(all_right, max(x_coords))
        final_length = all_right - all_left
        return final_length, placed_trans


    def search(self):
        first_part = max(range(self.num_parts), key=lambda p: self.part_areas[p])

        root_nodes = []
        for rot in range(self.num_rots):
            node = self.Node([first_part], [rot], [(0.0, 0.0)])
            root_nodes.append(node)

        current_nodes = root_nodes
        placed_count = 1
        total_parts = self.num_parts

        while placed_count < total_parts and current_nodes:
            # print(placed_count)
            all_candidates = []

            for node in current_nodes:
                placed_set = set(node.placed_parts)
                remaining = [p for p in range(self.num_parts) if p not in placed_set]
                if not remaining:
                    continue
                next_part = max(remaining, key=lambda p: self.part_areas[p])

                for rot in range(self.num_rots):
                    res = self._local_evaluation(node, next_part, rot)
                    if res is not None:
                        score, offset = res
                        all_candidates.append((score, offset, node, next_part, rot))

            if not all_candidates:
                break

            node_to_cands = {}
            for score, offset, node, part, rot in all_candidates:
                node_id = id(node)
                node_to_cands.setdefault(node_id, []).append((score, offset, node, part, rot))

            filtered_candidates = []
            for node_id, cand_list in node_to_cands.items():
                cand_list.sort(key=lambda x: x[0])
                filtered_candidates.extend(cand_list[:self.filter_width])

            if not filtered_candidates:
                break

            global_scores = []
            for score_local, offset, parent_node, part, rot in filtered_candidates:
                res = self._global_evaluation(parent_node, part, rot, offset)
                if res is not None:
                    length, full_offsets = res
                    global_scores.append((length, full_offsets, parent_node, part, rot))

            if not global_scores:
                break

            global_scores.sort(key=lambda x: x[0])
            best_globals = global_scores[:self.beam_width]

            next_nodes = []
            for length, full_offsets, parent_node, part, rot in best_globals:
                new_parts = parent_node.placed_parts + [part]
                new_rots = parent_node.placed_rots + [rot]
                new_offsets = full_offsets[:len(new_parts)]
                new_node = self.Node(new_parts, new_rots, new_offsets)
                next_nodes.append(new_node)

            current_nodes = next_nodes
            placed_count += 1

        if current_nodes:
            final_scores = []
            for node in current_nodes:
                if len(node.placed_parts) == total_parts:
                    all_left = float('inf')
                    all_right = -float('inf')
                    for (part, rot, (dx, dy)) in zip(node.placed_parts, node.placed_rots, node.offsets):
                        rings = self.parts_rot[part][rot]
                        for ring in rings:
                            closed = ring + [ring[0]] if ring[0] != ring[-1] else ring
                            x_coords = [p[0] + dx for p in closed]
                            all_left = min(all_left, min(x_coords))
                            all_right = max(all_right, max(x_coords))
                    length = all_right - all_left
                    final_scores.append((length, node))
            if final_scores:
                final_scores.sort(key=lambda x: x[0])
                best_length, best_node = final_scores[0]
                self.best_solution = (best_node.placed_parts, best_node.placed_rots, best_node.offsets, best_length)
            else:
                if current_nodes:
                    node = current_nodes[0]
                    self.best_solution = (node.placed_parts, node.placed_rots, node.offsets, None)
                    print("Partial solution only.")
        else:
            print("No feasible solution found.")

    def draw_result(self):
        if self.best_solution is None:
            return
        placed_parts, placed_rots, offsets, length = self.best_solution

        all_points = []
        for part, rot, (dx, dy) in zip(placed_parts, placed_rots, offsets):
            rings = self.parts_rot[part][rot]
            for ring in rings:
                for p in ring:
                    x = p[0] + dx
                    y = p[1] + dy
                    all_points.append((x, y))
        if not all_points:
            return
        xs, ys = zip(*all_points)
        min_x, min_y = min(xs), min(ys)
        offset_x = -min_x
        offset_y = -min_y
        max_y_trans = max(y + offset_y for y in ys)
        container_height = max_y_trans  
        container_width = length       
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(placed_parts)))

        for idx, (part, rot, (dx, dy)) in enumerate(zip(placed_parts, placed_rots, offsets)):
            rings = self.parts_rot[part][rot]
            color = colors[idx % len(colors)]  
            for ring in rings:
                x = [p[0] + dx + offset_x for p in ring]
                y = [p[1] + dy + offset_y for p in ring]
                if x[0] != x[-1] or y[0] != y[-1]:
                    x.append(x[0])
                    y.append(y[0])
                ax.fill(x, y, color='#FFD8B5', edgecolor='darkorange',
                        linewidth=1.5, alpha=0.7)
              
        rect_x = [0, container_width, container_width, 0, 0]
        rect_y = [0, 0, container_height, container_height, 0]
        ax.plot(rect_x, rect_y, 'k-', linewidth=2.5)

        ax.set_aspect('equal')
        margin = 5
        ax.set_xlim(-margin, container_width + margin)
        ax.set_ylim(-margin, container_height + margin)
        plt.show()


    
    
    
    
    


