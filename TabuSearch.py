import random
import copy
import time
import Packing
import Tool
import os
import numpy as np
class TS:
    def __init__(self, poly, repoly, numR, width, showplt):
        self.n_poly = len(poly)
        self.max_evaluations = 50 * self.n_poly
        self.tabu_tenure = 40
        self.NumR = numR
        self.width = width
        self.step = 1
        new_polys, KX, KY = Tool.rotate_all(poly, repoly, self.NumR)
        self.polys = new_polys
        self.KX = KX
        self.KY = KY
        self.repoly = repoly
        self.upbound = Tool.getup(repoly)
        self.NFP_history = [[[] for _ in range(len(poly) * self.NumR)] for _ in range(len(poly) * self.NumR)]
        self.bestvalue = float('inf')
        self.bestsolution = None
        self.number_evaluations = 0
        self.Time = None
        solution, value, _ = self.generate_initial_solution()
        self.current_solution = solution
        self.current_value = value
        self.bestsolution = copy.deepcopy(solution)
        self.bestvalue = value
        self.run()
        self.objective_func(self.bestsolution, showplt=showplt)
        self.Time = time.time() - self.Time
    
    def objective_func(self, X, idx=None, oldstate=None, showplt=False):
        self.number_evaluations += 1
        polys = [copy.deepcopy(self.polys[i][X[0][i]]) for i in range(len(self.polys))]
        polys = Tool.generatepoly(polys, X, self.repoly, self.KX, self.KY)
        X1 = copy.deepcopy(X[2])
        pid = [X[0][i] + i * self.NumR for i in range(len(self.polys))]
        res = Packing.LowerLeft(polys, X1, self.width, self.NFP_history, self.step, pid,
                                showplt, cached_state=oldstate, cached_index=idx)
        return res.current_length, res.toposscore, res.outstate

    def operator_func(self, solution, num, i=None):
        if num <= 4:
            new_solution = Tool.operator1(solution, num, self.NumR)
        elif num == 5:
            new_solution = Tool.MatchO(solution, self.similar_list, i)
        elif num == 6:
            polys = [copy.deepcopy(self.polys[i][solution[0][i]]) for i in range(len(self.polys))]
            polys = Tool.generatepoly(polys, solution, self.repoly, self.KX, self.KY)
            new_solution = Tool.CompactR(solution, polys, self.width, self.NFP_history, self.NumR, self.step, i)
        elif num == 7:
            new_solution = self.minlengthR(solution)
        elif num == 8:
            new_solution = self.minlengthO(solution)
        return new_solution

    def minlengthO(self, X):
        polys = [copy.deepcopy(self.polys[i][X[0][i]]) for i in range(len(self.polys))]
        polys = Tool.generatepoly(polys, X, self.repoly, self.KX, self.KY)
        Xp = copy.deepcopy(X[2])
        newX = copy.deepcopy(X)
        pid = [X[0][i] + i * self.NumR for i in range(len(self.polys))]
        if len(Xp) < 4:
            return newX
        idx1 = random.randint(2, len(Xp) - 2)
        pcopy = copy.deepcopy(Xp[idx1:min(idx1 + 4, len(Xp))])
        Xp = Xp[:idx1]
        res = Packing.LowerLeft(polys, Xp, self.width, self.NFP_history, self.step, pid)
        bestl = res.current_length
        try:
            cached = res.outstate
        except:
            return newX
        bestidx = None
        bestp = None
        for idx2, p in enumerate(pcopy):
            Xp[-1] = copy.deepcopy(p)
            res = Packing.LowerLeft(polys, Xp, self.width, self.NFP_history, self.step, pid,
                                    cached_state=cached, cached_index=idx1 - 2)
            if res.current_length < bestl:
                bestl = res.current_length
                bestp = copy.deepcopy(p)
                bestidx = idx2
        if bestidx is None:
            return newX
        newX[2][idx1 + bestidx] = copy.deepcopy(newX[2][idx1 - 1])
        newX[2][idx1 - 1] = copy.deepcopy(bestp)
        return newX

    def minlengthR(self, X):
        polys = [copy.deepcopy(self.polys[i][X[0][i]]) for i in range(len(self.polys))]
        polys = Tool.generatepoly(polys, X, self.repoly, self.KX, self.KY)
        Xp = copy.deepcopy(X[2])
        newX = copy.deepcopy(X)
        pid = [X[0][i] + i * self.NumR for i in range(len(self.polys))]
        if len(Xp) < 2:
            return newX
        idx1 = random.randint(2, len(Xp))
        Xp = Xp[:idx1]
        idx2 = random.randint(0, len(Xp[-1]) - 1)
        idx = Xp[-1][idx2]
        pcopy = copy.deepcopy(polys[idx])
        rold = pid[idx] - idx * self.NumR
        res = Packing.LowerLeft(polys, Xp, self.width, self.NFP_history, self.step, pid)
        bestl = res.current_length
        try:
            cached = res.outstate
        except:
            return newX
        for r in range(1, self.NumR):
            polys[idx] = Tool.rotate(pcopy, r, self.NumR)
            pid[idx] = (rold + r) % self.NumR + idx * self.NumR
            res = Packing.LowerLeft(polys, Xp, self.width, self.NFP_history, self.step, pid,
                                    cached_state=cached, cached_index=idx1 - 2)
            if res.current_length < bestl:
                bestl = res.current_length
                newX[0][idx] = (rold + r) % self.NumR
        return newX

    def get_move_signature(self, sol_before, sol_after):
        angles_before = sol_before[0]
        angles_after = sol_after[0]
        changed = []
        for i in range(len(angles_before)):
            if angles_before[i] != angles_after[i]:
                changed.append(('angle', i, angles_after[i]))
        pack_before = sol_before[2]
        pack_after = sol_after[2]
        if len(pack_before) != len(pack_after):
            changed.append(('pack_structure', len(pack_before), len(pack_after)))
        else:
            for idx, (gb, ga) in enumerate(zip(pack_before, pack_after)):
                if gb != ga:
                    changed.append(('pack_group', idx, tuple(ga)))
        if not changed:
            return None
        return tuple(changed)

    def generate_initial_solution(self):
        temp_p = [copy.deepcopy(self.polys[i][0]) for i in range(len(self.polys))]
        moves = Tool.Different_area(temp_p, self.repoly, self.KX, self.KY, self.upbound)
        temp_solution = [[0] * len(self.polys), moves, []]
        temp_p = Tool.generatepoly(temp_p, temp_solution, self.repoly, self.KX, self.KY)
        self.similar_list = Tool.matchpolys(temp_p, theta=9)

        def polygon_area(poly):
            s = 0.0
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                s += x1 * y2 - x2 * y1
            return abs(s) / 2.0

        areas = []
        for i in range(len(self.polys)):
            total = 0.0
            for comp in temp_p[i]:
                total += polygon_area(comp)
            areas.append((i, total))
        sorted_indices = [idx for idx, _ in sorted(areas, key=lambda x: -x[1])]

        angles = [0] * len(self.polys)
        for idx in sorted_indices:
            min_width = float('inf')
            best_rots = []
            for r in range(self.NumR):
                rotated = self.polys[idx][r]
                all_x = [pt[0] for comp in rotated for pt in comp]
                width = max(all_x) - min(all_x) if all_x else 0
                if width < min_width - 1e-9:
                    min_width = width
                    best_rots = [r]
                elif abs(width - min_width) < 1e-9:
                    best_rots.append(r)
            angles[idx] = random.choice(best_rots) if best_rots else 0

        packing = []
        for i in range(0, len(sorted_indices), 2):
            group = [sorted_indices[i]]
            if i + 1 < len(sorted_indices):
                group.append(sorted_indices[i + 1])
            packing.append(group)

        solution = [angles, moves, packing]
        value, _, _ = self.objective_func(solution)
        return solution, value, areas

    def run(self):
        tabu_list = []
        tabu_set = set()
        self.Time = time.time()

        while self.number_evaluations < self.max_evaluations:
            best_candidate = None
            best_candidate_value = float('inf')
            best_move_sig = None

            for op in range(1,9):  
                if op in [5, 6, 7, 8]:
                    parts_range = range(len(self.polys))
                else:
                    parts_range = [None]

                for i_param in parts_range:
                    if self.number_evaluations >= self.max_evaluations:
                        break

                    if i_param is None:
                        neighbor = self.operator_func(self.current_solution, op)
                    else:
                        neighbor = self.operator_func(self.current_solution, op, i_param)

                    val, _, _ = self.objective_func(neighbor)
                    move_sig = self.get_move_signature(self.current_solution, neighbor)

                    tabu = (move_sig is not None and move_sig in tabu_set)
                    if (not tabu) or (val < self.bestvalue):
                        if val < best_candidate_value:
                            best_candidate = neighbor
                            best_candidate_value = val
                            best_move_sig = move_sig

            if best_candidate is None:
                op = random.randint(1, 8)
                best_candidate = self.operator_func(self.current_solution, op)
                best_candidate_value, _, _ = self.objective_func(best_candidate)
                best_move_sig = self.get_move_signature(self.current_solution, best_candidate)

            self.current_solution = best_candidate
            self.current_value = best_candidate_value

            if best_move_sig is not None:
                tabu_list.append(best_move_sig)
                tabu_set.add(best_move_sig)
                if len(tabu_list) > self.tabu_tenure:
                    old = tabu_list.pop(0)
                    tabu_set.remove(old)

            if self.current_value < self.bestvalue:
                self.bestvalue = self.current_value
                self.bestsolution = copy.deepcopy(self.current_solution)

  
        

  