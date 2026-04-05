import copy
import random
import Packing
import math
from collections import Counter
import numpy as np

def read_data_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    namespace = {}
    exec(content, namespace)   
    newpolys = namespace['newpolys']
    newrepoly = namespace['newrepoly']
    numR = namespace['numR']
    Width = namespace['Width']    
    return newpolys, newrepoly, numR, Width

    
def getup(repolys):
    processed = set()
    result = []
    for i in range(len(repolys)):
        if repolys[i] and i not in processed:
            result.append(repolys[i][-1])
            processed.add(repolys[i][0])
    return result  


def get_move_directions(poly, idx1, idx2):
    n = len(poly)
    if (idx1 + 1) % n == idx2:
        x1, y1 = poly[idx1]
        x2, y2 = poly[idx2]
        dx = x2 - x1
        dy = y2 - y1
    elif (idx2 + 1) % n == idx1:
        x2, y2 = poly[idx2]
        x1, y1 = poly[idx1]
        dx = x1 - x2
        dy = y1 - y2
    else:
        raise ValueError("两点不相邻")
    def polygon_area(poly):
        s = 0.0
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[(i + 1) % n]
            s += xi * yj - xj * yi
        return s / 2

    area = polygon_area(poly)
    left = (-dy, dx)   
    right = (dy, -dx)  
    if area > 0:       
        nx, ny = right
    else:             
        nx, ny = left
    if abs(dx) > abs(dy):         
        kx_norm = 0
        ky_norm = 1 if ny > 0 else -1
    else:                          
        kx_norm = 1 if nx > 0 else -1
        ky_norm = 0
    result = {}
    for idx in [idx1, idx2]:
        prev = (idx - 1) % n
        nxt = (idx + 1) % n
        fixed = None
        if prev not in [idx1, idx2]:
            fixed = prev
        elif nxt not in [idx1, idx2]:
            fixed = nxt

        if fixed is None:        
            result[idx] = (kx_norm, ky_norm)
            continue
        xi, yi = poly[idx]
        xf, yf = poly[fixed]
        dx_f = xi - xf
        dy_f = yi - yf
        if abs(dx) > abs(dy):     
            ky = ky_norm
            if dy_f == 0:
                kx = 0
            else:
                kx = dx_f * ky / dy_f
        else:                       
            kx = kx_norm
            if dx_f == 0:
                ky = 0
            else:
                ky = dy_f * kx / dx_f
        result[idx] = (kx, ky)

    return result


def calculateKX(polys, catchpoly):
    KX = {}
    KY = {}
    for i, cp in enumerate(catchpoly):
        if not cp:
            continue
        part_idx = i
        comp = cp[1]
        p1 = cp[2]
        p2 = cp[3]
        poly = polys[part_idx][comp]
        directions = get_move_directions(poly, p1, p2)
        for idx, (kx, ky) in directions.items():
            KX[(part_idx, comp, idx)] = kx
            KY[(part_idx, comp, idx)] = ky
    return KX, KY



def rotate_all(polys, catchpoly, numR):
    
    num_parts = len(polys)
    new_polys = [[None] * numR for _ in range(num_parts)]  
    for part_idx, part_polys in enumerate(polys):
        for rot in range(numR):
            rotated_part = rotate(part_polys, rot, numR)
            new_polys[part_idx][rot] = rotated_part

    KX = {}
    KY = {}
    for i, cp in enumerate(catchpoly):
        if not cp:
            continue
        part_idx = i         
        comp = cp[1]
        p1 = cp[2]
        p2 = cp[3]
        for rot in range(numR):
            poly_rot = new_polys[part_idx][rot][comp]
            directions = get_move_directions(poly_rot, p1, p2)
            for pt, (kx, ky) in directions.items():
                KX[(part_idx, rot, comp, pt)] = kx
                KY[(part_idx, rot, comp, pt)] = ky

    return new_polys, KX, KY

def rotate_all2(poly, catchpoly, numR):
    
    num_parts = len(poly)
    polys = [[None] * numR for _ in range(num_parts)]  
    for part_idx, part_polys in enumerate(poly):
        for rot in range(numR):
            rotated_part = rotate(part_polys, rot, numR)
            polys[part_idx][rot] = rotated_part
    
    IR = copy.deepcopy(polys)
    index_i = 0
    items = []
    for o in range(len(polys)):
        for r in range(len(polys[o])):
            IR[o][r] = []
            for p in range(len(polys[o][r])):   
                a = copy.deepcopy(polys[o][r][p])         
                a.append(copy.deepcopy(polys[o][r][p][0]))
                items.append(a)
                IR[o][r].append(index_i)
                index_i += 1 
    I = len(items)            
    KX = [[[0 for i in range(I)] for c in range(len(items[j]))]for j in range(I)]
    
    KY = [[[0 for i in range(I)] for c in range(len(items[j]))]for j in range(I)]
    RP = [[]for i in range(I)]
    D = {}
    for  o, cp in enumerate(catchpoly):
        if not cp:
            continue
        for j1 in range(numR):
            i = IR[o][j1][cp[1]]
            p1 = cp[2]
            p2 = cp[3]
            for j2 in range(numR):
                j = IR[cp[0]][j2][catchpoly[cp[0]][1]]
                RP[i].append(j)
                poly_rot = polys[o][j1][cp[1]]
                directions = get_move_directions(poly_rot, p1, p2)
                for pt, (kx, ky) in directions.items():
                    KX[i][pt][j] = kx
                    KY[i][pt][j] = ky
                    D[i,j] = cp[-1]

    return polys, KX, KY, IR, items, RP, D

def generatepoly(polys, solution, catchpoly, KX, KY):
    angles = solution[0]         
    moves = solution[1]    
   
    nonempty = [(i, cp) for i, cp in enumerate(catchpoly) if cp]
    d = {}
    for pair_idx in range(len(moves)):
        idx1, cp1 = nonempty[2 * pair_idx]
        idx2, cp2 = nonempty[2 * pair_idx + 1]
        d[idx1] = moves[pair_idx]
        d[idx2] = round(cp1[4] - moves[pair_idx], 2)   
    
    for i, cp in nonempty:
        part_idx = i
        comp = cp[1]
        p1 = cp[2]
        p2 = cp[3]
        dist = d[i]
        
        rot = angles[part_idx] if part_idx < len(angles) else 0
        poly = polys[part_idx][comp]
        n = len(poly)
        
        if (p1 + 1) % n == p2:
            start_idx = p1
            end_idx = p2
            key_start = (part_idx, rot, comp, p1)
            key_end   = (part_idx, rot, comp, p2)
        elif (p2 + 1) % n == p1:
            start_idx = p2
            end_idx = p1
            key_start = (part_idx, rot, comp, p2)
            key_end   = (part_idx, rot, comp, p1)
        else:
            raise ValueError(f"Points {p1} and {p2} are not adjacent in polygon {part_idx} comp {comp}")
        
        kx_s, ky_s = KX[key_start], KY[key_start]
        kx_e, ky_e = KX[key_end], KY[key_end]
        
        x_s, y_s = poly[start_idx]
        x_e, y_e = poly[end_idx]
        new_start = [x_s + kx_s * dist, y_s + ky_s * dist]
        new_end   = [x_e + kx_e * dist, y_e + ky_e * dist]
        
        for tt in range(2):
            new_start[tt] = round(new_start[tt], 2)
            new_end[tt] = round(new_end[tt], 2)
        
        poly[start_idx] = new_start
        poly[end_idx] = new_end

    return polys

    
# def generatepoly(polys, solution, catchpoly, KX, KY):

#     angles = solution[0]         
#     moves = solution[1]    
   

#     nonempty = [(i, cp) for i, cp in enumerate(catchpoly) if cp]
#     d = {}
#     for pair_idx in range(len(moves)):
#         idx1, cp1 = nonempty[2 * pair_idx]
#         idx2, cp2 = nonempty[2 * pair_idx + 1]
#         d[idx1] = moves[pair_idx]
#         d[idx2] = round(cp1[4] - moves[pair_idx],2)   
#     for i, cp in nonempty:
#         part_idx = i
#         comp = cp[1]
#         p1 = cp[2]
#         p2 = cp[3]
#         dist = d[i]
        

#         rot = angles[part_idx] if part_idx < len(angles) else 0
#         poly = polys[part_idx][comp]
#         n = len(poly)
#         if (p1 + 1) % n == p2:
#             start_idx = p1
#             end_idx = p2
#             key_start = (part_idx, rot, comp, p1)
#             key_end   = (part_idx, rot, comp, p2)
#         elif (p2 + 1) % n == p1:
#             start_idx = p2
#             end_idx = p1
#             key_start = (part_idx, rot, comp, p2)
#             key_end   = (part_idx, rot, comp, p1)
#         else:
#             raise ValueError(f"Points {p1} and {p2} are not adjacent in polygon {part_idx} comp {comp}")
        

#         kx_s, ky_s = KX[key_start], KY[key_start]
#         kx_e, ky_e = KX[key_end], KY[key_end]
        
#         x_s, y_s = poly[start_idx]
#         x_e, y_e = poly[end_idx]
#         new_start = [x_s + kx_s * dist, y_s + ky_s * dist]
#         new_end   = [x_e + kx_e * dist, y_e + ky_e * dist]
        
#         for tt in range(2):
#             new_start[tt] = round(new_start[tt],2)
#             new_end[tt] = round(new_end[tt],2)
#         if start_idx == n - 1:
#             poly.append(new_start)
#             poly.append(new_end)
#         else:
#             poly.insert(start_idx + 1, new_start)
#             poly.insert(start_idx + 2, new_end)

#     return polys

def Same_area(polys, catchpoly, KX, KY,upbound):
    nonempty = [(i, cp) for i, cp in enumerate(catchpoly) if cp]
    if len(nonempty) % 2 != 0:
        raise ValueError("Number of non-empty catchpoly entries must be even.")
    weld = []
    rot = 0

    def compute_area(poly, start, end, k1, k2, d):
        """计算多边形在移动 d 后的面积（绝对值）"""
        n = len(poly)
        new_vertices = []
        for i in range(n):
            new_vertices.append(poly[i]) 
            if i == start:
                x_s, y_s = poly[start]
                x_e, y_e = poly[end]
                new_s = [x_s + k1[0] * d, y_s + k1[1] * d]
                new_e = [x_e + k2[0] * d, y_e + k2[1] * d]
                new_vertices.append(new_s)
                new_vertices.append(new_e)
        m = len(new_vertices)
        area = 0.0
        for i in range(m):
            x1, y1 = new_vertices[i]
            x2, y2 = new_vertices[(i + 1) % m]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    for group_idx in range(len(nonempty) // 2):
        idx_a, cp_a = nonempty[2 * group_idx]
        idx_b, cp_b = nonempty[2 * group_idx + 1]
        L = cp_a[4]  

        part_a = idx_a
        comp_a = cp_a[1]
        p1_a, p2_a = cp_a[2], cp_a[3]
        poly_a = polys[part_a][comp_a]
        n_a = len(poly_a)

        if (p1_a + 1) % n_a == p2_a:
            start_a, end_a = p1_a, p2_a
            key_start_a = (part_a, rot, comp_a, p1_a)
            key_end_a = (part_a, rot, comp_a, p2_a)
        elif (p2_a + 1) % n_a == p1_a:
            start_a, end_a = p2_a, p1_a
            key_start_a = (part_a, rot, comp_a, p2_a)
            key_end_a = (part_a, rot, comp_a, p1_a)
        else:
            raise ValueError(f"Points not adjacent in part {part_a} comp {comp_a}")

        kx_sa, ky_sa = KX[key_start_a], KY[key_start_a]
        kx_ea, ky_ea = KX[key_end_a], KY[key_end_a]
        k1_a = (kx_sa, ky_sa)
        k2_a = (kx_ea, ky_ea)

        part_b = idx_b
        comp_b = cp_b[1]
        p1_b, p2_b = cp_b[2], cp_b[3]
        poly_b = polys[part_b][comp_b]
        n_b = len(poly_b)

        if (p1_b + 1) % n_b == p2_b:
            start_b, end_b = p1_b, p2_b
            key_start_b = (part_b, rot, comp_b, p1_b)
            key_end_b = (part_b, rot, comp_b, p2_b)
        elif (p2_b + 1) % n_b == p1_b:
            start_b, end_b = p2_b, p1_b
            key_start_b = (part_b, rot, comp_b, p2_b)
            key_end_b = (part_b, rot, comp_b, p1_b)
        else:
            raise ValueError(f"Points not adjacent in part {part_b} comp {comp_b}")

        kx_sb, ky_sb = KX[key_start_b], KY[key_start_b]
        kx_eb, ky_eb = KX[key_end_b], KY[key_end_b]
        k1_b = (kx_sb, ky_sb)
        k2_b = (kx_eb, ky_eb)

        def f(x):
            area_a = compute_area(poly_a, start_a, end_a, k1_a, k2_a, x)
            area_b = compute_area(poly_b, start_b, end_b, k1_b, k2_b, L - x)
            return area_a - area_b

        a, b = 0.0, L
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            x_root = L / 2.0
        else:
            tol = 1e-6
            while b - a > tol:
                c = (a + b) / 2.0
                fc = f(c)
                if fc == 0:
                    break
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            x_root = (a + b) / 2.0
        weld.append(round(x_root,2))
    weld = [max(0, min(val, upbound[i])) for i, val in enumerate(weld)]
    return weld

def Different_area(polys, catchpoly, KX, KY,upbound):
    nonempty = [(i, cp) for i, cp in enumerate(catchpoly) if cp]
    if len(nonempty) % 2 != 0:
        raise ValueError("Number of non-empty catchpoly entries must be even.")
    weld = []
    rot = 0  

    def area_with_move(poly, start, end, k1, k2, d):
        new_vertices = []
        n = len(poly)
        for i in range(n):
            new_vertices.append(poly[i])
            if i == start:
                x_s, y_s = poly[start]
                x_e, y_e = poly[end]
                new_s = [x_s + k1[0] * d, y_s + k1[1] * d]
                new_e = [x_e + k2[0] * d, y_e + k2[1] * d]
                new_vertices.append(new_s)
                new_vertices.append(new_e)
        m = len(new_vertices)
        s = 0.0
        for i in range(m):
            x1, y1 = new_vertices[i]
            x2, y2 = new_vertices[(i + 1) % m]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

    def diff_func(poly_a, start_a, end_a, k1_a, k2_a,
                  poly_b, start_b, end_b, k1_b, k2_b, L, x):
        """计算差值 area_a(x) - area_b(L-x)"""
        area_a = area_with_move(poly_a, start_a, end_a, k1_a, k2_a, x)
        area_b = area_with_move(poly_b, start_b, end_b, k1_b, k2_b, L - x)
        return area_a - area_b

    for group_idx in range(len(nonempty) // 2):
        idx_a, cp_a = nonempty[2 * group_idx]
        idx_b, cp_b = nonempty[2 * group_idx + 1]
        L = cp_a[4] 
        part_a = idx_a
        comp_a = cp_a[1]
        p1_a, p2_a = cp_a[2], cp_a[3]
        poly_a = polys[part_a][comp_a]
        n_a = len(poly_a)
        if (p1_a + 1) % n_a == p2_a:
            start_a, end_a = p1_a, p2_a
            key_start_a = (part_a, rot, comp_a, p1_a)
            key_end_a = (part_a, rot, comp_a, p2_a)
        elif (p2_a + 1) % n_a == p1_a:
            start_a, end_a = p2_a, p1_a
            key_start_a = (part_a, rot, comp_a, p2_a)
            key_end_a = (part_a, rot, comp_a, p1_a)
        else:
            raise ValueError(f"Points not adjacent in part {part_a} comp {comp_a}")
        kx_sa, ky_sa = KX[key_start_a], KY[key_start_a]
        kx_ea, ky_ea = KX[key_end_a], KY[key_end_a]
        k1_a = (kx_sa, ky_sa)
        k2_a = (kx_ea, ky_ea)

        part_b = idx_b
        comp_b = cp_b[1]
        p1_b, p2_b = cp_b[2], cp_b[3]
        poly_b = polys[part_b][comp_b]
        n_b = len(poly_b)
        if (p1_b + 1) % n_b == p2_b:
            start_b, end_b = p1_b, p2_b
            key_start_b = (part_b, rot, comp_b, p1_b)
            key_end_b = (part_b, rot, comp_b, p2_b)
        elif (p2_b + 1) % n_b == p1_b:
            start_b, end_b = p2_b, p1_b
            key_start_b = (part_b, rot, comp_b, p2_b)
            key_end_b = (part_b, rot, comp_b, p1_b)
        else:
            raise ValueError(f"Points not adjacent in part {part_b} comp {comp_b}")
        kx_sb, ky_sb = KX[key_start_b], KY[key_start_b]
        kx_eb, ky_eb = KX[key_end_b], KY[key_end_b]
        k1_b = (kx_sb, ky_sb)
        k2_b = (kx_eb, ky_eb)

        x1 = 0.0
        x2 = L
        x3 = L / 2.0
        y1 = diff_func(poly_a, start_a, end_a, k1_a, k2_a,
                       poly_b, start_b, end_b, k1_b, k2_b, L, x1)
        y2 = diff_func(poly_a, start_a, end_a, k1_a, k2_a,
                       poly_b, start_b, end_b, k1_b, k2_b, L, x2)
        y3 = diff_func(poly_a, start_a, end_a, k1_a, k2_a,
                       poly_b, start_b, end_b, k1_b, k2_b, L, x3)

        if L != 0:
            A = (2 * (y2 - 2 * y3 + y1)) / (L * L)
            B = (y2 - y1) / L - A * L
        else:
            A = B = 0

        candidates = [(x1, abs(y1)), (x2, abs(y2))]
        if abs(A) > 1e-12:
            x0 = -B / (2 * A)
            if 0 <= x0 <= L:
                y0 = diff_func(poly_a, start_a, end_a, k1_a, k2_a,
                               poly_b, start_b, end_b, k1_b, k2_b, L, x0)
                candidates.append((x0, abs(y0)))
        best_x, _ = max(candidates, key=lambda t: t[1])
        weld.append(round(best_x,2))
    weld = [max(0, min(val, upbound[i])) for i, val in enumerate(weld)]
    return weld

def operator1(X, num, num2):
    new_X = copy.deepcopy(X)
    angles = new_X[0]
    packing = new_X[2]

    if num == 1:
        all_indices = []
        for sublist in packing:
            all_indices.extend(sublist)
        idx1, idx2 = random.sample(all_indices, 2)
        pos1 = pos2 = None
        for i, sub in enumerate(packing):
            for j, val in enumerate(sub):
                if val == idx1 and pos1 is None:
                    pos1 = (i, j)
                if val == idx2 and pos2 is None:
                    pos2 = (i, j)
        if pos1 and pos2:
            i1, j1 = pos1
            i2, j2 = pos2
            packing[i1][j1], packing[i2][j2] = packing[i2][j2], packing[i1][j1]

    elif num == 2:
        if len(angles) < 2:
            return new_X
        i, j = random.sample(range(len(angles)), 2)
        angles[i], angles[j] = angles[j], angles[i]

    elif num == 3:
        sub_idx = random.randint(0, len(packing) - 1)
        sub = packing[sub_idx]
        elem_pos = random.randint(0, len(sub) - 1)
        elem = sub[elem_pos]
        del sub[elem_pos]
        if len(sub) == 0:
            del packing[sub_idx]
        mode = random.randint(0, 1)
        if mode == 0:
            insert_pos = random.randint(0, len(packing))
            packing.insert(insert_pos, [elem])
        else:
            if not packing:               
                packing.append([elem])
            else:
                target_sub_idx = random.randint(0, len(packing) - 1)
                target_sub = packing[target_sub_idx]
                insert_pos = random.randint(0, len(target_sub))
                target_sub.insert(insert_pos, elem)

    elif num == 4:
        if not angles:
            return new_X
        idx = random.randint(0, len(angles) - 1)
        old_val = angles[idx]
        choices = [v for v in range(num2) if v != old_val]
        if choices:
            angles[idx] = random.choice(choices)
    elif num == 5:
        new_X = copy.deepcopy(X)
        packing = new_X[2]
        idx1, idx2 = random.sample(range(len(packing)), 2)
        packing[idx1], packing[idx2] = packing[idx2], packing[idx1]

    
    elif num == 6:
        new_X = copy.deepcopy(X)
        packing = new_X[2]
        group_idx = random.randint(0, len(packing)-1)
        packing[group_idx].reverse()
        
    elif num == 7:
        new_X = copy.deepcopy(X)
        angles = new_X[0]
        new_X[0] = [(a + 1) % num2 for a in angles]
    
    elif num == 8:
        new_X = copy.deepcopy(X)
        packing = new_X[2]
        idx = random.randint(0, len(packing)-2)
        packing[idx] = packing[idx] + packing[idx+1]
        del packing[idx+1]
        
    elif num == 9:
        new_X = copy.deepcopy(X)
        packing = new_X[2]
        candidates = [i for i, g in enumerate(packing) if len(g) >= 2]
        if not candidates:
            return new_X
        idx = random.choice(candidates)
        group = packing[idx]
        split = random.randint(1, len(group)-1)
        new_group1 = group[:split]
        new_group2 = group[split:]
        packing[idx:idx+1] = [new_group1, new_group2]


    return new_X



def operator0(X, upbound):
    new_X = copy.deepcopy(X)
    moves = new_X[1]
    for i in range(len(moves)):
        newval = random.uniform(0, upbound[i])
        moves[i] = round(newval, 2)
    indices = [i for i in range(len(upbound))]
    # new_X = copy.deepcopy(X)
    # moves = new_X[1]
    # L = len(moves)
    # k = random.randint(2, 5)
    # half = L // 2
    # num_to_change = min(k, half)
    # if num_to_change == 0:
    #     num_to_change = 1
    # indices = random.sample(range(L), num_to_change)
    # for idx in indices:
    #     new_val = None
    #     for _ in range(10):
    #         candidate = np.random.normal(moves[idx], 100)
    #         candidate = np.clip(candidate, 0, upbound[idx])  
    #         if abs(candidate - moves[idx]) >= 3:
    #             new_val = candidate
    #             break
    #     if new_val is None:
    #         new_val = np.random.uniform( 0, upbound[idx] )
    #     moves[idx] = round(new_val, 2)                    
    return new_X, indices

def MatchO(X, similar_list, i=None):
    new_X = copy.deepcopy(X)
    packing = new_X[2]
    all_parts = []
    for sub in packing:
        all_parts.extend(sub)
    if i is None:
        p = random.choice(all_parts)
        q = random.choice(similar_list[p])
        put_behind = random.choice([True, False])  
    else:
        p = i
        if i >= len(similar_list) or not similar_list[i]:
            return new_X
        q = similar_list[i][0]  
        put_behind = True 
    pos_p = None
    pos_q = None
    for r, sub in enumerate(packing):
        for c, part in enumerate(sub):
            if part == p:
                pos_p = (r, c)
            if part == q:
                pos_q = (r, c)

    if pos_p is None or pos_q is None:
        return new_X

    row_p, col_p = pos_p
    del packing[row_p][col_p]

    if not packing[row_p]:
        del packing[row_p]
    new_pos_q = None
    for r, sub in enumerate(packing):
        for c, part in enumerate(sub):
            if part == q:
                new_pos_q = (r, c)
                break
        if new_pos_q:
            break
    row_q, col_q = new_pos_q
    if put_behind:
        insert_col = col_q + 1
    else:
        insert_col = col_q

    if insert_col > len(packing[row_q]):
        packing[row_q].append(p)
    else:
        packing[row_q].insert(insert_col, p)

    return new_X

def rotate(p1, r, numR):
    p = copy.deepcopy(p1)
    if r == 0:
        return copy.deepcopy(p) 
    angle_deg = r * 360.0 / numR
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    new_p = []
    for part in p:
        new_part = []
        for (x, y) in part:
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            new_part.append([round(new_x,2), round(new_y,2)])
        new_p.append(new_part)
    return new_p


def CompactR(X, polys, width, NFP_history, numR, step,  i=None):
    new_X = copy.deepcopy(X)
    if i is None:
        rows_with_multiple = [r for r, sub in enumerate(new_X[2]) if len(sub) >= 2]
        if not rows_with_multiple:
            return new_X
        row = random.choice(rows_with_multiple)
        col = random.randint(0, len(new_X[2][row]) - 1)
        i = new_X[2][row][col]
    else:
        for row, sublist in enumerate(new_X[2]):
            if i in sublist:
                col = sublist.index(i)
                break

    row_indices = new_X[2][row]
    pid = [X[0][k] + k * numR for k in range(len(X[0]))]
    base_polys = copy.deepcopy(polys) 
    best_r = 0
    best_score = float('inf')
    
    for r in range(numR): 
        pid[i] = i * numR + (X[0][i] + r) % numR
        temp_polys = copy.deepcopy(base_polys)   
        A = copy.deepcopy(base_polys[i])              
        temp_polys[i] = rotate(A, r, numR)       
        restopos = Packing.TOPOS(temp_polys, row_indices , width, NFP_history, step,pid)
        if restopos.best_score <= best_score:
            best_score = restopos.best_score
            best_r = r

    new_X[0][i] = (new_X[0][i] + best_r) % numR
    return new_X


def matchpolys(polys, theta=0.9):
    new_polys = copy.deepcopy(polys)
    n_parts = len(new_polys)
    # angles = [0] * n_parts
    # solution = [angles, weld, []]
    # generatepoly(new_polys, solution, catchpoly, KX, KY)

    def get_edges_of_part(part_polys):
        edges = []
        for poly in part_polys:
            closed = poly + [poly[0]]
            for i in range(len(poly)):
                x1, y1 = closed[i]
                x2, y2 = closed[i+1]
                length = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
                edges.append(length)
        return edges

    part_edges = [get_edges_of_part(new_polys[i]) for i in range(n_parts)]

    def count_matched_edges(edgesA, edgesB, theta):
        usedB = [False] * len(edgesB)
        count = 0
        for a in edgesA:
            for j, b in enumerate(edgesB):
                if not usedB[j]:
                    sim = 1 - abs(a - b) / max(a, b)
                    if sim >= theta:
                        usedB[j] = True
                        count += 1
                        if sim >= 0.99:
                            count += 1  
                        break
        return count

    similar_list = []
    for i in range(n_parts):
        scores = []
        for j in range(n_parts):
            if i == j:
                continue
            score = count_matched_edges(part_edges[i], part_edges[j], theta)
            scores.append((j, score))
        scores.sort(key=lambda x: -x[1])
        top3 = [idx for idx, _ in scores[:6]]
        similar_list.append(top3)

    return similar_list   




# X = [[0,1,2,0,1,1],[5,15],[[0,2],[1,3,4],[5]]]    
# x1 = operator2(X, [20,30])
# print(X)
# print(x1)
# if __name__ == "__main__":
#     # 原始数据（与您提供的一致，但修正 catchpoly[0] 的点索引为 3）
#     polys = [[] for _ in range(6)]
#     polys[0] = [
#         [[0, 90], [0, 110], [30, 110], [30, 90]],
#         [[0, 80], [0, 90], [20, 90], [20, 80]],
#         [[0, 70], [0, 80], [20, 80], [25.0, 70]]
#     ]
#     polys[1] = [
#         [[0, 0], [0, 30], [50, 30], [50, 0]],
#         [[0, 30], [0, 50], [35.0, 50], [40, 40], [40, 30]]
#     ]
#     polys[2] = [
#         [[0, 0], [20, 20], [30, 20], [30, 0]],
#         [[0, 0], [0, 40], [20, 40], [20, 20]]
#     ]
#     polys[3] = [
#         [[0, 0], [0, 20], [10, 20], [30, 0]],
#         [[10, 20], [10, 40], [30, 40], [30, 0]]
#     ]
#     polys[4] = [[[0, 0], [0, 25], [25, 25], [25, 0]]]
#     polys[5] = [[[0, 0], [0, 40], [30, 0]]]
#     solution = [[0,0,0,0,0,0],[5,15],[[0,2],[1,3,4],[5]]]
#     KX = {(0, 0, 2, 0): -0.0, (0, 0, 2, 3): 0.5, (0, 1, 2, 0): 1, (0, 1, 2, 3): 1, (0, 2, 2, 0): 0.0, (0, 2, 2, 3): -0.5, (0, 3, 2, 0): -1, (0, 3, 2, 3): -1, (1, 0, 1, 1): 0.0, (1, 0, 1, 2): -0.5, (1, 1, 1, 1): -1, (1, 1, 1, 2): -1, (1, 2, 1, 1): -0.0, (1, 2, 1, 2): 0.5, (1, 3, 1, 1): 1, (1, 3, 1, 2): 1, (2, 0, 0, 2): 1, (2, 0, 0, 3): 1, (2, 1, 0, 2): 0.0, (2, 1, 0, 3): 0.0, (2, 2, 0, 2): -1, (2, 2, 0, 3): -1, (2, 3, 0, 2): -0.0, (2, 3, 0, 3): -0.0, (3, 0, 0, 0): -1, (3, 0, 0, 1): -1, (3, 1, 0, 0): -0.0, (3, 1, 0, 1): -0.0, (3, 2, 0, 0): 1, (3, 2, 0, 1): 1, (3, 3, 0, 0): 0.0, (3, 3, 0, 1): 0.0}
#     KY = {(0, 0, 2, 0): -1, (0, 0, 2, 3): -1, (0, 1, 2, 0): 0.0, (0, 1, 2, 3): 0.5, (0, 2, 2, 0): 1, (0, 2, 2, 3): 1, (0, 3, 2, 0): -0.0, (0, 3, 2, 3): -0.5, (1, 0, 1, 1): 1, (1, 0, 1, 2): 1, (1, 1, 1, 1): -0.0, (1, 1, 1, 2): -0.5, (1, 2, 1, 1): -1, (1, 2, 1, 2): -1, (1, 3, 1, 1): 0.0, (1, 3, 1, 2): 0.5, (2, 0, 0, 2): 0.0, (2, 0, 0, 3): 0.0, (2, 1, 0, 2): 1, (2, 1, 0, 3): 1, (2, 2, 0, 2): -0.0, (2, 2, 0, 3): -0.0, (2, 3, 0, 2): -1, (2, 3, 0, 3): -1, (3, 0, 0, 0): -0.0, (3, 0, 0, 1): -0.0, (3, 1, 0, 0): -1, (3, 1, 0, 1): -1, (3, 2, 0, 0): 0.0, (3, 2, 0, 1): 0.0, (3, 3, 0, 0): 1, (3, 3, 0, 1): 1}
#     catchpoly = [[] for _ in range(6)]
#     catchpoly[0] = [1, 2, 0, 3, 20]   # 修正：点索引应为 3 而非 4
#     catchpoly[1] = [0, 1, 1, 2, 20]
#     catchpoly[2] = [3, 0, 2, 3, 30]
#     catchpoly[3] = [2, 0, 0, 1, 30]
#     aa = matchpolys(polys, catchpoly, KX, KY, weld=[15,16], theta=0.9)
#     print(aa)
#     # polys = generatepoly(polys, solution, catchpoly, KX, KY)
#     weld = Different_area(polys, catchpoly, KX, KY)
#     print(weld)

   
    
    # 零件4和5无配对

    # new_polys, KX, KY = rotate_all(polys, catchpoly)
    # print(KX)
    # print(KY)
    # print(KX[(0,0,2,3)])
    # print(KY[(0,0,2,3)])
    # print(KX[(0,1,2,3)])    
    # print(KY[(0,1,2,3)])


    # 输出验证
    # print("新零件数量:", len(new_polys), "行，每行4个旋转版本")
    # print("原零件0的0°旋转（new_polys[0][0]）部件2:", new_polys[0][0][2])
    # print("原零件0的90°旋转（new_polys[0][1]）部件2:", new_polys[0][1][2])
    # print("原零件1的90°旋转（new_polys[1][1]）部件1:", new_polys[1][1][1])
    # print("KX 中 (0,0,2,0) 的值:", KX.get((0,0,2,0)))
    # print("KY 中 (0,0,2,0) 的值:", KY.get((0,0,2,0)))
    # print("KX 中 (0,1,2,0) 的值:", KX.get((0,1,2,0)))
    # print("KY 中 (0,1,2,0) 的值:", KY.get((0,1,2,0)))
    
    
    
# polys = [[]for i in range(6)]

# polys[0] = [[[0, 90], [0, 110], [30, 110], [30, 90]], [[0, 80], [0, 90], [20, 90], [20, 80]], [[0, 70], [0, 80], [20, 80], [25.0, 70]]]
# polys[1] = [[[0, 0], [0, 30], [50, 30], [50, 0]], [[0, 30], [0, 50], [35.0, 50], [40, 40], [40, 30]]]
# polys[2] = [[[0, 0], [20, 20], [30, 20], [30, 0]], [[0, 0], [0, 40], [20, 40], [20, 20]]]
# polys[3] = [[[0, 0], [0, 20], [10, 20], [30, 0]], [[10, 20], [10, 40], [30, 40], [30, 0]]]
# polys[4] = [[[0, 0], [0, 25], [25, 25], [25, 0]]]
# polys[5] = [[[0, 0], [0, 40], [30, 0]]]
# catchpoly = [[]for i in range(6)]
# catchpoly[0] = [1,2,0,3,20]
# catchpoly[1] = [0,1,1,2,20]
# catchpoly[2] = [3,0,2,3,30]
# catchpoly[3] = [2,0,0,1,30]
# catchpoly[4] = []
# catchpoly[5] = []    
# setroataion, KX, KY, IR, items, RP, D = rotate_all2(polys, catchpoly, 4)
# print(D)    
    
    
    
    
    
    