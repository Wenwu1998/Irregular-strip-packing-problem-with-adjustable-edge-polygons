import random
import numpy as np 
import copy
import os
import Packing
import time
import Tool
import pickle


class VNS:
    def __init__(self, poly, repoly,numR, width, parameters):
        self.maxgen = 50*len(poly)
        self.k_max = 10
        self.t1Max = 150
        self.theta = 0.1  
        self.step = 1
        self.NumR = numR
        new_polys, KX, KY = Tool.rotate_all(poly, repoly,self.NumR)
        self.polys = new_polys    
        self.KX = KX
        self.KY = KY
        self.repoly = repoly
        self.upbound = Tool.getup(repoly)
        self.NFPhistory = []
        self.bestvalue = 0
        self.bestsolution = []
        self.number_evaluations = 0
        self.Time = time.time()
        self.width = width                
        self.ql_beta = 0.995
        self.ql_num_actions = 9
        self.ql_num_states = 9
        self.ql_Q = np.zeros((self.ql_num_states, self.ql_num_actions))
        self.ql_CF = np.zeros(self.ql_num_actions, dtype=int)
        self.ql_state = 2        
        self.t1 = 0
        self.ql_last_action = None
        self.NFP_history = [[[]for _ in range(len(poly)*self.NumR)]for _ in range(len(poly)*self.NumR)]                 
        self.run()
        self.NFP_history = [[[]for _ in range(len(poly)*self.NumR)]for _ in range(len(poly)*self.NumR)]
        self.objective_func(self.bestsolution, showplt =True)
        self.Time = time.time()-self.Time 
        
    def clearNFP(self,indices):
        n = len(self.NFP_history) 
        used = set()  
        groups = []
        repolys = copy.deepcopy(self.repoly)
        for i, content in enumerate(repolys):
            if content and i not in used:
                j = content[0] 
                if 0 <= j < len(repolys) and repolys[j]:
                    group = sorted([i, j])
                    if group not in groups:
                        groups.append(group)
                    used.add(i)
                    used.add(j)
        ClearI = []
        for i in indices:
            ClearI += groups[i]      
        rows_cols_to_clear = set() 
        
        for g in ClearI:
            start = g * self.NumR
            for k in range(self.NumR):
                rows_cols_to_clear.add(start + k)
                
        for i in rows_cols_to_clear:
            for j in range(n):
                self.NFP_history[i][j] = []
                self.NFP_history[j][i] = []
         
        
    def shake(self, solution, value):
        if self.t1 > self.t1Max:
            solution, indices = Tool.operator0(solution, self.upbound)
            self.clearNFP(indices)           
            value, _ ,_s = self.objective_func(solution)
            self.t1 = 0
        s1 = copy.deepcopy(solution) 
        SI = random.randint(3,8)
       
        for i in range(SI):

            if random.random() > self.theta:          
                num = random.randint(1, 4)
                s1 = self.operator_func(s1,num )
            else:                
                _, scores,_r = self.objective_func(s1)
                if _ >= 999:
                    continue 
                min_val = min(scores)
                max_val = max(scores)
                if max_val == min_val:             
                    scores =  [0.5] * len(scores) 
                else:
                    scores = [(x - min_val) / (max_val - min_val) + 0.1 for x in scores]                         
                sublists = s1[2]
                total_score = sum(scores)
                probs = [s / total_score for s in scores]
                idx = random.choices(range(len(sublists)), weights=probs)[0]            
                selected = sublists[idx]              
                numl = random.randint(0, len(selected) - 1)  
                num = selected[numl]           
                s1 = self.operator_func(s1, 5, num)
                s1 = self.operator_func(s1, 6, num)   
            
        

        v1, _, state = self.objective_func(s1)  
       
        if v1>999:
            s1 = copy.deepcopy(self.bestsolution)   
            self.NFP_history = [[[]for _ in range(len(self.polys)*self.NumR)]for _ in range(len(self.polys)*self.NumR)]
            v1, _, state = self.objective_func(s1) 
           


        return solution, value, s1, v1, state
    
    def objective_func(self, X, idx=None, oldstate = None, showplt = False):    
        # self.NFP_history = [[[]for _ in range(10*self.NumR)]for _ in range(10*self.NumR)]
        self.number_evaluations += 1       
        polys = [[]for i in range(len(self.polys))]   
        for i in range(len(X[0])):
            polys[i] = copy.deepcopy(self.polys[i][X[0][i]])      
        polys = Tool.generatepoly(polys, X, self.repoly, self.KX, self.KY)   
        X1 = copy.deepcopy(X[2])
        pid = [X[0][i] + i * self.NumR for i in range(len(self.polys))]
        res = Packing.LowerLeft(polys, X1, self.width, self.NFP_history, self.step,pid, showplt,cached_state=oldstate, cached_index=idx) 
        return res.current_length, res.toposscore, res.outstate 


    def operator_func(self, solution, num, i = None ):
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
        polys = [[]for i in range(len(self.polys))] 
        for i in range(len(X[0])):
            polys[i] = copy.deepcopy(self.polys[i][X[0][i]]) 
        polys = Tool.generatepoly(polys, X, self.repoly, self.KX, self.KY)  
        Xp = copy.deepcopy(X[2])
        newX = copy.deepcopy(X)
        pid = [X[0][i] + i * self.NumR for i in range(len(self.polys))]
        if len(Xp) < 4:
            return newX
        idx1 = random.randint(2, len(Xp)-2)       
        pcopy = copy.deepcopy(Xp[idx1:min(idx1+4,len(Xp))])
        Xp = Xp[:idx1]
        res = Packing.LowerLeft(polys, Xp, self.width, self.NFP_history, self.step,pid)
        bestl = res.current_length
        try:
            cached = res.outstate 
        except:
            return newX
        bestidx = None
        for idx2, p in  enumerate(pcopy):
            Xp[-1] = copy.deepcopy(p)
            res = Packing.LowerLeft(polys, Xp, self.width, self.NFP_history, self.step,pid,cached_state=cached, cached_index=idx1-2)                    
            resl = res.current_length 
            if resl < bestl:
                bestl = resl
                bestp = copy.deepcopy(p)
                bestidx = idx2
        if not bestidx:
            return newX
        newX[2][idx1+bestidx] = copy.deepcopy(newX[2][idx1-1])      
        newX[2][idx1-1] = copy.deepcopy(bestp)      
        return newX
  
        
    
    
    def minlengthR(self, X):
        polys = [[]for i in range(len(self.polys))] 
        for i in range(len(X[0])):
            polys[i] = copy.deepcopy(self.polys[i][X[0][i]]) 
        polys = Tool.generatepoly(polys, X, self.repoly, self.KX, self.KY)   
        Xp = copy.deepcopy(X[2])
        newX = copy.deepcopy(X)
        pid = [X[0][i] + i * self.NumR for i in range(len(self.polys))]
        if len(Xp)<2:
            return newX
        else:
            idx1 = random.randint(2, len(Xp))     
        Xp = Xp[:idx1]
        idx2 = random.randint(0, len(Xp[-1])-1)
        idx = Xp[-1][idx2]        
        pcopy = copy.deepcopy(polys[idx])
        pid = [X[0][k] + k*self.NumR for k in range(len(polys))]
        rold = copy.deepcopy(pid[idx]) - idx * self.NumR 
        res = Packing.LowerLeft(polys, Xp, self.width,  self.NFP_history, self.step,pid)
        bestl = res.current_length
        try:
            cached = res.outstate
        except:
            return newX      
        for r in range(self.NumR)[1:]:             
            polys[idx] = Tool.rotate(pcopy, r, self.NumR)            
            pid[idx] = (rold + r)%self.NumR + idx * self.NumR   
            res = Packing.LowerLeft(polys, Xp, self.width, self.NFP_history, self.step,pid,cached_state=cached, cached_index=idx1-2)           
            resl = res.current_length          
            if resl < bestl:                             
                bestl = resl
                newX[0][idx] = (rold + r)%self.NumR            
        return newX
              
    
    
    def generate_initial_solution1(self):
        self.p = [copy.deepcopy(self.polys[i][0]) for i in range(len(self.polys))]    
        moves = Tool.Different_area(self.p, self.repoly, self.KX, self.KY,self.upbound)        
        solution = [[0 for i in range(len(self.polys))],moves,[]]        
        self.p = Tool.generatepoly(self.p, solution, self.repoly, self.KX, self.KY)       
        self.similar_list = Tool.matchpolys(self.p, theta=0.9)
        
        def polygon_area(poly):
            s = 0.0
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                s += x1 * y2 - x2 * y1
            return abs(s) / 2.0
        areas = []
        for i in range(len(self.p)):
            total_area = 0.0
            for comp_poly in self.p[i]:
                total_area += polygon_area(comp_poly)
            areas.append((i, total_area))
        sorted_indices = [idx for idx, _ in sorted(areas, key=lambda x: -x[1])]
        LP_O = copy.deepcopy(self.p)
        LP_P = []
        for i in range(len(self.p)):
            pcopy = copy.deepcopy(self.p[sorted_indices[i]])
            bestl = 9999           
            solution[2].append([sorted_indices[i]])
            LP_P.append([sorted_indices[i]])
            for r in range(self.NumR):                
                LP_O[sorted_indices[i]] = Tool.rotate(pcopy, r, self.NumR)
                pid = [solution[0][k] + k*self.NumR for k in range(len(LP_O))]
                pid[sorted_indices[i]] = r + sorted_indices[i] * self.NumR                             
                res = Packing.LowerLeft(LP_O, LP_P, self.width, self.NFP_history, self.step,pid)                    
                resl = res.current_length 
                if resl < bestl:
                    bestl = resl
                    solution[0][sorted_indices[i]] = r
            LP_O[sorted_indices[i]] = Tool.rotate(pcopy, solution[0][sorted_indices[i]], self.NumR)        
        
        value, _ = self.objective_func(solution) 
        for i in reversed(sorted_indices):
            Xnew = self.operator_func(solution, 5, i)
            Xnew = self.operator_func(Xnew, 6, i)
            valuenew, scorenew = self.objective_func(Xnew)
            if valuenew < value:
                value  = valuenew
                solution = copy.deepcopy(Xnew)       
        return solution, value, areas
    
    def generate_initial_solution(self):
       temp_p = [copy.deepcopy(self.polys[i][0]) for i in range(len(self.polys))]
       moves = Tool.Different_area(temp_p, self.repoly, self.KX, self.KY, self.upbound)
       temp_solution = [[0] * len(self.polys), moves, []]   # 暂时无 packing
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
           total_area = 0.0
           for comp in temp_p[i]:
               total_area += polygon_area(comp)
           areas.append((i, total_area))
       sorted_indices = [idx for idx, _ in sorted(areas, key=lambda x: -x[1])]
       angles = [0] * len(self.polys)
       for idx in sorted_indices:
           min_width = float('inf')
           best_rots = []
           for r in range(self.NumR):
               rotated = self.polys[idx][r]         
               all_x = [pt[0] for comp in rotated for pt in comp]
               if not all_x:
                   width = 0
               else:
                   width = max(all_x) - min(all_x)
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
       value, _,s_ = self.objective_func(solution)  
       return solution, value, areas  
      
    def checkidx(self, x1, x2):
        list1_2 = x1[2]
        list2_2 = x2[2]
        arr1 = x1[0]
        arr2 = x2[0]
        sub1_0 = list1_2[0]
        sub2_0 = list2_2[0]
        if abs(len(sub1_0) - len(sub2_0)) > 0.1:
            return None
        for j in range(len(sub1_0)):
            a = sub1_0[j]
            b = sub2_0[j]
            if abs(a - b) > 0.1:
                return None
            if abs(arr1[a] - arr2[b]) > 0.1:
                return None
        n = len(list1_2)
        for i in range(1, n):
            sub1 = list1_2[i]
            sub2 = list2_2[i]
            if abs(len(sub1) - len(sub2)) > 0.1:
                return i-1
            for j in range(len(sub1)):
                a = sub1[j]
                b = sub2[j]
                if abs(a - b) > 0.1:
                    return i-1
                if abs(arr1[a] - arr2[b]) > 0.1:
                    return i-1
        return n-1

                      
        
    def run(self, ):
        s_i, v_i,self.area = self.generate_initial_solution()
        self.Inisol = s_i
        self.Inival = v_i
        self.bestsolution = copy.deepcopy(s_i)        
        self.bestvalue = v_i       
        for iteration in range(10000):
            if  self.number_evaluations >= self.maxgen:
                break
            k = 1    
            s_i, v_i, s_1, v_1, state_1 = self.shake(s_i, v_i)   
            while k <= self.k_max:
                action = random.randint(1, 8)
                s_2 = self.operator_func(s_1, action)                                
                copyindex = self.checkidx(s_1, s_2)                   
                v_2, _ ,state_2 = self.objective_func(s_2, oldstate = state_1, idx = copyindex,)
                
                if v_2 < v_1:
                    s_1 = copy.deepcopy(s_2)
                    state_1 = copy.deepcopy(state_2)
                    v_1 = v_2
                    k = 1  
                else:
                    k += 1
                    
                if v_2 < self.bestvalue:
                    self.bestsolution = copy.deepcopy(s_2)
                    self.bestvalue = v_2
                    self.t1 = 0                  
                else:
                    self.t1 += 1            
            if v_1 < v_i:
                s_i = copy.deepcopy(s_1)
                v_i = v_1    



    


























