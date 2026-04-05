from BeamSearch_SA import BS_SA
from BeamSearch_DA import BS_DA
import Tool
from TabuSearch import TS
from QVNS import QVNS
from VNS import VNS
FileIndex = 1
AlgIndex = 1
Draw = True

filename = f"data/{FileIndex}.txt"
if AlgIndex == 1:
    # run beam seach algorithm with same area strategy
    polys, repoly, numR, Width = Tool.read_data_file(filename)
    res = BS_SA(polys, repoly, numR, Width, Draw )
    placed_parts, placed_rots, offsets, length = res.best_solution 
    time = res.Time
    print("container length:", round(length,2) )   
    print("computation time:", round(time,2)) 
elif AlgIndex == 2:
    # run beam seach algorithm with different area strategy
    polys, repoly, numR, Width = Tool.read_data_file(filename)
    res = BS_DA(polys, repoly, numR, Width, Draw )
    placed_parts, placed_rots, offsets, length = res.best_solution 
    time = res.Time
    print("container length:", round(length,2) )   
    print("computation time:", round(time,2)) 
elif AlgIndex == 3:
    # run tabu search
    polys, repoly, numR, Width = Tool.read_data_file(filename)
    res = TS(polys, repoly, numR, Width, Draw)
    print("container length:", round(res.bestvalue,2) )   
    print("computation time:", round(res.Time,2)) 
elif AlgIndex == 4:
    # run variable neighborhood search
    polys, repoly, numR, Width = Tool.read_data_file(filename)
    res = QVNS(polys, repoly, numR, Width, Draw)
    print("container length:", round(res.bestvalue,2) )   
    print("computation time:", round(res.Time,2))     
elif AlgIndex == 5:
    # run Q-leanring enhanced variable neighborhood search
    polys, repoly, numR, Width = Tool.read_data_file(filename)
    res = VNS(polys, repoly, numR, Width, Draw)
    print("container length:", round(res.bestvalue,2) )   
    print("computation time:", round(res.Time,2))     

    
      
       