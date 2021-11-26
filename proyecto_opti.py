from pyomo.environ import *
import numpy as np

cijr = []
fjr = []
with open("./Test_Instances/RAND2000_120-80.txt") as instanceFile:
  n = int(instanceFile.readline())
  clientsFacilitiesSizeStr = instanceFile.readline().strip().split(" ")
  instanceFile.readline()
  for phase in range(2):
    for i in range(1, len(clientsFacilitiesSizeStr)):
      if phase == 0:
        fjr.append([])
        for j in range(int(clientsFacilitiesSizeStr[i])):
          fjr[i-1].append(float(instanceFile.readline().strip().split(" ")[0]))
      else:
        nextLine = instanceFile.readline().strip()
        cijr.append([])
        j = 0
        while(nextLine != ""):
          nextLineNumbers = nextLine.split(" ")
          cijr[i-1].append([])
          for nextLinenumber in nextLineNumbers:
            cijr[i-1][j].append(float(nextLinenumber))
          j+=1
          nextLine = instanceFile.readline().strip()
    instanceFile.readline()
  instanceFile.close()
#cijr = np.array(
#  [
#    np.array([
#      [1,1000,1000],
#      [15000,2,15000],
#      [20000,20000,3],
#      [1,40000,40000]
#    ]),
#    np.array([
#      [10000,5,10000,10000,20000],
#      [30000,23000,3,24000,18000],
#      [28000,35000,21000,4,33000]
#    ]),
#    np.array([
#      [16, 14],
#      [2, 18000],
#      [20000, 3],
#      [20000, 2],
#      [24, 25]
#    ])
#  ]
#)

#fjr = np.array(
#  [
#    [10,15,20],
#    [17,20,25,30,18],
#    [48,50]
#  ]
#)

I = range(len(cijr[0]))

J = range(len(cijr[0][0]))

R = range(1, len(fjr)+1)
RM1 = range(1, len(fjr))

yjrIndexes = []
zirabIndexes = []
for r in R:
  for j in range(len(fjr[r-1])):
    yjrIndexes.append((r,j))

for i in I:
  for r in RM1:
    for a in range(len(cijr[r])):
      for b in range(len(cijr[r][0])):
        zirabIndexes.append((i,r,a,b))

model = ConcreteModel()

model.vij1 = Var(I, J, domain=Binary)
model.yjr = Var(yjrIndexes, domain=Binary)
model.zirab = Var(zirabIndexes, domain=Binary)

model.constraints = ConstraintList()

for i in I:
  model.constraints.add(sum(model.vij1[i, j] for j in J) == 1)
  for j1 in J:
    model.constraints.add(sum(model.zirab[i,1,j1,b] for b in range(len(cijr[1][0]))) == model.vij1[i,j1])
    model.constraints.add(model.vij1[i,j1] <= model.yjr[1,j1])
  for r in range(2, len(fjr) + 1):
    if(r <= len(fjr) - 1):
      for a in range(len(cijr[r])):
        model.constraints.add(sum(model.zirab[i,r,a,b] for b in range(len(cijr[r][0]))) == sum(model.zirab[i,r-1,bp,a] for bp in range(len(cijr[r-1]))))
    
    for b in range(len(cijr[r-1][0])):
      model.constraints.add(sum(model.zirab[i,r-1,a,b,] for a in range(len(cijr[r-1]))) <= model.yjr[r, b])

model.objective = Objective(
  expr = sum(sum(cijr[0][i][j1]*model.vij1[i,j1] for j1 in J) for i in I) + sum(sum(sum(sum(cijr[r][a][b]*model.zirab[i,r,a,b] for b in range(len(cijr[r][0])))for a in range(len(cijr[r]))) for r in RM1) for i in I) + sum(sum(fjr[r-1][j]*model.yjr[r,j] for j in range(len(fjr[r-1]))) for r in R),
  sense=minimize
)

results = SolverFactory('cplex').solve(model)
results.write()
#if results.solver.status:
#  model.pprint()

#model.constraints.display()