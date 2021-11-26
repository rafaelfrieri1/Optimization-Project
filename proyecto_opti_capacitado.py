from pyomo.environ import *
import numpy as np

cijr = []
fjr = []
di = []
ajr = []

with open("./Capacitated_Test_Instances/RAND2000_120-80.txt") as instanceFile:
  n = int(instanceFile.readline())
  clientsFacilitiesSizeStr = instanceFile.readline().strip().split(" ")
  instanceFile.readline()
  for i in range(int(clientsFacilitiesSizeStr[0])):
    di.append(float(instanceFile.readline().strip()))
  instanceFile.readline()
  for k in range(2):
    for i in range(1, len(clientsFacilitiesSizeStr)):
      if k == 0:
        fjr.append([])
        for j in range(int(clientsFacilitiesSizeStr[i])):
          fjr[i-1].append(float(instanceFile.readline().strip()))
      else:
        ajr.append([])
        for j in range(int(clientsFacilitiesSizeStr[i])):
          ajr[i-1].append(float(instanceFile.readline().strip()))
    instanceFile.readline()
  for k in range(len(clientsFacilitiesSizeStr)-1):
    cijr.append([])
    for i in range(int(clientsFacilitiesSizeStr[k])):
      cijr[k].append([])
      line = instanceFile.readline().strip().split()
      for j in line:
        cijr[k][i].append(float(j))
    instanceFile.readline()
  instanceFile.close()

#cijr = [
#  [
#    [1,2,3],
#    [6,4,2],
#    [5,9,7],
#    [10,1,2.5]
#  ],
#  [
#    [10,15,2,30,7],
#    [12,11,17,8,9],
#    [25,20,37,40,28]
#  ],
#  [
#    [50,38],
#    [33,62],
#    [27,70],
#    [50,60],
#    [100,5]
#  ]
#]

#fjr = [
#  [28,15,30],
#  [12,10,20,18,15],
#  [100,95]
#]

#di = [150,35,105,95]

#ajr = [
#  [200,200,200],
#  [100,57,250,130,170],
#  [250,400]
#]

I = range(len(cijr[0]))
J = range(len(cijr[0][0]))

R = range(1, len(fjr)+1)
RM1 = range(1, len(fjr))

yjrIndexes = []
wrabIndexes = []

for r in R:
  for j in range(len(fjr[r-1])):
    yjrIndexes.append((r,j))

for r in RM1:
  for a in range(len(cijr[r])):
    for b in range(len(cijr[r][0])):
      wrabIndexes.append((r,a,b))

model = ConcreteModel()

model.vij1 = Var(I, J, domain=NonNegativeReals)
model.yjr = Var(yjrIndexes, domain=Binary)
model.wrab = Var(wrabIndexes, domain=NonNegativeReals)

model.constraints = ConstraintList()

for i in I:
  model.constraints.add(sum(model.vij1[i, j] for j in J) >= di[i])

for j1 in J:
  model.constraints.add(sum(model.wrab[1, j1, j2] for j2 in range(len(cijr[1][j1]))) >= sum(model.vij1[i, j1] for i in I))

for r in range(2, len(fjr) + 1):
  if(r <= len(fjr) - 1):
    for a in range(len(cijr[r])):
      model.constraints.add(sum(model.wrab[r, a, b] for b in range(len(cijr[r][a]))) >= sum(model.wrab[r-1, b, a] for b in range(len(cijr[r-1]))))
      model.constraints.add(sum(model.wrab[r-1, b, a] for b in range(len(cijr[r-1]))) <= ajr[r-1][a]*model.yjr[r, a])
  else:
    for a in range(len(fjr[r-1])):
      model.constraints.add(sum(model.wrab[r-1, b, a] for b in range(len(cijr[r-1]))) <= ajr[r-1][a]*model.yjr[r, a])
  
  for a in range(len(cijr[r-1])):
    model.constraints.add(sum(model.wrab[r-1, a, b] for b in range(len(fjr[r-1]))) <= ajr[r-2][a]*model.yjr[r-1, a])
  
model.objective = Objective(
  expr = sum(sum(cijr[0][i][j1]*model.vij1[i,j1] for j1 in J) for i in I) + sum(sum(sum(cijr[r][jr][jr1]*model.wrab[r,jr,jr1] for jr1 in range(len(cijr[r][0]))) for jr in range(len(cijr[r]))) for r in RM1) + sum(sum(fjr[r-1][j]*model.yjr[r,j] for j in range(len(fjr[r-1]))) for r in R),
  sense = minimize
)

results = SolverFactory('cplex').solve(model)
results.write()

#if results.solver.status:
#  model.pprint()

#model.constraints.display()