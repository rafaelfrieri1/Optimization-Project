import haversine
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from pyomo.environ import *

annDataSet = pd.read_csv('./Database_And_Test_Instances_Feedback_System/3.TrainingTable.csv')

features = annDataSet[['Delivery_Norm', 'Shipping_Norm', 'Damage_rate_Norm', 'Population_Norm', 'Employment_Norm', 'Salary_Norm']].values
output = annDataSet['Pnum'].values

features_train, features_test, output_train, output_test = train_test_split(features, output, random_state = 7)

mlp = MLPRegressor(
  hidden_layer_sizes = (16, 16),
  activation = 'relu',
  solver = 'lbfgs',
  learning_rate = 'constant',
  max_iter = 10000,
  random_state = 3,
  n_iter_no_change = 10,
  max_fun = 15000
)

mlp.fit(features_train, output_train)

clientDataSet = pd.read_csv('./Database_And_Test_Instances_Feedback_System/1.Demand_Distribution.csv')
facilityDataSet = pd.read_csv('./Database_And_Test_Instances_Feedback_System/7.Candidate_set_VP_10_VQ_100.csv')

secondaryFacilitiesIndex = facilityDataSet[facilityDataSet['Primary Facility Candidate Index'] == 'Secondary Facility Candidate Index'].index.values[0]

supplierCoordinates = np.array([30.178336, 120.190069])
primaryFacilitiesCoordinates = facilityDataSet[['Latitude', 'Longitude']].values[0:secondaryFacilitiesIndex].astype(float)
secondaryFacilitiesCoordinates = facilityDataSet[['Latitude', 'Longitude']].values[secondaryFacilitiesIndex+1:].astype(float)
clientsCoordinates = clientDataSet[['Latitude', 'Longitude']].values

CS = 0.0006
CP = 0.0024
CK = 0.012
rP = 9 # Free to experiment
rQ = 2 # Free to experiment
l = 1 # Free to experiment

nP = len(primaryFacilitiesCoordinates)
nQ = len(secondaryFacilitiesCoordinates)

ettaiP = [250000]*nP
ettajQ = [25000]*nQ
gammaiP = [0.035]*nP
gammajQ = [0.140]*nQ

dK = clientDataSet['Demand'].values.tolist()

deltask = haversine.haversine_vector(clientsCoordinates, supplierCoordinates, unit = 'mi', comb = True).tolist()
deltasi = haversine.haversine_vector(primaryFacilitiesCoordinates, supplierCoordinates, unit = 'mi', comb = True).tolist()
deltaij = haversine.haversine_vector(secondaryFacilitiesCoordinates, primaryFacilitiesCoordinates, unit = 'mi', comb = True).tolist()
deltaiip = haversine.haversine_vector(primaryFacilitiesCoordinates, primaryFacilitiesCoordinates, unit = 'mi', comb = True).tolist()
deltajk = haversine.haversine_vector(clientsCoordinates, secondaryFacilitiesCoordinates, unit = 'mi', comb = True).tolist()

S = range(len(deltasi))
VP = range(len(deltasi[0]))
VQ = range(len(deltaij[0]))
K = range(len(dK))

previousResult = 0
currentResult = 10
epsilon = 0.000001

while abs(previousResult - currentResult) > epsilon:
  model = ConcreteModel()

  model.gip = Var(VP, domain = Binary)
  model.gjq = Var(VQ, domain = Binary)
  model.xsi = Var(S, VP, domain = NonNegativeReals)
  model.yij = Var(VP, VQ, domain = NonNegativeReals)
  model.zjk = Var(VQ, K, domain = NonNegativeReals)
  model.tiip = Var(VP, VP, domain = NonNegativeReals)

  model.constraints = ConstraintList()

  model.constraints.add(sum(model.gip[i] for i in VP) == rP)
  model.constraints.add(sum(model.gjq[j] for j in VQ) == rQ)

  for i in VP:
    model.constraints.add(sum(model.xsi[s, i] for s in S) <= sum(dK[k]*model.gip[i] for k in K))
    model.constraints.add(sum(model.yij[i, j] for j in VQ) <= sum(dK[k]*model.gip[i] for k in K))
    model.constraints.add(sum(model.tiip[i, ip] for ip in VP if i != ip) <= sum(dK[k]*model.gip[i] for k in K))
    model.constraints.add(sum(model.xsi[s, i] for s in S) + sum(model.tiip[ip, i] for ip in VP if i != ip) == sum(model.yij[i, j] for j in VQ) + sum(model.tiip[i, ip] for ip in VP if i != ip))
    model.constraints.add(sum(model.tiip[ip, i] for ip in VP if i != ip) == l*sum(model.yij[i, j] for j in VQ))

  for ip in VP:
    model.constraints.add(sum(model.tiip[i, ip] for i in VP if i != ip) <= sum(dK[k]*model.gip[ip] for k in K))

  for j in VQ:
    model.constraints.add(sum(model.yij[i, j] for i in VP) <= sum(dK[k]*model.gjq[j] for k in K))
    model.constraints.add(sum(model.zjk[j, k] for k in K) <= sum(dK[k]*model.gjq[j] for k in K))
    model.constraints.add(sum(model.yij[i, j] for i in VP) == sum(model.zjk[j, k] for k in K))

  for k in K:
    model.constraints.add(sum(model.zjk[j, k] for j in VQ) == dK[k])

  model.objective = Objective(
    expr = sum(sum(CS*deltasi[s][i]*model.xsi[s, i] for i in VP) for s in S) + sum(sum(CP*deltaij[i][j]*model.yij[i, j] for j in VQ) for i in VP) + sum(sum(CP*deltaiip[i][ip]*model.tiip[i, ip] for ip in VP if i != ip) for i in VP) + sum(sum(CK*deltajk[j][k]*model.zjk[j, k] for k in K) for j in VQ) + sum(ettaiP[i]*model.gip[i] for i in VP) + sum(ettajQ[j]*model.gjq[j] for j in VQ) + sum(gammaiP[i] * (sum(model.yij[i, j] for j in VQ) + sum(model.tiip[i, ip] for ip in VP if i != ip)) for i in VP) + sum(gammajQ[j] * sum(model.zjk[j, k] for k in K) for j in VQ),
    sense = minimize
  )

  results = SolverFactory('cplex').solve(model)

  newFeatures = features.copy()

  for k in K:
    for j in VQ:
      if model.zjk[j, k].value > 0.0001:
        newFeatures[k][0] = features[k][0]*deltajk[j][k]/deltask[0][k]
        break

  dK = mlp.predict(newFeatures)
  
  previousResult = currentResult
  currentResult = value(model.objective)
