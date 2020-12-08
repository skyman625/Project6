#!/usr/bin/python3
import numpy as np

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
#import numpy as np
from TSPClasses import *
import heapq
import itertools
import copy

#O(n^2) space and time
def fixMatrix(inMatrix, cost):
	matrix = copy.deepcopy(inMatrix)
	for i in range(len(matrix)):
		minNum = math.inf
		for j in range(len(matrix[i])):
			if matrix[i][j] == math.inf:
				pass
			elif matrix[i][j] < minNum:
				minNum = matrix[i][j]
		if minNum < math.inf and minNum != 0:
			for j in range(len(matrix[i])):
				matrix[i][j] -= minNum
			cost += minNum
	for j in range(len(matrix[0])):
		minNum = math.inf
		for i in range(len(matrix)):
			if matrix[i][j] == math.inf:
				pass
			elif matrix[i][j] < minNum:
				minNum = matrix[i][j]
		if minNum < math.inf and minNum != 0:
			for i in range(len(matrix)):
				matrix[i][j] -= minNum
			cost += minNum
	return matrix, cost

#O(n^2) space and time
def getNextMatrix(inMatrix, row, col, cost):
	matrix = copy.deepcopy(inMatrix)
	assert(row < len(matrix))
	assert(col < len(matrix[row]))
	cost += matrix[row][col]
	for i in range(len(matrix[row])):
		matrix[row][i] = math.inf
	for i in range(len(matrix)):
		matrix[i][col] = math.inf
	matrix[col][row] = math.inf
	matrix, cost = fixMatrix(copy.deepcopy(matrix), cost)
	return matrix, cost


class Node:
	def __init__(self, bound, partialPath, rcMatrix):
		self.bound = bound
		self.partialPath = partialPath
		self.rcMatrix = rcMatrix

	def __lt__(self, other):
		if self.bound // len(self.partialPath) < other.bound // len(other.partialPath):
			return True
		return False

class Edge:
	def __init__(self, origin, destination, cost):
		self.origin = origin
		self.destination = destination
		self.cost = cost


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < math.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def getGreedyPathAndCost(self, startCity):
		cities = self._scenario.getCities()
		city = startCity
		ncities = len(cities)
		path = []
		path.append(city)
		unvisitedCities = set(cities)
		unvisitedCities.remove(cities[0])
		while not len(path) == ncities:
			bestCity = None
			bestDistance = math.inf
			for i in range(ncities):
				if cities[i] in unvisitedCities and city.costTo(cities[i]) < bestDistance:
					bestCity = cities[i]
			if bestCity != None:
				path.append(bestCity)
				unvisitedCities.remove(bestCity)
				city = bestCity
			else:
				return path, math.inf

		bssf = TSPSolution(path)
		print(bssf.cost)
		return path, bssf.cost

	#O(nlogn) time and  O(n) space
	def greedy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		start_time = time.time()
		minPath, minCost = self.getGreedyPathAndCost(cities[0])
		for i in range(1, ncities):
			path, cost = self.getGreedyPathAndCost(cities[i])
			if cost < minCost:
				minPath = path
				minCost = cost

		'''
		city = cities[0]
		path = []
		path.append(city)
		unvisitedCities = set(cities)
		unvisitedCities.remove(cities[0])
		skipBestRoute = False
		bestCityIndex = None
		while not len(path) == ncities and time.time() - start_time < time_allowance:
			bestCity = None
			bestDistance = math.inf
			for i in range(ncities):
				if cities[i] in unvisitedCities and city.costTo(cities[i]) < bestDistance:
					bestCity = cities[i]
					bestCityIndex = i
			if skipBestRoute:
				skipBestRoute = False
				for i in range(ncities):
					if i != bestCityIndex and cities[i] in unvisitedCities and city.costTo(cities[i]) < bestDistance:
						bestCity = cities[i]
			if bestCity != None:
				path.append(bestCity)
				unvisitedCities.remove(bestCity)
				city = bestCity
			else:
				unvisitedCities.add(path[len(path) - 1])
				del path[len(path) - 1]
				skipBestRoute = True
				city = path[len(path) - 1]

		'''

		end_time = time.time()
		bssf = TSPSolution(minPath)
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	#O(n) space and time
	def expand(self, node):
		nodesInPath = node.partialPath
		lastNodeNum = nodesInPath[len(nodesInPath) - 1]
		newNodes = []
		for i in range(len(self._scenario.getCities())):
			if i not in nodesInPath:
				newPath = nodesInPath.copy()
				newPath.append(i)
				newMatrix, newCost = getNextMatrix(node.rcMatrix, i, lastNodeNum, node.bound)
				if newCost < math.inf:
					newNode = Node(newCost, newPath, newMatrix)
					newNodes.append(newNode)

		return newNodes

	def getCities(self, indices):
		cities = []
		for index in indices:
			cities.append(self._scenario.getCities()[index])
		return cities
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
	#O(n^2 * number of nodes checked) time complexity
	#O(n^2 * max number of states on the heap) space complexity
	def branchAndBound( self, time_allowance=60.0 ):
		start_time = time.time()
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		costMatrix = []
		count = 0
		pruned = 0
		total = 0
		maxQueueLength = 0

		#get the initial bssf

		# O(nlogn) space and O(n) time
		greedyResults = self.greedy()
		costSoFar = greedyResults['cost']
		bssf = greedyResults['soln']
		results['cost'] = costSoFar
		results['time'] = time.time() - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = maxQueueLength
		results['total'] = total
		results['pruned'] = pruned

		# Get the initial matrix
		for i in range(ncities):
			costs = []
			for j in range(0, ncities):
				costs.append(cities[i].costTo(cities[j]))
			costMatrix.append(costs)

		#Fix the matrix (0 out rows and columns and get cost)

		#O(n^2) space and time
		costMatrix, cost = fixMatrix(costMatrix, 0)
		partialPath = [0]
		node0 = Node(cost, partialPath, costMatrix)
		nodes = [node0]
		#O(logn) time, 0(1) space
		heapq.heapify(nodes)

		while len(nodes) > 0:
			#Timeout if necessary
			if time.time() - start_time > time_allowance:
				results['cost'] = costSoFar
				results['time'] = time.time() - start_time
				results['count'] = count
				results['soln'] = bssf
				results['max'] = maxQueueLength
				results['total'] = total
				results['pruned'] = pruned
				return results
			#Keep track of max queue length
			if len(nodes) > maxQueueLength:
				maxQueueLength = len(nodes)
			nextNode = heapq.heappop(nodes)
			#if this is a node we should check
			if nextNode.bound < costSoFar:
				newNodes = self.expand(nextNode)
				total += len(newNodes)
				for node in newNodes:
					#check if we have found a new solution
					if len(node.partialPath) == len(cities) and node.bound < costSoFar:
						costSoFar = node.bound
						bssf = TSPSolution(self.getCities(node.partialPath))
						count += 1
						results['cost'] = costSoFar
						results['time'] = time.time() - start_time
						results['count'] = count
						results['soln'] = bssf
						results['max'] = maxQueueLength
						results['total'] = total
						results['pruned'] = pruned
					# check if it is worth putting in the priority queue
					elif node.bound < costSoFar:
						nodes.append(node)
					else:
						pruned += 1
			else:
				pruned += 1
		return results












	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass
'''
row1 = [math.inf, 1, math.inf, 0, math.inf]
row2 = [math.inf, math.inf, 1, math.inf, 0]
row3 = [math.inf, 0, math.inf, 1, math.inf]
row4 = [math.inf, 0, 0, math.inf, 6]
row5 = [0, math.inf, math.inf, 9, math.inf]

myMatrix = [row1, row2, row3, row4, row5]
copyMatrix = copy.deepcopy(myMatrix)
for i in range(1,5):
	newMatrix, newCost = getNextMatrix(copyMatrix, 0, i, 21)
	print("Matrix: ")
	for j in range(len(newMatrix)):
		print(newMatrix[j])
	#print("The OG: ")
	#for j in range(len(newMatrix)):
		#print(myMatrix[j])
	print("Cost: " + str(newCost))
'''



