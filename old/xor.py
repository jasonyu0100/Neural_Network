from random import seed
from random import uniform
from math import exp,ceil

def sigmoid(x,derivative=False):
    if derivative:return x*(1-x)
    else:return 1 / (1 + exp(-x))

class NeuralNetwork:
    def __init__(self,*layers,outputMapping=None):
        self.layerInfo = layers
        self.n_inputs = self.layerInfo[0]
        self.n_outputs = self.layerInfo[-1]
        self.layers = self.initializeLayers()
        self.outputMapping = outputMapping
    
    def initializeLayers(self):
        newLayers = []
        prevLayerSize = self.n_inputs
        for layerSize in self.layerInfo[1:]:
            currentLayer = Layer(layerSize,prevLayerSize)
            prevLayerSize = layerSize
            newLayers.append(currentLayer)
        return newLayers

    def getActivations(self,inputs):
        activations = [inputs]
        for layer in self.layers:
            layerActivation = layer.activation(activations[-1])
            activations.append(layerActivation)
        return activations

    def forwardPropogation(self,inputs):
        return self.getActivations(inputs)[-1]

    def errorPropogation(self,expected):
        for layerIndex in reversed(range(len(self.layers))):
            layer = self.layers[layerIndex]
            errors = []
            if layerIndex != (len(self.layers)-1):
                for neuronIndex,neuron in enumerate(layer.neurons):
                    currentError = 0
                    for nextLayerNeuron in self.layers[layerIndex+1].neurons:
                        weight = nextLayerNeuron.weights[neuronIndex]
                        errorSignal = nextLayerNeuron.errorSignal
                        currentError += weight * errorSignal
                    errors.append(currentError)
            else:
                for outputIndex,neuron in enumerate(layer.neurons):
                    expectedOutput = expected[outputIndex]
                    output = neuron.currentActivation
                    errors.append(expectedOutput - output)
            for neuronIndex,neuron in enumerate(layer.neurons):
                neuron.errorSignal = errors[neuronIndex] * sigmoid(neuron.currentActivation,derivative=True)

    def updateWeights(self,inputs,learningRate):
        for layer in self.layers:
            for neuron in layer.neurons:
                for weightIndex in range(layer.prevLayerSize):
                    inputVal = inputs[weightIndex]
                    neuron.weights[weightIndex] += learningRate * neuron.errorSignal * inputVal
                neuron.bias += learningRate * neuron.errorSignal
            inputs = [neuron.currentActivation for neuron in layer.neurons]

    def trainNetwork(self,dataset,learningRate,numEpoch):
        for epoch in range(numEpoch):
            sumError = 0
            for row in dataset:
                outputs = self.forwardPropogation(row)
                expected = row[-1]
                sumError += sum((expected[index]-outputs[index])**2 for index in range(numOutputs))
                self.errorPropogation(expected)
                self.updateWeights(row,learningRate)
            if epoch % int(ceil(numEpoch/20)) == 0:
                print('>epoch={0}, lrate={1}, error={2}'.format(epoch, learningRate, sumError))

    def predict(self,inputs):
        return NeuralNetwork.finalOutputIndex(self.forwardPropogation(inputs))

    def accuracy(self,dataset):
        correct = 0
        outputMappingBool = self.outputMapping != None
        for row in dataset:
            prediction = self.outputMapping[NN.predict(row)] if outputMappingBool else NN.predict(row)
            expected = self.outputMapping[NN.finalOutputIndex(row[-1])] if outputMappingBool else NN.finalOutputIndex(row[-1])
            if prediction == expected:
                correct += 1
            print('Expected={0}, Got={1}'.format(expected, prediction), 'For Inputs {}'.format(row[:-1]))
        accuracy = 100 * (correct / len(dataset))
        print('Accuracy={0:.2f}%'.format(accuracy))
        return accuracy

    @staticmethod
    def finalOutputIndex(outputs):
        return outputs.index(max(outputs))

    @staticmethod
    def importNN(fileName):
        with open(fileName) as f:
            layerInformation = [int(layerSize) for layerSize in f.readline().strip().split()]
            NN = NeuralNetwork(*layerInformation)
            prevLayerSize = layerInformation[0]
            for layer in NN.layers:
                layerSize = len(layer.neurons)
                for neuron in layer.neurons:
                    weights = [float(weight) for weight in f.readline().strip().split()]
                    for weightIndex in range(prevLayerSize):
                        neuron.weights[weightIndex] = weights[weightIndex]
                    neuron.bias = weights[-1]
                prevLayerSize = layerSize
        return NN

    def exportNN(self,fileName):
        file = open(fileName,'w')
        file.write(' '.join(str(layerSize) for layerSize in self.layerInfo) + '\n')
        for layer in self.layers:
            for neuron in layer.neurons:
                file.write(' '.join(str(value) for value in (neuron.weights + [neuron.bias])) + '\n')
        file.close()

    def displayLayers(self):
        for layerIndex,layer in enumerate(self.layers):
            print("Layer Index: " + str(layerIndex))
            for neuronIndex,neuron in enumerate(layer.neurons):
                for weightIndex,weight in enumerate(neuron.weights):
                    indexes = (layerIndex,neuronIndex,weightIndex)
                    print(' Index: '+str(indexes) + ' Weight: ' + str(weight))
                print(' Bias: ' + str(neuron.bias))
    
    def displayErrorSignal(self):
        for layerIndex,layer in enumerate(self.layers):
            print("Layer Index: " + str(layerIndex))
            for neuronIndex,neuron in enumerate(layer.neurons):
                index = (layerIndex,neuronIndex)
                print(' Index: ' + str(index) + ' Error: ' + str(neuron.errorSignal))

class Layer:
    def __init__(self,layerSize,prevLayerSize):
        self.prevLayerSize = prevLayerSize
        self.neurons = [Neuron(index,prevLayerSize) for index in range(layerSize)]
    
    def activation(self,inputs):
        return [neuron.activation(inputs) for neuron in self.neurons]

class Neuron:
    def __init__(self,index,prevLayerSize):
        self.weights = [uniform(-1,1) for i in range(prevLayerSize)]
        self.bias = uniform(-1,1)
        self.errorSignal = 0
        self.currentActivation = 0

    def activation(self,inputs):
        activation = self.bias + sum(inputs[weightIndex] * weight for weightIndex,weight in enumerate(self.weights))
        self.currentActivation = sigmoid(activation)
        return self.currentActivation

seed(1)

#Information
dataset = [[0,0,[0,1]],
            [0,1,[1,0]],
            [1,0,[1,0]],
            [1,1,[0,1]]]
#dataset = [[10,[1,0]],
#            [20,[1,0]],
#            [-30,[0,1]],
#            [-10,[0,1]]]
outputMapping = {0:'True',1:'False'}
numInputs = len(dataset[0]) - 1
numOutputs = len(dataset[0][-1])
numEpochs = 1
learningRate = 0.01

#Setup and Training
NN = NeuralNetwork(numInputs, 2, numOutputs,outputMapping=outputMapping)
NN.trainNetwork(dataset, learningRate, numEpochs)
# accuracy = NN.accuracy(dataset)
# NN.exportNN('XOR.txt')
