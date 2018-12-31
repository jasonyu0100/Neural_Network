from math import exp,ceil,floor
from random import seed,uniform,gauss

def sigmoid(x,derivative=False):
  if derivative: return x * (1-x)
  else: return exp(x) / (exp(x) + 1)

def relu(x,derivative=False):
  if derivative:
    if x >= 0: return 1
    else: return 0
    return max(x,0)
  else: return max(x,0)

def leaky_relu(x,derivative=False):
  if derivative:
    if x >= 0: return 1
    else: return 0.1
  else:
    if x >= 0: return x
    else: return 0.1 * x

def linear(x,derivative=False):
  if derivative: return 1
  else: return x

def softmax(x,total,derivative=False):
  if derivative: pass
  else: return exp(x) / total

def encode_one_hot_vectors(unencoded_vectors,length):
  return [[1 if i == j else 0 for j in range(length)] for i in unencoded_vectors]

def learning_rate_formula(initial_learning_rate,iteration,adjustment_rate):
  val = initial_learning_rate / (1 + (iteration / adjustment_rate))
  return val

class Network:
  EPOCH_DISPLAY_COUNT = 20
  ACTIVATION_FUNCTIONS = {'relu':relu,'sigmoid':sigmoid,'linear':linear,'softmax':softmax,'leaky_relu':leaky_relu}
  def __init__(self,layers,network_type):
    self.layers = layers
    self.network_size = len(self.layers)
    self.input_size = layers[0].input_size
    self.output_size = layers[-1].size
    self.network_type = network_type
    self.init_layers()

  def init_layers(self):
    for layer_index,current_layer in enumerate(self.layers):
      if layer_index == 0: previous_layer_size = self.input_size
      else:  previous_layer_size = self.layers[layer_index-1].size
      current_layer.init_neurons(previous_layer_size)

  def forward_propogation(self,inputs):
    layer_activation = inputs
    network_activation = [layer_activation]
    for layer_index in range(self.network_size):
      layer_activation = self.layers[layer_index].layer_activation(layer_activation)
      network_activation.append(layer_activation)
    return network_activation

  def output_function(self,inputs):
    return self.forward_propogation(inputs)[-1]
  
  def cost_function(self,output,actual,derivative=False):
    if derivative == False: return (output - actual) ** 2
    else: return 2 * (float(output) - float(actual))

  def total_cost(self,outputs,actual_outputs):
    return float(sum(self.cost_function(outputs[i],actual_outputs[i]) for i in range(len(actual_outputs))))
  
  def cost_derivative(self,layer_index,neuron_index):
    return sum(neuron.cost_deriv * neuron.activation_deriv * neuron.weights[neuron_index] for neuron in self.layers[layer_index + 1].neurons)

  def backpropogation(self,network_activation,actual_outputs):
    for layer_index,layer in reversed(list(enumerate(self.layers))):
      if layer.activation_function == softmax: 
        softmax_total = sum(exp(x) for x in layer.neurons)
      for neuron_index,neuron in enumerate(layer.neurons):
        if layer_index == self.network_size - 1: neuron.cost_deriv = self.cost_function(neuron.output_activation,actual_outputs[neuron_index],derivative=True)
        else: neuron.cost_deriv = self.cost_derivative(layer_index,neuron_index)
        if layer.activation_function == softmax: neuron.activation_function(neuron.output_activation, softmax_total, derivative=True)
        else: neuron.activation_deriv = neuron.activation_function(neuron.output_activation,derivative=True)
        for weight_index,weight in enumerate(neuron.weights):
          neuron.weight_deltas[weight_index] -= neuron.cost_deriv * neuron.activation_deriv * network_activation[layer_index][weight_index]
        neuron.bias_delta -= neuron.cost_deriv * neuron.activation_deriv

  def train_network(self,data_set,learning_rate,adjustment_rate,batch_size,epochs):
    for epoch in range(epochs):
      sum_error,correct = 0,0
      for count,data in enumerate(data_set):
        actual_outputs,inputs = data
        network_activation = self.forward_propogation(inputs)
        outputs = network_activation[-1]
        sum_error += self.total_cost(outputs,actual_outputs)
        self.backpropogation(network_activation,actual_outputs)
        if self.classify(outputs) == self.classify(actual_outputs): correct += 1
        if count % batch_size: self.update_weights(learning_rate_formula(learning_rate,epoch,adjustment_rate))
      self.update_weights(learning_rate_formula(learning_rate,epoch,10000))
      if epoch % int(ceil(float(epochs)/self.EPOCH_DISPLAY_COUNT)) == 0: 
        print('Epoch: {} Total Cost: {} Accuracy: {}'.format(epoch,sum_error,float(correct)/len(data_set)))

  def update_weights(self,learning_rate):
    for layer in self.layers:
      for neuron in layer.neurons:
        for weight_index,weight_delta in enumerate(neuron.weight_deltas):
          neuron.weights[weight_index] += weight_delta * learning_rate
          neuron.weight_deltas[weight_index] = 0
        neuron.bias += neuron.bias_delta * learning_rate
        neuron.bias_delta = 0

  def show_weights(self):
    for layer_index,layer in enumerate(self.layers):
      print('Layer {}'.format(layer_index))
      for neuron_index,neuron in enumerate(layer.neurons):
        print('--- Neuron {}'.format(neuron_index))
        for weights in neuron.weights:
          print('------ Weights {}'.format(round(weights,3)))
        print('------ Bias {}'.format(round(neuron.bias,3)))
  
  def show_activations(self,network_activation):
    for layer_index,layer_activation in enumerate(network_activation):
      print('Layer {}'.format(layer_index))
      for neuron_index,neuron_activation in enumerate(layer_activation):
        print('Neuron {}: {}'.format(neuron_index,round(neuron_activation,3)))

  def classify(self,outputs):
    return max(range(len(outputs)),key=lambda index:outputs[index])

  def classify_verbose(self,outputs,output_mapping):
    return output_mapping[max(range(len(outputs)),key=lambda index:outputs[index])]

  def export_network(self,filename):
    with open(filename,'w') as f:
      f.write(str(len(self.layers)) + '\n')
      for layer_index,layer in enumerate(self.layers):
        if layer_index == 0: f.write(str(layer.input_size) + '\n')
        f.write(str(layer.size) + '\n')
        f.write(str(layer.activation_function_name) + '\n')
        weights = [neuron.weights for neuron in layer.neurons]
        biases = [neuron.bias for neuron in layer.neurons]
        f.write(str(weights) + '\n')
        f.write(str(biases) + '\n')

  @staticmethod  
  def import_network(filename,network_type):
    layers = []
    with open(filename) as f:
      network_size = eval(f.readline().strip())
      for layer_index in range(network_size):
        if layer_index == 0: input_size = eval(f.readline().strip())
        else: input_size = False
        layer_size = eval(f.readline().strip())
        activation_function = f.readline().strip()
        weights = eval(f.readline().strip())
        print(weights)
        biases = eval(f.readline().strip())
        layers.append(Layer(layer_size,activation_function,input_size=input_size,neuron_weights=weights,neuron_biases=biases))
    return Network(layers,network_type)

class Layer:
  def __init__(self,size,activation_function_name,input_size=None,neuron_weights=None,neuron_biases=None):
    self.size = size
    self.activation_function_name = activation_function_name
    self.activation_function = Network.ACTIVATION_FUNCTIONS[self.activation_function_name]
    self.input_size = input_size
    self.neuron_weights = neuron_weights
    self.neuron_biases = neuron_biases
  
  def init_neurons(self,previous_layer_size,neuron_weights=None,neuron_biases=None):
    if self.neuron_weights == None: self.neuron_weights = [self.xavier_weight_initialization(previous_layer_size) for neuron_index in range(self.size)]
    if self.neuron_biases  == None: self.neuron_biases  = self.xavier_weight_initialization(self.size)
    self.neurons = [Neuron(self.neuron_weights[neuron_index],self.neuron_biases[neuron_index],self.activation_function) for neuron_index in range(self.size)]

  def random_weight_initialization(self,previous_layer_size):
    return [uniform(-1,1) for weight_index in range(previous_layer_size)]

  def xavier_weight_initialization(self,previous_layer_size):
    x = 2 if self.activation_function == relu else 1 #Relu initialization has sd of 2/n while other of 1/n
    return [gauss(0,float(x)/float(previous_layer_size)) for i in range(previous_layer_size)]

  def layer_activation(self,previous_layer_activation):
    for neuron in self.neurons: neuron.output_activation = neuron.activation_function(neuron.input_function(previous_layer_activation))
    return [neuron.output_activation for neuron in self.neurons]

class Neuron:
  def __init__(self,weights,bias,activation_function):
    self.activation_function = activation_function
    self.weights = weights
    self.bias = bias
    self.output_activation = None
    self.activation_deriv = None
    self.cost_deriv = None
    self.weight_deltas = [0 for weight_index in range(len(weights))]
    self.bias_delta = 0

  def input_function(self,inputs):
    return sum([inputs[weight_index] * weight for weight_index,weight in enumerate(self.weights)]) + self.bias

