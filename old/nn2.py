from random import uniform,seed
from math import exp

def sigmoid(x,derivative=False):
  if derivative: return x * (1-x)
  else: return exp(x) / (exp(x) + 1)

def encode_one_hot_vectors(non_encoded_labels,length):
  encoded_vectors = []
  for index in non_encoded_labels:
    encoded_vectors.append([1 if i == index else 0 for i in range(length)])
  return encoded_vectors

class Network:
  def __init__(self,layers):
    self.layers = layers
    self.network_size = len(self.layers)
    self.input_size = layers[0].input_size
    self.output_size = layers[-1].size
    self.init_layers()

  def init_layers(self):
    for layer_index,current_layer in enumerate(self.layers):
      if layer_index == 0: previous_layer_size = self.input_size
      else: previous_layer_size = self.layers[layer_index-1].size
      current_layer.init_neurons(previous_layer_size)

  def forward_propogation(self,input_features,output_layer=True):
    layer_output = input_features
    network_activation = [layer_output]
    for layer_index in range(self.network_size):
      layer_output = self.layers[layer_index].layer_output(layer_output)
      network_activation.append(layer_output)
    return network_activation[-1] if output_layer else network_activation

  def cost_function(self,output,actual,derivative=False):
    if derivative == False: return (output - actual) ** 2
    else: return 2 * (output - actual)

  def total_cost(self,final_output_layer,label):
    return sum(self.cost_function(final_output_layer[i],label[i]) for i in range(len(label)))

  def update_network(self,learning_rate):
    for layer in self.layers: layer.update_layer(learning_rate)

  def show_weights(self):
    for layer in self.layers:
      print('  -  '.join(str(list(round(weight,3) for weight in neuron.weights)) for neuron in layer.neurons))


class Layer:
  def __init__(self,size,activation_function,input_size=False):
    self.size = size
    self.activation_function = activation_function
    self.input_size = input_size

  def init_neurons(self,previous_layer_size):
    self.neurons = [Neuron(self.activation_function,previous_layer_size) for neuron_index in range(self.size)]

  def layer_output(self,inputs):
    layer_output = []
    print('adfsafdsasdf',inputs)
    for neuron in self.neurons:
      neuron_input = neuron.neuron_input_function(inputs)
      print(neuron_input)
      neuron_output = neuron.activation_function(neuron_input)
      layer_output.append(neuron_output)
    return layer_output
  
  def update_layer(self,learning_rate):
    for neuron in self.neurons: neuron.update_neuron(learning_rate)

class Neuron:
  def __init__(self,activation_function,previous_layer_size):
    self.previous_layer_size = previous_layer_size
    self.activation_function = activation_function
    self.weights = self.random_weight_initialization()
    self.bias = 0

  def neuron_input_function(self,inputs):
    weight_values = [(self.weights[weight_index] * input_val) for weight_index,input_val in enumerate(inputs)]
    summed_weight_values = sum(weight_values) + self.bias
    return summed_weight_values

  def random_weight_initialization(self):
    return [uniform(-1,1) for i in range(self.previous_layer_size)]

#Network Information
seed(1)
nn = Network([Layer(2,activation_function=sigmoid,input_size=2), Layer(2,activation_function=sigmoid)])
labels = [0,1,1,1]
features = [[0,0],[0,1],[1,0],[1,1]]
one_hot_encoded_labels = encode_one_hot_vectors(labels,2)
data = list(zip(one_hot_encoded_labels,features))

print(nn.forward_propogation([0,0],output_layer=False))
nn.show_weights()
