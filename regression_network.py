from neuralnetwork import *
import random
random.seed(1)

function = lambda x,y: x * y
labels = []
features = []
for i in range(1,10):
  for j in range(1,10):
    labels.append([i * j])
    features.append([i,j])

data_set = list(zip(labels,features))

nn = Network([Layer(2,'leaky_relu',input_size=2),
              Layer(8,'leaky_relu'),
              Layer(1,'linear')],
              network_type='regression')

# epochs = 10000
# learning_rate = 0.001
# adjustment_rate = 1000
# batch_size = len(data_set)
# nn.train_network(data_set,learning_rate,adjustment_rate,batch_size,epochs)

# for i in range(1,5):
#   for j in range(1,5):
#     print(i,j,(nn.output_function([i,j])))

# nn.export_network('regression.txt')

nn2 = Network.import_network('regression.txt','regression')
for i in range(1,20):
  print(i,i,(nn2.output_function([i,i])))