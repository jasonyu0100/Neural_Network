from neuralnetwork import *
import random
random.seed(1)

#XOR FUNCTION 
labels   = [0,0,1,1]
encoded_labels = encode_one_hot_vectors(labels,2)
features = [[0,0],[1,1],[1,0],[0,1]]
data_set = list(zip(encoded_labels,features))

nn = Network([Layer(3,'leaky_relu',input_size=2),
              Layer(2,'sigmoid')],
              network_type='classification')

#Training and Learning and exporting
epochs = 100000
learning_rate = 0.001
adjustment_rate = 1000
batch_size = 1
nn.train_network(data_set,learning_rate,adjustment_rate,batch_size,epochs)
nn.show_weights()
print(nn.classify_verbose(nn.output_function([0,0]),{0:'False',1:'True'}))
nn.export_network('XOR.txt')


#Importing
nn2 = Network.import_network('XOR.txt','classification')
for i in range(2):
  for j in range(2):
    print(i,j)
    print(nn2.classify_verbose(nn2.output_function([i,j]),{0:'False',1:'True'}))