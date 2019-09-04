from neuralnetwork import *
import random
random.seed(1)

# Adding and finding if even
labels   = [1,0,0,1,0,1,1,0]
encoded_labels = encode_one_hot_vectors(labels,2)
features = [[0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1],
          ]
data_set = list(zip(encoded_labels,features))

nn = Network([Layer(5,'leaky_relu',input_size=3),
              Layer(2,'sigmoid')],
              network_type='classification')

#Training and Learning and exporting
epochs = 100000
learning_rate = 0.005
adjustment_rate = 1000
batch_size = 1
nn.train_network(data_set,learning_rate,adjustment_rate,batch_size,epochs)
nn.show_weights()
print(nn.classify_verbose(nn.output_function([0,0,0]),{0:'False',1:'True'}))
nn.export_network('EVEN.txt')


#Importing
nn2 = Network.import_network('EVEN.txt','classification')
for i in range(2):
  for j in range(2):
      for k in range(2):
        print(i,j,k)
        print(nn2.classify_verbose(nn2.output_function([i,j,k]),{0:'False',1:'True'}))