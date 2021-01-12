# Machine learning - Pulsar detector
# Ref (we all learn somewhere) https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# Implements a simple 'multi layer perceptron' neural network (fully connected), with variable number and size of layers
# Datset: https://www.kaggle.com/charitarth/pulsar-dataset-htru2
# By Alexis Angers [https://github.com/Alexsimulation]
import random
import math
import pickle
import csv


# Single hidden neuron in a network
class neuron:
    
    def __init__(self, size):
        # Class variables
        self.v = 0      # value, zero by default
        self.vp = 0     # derivative value, zero by default
        self.w = []     # weights, empty by default
        self.wu = []    # weights update, used for batch gradient descent
        self.b = 0      # bias, zero by default
        self.bu = 0     # bias update, used for batch gradient descent
        self.e = 0      # error signal, used for backpropagation
        
        self.b = random.uniform(-1, 1)
        for i in range(size):
            self.w.append( random.uniform(-1, 1) )
            self.wu.append( 0 )
    
    def get(self, x):
        self.v = self.b
        for i in range(len(self.w)):
            self.v += self.w[i]*x[i]
        
        # Activation
        try:
            self.v = 1/(1 + math.exp(-1*self.v))
        except OverflowError:
            self.v = float('inf')
        
        # Derivative of activation
        self.vp = self.v * (1 - self.v)
        
        return self.v


# Fully connected hidden/output layer
class layer:
    
    def __init__(self, num_neurons, input_size):
        # Class variables
        self.ns = [] # Array of neurons, start empty
        
        for i in range(num_neurons):
            ni = neuron(input_size)
            self.ns.append( ni )
    
    def get(self, x):
        v = []
        for i in range(len(self.ns)):
            v.append( self.ns[i].get(x) )
        
        return v


# Neural network class
class network:
    
    def __init__(self, layers_size ):
        # Class variables
        self.la = []  # Array of layers
        self.x = []   # last input values
        self.z = []   # value of the network output
        self.ls = []  # Network loss value
        self.lr = 0.5 # network learning rate
        
        num_layers = len(layers_size)
        
        # Num layers includes the input and output layers, but the network will actually have num_layers-1 layers
        for i in range(num_layers-1):
            # Input layer of hidden/output layers is always the output of the last layer
            lai = layer(layers_size[i+1], layers_size[i])
            self.la.append( lai )
    
    def get(self, x):
        self.x = x[:]
        self.z = x[:]
        z2 = x[:]
        
        # For each layer, compute the value of the layer output, and pass it to the next layer
        for i in range(len(self.la)):
            z2 = self.la[i].get(z2)
            self.z.append(z2)
        
        return self.z
    
    def loss(self, answ):
        # Make sure the answ is in a list object
        if not isinstance(answ, list):
            answ = [answ]
        
        self.ls = []
        for i in range(len(answ)):
            zi = self.z[len(self.z)-1][i]
            ai = answ[i]
            self.ls.append( -1*(ai*math.log(zi) + (1-ai)*math.log(1-zi)) )    # Log loss
        
        return self.ls
    
    def reset_error(self):
        endlay = len(self.la)-1
        
        # Work backward through all layers
        for i in range(endlay, -1, -1): # Loop over all layers, in reverse
            for j in range(len(self.la[i].ns)): # Loop over layer i's neurons
                self.la[i].ns[j].e = 0
    
    def backprop_error(self, answ):
        endlay = len(self.la)-1
        
        # Make sure the answ is in a list object
        if not isinstance(answ, list):
            answ = [answ]
        
        self.reset_error() # Resets error values to zero
        
        # Work backward through all layers
        for i in range(endlay, -1, -1): # Loop over all layers, in reverse
            if i == endlay:
                for j in range(len(self.la[i].ns)): # Loop over layer i's neurons
                    dLdv = ( self.la[i].ns[j].v - answ[j] ) / ( self.la[i].ns[j].v - self.la[i].ns[j].v**2 )    # Derivative of the loss function
                    self.la[i].ns[j].e = dLdv * self.la[i].ns[j].vp
            else:
                for j in range(len(self.la[i].ns)): # Loop over layer i's neurons
                    for k in range(len(self.la[i+1].ns)): # Loop over layer i+1's neuron
                        self.la[i].ns[j].e += self.la[i+1].ns[k].w[j] * self.la[i+1].ns[k].e * self.la[i].ns[j].vp
    
    def update_weights_stochastic(self):
        endlay = len(self.la)-1
        # Work backward through all layers
        for i in range(endlay, -1, -1): # Loop over all layers, in reverse
            if i != 0:
                for j in range(len(self.la[i].ns)): # Loop over layer i's neurons
                    self.la[i].ns[j].b -= self.lr * self.la[i].ns[j].e
                    for k in range(len(self.la[i].ns[j].w)): # Loop over layer i's neuron j's weights
                        self.la[i].ns[j].w[k] -= self.lr * self.la[i].ns[j].e * self.la[i].ns[j].v
            else:
                for j in range(len(self.la[i].ns)): # Loop over layer i's neurons
                    self.la[i].ns[j].b -= self.lr * self.la[i].ns[j].e
                    for k in range(len(self.la[i].ns[j].w)): # Loop over layer i's neuron j's weights
                        self.la[i].ns[j].w[k] -= self.lr * self.la[i].ns[j].e * self.x[i]
    
    def save_weight_updates_batch(self, batch_size):
        endlay = len(self.la)-1
        # Work backward through all layers
        for i in range(endlay, -1, -1): # Loop over all layers, in reverse
            if i != 0:
                for j in range(len(self.la[i].ns)): # Loop over layer i's neurons
                    self.la[i].ns[j].bu += (self.lr * self.la[i].ns[j].e) / batch_size
                    for k in range(len(self.la[i].ns[j].w)): # Loop over layer i's neuron j's weights
                        self.la[i].ns[j].wu[k] += (self.lr * self.la[i].ns[j].e * self.la[i].ns[j].v) / batch_size
            else:
                for j in range(len(self.la[i].ns)): # Loop over layer i's neurons
                    self.la[i].ns[j].bu += (self.lr * self.la[i].ns[j].e) / batch_size
                    for k in range(len(self.la[i].ns[j].w)): # Loop over layer i's neuron j's weights
                        self.la[i].ns[j].wu[k] += (self.lr * self.la[i].ns[j].e * self.x[i]) / batch_size
    
    def update_weights_batch(self):
        endlay = len(self.la)-1
        # Work backward through all layers
        for i in range(endlay, -1, -1): # Loop over all layers, in reverse
            for j in range(len(self.la[i].ns)): # Loop over layer i's neurons
                self.la[i].ns[j].b -= self.la[i].ns[j].bu # Update it's bias
                self.la[i].ns[j].bu = 0 # Reset bias update
                for k in range(len(self.la[i].ns[j].w)): # Loop over layer i's neuron j's weights
                    self.la[i].ns[j].w[k] -= self.la[i].ns[j].wu[k] # Update it's weight
                    self.la[i].ns[j].wu[k] = 0 # Reset weight update
    
    def train_step_stochastic(self, x, a):
        self.get(x)
        self.backprop_error(a)
        self.update_weights_stochastic()
    
    def train_step_batch(self, x, a, batch_size, step):
        if step == (batch_size-1):
            # If current step is the last one (batch size-1), update the weights
            self.update_weights_batch()
        else:
            # If current step is not a batch size multiple, compute and save the weights updates
            self.get(x)
            self.backprop_error(a)
            self.save_weight_updates_batch(batch_size)
    
    def disp(self):
        for i in range(len(self.la)):
            print(' - Layer',i)
            for j in range(len(self.la[i].ns)):
                print('  - Neuron',j)
                print('   - w:',self.la[i].ns[j].w)
                print('   - b:',self.la[i].ns[j].b)
            
            print('')
    
    def save(self, outfile):
        with open(outfile, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        
        print('Network saved to output file',outfile)
    
    def load(self, infile):
        with open(infile, 'rb') as input:
            new_n = pickle.load(input)
        
        print('Newtork loaded from file',infile)
        
        return new_n

# Utility functions

def meanc(x):
    # Check if member is a list
    if isinstance(x[0], list):
        y = []
        for i in range(len(x[0])):
            y.append(0)
        
        for i in range(len(x)):
            for j in range(len(y)):
                y[j] += x[i][j]/len(x)
        
    else:
        y = 0
        for i in range(len(x)):
            y += x[i]/len(x)
    
    return y


def sumc(x):
    # Check if member is a list
    if isinstance(x[0], list):
        y = []
        for i in range(len(x[0])):
            y.append(0)
        
        for i in range(len(x)):
            for j in range(len(y)):
                y[j] += x[i][j]
        
    else:
        y = 0
        for i in range(len(x)):
            y += x[i]
    
    return y


def sigmoid(x):
    return 1/(1 + math.exp(-1*x))


def create_train(n_in, n_out):
    # Create a training array of values to fit a function
    x = []
    a = []
    
    for v in range(10): # Loop over all the batches
        xb = []
        ab = []
        for i in range(50): # Loop over all the sets in the batch i
            xi = []
            ai = []
            for j in range(n_out):
                ai.append(0)
            
            for j in range(n_in):
                xi.append(random.uniform(0,1))
                for k in range(n_out):
                    ai[k] += xi[j]/n_in
            
            for k in range(n_out):
                if k == 0:
                    ai[k] = (1/math.pi * math.atan(-1*ai[k]) + 0.5)**2
                else:
                    ai[k] = 1/( 1 + math.exp(-1*ai[k]) )
            
            xb.append( xi )
            ab.append( ai )
        
        x.append(xb)
        a.append(ab)
    
    return x, a


def create_test(n_in, n_out):
    # Create a test array
    xt = []
    at = []
    
    for i in range(2000):
        xti = []
        ati = []
        for j in range(n_out):
            ati.append(0)
        
        for j in range(n_in):
            xti.append(random.uniform(0,1))
            for k in range(n_out):
                ati[k] += xti[j]/n_in
        
        for k in range(n_out):
            if k == 0:
                ati[k] = (1/math.pi * math.atan(-1*ati[k]) + 0.5)**2
            else:
                ati[k] = 1/( 1 + math.exp(-1*ati[k]) )
        
        xt.append( xti )
        at.append( ati )
    
    return xt, at


def load_data_train():
    print('Loading training data...')
    with open('data/pulsar_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        x = []
        xb = []
        a = []
        ab = []
        n0 = 0
        n1 = 1
        
        batch_size = 100
        for row in csv_reader:  # Read each row as a list
            arow = float(row[len(row)-1]) # Read the answer in the last line
            if arow == 1:
                if n0 >= n1:
                    # If the answer is 1, just add it
                    al = [ arow ]
                    ab.append(al)
                    xl = [] # Read the input in first line
                    for i in range(0, len(row)-2):
                        xl.append( float(row[i]) )
                    
                    xb.append(xl)
                    n1 += 1
            else:
                if n1 >= n0: # To unbias database, only add the 0 data is there's the same amount or more of 1s
                    al = [ arow ]
                    ab.append(al)
                    xl = [] # Read the input in first line
                    for i in range(0, len(row)-2):
                        xl.append( float(row[i]) )
                    
                    xb.append(xl)
                    n0 += 1
            
            # If batch is full, save and reset batch variables
            if len(xb) == batch_size:
                x.append(xb)
                a.append(ab)
                xb = []
                ab = []
            
            line_count += 1
    
    # Add last batch that isn't full
    if not len(xb) == 0:
        x.append(xb)
        a.append(ab)
    
    return x, a


def load_data_test():
    print('Loading testing data...')
    with open('data/pulsar_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        x = []
        a = []
        
        for row in csv_reader:  # Read each row as a list
            al = [ float(row[len(row)-1]) ] # Read the answer in the last line
            
            a.append(al)
            
            xl = [] # Read the input in first line
            for i in range(0, len(row)-2):
                xl.append( float(row[i]) )
            
            x.append(xl)
            
            line_count += 1
    
    return x, a


# Run function, handles the command line interface logic
def run(x, a, xt, at, n):
    run = 0
    
    while run != -1:
        
        choice = input('Enter a command: ')
        print('')
        
        if choice[:5] == 'train':
            
            # Get number of runs from input
            numvalid = 1
            try:
                num2run = int(choice[5:])
            except:
                numvalid = 0
                print('Invalid run number. Please enter an integer.')
            
            if numvalid == 1:
                # Loop over runs number
                for k in range(num2run):
                    
                    loss = []
                    # Check if training data is separated in batches or if it's just a stochastic run
                    if isinstance(x[0], list):
                        if isinstance(x[0][0], list):
                            # This means x[0] is a set of data sets -> do a batch run
                            batch_size = len(x[0])
                            
                            for i in range(len(x)):     # Loop over all training batches
                                for j in range(len(x[i])):      # Loop over all the sets in the batch i
                                    n.train_step_batch(x[i][j], a[i][j], batch_size, j)
                                    loss.append(n.loss(a[i][j]))
                                
                                print('Batch',i+1,'/',len(x),'done        ',end="\r")
                            
                        else:
                            # This means x[0] is a set of values -> stochastic run
                            for i in range(len(x)):     # Loop over training values
                                n.train_step_stochastic(x[i], a[i])
                                loss.append(n.loss(a[i]))
                        
                        if k%1 == 0:   # Each 1 run
                            if (k > 0)|(run == 0):
                                print(run,' : ',meanc(loss))
                        
                        # After a run, end
                        run += 1
                
                print(run,' : ',meanc(loss))
            
        elif choice[:4] == 'test':
            # Loop over the test batches
            dev = []
            dev0 = []
            dev1 = []
            c0 = 0
            c0r = 0
            c1 = 0
            c1r = 0
            for i in range(len(at[0])):
                dev.append(0)
                dev0.append(0)
                dev1.append(0)
            
            for i in range(len(xt)):
                z = n.get(xt[i]) # Get network output for xt[i]
                z = z[len(z)-1]
                for j in range(len(z)): # Loop over output
                    dev[j] += abs(z[j] - at[i][j])/len(xt)
                    if at[i][j] < 0.5:
                        dev0[j] += abs(z[j] - at[i][j])
                        c0 += 1
                        if round(z[j]) == at[i][j]:
                            c0r += 1
                    else:
                        dev1[j] += abs(z[j] - at[i][j])
                        c1 += 1
                        if round(z[j]) == at[i][j]:
                            c1r += 1
                
                if i%50 == 0:
                    print('Progress:',100*i/(len(xt)-1),'%         ',end="\r")
            
            for i in range(len(dev0)):
                dev0[i] = dev0[i]/c0
            
            for i in range(len(dev1)):
                dev1[i] = dev1[i]/c1
            
            print('Average overall deviation:',dev)
            print('Average dev. on negatives:',dev0)
            print('Average dev. on positives:',dev1)
            print('Right overall --- :',100*(c0r+c1r)/(c0+c1),'%')
            print('Right on negatives:',100*c0r/c0,'%')
            print('Right on positives:',100*c1r/c1,'%')
            
        elif choice[:3] == 'try':
            xstrtest = choice[4:].split()
            
            if len(xstrtest) == len(x[0]):
                xtest = []
                for i in range(len(xstrtest)):
                    xtest.append(float(xstrtest[i]))
                
                ztest = n.get(xtest)
                print('Answer: ',ztest[len(ztest)-1])
                
            else:
                print('Invalid input values length.')
                
        elif choice[:7] == 'example':
            try:
                totest = int(choice[7:])
            except:
                totest = random.randint(0, len(xt))
                print('No test index provided, using random integer ',totest)
            
            print('')
            print('Input:',xt[totest])
            print('Dataset answer:',at[totest])
            zc = n.get(xt[totest])
            print('Network answer:',zc[len(zc)-1])
            
        elif choice[:4] == 'edit':
            toedit = choice[4:]
            if toedit == ' learning rate':
                print('Current learning rate:',n.lr)
                lrin = input('Enter new learning rate: ')
                try:
                    lrin = float(lrin)
                    n.lr = lrin
                except:
                    print('Invalid learning rate.')
        elif choice[:5] == 'print':
            n.disp()
        
        elif choice[:4] == 'save':
            try:
                outfile = choice[4:]
                out_valid = 1
            except:
                out_valid = 0
                print('Invalid file string.')
            
            if out_valid == 1:
                n.save(outfile)
            
        elif choice[:4] == 'load':
            try:
                infile = choice[4:]
                in_valid = 1
            except:
                in_valid = 0
                print('Invalid file string.')
            
            if in_valid == 1:
                try:
                    n = n.load(infile)
                except:
                    print('Failed to load file',infile)
            
        elif choice[:4] == 'exit':
            run = -1
            
        else:
            print('Invalid command')
        
        if not run == -1:
            print('')


# Main function
if __name__ == '__main__':
    
    random.seed(10)
    
    # Print generic program info
    print('')
    print('A Machine Learning Project [version xx - public release]')
    print('(c) 2021 Alexis Angers [https://github.com/Alexsimulation]. Private and educational use only.')
    print('')
    
    x, a = load_data_train()   # Load training data
    xt, at = load_data_test()  # Load testing data
    print('')
    
    n_in = len(xt[0])    # Number of data inputs
    n_out = len(at[0])   # Number of data outputs
    
    n = network([n_in, 16, 8, n_out])    # Create new network sized for MNIST uses
    
    run(x, a, xt, at, n)    # Main run
    
    print('End training')


