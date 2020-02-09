import numpy as np
import sys
# #####################################################
# #####################################################
# #  TODO: Classes for Creating a Neural network    ###
# #####################################################
# #####################################################

# logic tables used to train networks
or_tbl = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])            # input table to train for an OR function
and_tbl = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])           # input table to train for an AND function
not_tbl = np.array([[0,1],[1,1]])                               # input table for NOT function
xor_tbl = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])           # input table to train for an XOR function

slide_input = np.array([[.05, .10]])
slide_output = np.array([[.01], [.99]])
slide_H = np.array([[.15, .20],[.25, .3]])
slide_o = np.array([[.4, .45],[.5, .55]])
s_x = slide_input
s_y = slide_output



# the usual binary inputs to a truth table
# just counting in binary from 0 to 3 for the four different
# patterns of the binary inputs for the two inputs to the gates
bin_in = or_tbl[:, [0,1]]

# the expected outputs for the binary inputs of the different logic gates
or_out = or_tbl[:, [2]]
and_out = and_tbl[:, [2]]
not_out = not_tbl[:, [1]]
xor_out = xor_tbl[:, [2]]


def handle_example(verbose=3):
    HLW = [[.15, .2],  # h1
          [.25, .30]]  # h2
    b1 = [.35, .35]

    OLW = [[.4, .45],  # o1
          [.5, .55]]  # o2
    b2 = [.6, .6]

    # input array
    input_size = 2
    # designate number of neurons in the hidden layer
    hidden_layers = 2
    # designate number of neurons in the put layer layer
    output_layers = 2
    # input array
    x0 = np.array([[.05, .1]])
    # output array
    yt = np.array([[.01], [.99]])
    eta1 = .5
    eta_min = .05
    epochs = 1
    kmax = int(epochs/.3)
    error1 = 'se'
    gnn = NeuralNetwork(input_size, number_layers=hidden_layers, neurons_layer=(hidden_layers, output_layers),
                        activations=('logistic', 'logistic'), error=error1, epochs=epochs, threshold=.0001,
                        eta=eta1, w=(HLW, OLW), b=[b1, b2], eta_min=eta_min, kmax=max,
                        verbose=verbose, weight_list=(-.1, .1))
    gnn.train(x0, yt)
    return gnn

def handle_example1000(verbose=-1, ):
    HLW = [[.15, .2],  # h1
          [.25, .30]]  # h2
    b1 = [.35, .35]

    OLW = [[.4, .45],  # o1
          [.5, .55]]  # o2
    b2 = [.6, .6]

    # input array
    input_size = 2
    # designate number of neurons in the hidden layer
    hidden_layers = 2
    # designate number of neurons in the put layer layer
    output_layers = 2
    # input array
    x0 = np.array([.05, .1])
    # output array
    yt = np.array([[.01], [.99]])
    eta1 = .5
    eta_min = .05
    epochs = 1000
    kmax = epochs/.3
    error1 = 'bce'
    gnn = NeuralNetwork(input_size, number_layers=hidden_layers, neurons_layer=(hidden_layers, output_layers),
                        activations=('linear', 'logistic'), error=error1, epochs=epochs, threshold=.0002,
                        eta=eta1, w=(HLW, OLW), b=[b1, b2], eta_min=eta_min, kmax=int(epochs/.3),
                        verbose=verbose, weight_list=(-.1, .1))
    gnn.train(x0, yt)
    return gnn

def handle_and(verbose=-1, ):

    HLW = [[.15, .2],  # h1
          [.25, .30]]  # h2
    b1 = [.35, .35]

    OLW = [[.4, .45],  # o1
          [.5, .55]]  # o2
    b2 = [.6, .6]

    # input array
    input_size = 2
    # designate number of neurons in the put layer layer
    output_layers = 1
    # input array
    x0 = np.array(bin_in)
    # output array
    yt = np.array([[0],[0],[0],[1]])
    print('binary inputs {}'.format(x0))
    print('and outputs {}'.format(yt))
    eta1 = 10
    eta_min = .05
    epochs = 300
    kmax = epochs/.3
    error1 = 'bce'
    gnn = NeuralNetwork(input_size, number_layers=1, neurons_layer=(output_layers, ),
                        activations=('logistic',), error=error1, epochs=epochs, threshold=.0002,
                        #eta=eta1, w=(HLW, OLW), b=[b1, b2], eta_min=eta_min, kmax=int(epochs/.3),
                        eta=eta1, w=None, b=[[-1]], eta_min=eta_min, kmax=int(epochs/.3),
                        verbose=verbose, weight_list=(-.5, .5, 1, 2, -2, -1))
    gnn.train(x0, yt)
    print(np.around(gnn.predict(x0,yt), 0))

    return gnn

def handle_xor(verbose=-1, ):
    HLW = [[.15, .2],  # h1
           [.25, .30]]  # h2
    b1 = [.35, .35]

    OLW = [[.4, .45],  # o1
           [.5, .55]]  # o2
    b2 = [.6, .6]

    # input array
    input_size = 2
    # designate number of neurons in the put layer layer
    hidden_layers=2
    output_layers = 1
    # input array
    x0 = np.array(bin_in, dtype=np.float64)
    # output array
    yt = np.array([[0], [1], [1], [0]])
    print('binary inputs {}'.format(x0))
    print('and outputs {}'.format(yt))
    eta1 = .4           # .9 no, works .5
    eta_min = .5
    epochs = 1500       # 1500,   1500
    #kmax =int(epochs / .5)
    kmax = 10
    error1 = 'bce'
    gnn = NeuralNetwork(input_size, number_layers=2, neurons_layer=(hidden_layers, output_layers),
                        activations=('logistic', 'logistic',), error=error1, epochs=epochs, threshold=.0002,
                        # eta=eta1, w=(HLW, OLW), b=[b1, b2], eta_min=eta_min, kmax=int(epochs/.3),
                        eta=eta1, w=None, b=[[-1, -1], [-1]], eta_min=eta_min, kmax=kmax,
                        verbose=verbose, weight_list=(-.5, .5, 1, 2, -2, -1), update_eta=False)
    gnn.train(x0, yt)
    y_loglike = np.around(gnn.predict(x0, yt), 0)
    yp = [ yll for yll in y_loglike]

    print('outputs from network:\n{}'.format(yp))
    print('accuracy: {}%'.format(accuracy(yt, yp) * 100))
    return gnn

def load_settings(method, verbose=True):
    """
        This method is used to process the command line argument and perform the desired task
    :param method: which of the 3 tasks to perform. options are:
                        * 'example'
                            - run 1 epoch of the class example
                            - Note: can be run for 1000 epochs for slide value comparison
                        * 'and'
                            - create and train a perceptron for the and function
                        * 'xor'
                            - a) create and attempt to train an perceptron for the xor function
                            - b) create and attempt to train an two layer NN with a perceptron output for the xor function
    :param verbose: how much of the training you wish to have displayed on the screen
                    Options:
                        - verbose:
                                    = -1 for none but the final error result
    :return:
    """
    print('running {}'.format(method))
    if method == 'example':
        gnn = handle_example(verbose)
    elif method == 'example1000':
        gnn = handle_example1000(verbose)
    elif method == 'and':
        gnn = handle_and(verbose)
    elif method == 'xor':
        gnn = handle_xor(verbose)
    return gnn

class Neuron:
    # source for equations used to code the activation functions:
    #                https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
    # source for the equations used to code the error functions:
    #                https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    #                https://towardsdatascience.com
    #                http://wiki.fast.ai/index.php/Log_Loss
    """
        Class object representing a neuron:
                This class represents a neuron in a artificial neural network (ANN).
                duties:
                    * process input
                    * calculate an produce output
                    * calculate error (store in 1x1 array for storage into layer object, same for next two)
                    * (if output layer) calculate error methods derivative with x*W + b for gradient decent
                    * calculate the backpropagation value, weight/bias update values,
                Has methods/abilities:
                    * calculate():
                        * takes a 1xn numpy array of values for the input
                        * calls in turn
                            * process_input():
                                performs summation operation on weigted inputs and adds the bias value
                            * activation():
                               is passed the result of process_input() as the parameter and calculates
                               the neurons response
                                * produces outputs based on the activation function selected, has several options
                                    * sigmoid/logistic
                                        * log likelihood in prob_pred array
                                        * binary output based on log likelihood
                                    * linear
                                        * real number value of summation
                    * calculate_loss:
                        used to calculate the loss of the neurons predicted outputs
                    * update_weights/bias:
                        * update the weights (sensitivity to or magnitude of influence of variable)
                        * update bias or intercept(linear)/threshold(logistic/sigmoid) value for neuron
                has class variables:
                    * activation_func: the activation function used for this neuron
                    * eta: learning rate
                    * input_size: the number of inputs
                    * loss: the loss calculated for this neuron
                    * loss_prim: the deriviative loss calculated for this neuron used for GD
                    * w: initial weight values if so desired
                    * b: initial bias value if so desired
                    * ID: index into a layers neurons that id's this neuron


            Instantiate with: Neuron( input_size, eta=.01, w=None, activation='linear', error='mse', verbose=False)
             *           input_size: the number of inputs (int >= 0), pass 0 to test with just 1, for future
                                     statistical calculations. NOTE: this can also be ignored if desired if and only
                                     if a weights array is the length of number of inputs
             *       (optional) eta: learning rate for gradient descent/training (default=.01)
             *         (optional) w: numpy weights array, size of array must match input_size (default=None)
                                     if set to None the weights will be random set to either -.01 or .01
             *(optional) activation: string or integer selecting the activation function for this neuron. The
                                     options are:
                                         * 0 or 'sigmoid' for the logistic activation function
                                         * 1 or 'linear' for the linear activation function
                                         * 2 or 'relu' for the relu activation function
                                         * 3 or 'tanH' for the tanH activation function
                                         * TODO: fix this -- 4 or 'relu' for the  activation function
                                         * 5 or 'softplus' for the softplus activation function
                                         * 6 or 'arctan' for the arctan activation function
                                         * 7 or 'perceptron' for the perceptron activation function

             *     (optional) error: string, the error function for the gradient descent (default 'mse')
                                     options are:
                                        * 'mse': mean square error
                                        * 'bce': binary cross entropy loss
                                        * 'mae': mean absolute error
            *    (optional) verbose: boolean, controls how much is printed to the screen during the process
                                     (default, False), If False nothing is printed except the end results, otherwise
                                     various things during the training, process will be printed to standard out


    """
    # #####################################################################################################
    #                                 class variables for testing and such
    # #####################################################################################################
    # a dictionary used to convert from numerical or string based activation function selection
    activation_dictS = {'sigmoid':8, 'linear':1, 'relu':2, 'tanh':3, 'softplus':5, 'arctan':6, 'perceptron':7,
                        'logistic':0,}
    error_dictS = {'se':0, 'crossentropy':1, 'mae':2, 'rmse':3, 'hinge':4, 'huber':5, 'kullback':6, 'mse':2, 'acc':7,}
    ########################################################################################################
    ########################################################################################################
    def __init__(self, input_size, eta=.01, activation_fnc=0, error='se', b=1, w=None, ID=0, verbose=-1,
                 pweights=(-.01, .01), eta_min=.001, kmax=100, ):
        # intialize using passed parameters
        self.verbose=verbose
        self.input_size = input_size                              # number of input signals to process
        self.eta=eta                                              # learning rate, how fast we will attempt to approach the minimum
        self.orig_eta = eta
        self.eta_min = .5
        self.k = 0
        self.kmax = 50
        self.error = self.type_check(error, '', self.Lcase_adj)   # error method to use, use type check and Lcase_adj to set to lower case if a string
        self.activation_fnc = self.type_check(activation_fnc, '', self.Lcase_adj) # the type of activation function to use
        self.loss = np.array([0], dtype=np.float)                 # will store the loss of the neurons predicted output
        # handle the cases when there are no weights given, randomize weight values pulling from pweights list,
        # or check the passed array for error and store it
        self.w = self.process_w_param(w, pweights, input_size, verbose)
        '''
        if w is None:
            pweights = list(pweights)
            pweights.sort()
            #weights = np.linspace(weights[0], weights[1], input_size+int(np.around((.5*input_size), 0)), endpoint=True)
            self.w = np.array(np.random.choice(pweights, input_size, replace=True))
            print('w',self.w)
        else:
            # check the type of the array of the array matches the number of inputs
            if type(w) != type(np.array([0])):
                if self.verbose:
                    print('converting to numpy array...')
                w = np.array(w)
            if w.shape[0] != self.input_size:
                print('ERROR: size of weights array {} != length w array {}'.format(len(w), self.input_size))
                quit(-155)
            self.w = w
        '''
        self.b = np.array([b], dtype=np.float)                    # bias or threshold to overcome, treat as a value/weight

        self.x=np.zeros(input_size, dtype=np.float)               # will store the input to the neuron
        self.y = np.zeros(1, dtype=np.float)                      # will hold the ground truth for the current input
        # self.b = np.array([-1], dtype=np.float) # bias or threshold to overcome

        self.loss_Prime = np.array([0], dtype=np.float)           # will store the loss of the neurons predicted output
        self.act_prime = np.array([0], dtype=np.float)
        self.ID = ID                                      # ID is an integer value used to identify it in a layer
        self.pred_prob = np.array([0], dtype=np.float)    # will be used to store predicted probability for sigmoid etc.
        self.output = np.array([0], dtype=np.float)               # will be used to store output value
        self.del_w = np.zeros(input_size, dtype=np.float)         # used to update weight array
        self.del_b = np.zeros(1, dtype=np.float)                  # used to update bias value
        self.w_updates = list()                                   # will hold past update values
        self.b_updates = list()                                   # will hold past update values
        self.verbose = verbose
        self.z=None                                               # will store the result of the activation function
        # if no initial weights given
        # then set them to random values of either
        # -.1 or .1, with input_size elements

        # check for inputs for errors and quit if some are found
        self.process_activation_fnc()
        self.error_check()

    def process_w_param(self,w, pweights, input_size, verbose):
        """
            will process the weight array input parameter. If it is None the
            weights will be set to random values selected from the pweights (possible weights)
            array. Uses the input size to error check the size of given array
        :param w:
        :param pweights:
        :param input_size:
        :param verbose:
        :return:
        """
        verbose= self.verbose
        if w is None:
            if self.verbose > 1:
                print('Creating randomized initial weights....')
            pweights = list(pweights)
            #pweights.sort()
            #weights = np.linspace(weights[0], weights[1], input_size+int(np.around((.5*input_size), 0)), endpoint=True)
            self.w = np.array(np.random.choice(pweights, input_size, replace=True))
            if self.verbose > 1:
                print('weights array w set to {}'.format(self.w))
            return self.w
        else:
            # check the type of the array and convert if needed
            if type(w) != type(np.array([0])):
                if verbose > 1:
                    print('converting to numpy array...')
                w = np.array(w)
                if verbose > 1:
                    print('weights array w set to {}'.format(w))
            if w.shape[0] != self.input_size:
                print('ERROR: size of weights array {} != length w array {}'.format(len(w), self.input_size))
                quit(-155)
            # self.w = w[:]
            return w[:]

    def type_check(self, tocheck, vtype, dothis, ):
        if type(tocheck) == type(vtype):
            return dothis(tocheck)
        return tocheck

    def Lcase_adj(self, toadj):
        return toadj.lower()

    def process_activation_fnc(self, ):
        """ Will convert a string version of the activation function into a numerical one
        :return: None
        """
        if type(self.activation_fnc) == type(''):
            self.activation_fnc = self.activation_dictS[self.activation_fnc]

    def error_check(self, ):
        """ Checks input arguments for errors and stops the program if some are found
        :return: None
        """
        err_options = ['mse', 'bce', 'mse2', 'se', 'acc','mae']
        act_options = ['sigmoid', 'linear', 'relu', 'tanH', 'arctan', 'perceptron', 'softmax', 'logistic']
        if self.input_size < 0:
            print('Error: Bad Input size, the input size must be >= 0, got a value of {}'.format(self.input_size))
            quit(-96)
        if self.eta < 0:
            print('Error: Bad learning rate(eta). Eta must be greater than zero but recieved {} '.format(self.eta))
            quit(-100)
        if self.activation_fnc not in self.activation_dictS.keys() and self.activation_fnc not in self.activation_dictS.values():
            print('Error: Unknown activation function {}, must be one of:'.format(self.activation_fnc))
            print(act_options)
            quit(-104)
        if self.error not in err_options:
            print('413Error: Unknown error function {}'.format(self.error))
            quit(-107)
        return

    def processInput(self, x, w, b):
        """
                Performs summation operation for the neuron and adds bias
        :param x: inputs signals to sum
        :param w: weights of the various inputs
        :param b: bias to overcome
        :return:   the value of the summation operation on the weighted inputs
                   with the biase added (x0*w0 **** + b)
        """
        self.sigma =np.dot(x, w) + b
        return self.sigma

    def activation(self, x, w, b):
        """  Performs the activation function calculation selected when neuron
             was instantiated
             {'sigmoid':8, 'linear':1, 'relu':2, 'tanH':3, 'softplus':5, 'arctan':6, 'perceptron':7, 'logistic':0,}
        :param x:  input value numpy array from inputing neurons
        :param w:  weights on inputs
        :return:
        """
        #if self.activation_fnc == 0 or self.activation_fnc == 'sigmoid' or self.activation_fnc == 'logistic':
        if self.activation_fnc in [0, 'logistic']:
                return self.logistic(self.processInput(x,w.transpose(),b))
        elif self.activation_fnc in [8, 'sigmoid']:
                return self.sigmoid(self.processInput(x,w.transpose(),b))
        elif self.activation_fnc in [1, 'linear']:
            return self.linear(self.processInput(x, w.transpose(), b))
        elif self.activation_fnc in [2, 'relu']:
            return self.relu(self.processInput(x,w.transpose(),b))
        elif self.activation_fnc in [3, 'tanH']:
            return self.tanH(self.processInput(x,w.transpose(),b))
        elif self.activation_fnc in [4, 'relu']:
            return self.relu(self.processInput(x,w.transpose(),b))
        elif self.activation_fnc in [5, 'softplus']:
            return self.softplus(self.processInput(x,w.transpose(),b))
        elif self.activation_fnc in [6, 'arctan']:
            return self.garctan(self.processInput(x,w.transpose(),b))
        elif self.activation_fnc in [7, 'perceptron']:
            if self.verbose > 1:
                print('perception:')
                print('activation function {}'.format(self.activation_fnc))
            return self.perceptron(self.processInput(x,w.transpose(),b))

    def activation_funcPrime(self, x=None, w=None, b=None):
        """  Performs the derivative of the activation function calculation selected when neuron
             was instantiated
        :param x:  input value numpy array from inputing neurons
        :param w:  weights on inputs
        :return:
        """
        print('activation function', self.activation_fnc)
        if x is None:
            x = self.x[:]
        if w is None:
            w = self.w[:]
        if b is None:
            # print('bbbbbbb', self.b)
            b = self.b[:]

        if self.activation_fnc in [0, 'logistic']:
            return self.logistic_prime()
        elif self.activation_fnc == 1 or self.activation_fnc == 'linear':
            return self.linear_prime()
        elif self.activation_fnc == 2 or self.activation_fnc == 'relu':
            #return self.relu_prim(self.processInput(x, w.transpose(), b))
            return self.relu_prim()
        elif self.activation_fnc == 3 or self.activation_fnc == 'tanH':
            return self.tanH_prime()
        elif self.activation_fnc == 4 or self.activation_fnc == 'relu':
            return self.relu_prim()
        elif self.activation_fnc == 5 or self.activation_fnc == 'softplus':
            return self.softplus_prime()
        elif self.activation_fnc == 6 or self.activation_fnc == 'arctan':
            return self.arctan_prime()
        elif self.activation_fnc == 7 or self.activation_fnc == 'perceptron':
            return self.perceptron_prime(self.processInput(x,w.transpose(),b))
        elif self.activation_fnc == 8 or self.activation_fnc == 'sigmoid':
            return self.sigmoid_prime()

    def update_eta(self, k):
        self.eta = epsilon(self.orig_eta, self.eta_min, k, self.kmax)
    # called when an input is received
    # calls activation with the given input as x and stores
    # the value of the given input
    def calculate(self, x, w=None, b=None , verbose=True):
        """
                Can be called to process an input vector
                calls activation with the given input as x and stores
                the value of the given input
        :param x:
        :param w:
        :param b:
        :return: the result of calling the activation function
                 on the given input vector x with the current weight vector and bias
        """
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        self.x = x
        val = np.around(self.activation(x, w, b), 6)
        if self.verbose > 1:
            print('stuff w, x, b', self.w, self.x, self.b)
            print('val', val)
            print(self.output)
        return self.output[:]

    def calculate_error(self,yt, error=None):
        """ will calculate the error of the perceptron's
            output based on the error function chosen
        :param yt: ground truth output value
        :param yp: predicted output value from perceptron
        :param error: the error method to use
        :return:
        """
        if error is None:
            error = self.error

        if error == 'mse':
            self.loss = MSE(yt, self.output)
            print(' ****** error:', self.loss)
            return self.loss
        elif error == 'bce':
            self.loss = BCE(yt, self.output)
            return self.loss
        elif error == 'se':
            self.loss = SE(yt, self.output)
            return self.loss
        elif error == 'mae':
            self.loss = MAE(yt, self.output)
            return self.loss
        elif error == 'acc':
            acc =accuracy(yt, np.around(self.output, 0))
            self.loss = 1 - acc
            print('loss', self.loss)
            print('acc', acc)
            return self.loss

    def error_Prime(self, X=None, ytruth=None, ypred=None, error=None, verbose=False):
        """ Method will calculate the derivative ot the error function
        :param X: input vector
        :param ytruth: ground truth output
        :param ypred:  predicted output value
        :param error:  type of error/cost function to use
        :return:
        """
        if X is None:
            X = self.x[:]
        if ytruth is None:
            ytruth = self.y[:]
        if ypred is None:
            ypred = self.output[:]

        ytruth.reshape(ytruth.shape[0], 1)
        ypred.reshape(ytruth.shape[0], 1)
        if error is None:
            error = self.error
        if error == 'mae':
            print('mae')
            msePrime_w = -1 / len(ytruth) * np.dot((ytruth - ypred) / (abs(ytruth - ypred)), X)
            msePrime_b = -1 / len(ytruth) * sum([(yt - yp) / abs(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            self.del_w = msePrime_w
            self.del_b = msePrime_b
            return [msePrime_w, msePrime_b]
        elif error == 'mse':
            print('mse')
            print('pred', ypred.shape)
            print('truth', ytruth.shape)
            #msePrime_w = -2 / len(ytruth) * np.dot((ytruth - ypred), X)
            msePrime_w = -2 / len(ytruth) * sum([(yt - yp)*x for yt, yp, x in zip(ytruth, ypred, X)])
            msePrime_b = -2 / len(ytruth) * sum([(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            self.del_w = msePrime_w
            self.del_b = msePrime_b
            return [msePrime_w, msePrime_b]
        elif error == 'se':
            print('se')
            print()
            #print('pred', ypred.shape)
            #print('pred', ypred)
            #print('truth', ytruth.shape)
            #print('truth', ytruth)
            #msePrime_w = -2 / len(ytruth) * np.dot((ytruth - ypred), X)
            msePrime_w = -1 / len(ytruth) * sum([(yt - yp)*x for yt, yp, x in zip(ytruth, ypred, X)])
            msePrime_b = -1 / len(ytruth) * sum([(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            self.del_w = np.array(msePrime_w)
            self.del_b = np.array(msePrime_b)
            return [msePrime_w, msePrime_b]
        elif error == 'bce':
            print('bce')
            msePrime_w = sum([(-yt/(max(1e-15, yp))) + ((1-yt)/max((1-yp), 1e-15))  for yt, yp in zip(ytruth, ypred)])
            msePrime_b = sum([(-yt/(max(1e-15, yp))) + ((1-yt)/(max(1e-15, 1-yp)) ) for yt, yp in zip(ytruth, ypred)])
            self.del_b = msePrime_b
            #self.dels_past.append(self.del_b)
            #self.del_w = msePrime_w
            return [msePrime_w, msePrime_b]
        elif error == 'acc':
            print('accuracy')
            accPrime_w = -(1/len(ytruth))
            accPrime_b = -(1/len(ytruth))
            self.loss = -(1/len(ytruth))
            return self.loss
        else:
            print('ERROR: Unknown error method {}, must be one of:'.format(error))
            print(list(self.error_dictS.keys()))
            quit(-215)

    def process_Binary_output1(self, val, thresh=.5):
        self.pred_prob[:] = val            # store current predicted val
        if val > thresh:
            return 1
        return 0

    def update_weights(self, delta, lr=None):
        if self.verbose > 1:
            print('wdelta', delta)
        if lr is None:
            lr = self.eta
        self.w_updates.append(delta)
        self.w[:] = self.w - lr*delta
        return

    def update_bias(self, delta, lr=None):
        if self.verbose > 1:
            print('b delta', delta)
        if lr is None:
            lr = self.eta
        self.b_updates.append(delta)
        self.b[:] = self.b - lr*delta
        return

    def backpropagate(self, x,y, yp, error=None):
        # set up the arrays for updateing
        # the weights and biases

        delta = self.error_Prime(x,y,yp, error=None)
        pass

    def logistic(self, z, verbose=-1):
        self.z = z
        print('      ------------------------------    verbose ', self.verbose)
        if self.verbose > 1:
            print('logistic')
            print('z', z)
        """The Logistic function."""
        self.output[:] = 1.0 / (1.0 + np.exp(-z))
        return self.output[0]
    def logistic_prime(self):
        """Derivative of the sigmoid function."""
        # return self.sigmoid(z) * (1 - self.sigmoid(z))
        self.act_prime = self.output * (1 - self.output)
        return self.output * (1 - self.output)

    def linear(self, z):
        self.z = z      # store summation result
        self.output[:] = z  # store output value
        return z

    def linear_prime(self):
        return 1

    '''
    # loss suggested in class
    def loss_prime(self):
        return -(self.y - self.output)
    def activation_prime(self):
        if self.activation_fnc in [0, 'sigmoid', 'logistic']:
            self.act_prime = self.output*(1 - self.output)
            return self.output*(1 - self.output)
    def calculated_my_del(self,):
        self.my_del = self.activation_prime() * self.loss_prime()
    
    spczz = 0
   '''
    ### Miscellaneous functions
    def sigmoid(self, z, verbose=-1):
        self.z = z
        if self.verbose > 1:
            print('sigmoid')
        """The sigmoid function."""
        self.output[:] = self.process_Binary_output1(1.0 / (1.0 + np.exp(-z)), )
        return self.output[0]
    def sigmoid_prime(self):
        """Derivative of the sigmoid function."""
        #return self.sigmoid(z) * (1 - self.sigmoid(z))
        self.act_prime = self.output * (1 - self.output)
        return self.output * (1 - self.output)

    def tanH(self, z):
        """the Tanh activation function"""
        self.z = z
        self.output[:] = (2.0 / (1.0 + np.exp(-2 * z))) - 1
        return self.output[0]
    def tanH_prime(self):
        return 1 - (self.output[0] ** 2)

    def softplus(self, z):
        self.z = z
        self.output[:] = np.log(1 + np.exp(z))
        return self.output[0]
    def softplus_prime(self,):
        return 1 / (1 + np.exp(-self.z))

    def garctan(self, z):
        self.z = z
        self.output[:] = np.arctan(z)
        return self.output[0]
    def arctan_prime(self, ):
        return 1 / ((self.z ** 2) + 1)

    def perceptron(self, z):
        """
            perceptron thresholding function, returns 1 iff
            z is non negative, otherwise returns 0
        :param z: input to threshold
        :return:
        """
        self.z = z      # store summation result
        print()
        if z >= 0:
            self.output[:] = 1
            return 1
        else:
            self.output[:] = 0
            return 0
    def perceptron_prime(self, z=None):
        if z is None:
            self.z = z
        if self.z != 0:
            return 0
        print('strange input to perceptron prime {}'.format(self.z))
        return 0

    def relu(self, z):
        self.z = z
        if z < 0:
            self.output[:] = 0
            return 0
        else:
            self.output[:] = z
            return z
    def relu_prim(self, ):
        if self.z < 0:
            return 0
        else:
            return 1

class FullyConnectedLayer():
    """
        Represents a fully connected layer (collection of neurons each connected to the same set of inputs,
        and all outputting into the layer/list  in an NN
        should have the same abilities of a neuron only applied to a collection of neurons
    """
    def __init__(self, input_size, number_neurons=1, eta=.01, w=None, activation_fnc=0, verbose=-1,
                 weight_list=(-.01, .01), error='se', b=None, thresh=.01, ID=0, update_eta=False):
        self.update_eta = update_eta
        self.ID = ID                                # the id in the network of the layer
        self.input_size=input_size                  # number of inputs for each neuron
        self.number_neurons=number_neurons          # number of neurons in layer
        self.eta=eta                                # learning rate for layers neurons
        self.w = w                                  # list of weight vectors for each neuron
        if self.w is None:                          # if no weights given generate random weights for each neuron
            self.w = np.array([np.array(np.random.choice(weight_list, input_size,replace=True))
                               for i in range(number_neurons)])
        # print('W is now {}'.format(self.w))
        self.activation_fnc = activation_fnc        # the type of activation function for each neuron
        self.verbose=verbose                        # used for printing to standard out if desired/ debugging
        self.neurons=list()                           # list of neurons in layer
        self.weights=weight_list                    # optional list of possible random variables to initialze weights to
        self.bias = [np.array([1.0]) for i in range(number_neurons)]   # list of bias values for neurons in layer
        self.bias = np.array(self.bias)
        self.thresh=thresh                          # threshold for sigmoid or similar
        self.outputs = [np.array([0.0]) for i in range(number_neurons)]     # will hold the output values for each neuron
        self.outputs = np.array(self.outputs)
        self.pred_probs_ = [np.array([0.0]) for i in range(number_neurons)] # will hold the predicted probabilites if using sigmoid
        self.pred_probs_ = np.array(self.pred_probs_)
        self.verbose=verbose
        self.handle_bias_array(b)                   # this line handels creating the list of bias values

        # make a list of neurons of the neccesary size
        # either loading the given weights and/or bias values
        # or generating random ones
        print('number of neurons ', number_neurons)
        for i in range(number_neurons):
            # create a new neuron
            # if given None create random weights
            if w is None:
                print('w is none')
                print('bb', b, i)
                self.neurons.append(Neuron(input_size, eta=eta, activation_fnc=activation_fnc, error=error, b = self.bias[i],
                                           w=np.random.choice(weight_list, input_size), ID=i, verbose=verbose))
                #self.neurons[-1].b = np.array(self.bias[i], dtype=np.float)
            else:
                self.neurons.append(
                    Neuron(input_size, eta=eta, activation_fnc=activation_fnc, error=error, w=self.w[i], ID=i,
                           verbose=verbose))
                self.neurons[-1].b = np.array([self.bias[i]], dtype=np.float)
            # now store a reference to the new neurons weights and bias arrays
            self.w[i] = self.neurons[-1].w[:]       # store slice of last created neuron's bias
            #print('bias', self.bias)
            #print('i', i)
            print('neurons {}'.format(self.neurons))
            #self.bias[i] = self.neurons[-1].b[:]       # store slice of last created neuron's bias
            self.outputs[i] = self.neurons[-1].output[:]
        self.w = np.array(self.w)
        self.bias = np.array(self.bias)

    def handle_bias_array(self, b):
        """ Generates the list of bias values for the
            neurons in the layer
        :param b: either None (generate all 1 bias values) or an array of
                  bias values to use for each neuron
        :return: None
        """
        if b is None:
            pass
        else:
            self.bias = b

    def calculate(self, X):
        """
            Feeds input X into all the neurons in the layer, by calling calculate on them.
            The outputs are stored in the neurons output variable
        :param X:
        :return:
        """
        #inpts = X.tolist()
        print()
        print('   ************************************************************ verbose', self.verbose)
        print()
        cnt = 0
        for n in self.neurons:
            n.calculate(X)      # feed input x into neuron for processing, output stored by neuron in output
            self.outputs[cnt] = n.output[:][0]
            print('out at c {}, {}'.format(cnt, self.outputs[cnt]))
            if self.verbose > 1:
                print()
                print('----------------------------------')
                print('neuron {}'.format(n.ID))
                print('neuron w {}'.format(n.w))
                print('neuron b {}'.format(n.b))
                print('neuron output {}'.format(n.output))
                print('----------------------------------')
                print()
            cnt += 1

    def calculateA(self, X, verbose=False):
        """ Returns an array of outputs from the activations of  each neuron
        :param X:
        :param verbose:
        :return:
        """
        yp = list()
        # go through my neurons getting predicted outputs
        # by givinge each neuron the input vector X
        for ni in range(len(self.neurons)):
            # get a prediction from the current neuron and store it
            self.outputs[ni] = self.neurons[ni].calculate(X, self.neurons[ni].w, self.neurons[ni].b, verbose)
            if self.activation_fnc == 0 or self.activation_fnc == 'sigmoid':
                self.pred_probs_[ni] = self.neurons[ni].pred_prob
            # self.outputs[ni] = self.neurons[ni].activation(X, self.neurons[ni].w, self.neurons[ni].b)
        return self.outputs.copy()

    def update_weights(self, updates):
        for ni in range(len(self.neurons)):
            self.neurons[ni].update_weights(updates)

    def update_bias(self, updates):
        for ni in range(len(self.neurons)):
            self.neurons[ni].update_bias(updates)

    def pass_error_prime_to_network(self, network, yt):
        """
            If this is the output layer will collect the error function
            derivative (delta i.e. activation' * error')  values from its
            neurons and return it to it's calling network
        :param network: empty list passed from calling network
        :param yt:     ground truth value for output('s)
        :return:  list of delta values for the layers neurons
        """
        for on, y in zip(self.neurons, yt):
            wu1, delni = on.error_Prime(ytruth=y)           # delni = del_b
            network.append(on.del_b)
        return network

    def pass_activation_prime_to_network(self, network=None,):
        if network is None:
            network = list()
        """
            If this is the output layer will collect the activation function
            derivative values from its neurons and return it to it's calling network
        :param network: empty list passed from calling network
        :return:  list of the derivative of the activation function  values for the layers neurons
        """
        for on in self.neurons:
            on.activation_funcPrime()
            network.append(on.act_prime)
        return network

    def pass_output_layer_backpropagation_array(self, dels, verbose=True, k=1):
        bpl = list()  # list of back propagation values
        # now update output layer and store backpropagation values to pass back
        for on, dta in zip(self.neurons, dels):
            # update this neurons weights array
            upda = dta * on.x
            if self.verbose > 1:
                print('neuron {}'.format(on.ID))
                print('Input x: {}'.format(on.x))  # input
                print('Initial weights w: {}'.format(on.w[:]))  # initial weights
                print('Initial bias {}', on.b[:])
                print('learning rate: {}'.format(on.eta))
                print('delta for w: {}'.format(on.del_w))
                print('delta for b: {}'.format(on.del_b))
                print('dta:', dta)
                print('update: ', upda)
                print('         ******')

            # on.w[:] = on.w - on.eta * dta * on.x
            # update weights and bias
            on.w[:] = on.w[:] - (on.eta * upda)
            on.b[:] = on.b[:] - (on.eta * dta)
            on.w_updates.append(upda)
            on.b_updates.append(dta)
            if self.verbose > 1:
                print('Updated weights w: {}'.format(on.w[:]))  # initial weights
                print('Updated bias {}', on.b[:])
                print('------------')
                print('')
            # store back propagation values for current neuron
            bpl.append(on.w * dta)
            if self.update_eta:
                on.update_eta(k)
        if self.verbose > 0:
            print('                        deltas for out n1                    deltas for out n2')
            print('back prop', bpl)
        return bpl
        '''
        # now sum them and pass to network
        back_prop_val1 = sum(bpl)
        print('passing this back through network {}'.format(back_prop_val1))
        '''

    def back_propagate(self, val):
        return val

    def display_layer(self):
        for n, i in zip(self.neurons, range(self.number_neurons)):
            print('{}) neuron weights and bias: {} {}'.format(n.ID, n.w, n.b))
            print('Layer weights & bias at {}: {} {}'.format(i, self.w[i], self.bias[i]))
            print('--------------------------------------------')

class NeuralNetwork:
    """ Represents a neural network. Made of several parts
        * a collection  of fully connected layers
        * each fully connected layer contains some
          number of neuron objects
    """
    activation_dictS = {'logistic': 0, 'linear': 1, 'relu': 2, 'tanh': 3, 'softplus': 4, 'arctan': 6, 'perceptron': 7,
                        'sigmoid':8,}
    error_dictS = {'se': 0, 'crossentropy': 1, 'mae': 2, 'rmse': 3, 'hinge': 4, 'huber': 5, 'kullback': 6, 'bce':1,
                   'mse':2, 'acc':7,}
    or_tbl = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]  # input table to train for an OR function
    and_tbl = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]  # input table to train for an AND function
    not_tbl = [[0, 1], [1, 1]]  # input table for NOT function
    xor_tbl = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]  # input table to train for an XOR function

    def __init__(self, input_size, number_layers=1, neurons_layer=(1,), activations=('logistic',), error='mse',
                 eta=.01, w=None, b=None, eta_min=.0001, kmax=2, verbose=-1, weight_list=(-.1, .1,), epochs=1,
                 threshold=.001, update_eta=False):
        # process parameters and do some error checking
        self.update_eta=update_eta
        self.out_re = list()
        self.num_inputs=int(input_size)                              # the number of input features
        if self.num_inputs < 1:
            print('ERROR: number of inputs must be >= 1, was give {}'.format(self.num_inputs))
            quit(387)

        self.number_layers=int(number_layers)                        # number of fully connected layers
        if self.number_layers < 1:
            print('ERROR: number of layers must be >= 1, was give {}'.format(self.number_layers))
            quit(391)

        self.neuron_layer=neurons_layer                         # list where [# neurons layer 1, ***, # neurons layer N]
        if len(self.neuron_layer) < 1:
            print('ERROR: There must be at least 1 entry in the number of neurons per layer array'.format())
            quit()
        else:
            cnt = 0
            for i in self.neuron_layer:
                if i < 1:
                    print('ERROR: number of neurons must be >= 1, was given {}\nfor layer {}'.format(i, cnt))
                cnt += 1

        self.activations=activations                            # the activation function for each layer
        if len(self.activations) < 1:
            print('ERROR: There must be at least 1 entry in the activations per layer array'.format())
            quit(-415)
        else:
            for af in self.activations:
                if af not in self.activation_dictS.keys() and af not in self.activation_dictS.values():
                    print('ERROR: Unknown activation option {}, options are:'.format(af))
                    print(self.activation_dictS)
                    quit(969)

        if (len(activations) - number_layers - len(neurons_layer)) != -len(neurons_layer):
            print('ERROR: the number of layers, number of neurons per layer, and activation functions must match.')
            print('Was given lists of sizes {}, {}, and {} for the # of layers, # neurons per layer, and activations\n'
                  'arrays.'.format(number_layers, len(neurons_layer), len(activations)))
            quit(704)
        self.loss = error                                       # error/loss method for gradient descent
        if self.loss not in self.error_dictS.keys():
            print('ERROR: Unknown error method {}'.format(self.loss))
            quit(-715)

        self.eta = eta                                          # learning rate
        self.eta_min = eta_min                                  # minimum learning rate if adjusted learning rate desired
        self.epochs = epochs                                    # number of training epochs to run
        self.kmax = kmax                                        # used with epsilon method to adjust learning rate
        self.verbose=verbose                                    # used for debugging
        self.layers = []                                        # will hold the connected layer objects
        self.inputs=None                                        # will be used to hold a set of inputs to train with
        self.threshold = threshold
        self.outs = np.zeros(neurons_layer[-1])                 # get redy to store the outputs from the network
        self.losses = list()
        self.epochL = list()
        self.best_loss=1e-9
        self.ypred = list()
        self.w = w
        self.b = b
        # if none are for the weights and or bias values for the neurons in each layer
        # set an array of nones to pass to the layers
        # the the neurons in the layer will randomize the initial weights and or bias values
        if self.w is None:

            self.w = list()
            # get a none for each layer so we
            # can get the layers to tell the neurons
            # to randomly initialize
            for i in range(len(neurons_layer)):
                self.w.append(None)
        if self.b is None:
            self.b = list()
            for i in range(len(neurons_layer)):
                self.b.append(None)
        self.X, self.Y = None, None         # set up storage space for the input and output array and
        # create need number of layers
        in_size = input_size                    # this will be used to determine the size of the next layer
        ID = 0
        # go through the number of neurons per layer (neuron_layer), and activation functions for
        # each layer (activations) setting up the layers of the network accordingly
        for ne, af, w, b in zip(self.neuron_layer, self.activations, self.w, self.b):
            self.layers.append(FullyConnectedLayer(in_size, number_neurons=ne, eta=eta, w=w,
                                                   activation_fnc=af, verbose=verbose, update_eta=update_eta,
                                                   weight_list=weight_list, error=self.loss, b=b, ID=ID))
            in_size = ne      # grab the # of neurons in the last created layer to know how many inputs for the next
            ID += 1           # each layer is given an interger ID that is it's index in the network layer array

    '''
    def fit(self, X, Y, threshold = .01):
        self.X = X
        self.Y = Y

        # need to move through samples adjusting wieghts with gradient descent
        # and backpropagation
        # itearte through x making predicionts
        err = 100
        epoch = 0
        while err > threshold and epoch < self.epochs:
            cnt = 0
            # use each sample to generate a set of outputs for each layer
            for sample, response in zip(X,Y):
                # perform forward pass passing successive output/input
                # for each layer
                for lyr in range(len(self.layers)):
                    # give current sample to current layer
                    yp = self.layers[lyr].calculate(X, )
                    if (lyr == len(self.layers) - 1):
                        ypred = yp.copy()
                        if self.layers[lyr].activation_fnc == 'bce':
                            ypred[ypred < self.layers[lyr].thresh] = 0
                            ypred[ypred >= self.layers[lyr].thresh] = 1
                        self.ypred[lyr] = ypred
                # once you have went through all layers
                # calculate the error and backpropagate
                # store this layers predictions
                # if doing
    def train2(self, X, Y):
        self.fit(X,Y)
    '''

    def train(self, X, Y, epochs=None, verbose=True):
        """
            trains network for either a set number of epochs or once an error threshold is met
            calls forward_pass() then back_propagate() repeatedly until above conditions met
        :param X: Inputs to network
        :param Y: True response values
        :param epochs: number of training epochs to run
        :param verbose: how verbose the training process is
        :return: None
        """
        if epochs is not None:
            self.epochs = epochs
            if self.verbose > -1:
                print(' ----------------   Epochs set to {}'.format(self.epochs))
        err = 100
        epoch = 0
        self.best_loss = 1e9
        best_epoch=None
        err_tot = 0
        step = 0
        while epoch < self.epochs:
            if self.verbose > -1:
                print()
                print('# ####################################################################')
                print('# ####################################################################')
                print('# ####################    Epoch {}     ###############################'.format(epoch))
                print('# ####################################################################')
                print('# ####################################################################')
                print()
            # TODO: perform forward pass putting input through network
            #       and calculating the total error
            print('X', X)
            err_tot = 0
            cnt = 1
            for x, y in zip(X,Y):
                print(x)
                err_tot += self.forward_pass([x], [y])
                cnt += 1
                # store values for learning visualizations
                self.losses.append(err_tot)
                self.epochL.append(step)
                # perform backpropagation
                self.back_propagate([y], k=epoch)
                step += 1
            err_tot /= cnt
            if verbose > -1:
                print('Epoch: {}, Total {} error: {}'.format(epoch + 1, self.loss, err_tot))
            # check the error/loss for new best and see if training can end
            if err_tot < self.best_loss:
                self.best_loss = err_tot
                best_epoch = epoch

                if verbose > -1:
                    print()
                print(' ****************************************          new best loss {} at epoch {}'.format(
                    self.best_loss, best_epoch))
                print()
            if err_tot <= self.threshold:
                if verbose > -1:
                    print('')
                    print(' *****************                                  Error threshold met {}'.format(err_tot))
                    print('')
                    break
            epoch += 1


    def predict(self, X, Y):
        yp = list()
        for x, y in zip(X, Y):
            self.forward_pass([x],[y])
            yp.append(self.out_re)
        return yp

    def forward_pass(self, X, Y):
        """
            Will run through one pass through of the network form input layer to output layer
            and calculate the total error of the output layer and return that
        :param X: input array
        :param Y: ground truth response variabels
        :return: total error of network output
        """
        Layer_results = list()
        inputs = X
        old_inputs = None
        for lyr in self.layers:
            # grab the layer and apply the inputs to the layer
            lyr.calculate(inputs)
            if self.verbose > -1:
                print('layer {} outputs: {}'.format(lyr.ID, lyr.outputs))
            inputs = list()                 # will be used to store the result of last layer to feed into next

            # go through neurons of current layer processing input,
            # producing outputs to feed into the next layer if not the output/last layer
            for n in lyr.neurons:
                inputs.append(n.output[0])                  # store the last layers output
                if self.verbose > -1:
                    print('inputs i ', inputs[-1])
            inputs = np.array(inputs, dtype=np.float)
            old_inputs = inputs.copy()
            self.out_re = old_inputs
            # pass the outputs to the layer
            if self.verbose > -1:
                print('--------------------')
                print('--------------------')
                print('--------------------')
        if self.verbose > -1:
            print('inputs from last layer  = {}, pass to network for error calculation and decisions'.format(old_inputs))
        E_total = 0
        # go through last layer calculateing total error
        for n, y in zip(self.layers[-1].neurons, Y):
            n.calculate_error(y)        # call the neurons calculate_error method with stores the loss in the neuron's loss variable
            E_total += n.loss           # sum the error from all of the output layers neurons
        if self.verbose > -1:
            print('*************              Total Error {}'.format(E_total))
        return E_total                  # return the total error for that run to be tested against the threshold

    def back_propagate(self, yt, verbose=True, k=1):
        #r = self.w * previous
        # TODO: now perform Back propagation

        # ******************** TODO: handle output
        #                            Calculate Error derivative:
        #                                     calculate error prime for the last layer for each neuron
        #                                     and pass to network
        net_e_prime = list()                 #will contain the values TODO: make the actual neuron stuff happen in layer
        net_e_prime = self.layers[-1].pass_error_prime_to_network(list(), yt)
        '''
        for on, y in zip(self.layers[-1].neurons, yt):
            wu1, delni = on.error_Prime(ytruth=y)           # delni = del_b
            net_e_prime.append(on.del_b)
        '''
        # calculate activation prime for the last layer
        # and pass to network
        net_a_prime = self.layers[-1].pass_activation_prime_to_network(list())
        '''
        for on, y in zip(self.layers[-1].neurons, yt):
            on.activation_funcPrime()
            net_a_prime.append(on.act_prime)
        '''

        # now calculate the delta's for
        # the neurons in the last layer
        net_del = list()
        print('eprime', net_e_prime)
        print('aprime', net_a_prime)
        for ap, ep in zip(net_a_prime, net_e_prime):
            net_del.append(ap * ep)
        # convert to a numpy array
        net_del = np.array(net_del, dtype=np.float)
        dels = net_del
        if verbose:
            print('------------------------------------')
            print('------------------------------------')
            print('rd {}'.format(net_del))
            print('------------------------------------')
            print('------------------------------------')
            print('original deltas', dels)
            print()
        bpl = list()                # list of back propagation values

        # now update output layer and store backpropagation values to pass back
        bpl = self.layers[-1].pass_output_layer_backpropagation_array(dels, verbose=verbose, k=k)               # list of back propagation values

        '''
        # now update output layer and store backpropagation values to pass back
        for on, dta in zip(self.layers[-1].neurons, dels):
            # update this neurons weights array
            upda = dta * on.x
            if verbose:
                print('Input x: {}'.format(on.x))                    # input
                print('Initial weights w: {}'.format(on.w[:]))       # initial weights
                print('Initial bias {}', on.b[:])
                print('learning rate: {}'.format(on.eta))
                print('delta for w: {}'.format(on.del_w))
                print('delta for b: {}'.format(on.del_b))
                print('dta:', dta)
                print('update: ', upda)
                print('         ******')

            # on.w[:] = on.w - on.eta * dta * on.x
            # update weights and bias
            on.w[:] = on.w[:] - (on.eta * upda)
            on.b[:] = on.b[:] - (on.eta * dta)
            if verbose:
                print('Updated weights w: {}'.format(on.w[:]))  # initial weights
                print('Updated bias {}', on.b[:])
                print('------------')
                print('')
            # store back propagation values for current neuron
            bpl.append(on.w * dta)
        if verbose:
            print('                        deltas for out n1                    deltas for out n2')
            print('back prop', bpl)
        # now sum them and pass to network
        '''
        back_prop_val1 = sum(bpl)
        print('passing this back through network {}'.format(back_prop_val1))

        # now pass back iteratively to the previous layers
        # starting at penultimate layer
        # calculate the update values for each neuron
        # calculate the summed weighted sigma to pass back
        # and continue
        next_del = back_prop_val1
        for lyr in range(2, len(self.layers) + 1, 1):
            if verbose > 0:
                print('looking at layer {}'.format(-lyr))
            to_sum = list()
            for hn, bpv in zip(self.layers[-lyr].neurons, next_del):
                # get its activation function
                hn.activation_funcPrime()
                dta = bpv * hn.act_prime[0]
                if verbose > 1:
                    print('X {}'.format(hn.x))
                    print('dta', dta)
                    print('activation', hn.act_prime[0])
                    print('bpv', bpv)
                    # update the weights and biase
                    print('w b4 {}'.format(hn.w))
                    print('b b4 {}'.format(hn.b))
                    print('  *************************************   ')
                    print('  *************************************   ')
                    print('  *************************************   ')
                to_sum.append(dta*hn.w)
                print(hn.x)
                print(dta)
                hn.update_weights(dta * np.array(hn.x))
                if self.update_eta:
                    hn.update_bias(dta)
                #hn.b[:] = hn.b[:] - hn.eta * dta
                if self.verbose > 1:
                    print('w b4 {}'.format(hn.w))
                    print('b b4 {}'.format(hn.b))
                    print(' -------------------------------------------')
                    print(' -------------------------------------------')
                    print()
                hn.update_eta(k)
            next_del = sum(to_sum)

    def calculate_error(self,yt, yp, error=None):
        """ will calculate the error of the perceptron's
            output based on the error function chosen
        :param yt: ground truth output value
        :param yp: predicted output value from perceptron
        :param error: the error method to use
        :return:
        """
        if error is None:
            error = self.activations[-1]
        if error.lower() == 'mse':
            return MSE(yt, yp)
        elif error.lower() == 'bce':
            return binary_cross_entropy(yt, yp)

    def error_Prime(self, X, ytruth, ypred, error=None, verbose=False):
        """ Method will calculate the derivative ot the error function
        :param X: input vector
        :param ytruth: ground truth output
        :param ypred:  predicted output value
        :param error:  type of error/cost function to use
        :return:
        """
        if error is None:
            error = self.activations[-1].lower()
        if error == 'mae':
            print('mae')
            maePrime_w = -1 / len(ytruth) * np.dot((ytruth - ypred) / (abs(ytruth - ypred)), X)
            maePrime_b = -1 / len(ytruth) * sum([(yt - yp) / abs(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            return [maePrime_w, maePrime_b]
        elif error == 'mse':
            print('mse')
            msePrime_w = -2 / len(ytruth) * np.dot((ytruth - ypred), X)
            msePrime_b = -2 / len(ytruth) * sum([(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            return [msePrime_w, msePrime_b]
        elif error == 'bce':
            print('bce')
            msePrime_w = np.dot((ytruth - ypred), X)
            msePrime_b = sum([(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            return [msePrime_w, msePrime_b]
        else:
            print('ERROR: Unknown error method {}, must be one of:'.format(error))
            print(list(self.error_dictS.keys()))
            quit(-215)

    def gradient_descent(self, X, yt, yp, verbose=False):
        dels = self.error_Prime(X, yt, yp, error=self.loss, verbose=verbose)
        self.w[:] = self.w - self.eta * dels[0]
        self.b[:] = self.b - self.eta * dels[1]

    def calculate_loss(self):
        # calculate loss from last layer
        # grab last layers outputs
        if self.activations[-1] == 'MSE':
            err = MSE(self.Y, self.layers[-1].outputs)
            err_primeW, err_primeb = self.error_Prime(self.X, self.Y, self.layers[-1].outputs)
            self.layers[-1].update_weights(err_primeW)
            self.layers[-1].update_bias(err_primeb)
        # now do the back prop from the second the last to the first
        for i in range(-2, -len(self.layers)+1, -1):
            pass

# #####################################################
# #####################################################
# #########   TODO: Regression Performance     ########
# #####################################################
# #####################################################
def Rvar(ytrue, ypred):
    ymean = ypred.mean(axis=0)
    ssreg = SSREG(ytrue, ymean=ymean)
    ssres = SSRES(ytrue=ytrue, ypred=ypred)
    return (SSREG(ypred, ymean) / len(ypred)) / (SSTOT(ytrue) / len(ypred))

def binary_cross_entropy(ytrue, yprob):
    N = len(ytrue)
    ytrue = ytrue.reshape(N,1)
    yprob = ytrue.reshape(N,1)
    return -sum([BCE(yt, yp) for yt,yp in zip(ytrue, yprob)]) / N

def log_loss(ytrue, yprob):
    return binary_cross_entropy(ytrue, yprob)

def BCE(ytrue, yprob):
        return -sum([yt*np.log(max(yp, 1e-15)) + (1-yt)*np.log(max(1-yp, 1e-15)) for yt, yp in zip(ytrue, yprob)])/len(ytrue)

def SSE2( ytrue, ypred):
    sm = sum([.5 * ((yt - yp) ** 2) for yp, yt in zip(ytrue, ypred)])
    return sm

def SE(ytrue, ypred):
    n = len(ytrue)
    print('n', n)
    return SSE2(ytrue, ypred) / n

def SSE( ytrue, ypred):
    sm = sum([(yt - yp) ** 2 for yp, yt in zip(ytrue, ypred)])
    return sm

def MSE(ytrue, ypred):
    n = len(ytrue)
    return SSE(ytrue, ypred) / n

def RMSE(ytrue, ypred):
    return np.sqrt(MSE(ytrue, ypred))

def MAD(ytrue, ypred):
    n = len(ytrue)
    return sum([abs(yt - yp) for yp, yt in zip(ytrue, ypred)]) / n

def MAE(ytrue, ypred):
    n = len(ytrue)
    return sum([abs(yt - yp) for yp, yt in zip(ytrue, ypred)]) / n

def SSREG(ypred, ymean):
    return sum([(yp - ymean) ** 2 for yp in ypred])

def SSRES(ytrue, ypred):
    return sum([(yt - yp) ** 2 for yp, yt in zip(ytrue, ypred)])

def COD(ytrue, ypred):
    return 1 - (SSRES(ytrue, ypred)/SSTOT(ytrue))

def SSTOT(ytrue):
    ymean = ytrue.mean(axis=0)
    return sum([(yt - ymean) ** 2 for yt in ytrue])  # scatter total (sum of squares)

def calculate_log_like(attribs, params):
    #attribs.append('const')
    l = []
    for attrib in attribs:
        l.append(params[attrib])
    return np.exp(l).tolist()

def correct(ytrue, ypredict):
    # count the predictions that are correct
    return sum(yt == yp for yt, yp in zip(ytrue, ypredict))

def shape_check(yc, yd):
    if yc.shape != yd.shape:
        yc =  yc.reshape(yd.shape[0], yd.shape[1])
    return yc


def accuracy(ytrue, ypredict):
    ypredict = shape_check(ypredict, ytrue)
    return correct(ytrue, ypredict)/len(ytrue)


# ##################################################################
# ##################################################################
# ################TODO: machine learning tools  ####################
# ##################################################################
# ##################################################################
def epsilon(emax, emin, k, kmax):
    """ Can be used to modify the learning rate as training occurs
    :param emax: starting learning rate
    :param emin: the final learning rate
    :param k:    current step
    :param kmax: controls how many steps it takes to get to emin
    :return: new learning rate
    """
    return emax * ((emin/emax)**(min(k, kmax)/kmax))



# handles the command line arguments if there are any
def handle_cmd_line(method='example', verbose=False):
    """ This method just looks for the first command line argument. If none are given
        the program will run all three options for Project 1

    :param method:
    :param verbose:
    :return:
    """
    if len(sys.argv) == 2:
        return sys.argv[1]
    return method