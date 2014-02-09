__author__ = 'sharmrav'
import numpy as np
class BackPropgationNetwork:
    """A Backpropagation Network"""

    #Class Members
    layercount=0
    shape =None  #tuple
    weights=[] #list

    #Class Methods
    def __init__(self,layerSize):
        """Initialize the network"""

        #layer Information
        self.layercount =len(layerSize) -1 #If you pass a list 2 2 1 .. Its actually only 2 layers.Hence the -1
        self.shape =layerSize


        #Input/Output Data
        self._layerInput=[]
        self._layerOutput=[]
        self._previousWeightDelta=[]

        #Creating the weight Arrays
        for(l1,l2) in zip(layerSize[:-1],layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))
            self._previousWeightDelta.append(np.zeros((l2,l1+1)))


    #Run Method
    def Run(self,input):
        """Run the network based on input Data """
        lnCases=input.shape[0]

        #Clear previous intermediate value lists
        self._layerInput=[]
        self._layerOutput=[]

        #Go through each layer anc pull data from previos layer
        for(index) in range(self.layercount):
            #Detrmine Layer Input
            if index ==0:
                layerInput =self.weights[0].dot(np.vstack([input.T,np.ones([1,lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,lnCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))

        return self._layerOutput[-1].T


    def TrainEpoch(self,input,target,trainingRate=0.2,momentum = 0.5):
        """This method trains the network for one epoch"""

        delta =[]
        lnCases = input.shape[0]

        #First Run the Network
        self.Run(input)

        #Calculate Deltas
        for index in reversed(range(self.layercount)):
            if index == self.layercount-1:
                #Compare to the target Values
                output_delta=self._layerOutput[index] -target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta*self.sgm(self._layerInput[index],True))
            else:
                #Compare to the following Layer's delta
                delta_pullback = self.weights[index +1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1,:] *  self.sgm(self._layerInput[index],True))

        #COmpute Weight Deltas
        for index in range(self.layercount):
            delta_index = self.layercount -1 -index

            if index ==0:
                layerOutput = np.vstack([input.T, np.ones([1,lnCases])])
            else:
                layerOutput =  np.vstack([self._layerOutput[index - 1 ], np.ones([1,self._layerOutput[index - 1 ].shape[1]])])

            # Implementation for transfer fucntion
            #weightDelta = np.sum(\
            #    layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0)
            #    ,axis =0)

            #self.weights[index] -=trainingRate * weightDelta

            #Implementation for Adding Momentum
            curweightDelta = np.sum(\
                layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0)
                ,axis =0)


            weightDelta = trainingRate * curweightDelta + momentum *self._previousWeightDelta[index]

            self.weights[index] -=trainingRate * weightDelta
            self._previousWeightDelta = weightDelta


        return error




    #Transfer Function
    def sgm(self,x,Derivative=False):
        if not Derivative:
            return 1/(1+np.exp(-x))
        else:
            out=self.sgm(x)
            return out*(1-out)


#If Run as a script create a test Object.

if __name__ == "__main__":
    bpn = BackPropgationNetwork((2,2,2))
    print(bpn.shape)
    print(bpn.weights)

    #lvInput = np.array([[0,0],[1,1],[-1,0.5]])
    #lvOutput = bpn.Run(lvInput)

    #print("Input :{0} \nOutput:{1}".format(lvInput,lvOutput))

    lvInput = np.array([[0,0],[1,1],[0,1],[1,0]])
    lvTarget = np.array([[0.05],[0.05],[0.95],[0.95]])

    lnMax =100000
    lnErr = 1e-5
    for i in range(lnMax-1):
        err = bpn.TrainEpoch(lvInput,lvTarget)
        if i%2500 ==0:
            print("Iteration {0} \t Error : (1:0.5f)".format(i,err))
        if err <=lnErr:
            print("Minimum Error reached at iteration{0}".format(i))
            break

    #Display Output
    lvOutput = bpn.Run(lvInput)
    print("Input :{0} \nOutput:{1}".format(lvInput,lvOutput))



