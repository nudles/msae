"""
Copyright (C) 2014 Wei Wang (wangwei@comp.nus.edu.sg)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import gnumpy as gp
import numpy as np
import sys
from model import Model


class AE(Model):
    """
    autoencoder
    matrix are in row fashion, i.e., one row per case
    weightDecay is enabled
    loss=0.5*||a2-a0||^2+0.5*weightCost*(||W1||^2+||W2||^2)
    """

    def __init__(self, config, name):
        super(AE, self).__init__(config, name) 

        #dimension of hidden layer
        self.hDim = int(self.readField(config, name, "hidden_dimension"))

        #dimension of visible layer
        self.vDim = int(self.readField(config, name, "visible_dimension"))

        #baise for hidden layer
        if self.hDim>0:
            self.b1 = gp.zeros(self.hDim)

        #biase for visible layer
        if self.vDim>0:
            self.b2 = gp.zeros(self.vDim)

        #init weight: uniform between +-sqrt(6)/sqrt(v+h+1)
        if self.hDim*self.vDim>0:
            gp.seed_rand()
            r=gp.sqrt(6)/gp.sqrt(self.hDim+self.vDim+1)
            self.W1 = gp.randn(self.vDim, self.hDim) * 2 * r - r
            self.W2 = gp.randn(self.hDim, self.vDim) * 2 * r - r
            self.initUpdate()
            self.initHyperParam(config, name)

    def initUpdate(self):
        #increment for hidden biase
        self.incb1 = gp.zeros(self.hDim)

        #increment for visible biase
        self.incb2 = gp.zeros(self.vDim)

        #increment for weight
        self.incW1 = gp.zeros((self.vDim, self.hDim))
        self.incW2 = gp.zeros((self.hDim, self.vDim))

    def initHyperParam(self, config, name):
        #learning rate, decay steps
        self.epsilon, self.epsDecayHalfEpochs = self.readLearningRate(config, name)

        #momentum
        self.momentumStart, self.momentumEnd, self.momDecayEpochs = self.readMomentum(config, name)

        #weight decay cost
        self.weightCost = float(self.readField(config, name, "weight_cost"))

        #corrputionLevel:0-1, e.g., 0.1 indicate 0.1 of input is set to 0
        self.corrputionLevel=float(self.readField(config, name, "corruption_level"))

    def resetHyperParam(self, config, section):
        if self.verbose:
            print '********reset hyper-parameters**************'
        if section=="Self":
            section=self.name
        self.initHyperParam(config, section)

    def updateParam(self,epoch, param, inc, grad):
        epsilon, momentum = self.getEpsilonAndMomentum(epoch)
        inc*=momentum
        inc+=grad*epsilon
        param-=inc

    def getEpsilonAndMomentum(self, epoch):
        if epoch==0:
            momentum=0.0
        elif epoch >= self.momDecayEpochs:
            momentum = self.momentumEnd
        else:
            f=float(epoch)/self.momDecayEpochs
            momentum = (1.0-f)*self.momentumStart+f*self.momentumEnd

        if self.epsDecayHalfEpochs==0:
            epsilon = self.epsilon
        else:
            epsilon = self.epsilon / (1 + float(epoch) / self.epsDecayHalfEpochs)

        return epsilon, momentum

    def applyBackActivation(self,z):
        return z.logistic()

    def getWGradient(self, d, a, W):
        """
        dJ/dW=dJ/da*da/dz*dz/dw=d*dz/dW=a.T*d
        since d has batchsize cases, the gradient should be divided by batchsize
        weight decay is added
        """
        return gp.dot(a.T, d)/d.shape[0]+self.weightCost*W

    def getbGradient(self,d, layer=2):
        grad= d.mean(axis=0)
        return grad

    def computeWeightNorm(self, W=None):
        if W==None:
            return self.W1.euclid_norm()
        else:
            return W.euclid_norm()

    def computeSparsity(self, h=None):
        if h!=None:
            return h.mean()
        else:
            return 0.0

    def getWeightLoss(self,W1,W2):
        loss=0.5*self.weightCost*(gp.sum(W1**2)+gp.sum(W2**2))
        return loss


    def getActivationGradient(self, a):
        """
        a: layer output
        compute gradient of output a w.r.t input z
        for sigmoid function, it is (1-a)*a
        """
        return (1-a)*a

    def forwardOneStep(self, a):
        z=gp.dot(a,self.W1)+self.b1
        return z.logistic()

    def backwardOneStep(self,a):
        z=gp.dot(a,self.W2)+self.b2
        return z.logistic()

    def getCorruptedInput(self, input):
        if self.corrputionLevel>0:
            rnd=gp.rand(self.batchsize, self.vDim)>self.corrputionLevel
            output=rnd*input
            return output
        else:
            return input

    def forward(self, a0, training=False):
        """
        forwar up and then down
        compute a1 and a2 from input 
        different decoder should have different way to compute a2
        """
        x=a0
        if training:
            x=self.getCorruptedInput(x)
        a1=self.forwardOneStep(x)
        a2=self.backwardOneStep(a1)
        return a0,a1,a2

    def getErrorLoss(self,a0,a2,factor=1):
        """
        compute error/reconstruction error
        a2: reconstruction
        a0: input
        one row per case
        """
        loss=factor*0.5*gp.sum((a2-a0)**2)/a0.shape[0]
        return loss 

    def getCost(self, param, a0,factor=1.0):
        """
        total cost
        """
        self.W1,self.b1,self.W2,self.b2=self.splitParam(param)
        a0,a1,a2=self.forward(a0)
        cost=self.getErrorLoss(a0,a2,factor)
        cost+=self.getWeightLoss(self.W1,self.W2)
        return cost,a1

    def splitParam(self, param):
        """
        split parameters in array into W1,b1,W2,b2
        """
        s1=self.W1.size
        s2=s1+self.b1.size
        s3=s2+self.W2.size
        s4=s3+self.b2.size
        W1=param[0:s1].reshape(self.W1.shape)
        b1=param[s1:s2].reshape(self.b1.shape)
        W2=param[s2:s3].reshape(self.W2.shape)
        b2=param[s3:s4].reshape(self.b2.shape)
        return W1,b1,W2,b2

    def computeNumericGrads(self,input, eps=1e-4, sampleNum=500):
        """
        gradient of J w.r.t. x computed by (J(x+eps)-J(x-eps))/2eps
        only check param at sampleNum positions
        """
        param=self.vectorParam([self.W1,self.b1,self.W2,self.b2])
        sample=np.random.randint(0,param.size,sampleNum)
        grad=gp.zeros(sampleNum)
        for (i,idx) in enumerate(sample):
            if i%100==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            q=gp.zeros(param.shape)
            q[idx]=eps
            p1=param+q
            p2=param-q
            c1,_=self.getCost(p1, input)
            c2,_=self.getCost(p2, input)
            grad[i]=(c1-c2)/(2.0*eps)
        print "end"
        return grad, sample

    def computeDlast(self, a0,a2,factor=1.0):
        """d2=dJ/dz2, z2 is the activation of reconstruction of input layer"""
        d2=factor*(a2-a0)*self.getActivationGradient(a2)
        return d2

    def computeD(self,a1,d2,W2):
        """d2=dJ/dz1, z1 is activation of (reconstruction) of layers"""
        d1=gp.dot(d2,W2.T)*self.getActivationGradient(a1)
        return d1

    def computeGrads(self, a0,a1,a2):
        """
        compute grads of W and b from derivatives
        """
        d2=self.computeDlast(a0,a2)
        d1=gp.dot(d2,self.W2.T)*self.getActivationGradient(a1)
        W1grad=self.getWGradient(d1,a0,self.W1)
        W2grad=self.getWGradient(d2,a1,self.W2)
        b1grad=self.getbGradient(d1)
        b2grad=self.getbGradient(d2)
        return [W1grad,b1grad,W2grad,b2grad]

    def gradientCheck(self, dat):
        """
        check gradient by comparing with numeric computing
        it should be done on cpu
        """
        print "doing gradient check..."
        a0,a1,a2=self.forward(dat)
        grads=self.computeGrads(a0,a1,a2)
        pgrad=self.vectorParam(grads)
        
        #Note: compute numberic gradient after derivatives!!!
        pnumeric,idx=self.computeNumericGrads(dat)
        pgrad=pgrad[idx]
        diff= (pgrad-pnumeric).euclid_norm()/(pgrad+pnumeric).euclid_norm()
        print "the diff is %.15f, which should be very small" % diff

    def trainOneBatch(self, input, epoch, computeStat=True):
        assert (self.batchsize == input.shape[0])

        dat = gp.as_garray(input)

        if self.debug:
            self.gradientCheck(dat)
            sys.exit(0)

        a0,a1,a2=self.forward(dat, training=True)
        grads=self.computeGrads(a0,a1,a2)
        param=[[self.W1,self.incW1],[self.b1,self.incb1],[self.W2,self.incW2],[self.b2,self.incb2]]
        for i in range(len(param)):
            self.updateParam(epoch, param[i][0],param[i][1],grads[i])

        if computeStat:
            rloss =  self.getErrorLoss(a0, a2)
            wloss= self.getWeightLoss(self.W1,self.W2) 
            norm = self.computeWeightNorm()
            ratio = self.computeIncRatio(norm)
            sparsity=self.computeSparsity(a1)
            return np.array([rloss, wloss, norm, ratio, sparsity])

    def getDisplayFields(self):
        s="neigbor dist,epoch:rec err, L2  , weight(w), +w ratio, sparsity"
        return 5,5,s
   
    def getReps(self, v):
        """
        v: input to first layer
        return: top layer vector
        """
        a=self.forwardOneStep(v)
        return a.as_numpy_array()
