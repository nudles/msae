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
    
from model import Model
import gnumpy as gp
import numpy as np
from sae import SAE
import evaluate
import sys
import os
from datahandler import DataHandler


class MSAE(Model):
    """
    multi-modal stacked autoencoder
    """

    def __init__(self, config, name="msae"):
        super(MSAE,self).__init__(config, name)

        self.depth=int(self.readField(config, name, "depth"))

        #image path
        self.isae=self.createsae("iae", "isae")
        #text path
        self.tsae=self.createsae("tae", "tsae")

        self.epoch=0
        self.statesIdx=0
        states=self.readField(config, name, 'states')
        fields=states.split(',')
        self.states=[] 
        assert(len(fields)%7==0)

        #states indicate which sae to fix, which to adjust
        for i in range(len(fields)/7):
            k=0
            state=[]
            state.append(fields[i*7+k])
            state.append(self.str2bool(fields[i*7+k+1]))
            state.append(self.str2bool(fields[i*7+k+2]))
            state.append(float(fields[i*7+k+3]))
            state.append(float(fields[i*7+k+4]))
            state.append(float(fields[i*7+k+5]))
            state.append(int(fields[i*7+k+6]))
            self.states.append(state)
        #print self.states

        self.sections.extend(self.isae.sections)
        self.sections.extend(self.tsae.sections)

    def createsae(self, prefix, saeName):
        if self.config.has_option(self.name, saeName):
            saepath=self.readField(self.config, self.name, saeName)
            sae=self.loadModel(self.config, saepath)
            reset=self.readField(self.config, self.name, "reset_hyperparam")
            if reset!="False":
                for ae in sae.ae[1:]:
                    ae.resetHyperParam(self.config, reset)
            return sae 
        else:
            return SAE(self.config, self.name, prefix=prefix)

    def singlePathNumericGrad(self, saes, inputs, factor=1,sampleNum=500,eps=1e-4): 
        """
        get gradient for single path by numeric computing
        aes: (my_aes, other_aes), autoencoders for this path and the other path
        inputs:(my_input, other_input), input data for this path and the other_path
        Since the param of the other path is fixed, no need to compute its cost
        """
        mysae, osae=saes
        myinput, oinput=inputs
        myparam=mysae.combineParam(down=False)
        #aes[0] is None
        oas=osae.forward(oinput)
        plen=myparam.size
        sample=np.random.randint(0,plen,sampleNum)
        grad=gp.zeros(sampleNum)
        for (i,idx) in enumerate(sample):
            if i%100==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            q=gp.zeros(myparam.shape)
            q[idx]=eps
            p1=myparam+q
            p2=myparam-q
            c1,a=mysae.getCost(p1,myinput,factor)
            c1+=self.getDiffLoss(a[self.depth-1],oas[self.depth-1])
            c2,a=mysae.getCost(p2,myinput,factor)
            c2+=self.getDiffLoss(a[self.depth-1],oas[self.depth-1])
            grad[i]=(c1-c2)/(2.0*eps)
        return grad, sample

    def getSinglePathGradVec(self,sae,a,oa,factor):
        g,_=self.getSinglePathGrad(sae,a,oa,factor)
        return self.vectorParam(g)

    def gradientCheck(self, img, txt):
        """
        check gradient by comparing with numeric computing
        it should be done on cpu
        """
        print "doing gradient check..."
        ia=self.isae.forward2Top(img)
        ta=self.tsae.forward2Top(txt)
        if not self.fix_img_path:
            igrad=self.getSinglePathGradVec(self.isae, ia, ta, self.imgCost)
        if not self.fix_txt_path:
            tgrad=self.getSinglePathGradVec(self.tsae, ta, ia, self.txtCost) 

        if not self.fix_img_path:
            isgrad,isample=self.singlePathNumericGrad((self.isae,self.tsae),(img,txt),factor=self.imgCost)
            x=igrad[isample]-isgrad
            y=igrad[isample]+isgrad
            print "the diff for img path is %.15f, which should be very small" % (x.euclid_norm()/y.euclid_norm())
        if not self.fix_txt_path:
            tsgrad,tsample=self.singlePathNumericGrad((self.tsae,self.isae),(txt,img),factor=self.txtCost)
            x=tgrad[tsample]-tsgrad
            y=tgrad[tsample]+tsgrad
            print "the diff for txt path is %.15f, which should be very small" % (x.euclid_norm()/y.euclid_norm())


    def getSinglePathGrad(self, sae, a, oa, rec_factor, diff_factor=1.0):
        """
        compute gradients for the sae that is being adjusted 
        return g1,g2, reconstruction error
        g1: [w1,b1] grads for encoders from bottom to top
        g2: [w2,b2] grads for decoders from bottom to top
        if rec_factor=0, g2=None
        """
        #rec_factor is the weight, alpha/beta
        if rec_factor>0:
            a=sae.backward2Bottom(a)
            recloss=sae.ae[1].getErrorLoss(a[0],a[-1],rec_factor)
        else:
            recloss=0
        if diff_factor==0:
            diffgrad=None
        else:
            diffgrad=diff_factor*(a[self.depth-1]-oa[self.depth-1])
        g=sae.computeGrads(a,diffgrad=diffgrad,factor=rec_factor)
        return g,recloss

    def getReps(self, imgData, txtData):
        """
        forward input data to top layer, then do sampling
        """
        ia=self.isae.forward2Top(imgData)
        ta=self.tsae.forward2Top(txtData)
        imgcode=ia[-1]
        txtcode=ta[-1]
        return imgcode.as_numpy_array(), txtcode.as_numpy_array()

    def getDiffLoss(self, x,y):
        loss=gp.sum((x-y)**2)*(0.5/x.shape[0])
        return loss

    def checkPath(self, epoch):
        """get state info about fix which sae, adjust which sae"""
        epoch=epoch-self.epoch
        idx=self.statesIdx
        if epoch==self.states[idx][6]:
            self.epoch+=epoch
            idx=(self.statesIdx+1)%len(self.states)
            self.statesIdx=idx
            info=self.states[idx][0]
            print info
        k=1
        self.fix_img_path=self.states[idx][k]
        self.fix_txt_path=self.states[idx][k+1]
        imgcost=self.states[idx][k+2]
        txtcost=self.states[idx][k+3]
        diffcost=self.states[idx][k+4]
        return epoch,imgcost,txtcost,diffcost

    def trainOneBatch(self,img, txt, epoch, imgcost,txtcost,diffcost=1.0):
        img=gp.as_garray(img)
        txt=gp.as_garray(txt)
        if self.debug:
            self.gradientCheck(img,txt)
            sys.exit(0)
 
        ia=self.isae.forward2Top(img, training=True)
        ta=self.tsae.forward2Top(txt, training=True)
        if not self.fix_img_path and (imgcost>0 or diffcost>0):
            g,irecloss=self.getSinglePathGrad(self.isae,ia,ta,imgcost, diffcost)
            self.isae.updateParams(epoch,g,self.isae.ae)
        else:
            irecloss=0
                 
        if not self.fix_txt_path and (txtcost>0 or diffcost>0):
            g,trecloss=self.getSinglePathGrad(self.tsae,ta,ia,txtcost, diffcost)
            self.tsae.updateParams(epoch,g,self.tsae.ae)
        else:
            trecloss=0

        perf=[irecloss,trecloss]
        for i in range(1,self.depth):
            perf.append(self.getDiffLoss(ia[i],ta[i]))
        a=ia[1:self.depth]+ta[1:self.depth]
        ae=self.isae.ae[1:]+self.tsae.ae[1:]
        for i in range(len(a)):
            perf.append(ae[i].computeSparsity(a[i]))
        return np.array(perf)

    def doCheckpoint(self, outdir):
        """
        checkpoint for autoencoders along both two paths
        save them as 'modelcd' file under the same directory where the original model file locates
        """
        aes=self.isae.ae[1:]+self.tsae.ae[1:]
        for ae in aes:
            path=os.path.join(outdir,ae.name)
            ae.save(path)
        super(MSAE,self).doCheckpoint(outdir)

    def inference(self, imgpath, txtpath, statpath=None):
        """map input featuers into latent features, do normalization if statpath is available"""
        imgData=gp.garray(np.load(imgpath))
        txtData=gp.garray(np.load(txtpath))
        if statpath:
            stat=np.load(statpath)
            mean=gp.as_garray(stat['mean'])
            std=gp.as_garray(stat['std'])
            imgData-=mean
            imgData/=std
        imgcode,txtcode=self.getReps(imgData, txtData)
        return imgcode, txtcode

    def extractValidationReps(self,imgData, txtData, reps_input_field,reps_output_field,outputPrefix=None):
        """evaluation data are small, thus stored in single file"""
        imgoutpath=self.readField(self.isae.ae[-1].config, self.isae.ae[-1].name, reps_output_field)
        txtoutpath=self.readField(self.tsae.ae[-1].config, self.tsae.ae[-1].name, reps_output_field)
        imgcode,txtcode=self.getReps(imgData, txtData)
        if not outputPrefix:
            np.save(imgoutpath,imgcode)
            np.save(txtoutpath,txtcode)
        else:
            np.save(outputPrefix+"img",imgcode)
            np.save(outputPrefix+"txt",txtcode)

    def extractTrainReps(self,imgDH, txtDH, numBatch):
        """training data may be large, thus use DataHandler to load them"""
        imgDH.reset()
        txtDH.reset()
        for i in range(numBatch):
            imgBatch=imgDH.getOneBatch()
            txtBatch=txtDH.getOneBatch()
            if imgBatch is None:
                break
            imgcode,txtcode=self.getReps(imgBatch, txtBatch)
            imgDH.write(imgcode)
            txtDH.write(txtcode)
        imgDH.flush()
        txtDH.flush()


    def getDisplayFields(self):
        s="neigbor dist(I->I,T->T,I->T,T->I),epoch , Img/Txt rec err,"
        format="%%%-ds, %%%-ds" %(7*(self.depth-1), 7*(self.depth-1))
        s+=format % ('layer-wise diff', '--img/txt layer-wise sparsity')
        return self.depth+1,self.depth*3-1,s

    def train(self):
        outputPrefix=self.readField(self.config,self.name,"output_directory")
        outputDir=os.path.join(outputPrefix,self.name)
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        
        imageinput = self.readField(self.isae.ae[1].config, self.isae.ae[1].name, "train_data")
        textinput = self.readField(self.tsae.ae[1].config, self.tsae.ae[1].name, "train_data")

        if self.readField(self.config, self.name,"extract_reps")=="True":
            imageoutput=self.readField(self.isae.ae[-1].config, self.isae.ae[-1].name, "train_reps")
            textoutput=self.readField(self.tsae.ae[-1].config, self.tsae.ae[-1].name, "train_reps")
        else:
            imageoutput=None
            textoutput=None

        maxEpoch = int(self.readField(self.config, self.name, "max_epoch"))
        trainSize=int(self.readField(self.config, self.name, "train_size"))
        numBatch = int(trainSize / self.batchsize)
 
        normalizeImg=self.str2bool(self.readField(self.config, self.name, "normalize"))
        imgTrainDH=DataHandler(imageinput, imageoutput, self.isae.ae[1].vDim, self.isae.ae[-1].hDim, self.batchsize, numBatch,normalizeImg)
        txtTrainDH=DataHandler(textinput, textoutput, self.tsae.ae[1].vDim, self.tsae.ae[-1].hDim, self.batchsize, numBatch)

        showFreq = int(self.readField(self.config, self.name, "show_freq"))
        if showFreq > 0:
            visDir = os.path.join(outputDir, "vis")
            if not os.path.exists(visDir):
                os.makedirs(visDir)

        evalFreq = int(self.readField(self.config, self.name, "eval_freq"))
        if evalFreq!=0:
            qsize=int(self.readField(self.config, self.name, "query_size"))
            labelPath=self.readField(self.config,self.name,"label")
            label=np.load(labelPath)
            queryPath=self.readField(self.config, self.name, "query")
            validation=evaluate.Evaluator(queryPath,label,os.path.join(outputDir,'perf'), self.name, query_size=qsize,verbose=self.verbose)
            validateImagepath = self.readField(self.isae.ae[1].config, self.isae.ae[1].name, "validation_data")
            validateTextpath = self.readField(self.tsae.ae[1].config, self.tsae.ae[1].name, "validation_data")
            validateImgData = gp.garray(np.load(validateImagepath))
            if normalizeImg:
                validateImgData=imgTrainDH.doNormalization(validateImgData)
            validateTxtData = gp.garray(np.load(validateTextpath))
        else:
            print "Warning: no evluation setting!"

        nCommon, nMetric, title=self.getDisplayFields()
        if self.verbose:
            print title
 
        for epoch in range(maxEpoch):
            perf=np.zeros( nMetric)
            epoch1, imgcost, txtcost, diffcost=self.checkPath(epoch)
            imgTrainDH.reset()
            txtTrainDH.reset()
            for i in range(numBatch):
                img = imgTrainDH.getOneBatch() 
                txt = txtTrainDH.getOneBatch()
                curr= self.trainOneBatch(img, txt, epoch1, imgcost, txtcost, diffcost)
                perf=self.aggregatePerf(perf, curr)

            if evalFreq!=0 and (1+epoch) % evalFreq == 0:
                imgcode,txtcode=self.getReps(validateImgData, validateTxtData)
                validation.evalCrossModal(imgcode,txtcode,epoch,'V')

            if showFreq != 0 and (1+epoch) % showFreq == 0:
                imgcode,txtcode=self.getReps(validateImgData, validateTxtData)
                np.save(os.path.join(visDir,'%simg' % str((epoch+1)/showFreq)),imgcode)
                np.save(os.path.join(visDir,'%stxt' % str((epoch+1)/showFreq)),txtcode)

            if self.verbose:
                self.printEpochInfo(epoch, perf, nCommon)

        if self.readField(self.config, self.name, "checkpoint")=="True":
            self.doCheckpoint(outputDir)

        if self.readField(self.config, self.name,"extract_reps")=="True":
            if evalFreq!=0:
                self.extractValidationReps(validateImgData, validateTxtData, "validation_data","validation_reps")
            self.extractTrainReps(imgTrainDH, txtTrainDH, numBatch)

        self.saveConfig(outputDir)
