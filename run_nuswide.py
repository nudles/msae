#!/usr/bin/python

#example script for training and test on NUS-WIDE dataset

import os
import glob
import shutil
import subprocess
import drawVis
import re

def train(dataset, iaes, taes, dim):
    """train image sae & text sae simultaneously; train msae"""
    print '###########Training#################'
    #output_dir=os.path.join('../data/',dataset,'output','output'+str(dim))
    config=os.path.join("config/",dataset,"config"+str(dim)+".ini")

    cmd1="python main.py -a "
    for ae in iaes:
        cmd1+=ae+" "
    cmd1+=config
    print cmd1
    isae=subprocess.Popen(cmd1, shell=True)
    isae.wait()

    cmd2="python main.py -a "
    for ae in taes:
        cmd2+=ae+" "
    cmd2+=config
    print cmd2
    tsae=subprocess.Popen(cmd2, shell=True)
    tsae.wait()

    cmd3="python main.py -a msae "+config
    print cmd3
    subprocess.call(cmd3,shell=True)

def test(dataset, dim, qsize, statfile=None):
    """ do multi-modal retrieval on test dataset; save MAP to result.txt """
    print '#############Test###############'
    input_dir=os.path.join("../data", dataset, "input")
    output_dir=os.path.join("../data", dataset, "output", "output"+str(dim)) 
    
    inputImg=os.path.join(input_dir,"testImg.npy")
    inputTxt=os.path.join(input_dir, "testTxt.npy")
    modelfile=os.path.join(output_dir,"msae","model")
    if statfile:
        statfile=os.path.join(input_dir, statfile)
        subprocess.call(["python","main.py", "-e", modelfile, inputImg, inputTxt, statfile, output_dir])
    else:
        subprocess.call(["python","main.py", "-e", modelfile, inputImg, inputTxt, output_dir])

    imgft=os.path.join(output_dir,"img.npy")
    txtft=os.path.join(output_dir,"txt.npy")
    querypath=os.path.join(input_dir,"query.npy")
    gndpath=os.path.join(input_dir,"testGnd.npy")
    subprocess.call(["python", "main.py", "-s", querypath, gndpath, imgft, txtft, "euclidean", qsize])
    subprocess.call(["python", "main.py", "-p", "tmp/","map"])


def last_vis_file(vis_dir):
    vis_files=glob.glob(vis_dir+"/*.npy")
    vis_files.sort()
    vis_files.sort(key=len)
    print vis_files[-1]
    return  vis_files[-1]

def visualize(dataset, iaes, taes, dim, validation_size=None):
    """visualize training process by project features into 2-d space;
    1. sample and plot latent features after each training epoch
    2. plot MAPs after each training epoch
    both features and MAPs are for validation dataset
    """
    train(dataset, iaes, taes, dim)

    dim=str(dim)
    output_dir=os.path.join("../data", dataset, "output", "output"+dim) 
    isae_vis_dir=os.path.join(output_dir,"isae"+dim, "vis")
    tsae_vis_dir=os.path.join(output_dir,"tsae"+dim, "vis")

    msae_vis_dir=os.path.join(output_dir,"msae","vis")
    shutil.copy(last_vis_file(isae_vis_dir),os.path.join(msae_vis_dir,"0img.npy"))
    shutil.copy(last_vis_file(tsae_vis_dir),os.path.join(msae_vis_dir,"0txt.npy"))

    if not os.path.exists("vis"):
        os.mkdir("vis")

    if os.path.exists("sample.npy"):
        subprocess.call(["python", "drawVis.py", msae_vis_dir, "sample.npy", "vis", "-f", ".png"])
    elif validation_size:
        subprocess.call(["python", "drawVis.py", msae_vis_dir, "sample.npy", "vis", "-f", ".png", "-s", str(validation_size)])
        print "images are in vis dir"
    else:
        print "for sample.npy and validation_size, one must be provided"

    #plot MAPs w.r.t. epoch
    perf=subprocess.check_output(['python','main.py','-p', os.path.join(output_dir,'msae','perf'),'map'])
    lines=perf.split('\n')
    maps=[]
    pat=re.compile(r'0\.[0-9]{4}')
    for line in lines:
        if 'qimg' in line or 'qtxt' in line:
            vals=line.split(' ')
            map=[float(val) for val in vals[1:] if pat.match(val)]
            maps.append(map)
    print maps
    drawVis.drawMAP(maps, "vis/map.png")

if __name__=='__main__':
    """to run on other datasets, just update the following information"""
    dataset='nuswide'
    qsize='1000'
    for dim in [32,24,16]:
        iaes=['iae500-128','iae128-%d' % dim, 'isae%d' % dim]
        taes=['tae1k-128', 'tae128-%d' % dim, 'tsae%d' % dim]
        train(dataset, iaes, taes, dim)
        test(dataset, dim, qsize)

    #for training visualization
    dim=2
    iaes=['iae500-128', 'iae128-16', 'iae16-%d' % dim, 'isae%d' % dim]
    taes=['tae1k-128','tae128-16', 'tae16-%d' % dim, 'tsae%d' % dim]
    visualize(dataset, iaes, taes, dim, 10000)
