#!/usr/bin/env python3.9



import pdb
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import reduce
from shutil import copytree, rmtree
from operator import add, itemgetter
from typing import Any, Callable, Tuple
from shutil import copyfile
import os,sys
# from time import time

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn, einsum

from torch import Tensor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataloader_2 import get_loaders
from utils import map_
from utils import depth, weights_init
from utils import class2one_hot,probs2one_hot, probs2class
from utils import dice_coef, save_images, tqdm_, dice_batch
from utils import EarlyStopping

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader_2 import (SliceDataset)
from torchvision import transforms
from networks import ResidualUNet
from losses import CrossEntropy


import pdb




# .................... PLOT UTILS ########################################

import math
import matplotlib.pyplot as plt
params2 = { 'axes.labelsize': 24,
  'text.fontsize': 24,
  'legend.fontsize': 18,
  'xtick.labelsize': 18,
  'ytick.labelsize': 18,
  'labelsize': 24,
'axes.titlesize':30
  }
params = { 'axes.labelsize': 24,
  'text.fontsize': 24,
  'legend.fontsize': 18,
  'xtick.labelsize': 18,
  'ytick.labelsize': 18,
'axes.titlesize':30
  }
params3 = { 'axes.labelsize': 24,
  'legend.fontsize': 18,
  'xtick.labelsize': 18,
  'ytick.labelsize': 18,
'axes.titlesize':30
  }

def arrayToFile(myArray,fn):
    myFile = open( fn , "w" )
    for elem in myArray:

        if(np.issubdtype(elem,np.dtype(int).type)):
            myFile.write("%d "%elem)
        elif(np.issubdtype(elem,np.dtype(np.float32).type)):
            myFile.write("%.3f "%elem)
        else:
            intelem = int(elem)
            myFile.write("%d "%intelem)

    myFile.write("\n")
    myFile.close()
def PlotLists(ListOfLists,fileName,display,xlabel="x",ylabel="pixel value",mylegends=[],ylimit=0,title=None):
    plt.clf()
    if(mylegends != "None"):
        if(mylegends==[]):
            mylegends = range(len(ListOfLists))
    plt.rcParams.update(params3)
    plt.grid(True)
    if(xlabel !=None):
        plt.xlabel(xlabel)
    if(ylabel !=None):
        plt.ylabel(ylabel)
    if(title !=None):
        plt.title(title)


    markerList = ['o','.','+','*','p','s','x','D','h','^']

    i = 0

    for myList in ListOfLists:
        if(mylegends != "None"):
            tmp_legend = mylegends[i]
        else:
            tmp_legend = [str(i)]

        marker_nb = int(math.fmod(i,len(markerList)))
        #        plt.plot(myList, linewidth=2.0,marker=markerList[marker_nb],label = tmp_legend)
        plt.plot(myList, linewidth=2.0,label = tmp_legend)

        i = i + 1
        plt.legend()
        
#        plt.legend(mylegend1,mylegend2,loc='lower left')
        # if(mylegends != "None"):
        #     plt.legend(mylegend1,mylegend2,loc='lower center')

    if(ylimit > 0):
        plt.ylim(ymax=ylimit)
    plt.tight_layout()        
    plt.savefig(fileName)
    plt.clf()
    if(display == 1):
        pdb.set_trace()
        imtoto = Image(fileName)
        file_i = os.path.splitext(os.path.split(fileName)[1])[0]

        ImDisplayX(imtoto,"tmp_"+file_i)
        pdb.set_trace()

    

# .................... UTILS ########################################



def setLossesList(args):
    losses_list = []
    if(args.S):
        losses_list.append(('LogBarrierLoss', {'idc': [0, 1], 't': 1}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2))
    if(args.L):
        losses_list.append(('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_length', 1e-2))
    if(args.C):
        losses_list.append(('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_centroid', 1e-2))
    if(args.CP):
        losses_list.append(('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_compactness', 1e-2))       

    return losses_list

import time
def get_time_string():
    """ return time in string format"""
    time_list = time.localtime()
    return time_list,"23_%s_" % str(time_list[1]) + "_".join(["%02d" % i for i in time_list[2:5]])
def  preciseDirName(savedir,add_str):
    time_list,time_str = get_time_string()

    tmpdir = Path(str(savedir)+"/"+time_str+"_"+add_str)
    if tmpdir.exists():
        savedir = Path(str(savedir)+"/"+time_str+"_"+str(time_list[5])+"_"+add_str)# add seconds!
    else:
        savedir = Path(str(savedir)+"/"+time_str+"_"+add_str)
    return savedir
def getAddStr(args):
    add_str = "E"+str(args.n_epoch)
    if(args.ob_ce):
        add_str=add_str+"_OCE"
    else:
        if (args.dataset == "data/capillaires"):
            add_str=add_str+"_capillaires"
        if (args.dataset == "data/TOY10/cercle"):
            add_str=add_str+"_cercle"
        if (args.dataset == "data/carrenoir"):
            add_str=add_str+"_carrenoir"
        
        if(args.S):
            add_str=add_str+"_S"
        if(args.L):
            add_str=add_str+"_L"
        if(args.C):
            add_str=add_str+"_C"
        if(args.L):
            add_str=add_str+"_L"
        if(args.b_ce):
            add_str=add_str+"_CE"    
        if(args.CP):
            add_str=add_str+"_CP"    
        if(args.e):
            add_str=add_str+"_empt" 
        if (args.hwBB):
            add_str=add_str+"_hwBB"
        if (args.ratiohwBB):
            add_str=add_str+"_ratiohwBB"
        if (args.sizewgt):
            add_str=add_str+"_sizewgt"
        if (args.bb):
            add_str=add_str+"_bb"
        if (args.bb_tol):
            add_str=add_str+"_bb_tol"
            
    return add_str
def infoToFile(fn,savedir,best_epoch,best_dice,metrics):
    myF= open(fn, "a")
    myF.write(str(savedir))
    myF.write("\n")

    
    for metric in metrics:
        # Do not care about training values, nor the loss (keep it simple):
        if "val" in metric or "loss" in metric:
            myF.write(f"\t{metric}: {metrics[metric][best_epoch].mean(dim=0)}")
    myF.write("\n##################################\n")                        
    myF.close()

    
def setup(args, n_class: int) -> Tuple[Any, Any, Any, list[list[Callable]], list[list[float]], Callable]:
    print("\n>>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")
    K = 2  # K for the number of classes # two variables for the same thing? n_class and K?

    # SELECT DATABASE
    if (args.dataset.find("TOY")>=0):
        initial_kernels = 4
        #net = shallowCNN(1, initial_kernels, K)

    root_dir = Path(".") / args.dataset

    
    # .......... MODEL ....................
    if args.weights:# FROM A PREVIOUS MODEL?
        if cpu:
            net = torch.load(args.weights, map_location='cpu')
        else:
            net = torch.load(args.weights)
            print(f">> Restored weights from {args.weights} successfully.")
    else:
        net_class = getattr(__import__('networks'), args.network)
        net = net_class(args.modalities, n_class).to(device)
        net.init_weights()
    net.to(device)

    # .......... OPTIMIZER ....................
    optimizer: Any  # disable an error for the optmizer (ADAM and SGD not same type)
    if args.use_sgd:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.99), amsgrad=False)

    # .......... LOSSES ....................        
    if():
        losses_list = eval(args.losses)
    else:
        losses_list = setLossesList(args)
    #if depth(losses_list) == 1:  # For compatibility reasons, avoid changing all the previous configuration files
    #        losses_list = [losses_list]

    nd: str = "whd" if args.three_d else "wh"


    loss_fns: list[Callable] = []
    
    for loss_name, loss_params, _, _, fn, _ in losses_list:
        loss_class = getattr(__import__('losses'), loss_name)
        loss_fns.append(loss_class(**loss_params, fn=fn, nd=nd))

    loss_weights: list[Callable] = []
    for losses in losses_list:
        loss_weights.append(losses[5])



    #loss_weights: list[list[float]] = [map_(itemgetter(5), losses) for losses in losses_list]


    # .......... SCHEDULER ....................        
    scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))
    # Dataset part
    batch_size = 1
        
    transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    mask_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # and {0, 85, 170, 255} for 4 classes
        lambda nd: nd / (255 / (K - 1)),  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        lambda t: class2one_hot(t, K=K),
        itemgetter(0)
    ])

    train_set = SliceDataset('train',
                             root_dir,
                             n_class = n_class,
                             losses_list = losses_list,
                             transform=transform,
                             mask_transform=mask_transform,
                             augment=True,
                             equalize=False)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=0,# bmi num_workers = 5
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           n_class = n_class,
                           losses_list = losses_list,
                           transform=transform,
                           mask_transform=mask_transform,
                           equalize=False)
    val_loader = DataLoader(val_set,
                            batch_size=1,
                            num_workers=0,# bmi num_workers = 5
                            shuffle=False)




    e_class = getattr(__import__('losses'), 'EmptinessLoss')
    e_loss = e_class(nd=nd,** {'idc': [1], 't': 1})
    
    hwBB_class = getattr(__import__('losses'), 'HeightWidthBoundingBoxLoss')
    hwBB_loss = hwBB_class(nd=nd,** {'idc' : [1], 't' : 1})
    
    ratiohwBB_class = getattr(__import__('losses'), 'ratioHeighthWidthBoundingBoxLoss')
    ratiohwBB_loss = ratiohwBB_class(nd=nd,** {'idc' : [1], 't' : 1})
    
    sizewgt_class = getattr(__import__('losses'), 'SizeWithoutGroundTruth')
    sizewgt_loss = sizewgt_class(nd=nd,** {'idc' : [1], 't' : 1})
    
    distancewgt_class = getattr(__import__('losses'), 'DistanceCentroidWithoutGroundTruth2')
    distancewgt_loss = distancewgt_class(nd=nd,** {'idc' : [1], 't' : 1},fn = 'soft_dist_centroid')
    
    bb_class = getattr(__import__('losses'), 'BoundingBoxLoss5')
    bb_loss = bb_class(nd=nd,** {'idc' : [1], 't' : 1})
    
    bb_class_tol = getattr(__import__('losses'), 'BoundingBoxLoss')
    bb_loss_tol = bb_class_tol(nd=nd,** {'idc' : [1], 't' : 1})
    
    
    if (args.e):
        loss_fns.append(e_loss)
        loss_weights.append(0.01)
        
    if (args.sizewgt):
        loss_fns.append(sizewgt_loss)
        loss_weights.append(0.01)
    if (args.d):
        loss_fns.append(distancewgt_loss)
        loss_weights.append(0.01)
    if (args.bb):
        loss_fns.append(bb_loss)
        loss_weights.append(0.01)    
    if (args.bb_tol):
        loss_fns.append(bb_loss_tol)
        loss_weights.append(0.01)
    
    
    
    print("END SETUP")

    return net, optimizer, device, loss_fns, loss_weights, scheduler, train_loader, val_loader, e_loss, hwBB_loss,ratiohwBB_loss,sizewgt_loss,bb_loss,bb_loss_tol,distancewgt_loss






def do_epoch(mode: str, net: Any, device: Any, loader: DataLoader, epc: int,
             loss_fns: list[Callable], loss_weights: list[float],e_loss,hwBB_loss,ratiohwBB_loss,sizewgt_loss,bb_loss,bb_loss_tol,distancewgt_loss, K: int,
             savedir: Path = None, optimizer: Any = None,
             b_ce:bool=False, ob_ce:bool=False,metric_axis: list[int] = [1], requested_metrics: list[str] = None,
             temperature: float = 1) -> dict[str, Tensor]:
    assert mode in ["train", "val", "dual"]
    if requested_metrics is None:
        requested_metrics = []
    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"
    elif mode == "dual":
        net.eval()
        desc = f">> Dual       ({epc})"

    total_iteration: int = len(loader)#) for loader in loaders)  # U
    total_images: int = len(loader.dataset)# for loader in loaders)  # D

    n_loss: int = len(loss_fns)#max(map(len, list_loss_fns))
    if(b_ce or ob_ce):
        n_loss = n_loss + 1
    if (args.hwBB):
        n_loss = n_loss + 1
    if (args.ratiohwBB):
        n_loss = n_loss + 1

    epoch_metrics: dict[str, Tensor]
    epoch_metrics = {"loss": torch.zeros((total_iteration, n_loss), dtype=torch.float32, device=device)}

    if "3d_dsc" in requested_metrics:
        epoch_metrics["3d_dsc"] = torch.zeros((total_iteration, K), dtype=torch.float32, device=device)

    few_axis: bool = len(metric_axis) <= 4

    # time_log: np.ndarray = np.ndarray(total_iteration, dtype=np.float32)

    done_img: int = 0
    done_batch: int = 0
        
    if(0):
        tq_iter = tqdm_(total=total_iteration, desc=desc)
    else:#BMI
        tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)

    #        for i, (loader, loss_fns, loss_weights) in enumerate(zip(loaders, list_loss_fns, list_loss_weights)):

        


    L_dices=[]
    for i, data in tq_iter:
    
        # t0 = time()
        image: Tensor = data["images"].to(device)

        filenames: list[str] = data["filenames"]


        
   
        
        
        loss_log: Tensor = torch.zeros(( n_loss), 
                                       dtype=torch.float32, device=device)
        
        B, C, *_ = image.shape
        
        # Reset gradients
        if optimizer:
            optimizer.zero_grad()

        # Forward
        pred_logits: Tensor = net(image)
        pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
        

        # Used only for dice computation:
        predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  
        assert not predicted_mask.requires_grad


        losses = []      

        if (args.e):
            w=0.01
            if (args.dataset == "data/capillaires"):
                x_min,y_min,x_max,y_max = 60,60,200,166
            if (args.dataset == "data/TOY10/cercle"):
                x_min,y_min,x_max,y_max = 60,60,220,220
            if (args.dataset == "data/carrenoir"):
                x_min,y_min,x_max,y_max = 60,30,190,190
            box = [x_min,y_min,x_max,y_max]
            toto_e = e_loss(pred_probs, box) 

            losses.append(w*toto_e)
        
        if (args.hwBB):
            if (args.dataset == "data/capillaires"):
                h,w = 107,127
            if (args.dataset == "data/TOY10/cercle"):
                h,w = 64,64
            if (args.dataset == "data/carrenoir"):
                h,w = 138,102
            percentage = 0.10
            h_min, h_max = h - h*percentage, h + h*percentage
            w_min, w_max = w - w*percentage, w + w*percentage
            toto_hwBB = hwBB_loss(pred_probs,h_min,h_max,w_min,w_max) 
            losses.append(toto_hwBB)
            
        if (args.ratiohwBB):
            h,w = 138,102
            toto_ratiohwBB = ratiohwBB_loss(pred_probs,h,w) 
            losses.append(toto_ratiohwBB)
            
        if (args.sizewgt):
            w=0.01
            p = 0.20
            if (args.dataset == "data/capillaires"):
                Size = 1800
            if (args.dataset == "data/TOY10/cercle"):
                Size = 3200
            if (args.dataset == "data/carrenoir"):
                Size = 14000
                p = 0.05
            bounds = torch.tensor([[[Size - Size*p,Size + Size*p]]])
            toto_sizewgt = sizewgt_loss(pred_probs,bounds)
            losses.append(w*toto_sizewgt)
            
        
        if (args.bb):
            weights=0.01
            if (args.dataset == "data/capillaires"):
                h,w = 107,127
            if (args.dataset == "data/TOY10/cercle"):
                h,w = 64,64
            if (args.dataset == "data/carrenoir"):
                h,w = 138,102
            toto_bb = bb_loss(pred_probs,h,w) 
            losses.append(weights*toto_bb)
            
            
        if (args.bb_tol):
            weights=0.01
            if (args.dataset == "data/capillaires"):
                h,w = 107,127
            if (args.dataset == "data/TOY10/cercle"):
                h,w = 64,64
            if (args.dataset == "data/carrenoir"):
                h,w = 138,102
            p = 0.05
            h_min,w_min,h_max,w_max = h-h*p,w-w*p,h+h*p,w+w*p
            toto_bb_tol = bb_loss_tol(pred_probs,h_min,w_min,h_max,w_max) 
            losses.append(weights*toto_bb_tol)
           
            
        
        if(args.d):
            w=1
            bounds = torch.tensor([[[29.9156,39.8784]]])
            toto_distancewgt = distancewgt_loss(pred_probs,bounds)
            losses.append(w*toto_distancewgt)
              
        



                
        if(b_ce or ob_ce):#BMI add cross entropy too 17.01.2022
            targetf=target.to(torch.float32)
            ce_loss = nn.CrossEntropyLoss()#idc=list(range(K)))  # Supervise both background and foreground
            ce_val = ce_loss(pred_logits, targetf)#pytorch crossEntropy includes softmax, but it is redefined in losses.py and it does not include softmax...

            #toto: Tensor = torch.tensor(0, dtype=torch.float32,requires_grad=True)

            ce_val = torch.mul(ce_val,0.01)#np.amax(loss_weights))
            losses.append(ce_val)

        pdb.set_trace()

        if(ob_ce):
            # Backward
            if optimizer:
                ce_val.backward()
                optimizer.step()
              
        
        else:
            loss = reduce(add, losses)
            # Backward
            

            if optimizer:
                loss.backward()
                optimizer.step()

        # Compute and log metrics
        for j in range(len(loss_fns)):
            loss_log[j] = losses[j].detach()
        if(b_ce or ob_ce):
            loss_log[len(loss_fns)] = ce_val
        
        index = 0
        names = []
        if(args.e):
            names.append("e")
        if(args.hwBB):
            names.append("hwBB")
        if(args.ratiohwBB):
            names.append("ratiohwBB")
        if(args.sizewgt):
            names.append("sizewgt")
        if(args.d):
            names.append("distancewgt")
        if(args.bb):
            names.append("bb")
        if(args.bb_tol):
            names.append("bb_tol")

            
            

        if(loss_log.shape == torch.Size([1])):
            reduced_loss_log = loss_log
        else:
            reduced_loss_log: Tensor = loss_log.sum(dim=0)
            #assert reduced_loss_log.shape == (len(loss_fns)), (reduced_loss_log.shape, len(loss_fns))

        epoch_metrics["loss"][done_batch, ...] = loss_log[...]
        #del loss_sub_log

        sm_slice = slice(done_img, done_img + B)  # Values only for current batch

      
            

        if "3d_dsc" in requested_metrics:
            three_d_DSC: Tensor = dice_batch(predicted_mask, target)
            assert three_d_DSC.shape == (K,)
            
            epoch_metrics["3d_dsc"][done_batch] = three_d_DSC  # type: ignore

        # Save images
        if savedir:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                predicted_class: Tensor = probs2class(pred_probs)
                if(epc>300000000000000000000000):
                    pdb.set_trace()
                    import imageio
                    fn = data["filenames"][0]
                    im_name = os.path.splitext(os.path.split(fn)[1])[0]
                    imageio.imwrite("prob_"+im_name+".png", pred_probs[0,1].detach().cpu().numpy())
                else:
                    
                    save_images(predicted_class, 
                                data["names"], 
                                savedir / f"iter{epc:03d}" / "predicted_class" )
                    
                    save_images(pred_probs[:,0,:,:], 
                                data["names"], 
                                savedir / f"iter{epc:03d}" / "predicted_probs" )

                    


        # Logging
        big_slice = slice(0, done_img + B)  # Value for current and previous batches

        stat_dict: dict[str, Any] = {}
        # The order matters for the final display -- it is easy to change
        if few_axis:

            if "3d_dsc" in requested_metrics:
                stat_dict |= {f"3d_DSC{n}": epoch_metrics["3d_dsc"][:done_batch, n].mean()
                              for n in metric_axis}

        if len(metric_axis) > 1:

            stat_dict |= {f"loss_{names[j]}": epoch_metrics["loss"][:done_batch].mean(dim=0)[j] 
                          for j in range(n_loss)}

            nice_dict = {k: f"{v:.5f}" for (k, v) in stat_dict.items()}

            # t1 = time()
            # time_log[done_batch] = (t1 - t0)

            done_img += B
            done_batch += 1
            tq_iter.set_postfix({**nice_dict, "loader": str(i)})
            tq_iter.update(1)
    tq_iter.close()

    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    return {k: v.detach().cpu() for (k, v) in epoch_metrics.items()}
def run(args: argparse.Namespace) -> dict[str, Tensor]:
    n_class: int = args.n_class
    lr: float = args.l_rate
    savedir: Path = Path(args.workdir)
    n_epoch: int = args.n_epoch
    val_f: int = args.val_loader_id# not used, try simple setup, with single val set BMI
    loss_fns: list[list[Callable]]
    loss_weights: list[float]
    net, optimizer, device, loss_fns, loss_weights, scheduler,train_loader,val_loader,e_loss,hwBB_loss,ratiohwBB_loss,sizewgt_loss,bb_loss,bb_loss_tol,distancewgt_loss = setup(args, n_class)
    
    
    add_str = getAddStr(args)
    savedir = preciseDirName(savedir,add_str)
    myF= open(Path(str(".")+"/"+ args.current_res), "a")
    myF.write(f"savedir:{savedir}\n")
    myF.close()

    #train_loaders: list[DataLoader]
    #val_loaders: list[DataLoader]
    #train_loaders, val_loaders = get_loaders(args, args.dataset,
    #                                         args.batch_size, n_class,
    #                                         args.debug, args.in_memory, args.dimensions, args.use_spacing)

    n_tra: int = len(train_loader.dataset)#sum(len(tr_lo.dataset) for tr_lo in train_loaders)  # Number of images in dataset
    l_tra=n_tra#BATCH SIZE =1 : int = sum(len(tr_lo) for tr_lo in train_loaders)  # Number of iteration per epc: different if batch_size > 1
    n_val: int = len(val_loader.dataset)#sum(len(vl_lo.dataset) for vl_lo in val_loaders)
    l_val: int = n_val#sum(len(vl_lo) for vl_lo in val_loaders)
    
    n_loss: int = len(loss_fns)# BMI max(map(len, loss_fns))
    losslist=[]
    if(args.b_ce or args.ob_ce):
        n_loss = n_loss + 1
    if(args.e):
        losslist.append("e")
    if (args.hwBB):
        losslist.append("hwBB")
    if (args.ratiohwBB):
        n_loss+=1
    if (args.sizewgt):
        losslist.append("sizewgt")
    if (args.d):
        losslist.append("distancewgt")
    if (args.bb):
        losslist.append("bb")
    if (args.bb_tol):
        losslist.append("bb_tol")


    best_dice: Tensor = torch.tensor(0, dtype=torch.float32)
    best_epoch: int = 0
    metrics: dict[str, Tensor] = {"val_dice": torch.zeros((n_epoch, n_val, n_class), dtype=torch.float32),
                                  "val_loss": torch.zeros((n_epoch, l_val, n_loss), dtype=torch.float32),#BMI len(loss_fns[val_f])
                                  "tra_dice": torch.zeros((n_epoch, n_tra, n_class), dtype=torch.float32),
                                  "tra_loss": torch.zeros((n_epoch, l_tra, n_loss), dtype=torch.float32)}

    if args.compute_3d_dice:
        metrics["val_3d_dsc"] = torch.zeros((n_epoch, l_val, n_class), dtype=torch.float32)
        best_3d_dsc: Tensor = torch.tensor(0, dtype=torch.float32)

    tra_req_metrics: list[str] = [k.removeprefix("tra_") for k in metrics.keys() if "tra_" in k]
    val_req_metrics: list[str] = [k.removeprefix("val_") for k in metrics.keys() if "val_" in k]

    print("\n>>> Starting the training")

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
         
    for i in range(n_epoch):
        # Do training and validation loops
        tra_metrics = do_epoch("train", net, device, train_loader, i,
                               loss_fns, loss_weights,e_loss, hwBB_loss,ratiohwBB_loss,sizewgt_loss, bb_loss,bb_loss_tol,distancewgt_loss,n_class,
                               savedir=savedir if args.save_train else None,
                               optimizer=optimizer,
                               b_ce=args.b_ce,
                               ob_ce=args.ob_ce,
                               metric_axis=args.metric_axis,
                               requested_metrics=tra_req_metrics,
                               temperature=args.temperature)
        
        loss_values = []
        for k in range(n_loss):
            loss_values.append(tra_metrics["loss"].mean(dim=0)[k].item())

        # Ajouter les valeurs de loss et la moyenne à la fois à l'aide de add_scalars
        loss_dict = {losslist[k]: loss_values[k] for k in range(n_loss)}
        loss_dict["mean"] = tra_metrics["loss"].mean(dim=0).mean().item()
        writer.add_scalars("Loss/train", loss_dict, i)
        
        
        
        
        
        with torch.no_grad():
            val_metrics = do_epoch("val", net, device, val_loader, i,
                                   loss_fns,
                                   loss_weights,
                                   e_loss,hwBB_loss,ratiohwBB_loss,sizewgt_loss,bb_loss,bb_loss_tol,distancewgt_loss,
                                   n_class,
                                   savedir=savedir,
                                   b_ce=args.b_ce,
                                   ob_ce=args.ob_ce,
                                   metric_axis=args.metric_axis,
                                   requested_metrics=val_req_metrics,
                                   temperature=args.temperature)
            
            loss_values = []
            for k in range(n_loss):
                loss_values.append(val_metrics["loss"].mean(dim=0)[k].item())

            # Ajouter les valeurs de loss et la moyenne à la fois à l'aide de add_scalars
            loss_dict = {losslist[k]: loss_values[k] for k in range(n_loss)}
            loss_dict["mean"] = val_metrics["loss"].mean(dim=0).mean().item()
            writer.add_scalars("Loss/val", loss_dict, i)


        # Sort and save the metrics
        for mode, mode_metrics in zip(["tra_", "val_"],
                                              [tra_metrics, val_metrics]):
            for k in mode_metrics:
                key: str = f"{mode}{k}"
                assert metrics[key][i].shape == mode_metrics[k].shape, \
                    (metrics[key][i].shape, mode_metrics[k].shape, k)
                metrics[key][i] = mode_metrics[k]

        if(0):#DEBUG
            pdb.set_trace()
            print(i,metrics["val_loss"][i],metrics["val_loss"][i].mean(dim=1).mean(dim=0))
            print(metrics["val_loss"][i].mean(dim=0))
        for k, e in metrics.items():
            np.save(savedir / f"{k}.npy", e.cpu().numpy())
        

        dict = {}
        for j in range(n_loss):
            dict[f"val_loss{losslist[j]}"] = metrics["val_loss"].mean(dim=1)[:,j].numpy()
        dict["tra_loss_mean"] = metrics["tra_loss"].mean(dim=(1,2)).numpy()
        dict["val_loss_mean"] = metrics["val_loss"].mean(dim=(1,2)).numpy()
        df = pd.DataFrame(dict,index=range(n_epoch))
        

        df.to_csv(savedir / args.csv, float_format="%.4f", index_label="epoch")

        # Save model if better
       

                   
        optimizer, loss_fns, loss_weights = scheduler(i, optimizer, loss_fns, loss_weights,
                                                      net, device, train_loader, args)

        # if args.schedule and (i > (best_epoch + 20)):
        if args.schedule and (i % (best_epoch + 20) == 0):  # Yeah, ugly but will clean that later
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group['lr'] = lr
                print(f'>> New learning Rate: {lr}')
                
    for metric in metrics:
        # Do not care about training values, nor the loss (keep it simple):
        if "val" in metric or "loss" in metric:
            print(f"\t{metric}: {metrics[metric][best_epoch].mean(dim=0)}")
    this_file_name = os.path.basename(__file__)
    copyfile(__file__, os.path.join(savedir, this_file_name))
    copyfile( os.path.join(".", "losses.py"), os.path.join(savedir, "losses.py"))
    copyfile( os.path.join(".", "utils.py"), os.path.join(savedir, "utils.py"))
    copyfile( os.path.join(".", "dataloader.py"), os.path.join(savedir, "dataloader.py"))
    fn = Path(str(savedir)+"/"+ args.csv)
  
    
    titititi = list(metrics['tra_loss'][:,:,0].mean(dim=1))
    totototo = list(metrics['val_loss'][:,:,0].mean(dim=1))

    fn = os.path.join(savedir,add_str+"_loss_mean.png")
    display = False
    PlotLists([titititi,totototo],fn,display,xlabel="epoch",ylabel="loss",mylegends=["train","val"],ylimit=0,title=None)
    if(0):#DEBUG
        pdb.set_trace()
        print("BEST EPOCH",best_epoch)
    return best_dice,metrics,savedir


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--csv", default = "csv", type=str)
    parser.add_argument("--workdir", type=str, default="results")
    parser.add_argument("--losses", type=str, required=True,
                        help="list of list of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")

    parser.add_argument("--folders", type=str,
                        help="list of list of (subfolder, transform, is_hot)")
    parser.add_argument("--network", type=str, default="ResidualUNet", help="The network to use")
    parser.add_argument('--n_class', type=int, default=2, help='# number of classes')

    
    parser.add_argument("--metric_axis", type=int, nargs='*', default=[0,1], help="Classes to display metrics. \
    Display only the average of everything if empty")
    
    parser.add_argument("--b_ce", action='store_true')#crossEntrpy
    parser.add_argument("--ob_ce", action='store_true')#crossEntrpy
    parser.add_argument("--S", action='store_true')#Loss size
    parser.add_argument("--L", action='store_true')#Loss length
    parser.add_argument("--C", action='store_true')#Loss centroid
    parser.add_argument("--D", action='store_true')#Loss std deviation
    
    parser.add_argument("--losslist", type=str, required=True)
    
    parser.add_argument("--augment_blur", action="store_true")
    parser.add_argument("--blur_onlyfirst", action="store_true",
                        help="Blur only the first image, i.e., the image, and not the ground truth.")
    parser.add_argument("--augment_rotate", action="store_true")
    parser.add_argument("--augment_scale", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--schedule", action='store_true')
    parser.add_argument("--use_sgd", action='store_true')
    parser.add_argument("--compute_3d_dice", action='store_true')
    parser.add_argument("--save_train", action='store_true')
    parser.add_argument("--use_spacing", action='store_true')
    parser.add_argument("--three_d", action='store_true')
    parser.add_argument("--no_assert_dataloader", action='store_true')
    parser.add_argument("--ignore_norm_dataloader", action='store_true')
    parser.add_argument("--group", action='store_true', help="Group the patient slices together for validation. \
    Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")
    parser.add_argument("--group_train", action='store_true', help="Group the patient slices together for training. \
    Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")
    
    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--patience', nargs='?', type=int, default=10,
                        help='# patience')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,#5e-4,
                        help='Learning Rate')
    parser.add_argument("--grp_regex", type=str, default=None)
    parser.add_argument('--temperature', type=float, default=1, help="Temperature for the softmax")
    parser.add_argument("--scheduler", type=str, default="MultiplyT")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--modalities", type=int, default=1)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--weights", type=str, default='', help="Stored weights to restore")
    parser.add_argument("--training_folders", type=str, nargs="+", default=["train"])
    parser.add_argument("--validation_folder", type=str, default="val")
    parser.add_argument("--val_loader_id", type=int, default=-1, help="""
    Kinda housefiry at the moment. When we have several train loader (for hybrid training
    for instance), wants only one validation loader. The way the dataloading creation is
    written at the moment, it will create several validation loader on the same topfolder (val),
    but with different folders/bounds ; which will basically duplicate the evaluation.
    """)

    args = parser.parse_args()
    if args.metric_axis == []:
        args.metric_axis = list(range(args.n_class))
    print("\n", args)
    return args

def runN(args: argparse.Namespace) -> dict[str, Tensor]:
    fn = Path(str(".")+"/"+ args.current_res)
    myF= open(fn, "a")
    myF.write(f"############...............##################\n")
    myF.close()

    N = 1
    bestDiceN=0
    for i in range(N):
        myF= open(fn, "a")
        myF.write(f"ITERATION:{i}/{N} \n")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"ITERATION:{i}/{N} \n")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        myF.close()
        best_dice,metrics,savedir=run(args)

        bestDiceN += best_dice
    bestDiceN = bestDiceN /N

    myF= open(fn, "a")
    myF.write(f"AVG:{N}: DSC: {bestDiceN:.3f}")
    
    myF.write("\n##################################\n")
    myF.write("\n##################################\n")
    myF.close()
    print(f"AVG:{N}: DSC: {bestDiceN:.3f}")


def setArgParam(args):
    losslist = args.losslist
    losslist_e=eval(losslist)
    args.b_ce,args.ob_ce,args.S,args.C,args.L,args.CP,args.e,args.hwBB,args.ratiohwBB,args.sizewgt,args.bb,args.bb_tol,args.d = False,False,False,False,False,False,False,False,False,False,False,False,False
    for loss in losslist_e:
        if(loss == "size"):
            args.S = True
        elif (loss == "length"):
            args.L = True
        elif(loss == "centroid"):
            args.C = True
        elif(loss == "distancewgt"):
            args.d = True
        elif(loss == "sizeCE"):
            args.S = True
            args.b_ce = True
        elif(loss == "crossentropy"):
            args.b_ce = True
        elif(loss == "compactness"):
            args.CP = True
        elif (loss == "OCE"):
            args.ob_ce = True
        elif(loss == "emptiness"):
            args.e = True
        elif(loss == "height_width_bounding_box"):
            args.hwBB = True
        elif(loss == "ratio_height_widthbounding_box"):
            args.ratiohwBB = True
        elif(loss == "sizewgt"):
            args.sizewgt = True
        elif(loss == "bb"):
            args.bb = True
        elif(loss == "bb_tol"):
            args.bb_tol = True
        else:
            print("segArgParam ERROR")
            pdb.set_trace()
            
    return args




def startloss(args):
    losslist=args.losslist;
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("loss=", losslist)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    args = setArgParam(args)

    args.current_res = "current_res3.txt"
    runN(args)
    writer.flush()
    writer.close()
if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter('runs/capillaires_tensorboard')

    startloss(args)





