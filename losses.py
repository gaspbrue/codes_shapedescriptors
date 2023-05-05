#!/usr/env/bin python3.8
import pdb
from pprint import pprint
from functools import reduce
from operator import mul
from typing import Dict, List, cast

import torch
import numpy as np
from torch import Tensor, einsum
import torch.nn.functional as F

from utils import simplex
from utils import soft_length
from utils import soft_size_outside_box
from utils import soft_size 
from utils import soft_size_inside_box
from utils import soft_size_total_box
from utils import diff_h_bounding_box
from utils import diff_w_bounding_box
from utils import diff_size_bounding_box
from utils import diff_size_bounding_box_min
from utils import diff_size_bounding_box_max
from utils import find_size_bounding_box
from utils import diff_ratio_hw_bounding_box
from utils import diff_mul_bounding_box
from utils import soft_dist_centroid




class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        #BMI self.nd: str = kwargs["nd"]
        print(f"> Initialized {self.__class__.__name__} with kwargs:")
        pprint(kwargs)

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        #BMIloss = - einsum(f"bk{self.nd},bk{self.nd}->", mask, log_p)
        loss = - einsum("bkwh,bkwh->", mask, log_p)

        loss /= mask.sum() + 1e-10

        return loss


class AbstractConstraints():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        #print(f"> Initialized {self.__class__.__name__} with kwargs:")
        #pprint(kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        """
        id: int - Is used to tell if is it the upper or the lower bound
                  0 for lower, 1 for upper
        """
        raise NotImplementedError

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor, filenames: List[str]) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        assert probs.shape == target.shape

        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        value: Tensor = cast(Tensor, self.__fn__(probs[:, self.idc, ...]))
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape
        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).reshape(b, self.C * k)
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).reshape(b, self.C * k)
        if(isinstance(filenames,str)):
            len_fn = 1
        else:
            
            len_fn = len(filenames)
        assert len(upper_z) == len(lower_b) == len_fn

        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)
        assert upper_penalty.numel() == lower_penalty.numel() == upper_z.numel() == lower_z.numel()

        # f for flattened axis
        res: Tensor = einsum("f->", upper_penalty) + einsum("f->", lower_penalty)

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss


class NaivePenalty(AbstractConstraints):
    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()

        return F.relu(z)**2


class LogBarrierLoss(AbstractConstraints):
    def __init__(self, **kwargs):
        self.t: float = kwargs["t"]
        super().__init__(**kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()
        z_: Tensor = z.flatten()
        # del z

        barrier_part: Tensor = - torch.log(-z_) / self.t  # Careful, this part can produce NaN
        barrier_part[torch.isnan(barrier_part)] = 0
        linear_part: Tensor = self.t * z_ + -np.log(1 / (self.t**2)) / self.t + 1 / self.t
        assert barrier_part.dtype == linear_part.dtype == torch.float32

        below_threshold: Tensor = z_ <= - 1 / self.t**2
        assert below_threshold.dtype == torch.bool

        assert barrier_part.shape == linear_part.shape == below_threshold.shape
        res = barrier_part * below_threshold + linear_part * (~below_threshold)
        assert res.dtype == torch.float32

        # if z <= - 1 / self.t**2:
        #     res = - torch.log(-z) / self.t
        # else:
        #     res = self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

        assert res.requires_grad == z.requires_grad
        return res


class LengthRatioLoss():
    def __init__(self, **kwargs):
        self.class_pair: tuple[int, int] = kwargs["class_pair"]
        self.bounds: tuple[int, int] = kwargs["bounds"]
        self.nd: str = kwargs["nd"]
        self.t: float = kwargs["t"]
        print(f"> Initialized {self.__class__.__name__} with kwargs:")
        pprint(kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()
        z_: Tensor = z.flatten()
        # del z

        barrier_part: Tensor = - torch.log(-z_) / self.t  # Careful, this part can produce NaN
        barrier_part[torch.isnan(barrier_part)] = 0
        linear_part: Tensor = self.t * z_ + -np.log(1 / (self.t**2)) / self.t + 1 / self.t
        assert barrier_part.dtype == linear_part.dtype == torch.float32

        below_threshold: Tensor = z_ <= - 1 / self.t**2
        assert below_threshold.dtype == torch.bool

        assert barrier_part.shape == linear_part.shape == below_threshold.shape
        res = barrier_part * below_threshold + linear_part * (~below_threshold)
        assert res.dtype == torch.float32

        # if z <= - 1 / self.t**2:
        #     res = - torch.log(-z) / self.t
        # else:
        #     res = self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

        assert res.requires_grad == z.requires_grad
        return res

    def __call__(self, probs: Tensor, _: Tensor, __: Tensor, ___) -> Tensor:
        assert simplex(probs)

        B, K, *_ = probs.shape  # type: ignore

        lengths: Tensor = soft_length(probs[:, self.class_pair, ...])
        assert lengths.shape == (B, 2, 1), lengths.shape


        loss: Tensor = self.penalty(self.bounds[0] - lengths[0]) + self.penalty(lengths[1] - self.bounds[1])
        assert loss.shape == (2,), loss.shape

        return loss.mean()
    
    
    
    
    
    
    
    
    
 
    
 
class LogBarrierLossWithoutGroundTruth:
    
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        self.t: float = kwargs["t"]
        self.C = len(self.idc)
        pprint(kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()
        z_: Tensor = z.flatten()
        
        barrier_part: Tensor = - torch.log(-z_) / self.t  # Careful, this part can produce NaN
        barrier_part[torch.isnan(barrier_part)] = 0
        linear_part: Tensor = self.t * z_ + -np.log(1 / (self.t**2)) / self.t + 1 / self.t
        assert barrier_part.dtype == linear_part.dtype == torch.float32

        below_threshold: Tensor = z_ <= - 1 / self.t**2
        assert below_threshold.dtype == torch.bool

        assert barrier_part.shape == linear_part.shape == below_threshold.shape
        res = barrier_part * below_threshold + linear_part * (~below_threshold)
        assert res.dtype == torch.float32

        # if z <= - 1 / self.t**2:
        #     res = - torch.log(-z) / self.t
        # else:
        #     res = self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

        assert res.requires_grad == z.requires_grad
        return res    
    
    
class EmptinessLoss(LogBarrierLossWithoutGroundTruth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, probs: Tensor, box) -> Tensor:
        assert simplex(probs)
        B, K, *_ = probs.shape  # type: ignore
        size_outside: Tensor = soft_size_outside_box(probs[:,self.idc,...], box)   
        assert size_outside.shape == (B, self.C, 1), size_outside.shape

        loss: Tensor = self.penalty(size_outside)
        assert loss.shape == (1,), loss.shape
        
        loss = loss/(256*256)
        return loss.mean()






class HeightWidthBoundingBoxLoss(LogBarrierLossWithoutGroundTruth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
    def __call__(self, probs: Tensor, h_min,h_max, w_min,w_max) -> Tensor:
        
        """
        assert simplex(probs)
        B, K, *_ = probs.shape  # type: ignore
        h_pred, w_pred = find_size_bounding_box(probs)
        
        diff_size_h_min: Tensor = diff_h_bounding_box(probs[:,self.idc,...],h_min,h_pred)
        diff_size_h_max: Tensor = diff_h_bounding_box(probs[:,self.idc,...],h_pred,h_max)
        diff_size_w_min: Tensor = diff_w_bounding_box(probs[:,self.idc,...],w_min,w_pred)
        diff_size_w_max: Tensor = diff_w_bounding_box(probs[:,self.idc,...],w_pred,w_max)
        
        assert diff_size_h_min.shape == (B, 1, 1), diff_size_h_min.shape
        
        loss_h_min: Tensor = self.penalty(diff_size_h_min)
        loss_h_max: Tensor = self.penalty(diff_size_h_max)
        loss_w_min: Tensor = self.penalty(diff_size_w_min)
        loss_w_max: Tensor = self.penalty(diff_size_w_max)        
        loss = loss_h_min + loss_h_max + loss_w_min + loss_w_max 
        
        assert loss.shape == (1,), loss.shape
        return loss.mean()
        """

        assert simplex(probs)
        B, K, *_ = probs.shape  # type: ignore
            
        h_pred, w_pred = find_size_bounding_box(probs)
        
        diff_size_min: Tensor = diff_mul_bounding_box(probs[:,self.idc,...],h_min,w_min,h_pred,w_pred)
        diff_size_max: Tensor = diff_mul_bounding_box(probs[:,self.idc,...],h_pred,w_pred,h_max,w_max)
        
        loss_min: Tensor = self.penalty(diff_size_min)
        loss_max: Tensor = self.penalty(diff_size_max)
        loss = loss_min + loss_max 
        
        print(diff_size_min, diff_size_max)
        
        assert loss.shape == (1,), loss.shape
        return loss.mean()
        




class BoundingBoxLoss5(LogBarrierLossWithoutGroundTruth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, probs: Tensor, h, w) -> Tensor:
        assert simplex(probs)

        B, K, *_ = probs.shape  # type: ignore

        diff_size: Tensor = diff_size_bounding_box(probs[:,self.idc,...],h,w)
        
        assert diff_size.shape == (B, 1, 1), diff_size.shape

        loss: Tensor = self.penalty(diff_size)
        assert loss.shape == (1,), loss.shape
        loss = loss/(256*256)        

        return loss.mean()
    
    
    
class BoundingBoxLoss(LogBarrierLossWithoutGroundTruth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, probs: Tensor, h_min, w_min,h_max,w_max) -> Tensor:
        assert simplex(probs)

        B, K, *_ = probs.shape  # type: ignore

        diff_size_min: Tensor = diff_size_bounding_box_min(probs[:,self.idc,...],h_min,w_min)
        diff_size_max: Tensor = diff_size_bounding_box_max(probs[:,self.idc,...],h_max,w_max)
            
        assert diff_size_min.shape == (B, 1, 1), diff_size_min.shape

        loss: Tensor = self.penalty(diff_size_min) + self.penalty(diff_size_max)
        assert loss.shape == (1,), loss.shape
        loss = loss/(256*256)        

        return loss.mean()
    
    




class ratioHeighthWidthBoundingBoxLoss(LogBarrierLossWithoutGroundTruth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, probs: Tensor, h, w) -> Tensor:
        assert simplex(probs)

        B, K, *_ = probs.shape  # type: ignore

        diff_ratio_hw: Tensor = diff_ratio_hw_bounding_box(probs[:,self.idc,...],h,w)
        
        assert diff_ratio_hw.shape == (B, 1, 1), diff_ratio_hw.shape

        loss: Tensor = self.penalty(diff_ratio_hw)
        assert loss.shape == (1,), loss.shape      

        return loss.mean()
    
    

    
    
    
class SizeWithoutGroundTruth(LogBarrierLossWithoutGroundTruth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, probs: Tensor, bounds) -> Tensor:
        assert simplex(probs)

        B, K, *im_shape = probs.shape  # type: ignore

        value: Tensor = soft_size(probs[:,self.idc,...])
        lower_b = bounds[:,0, 0]
        upper_b = bounds[:,0, 1]

        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).reshape(B, self.C)
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).reshape(B, self.C)

        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)
        assert upper_penalty.numel() == lower_penalty.numel() == upper_z.numel() == lower_z.numel()

        # f for flattened axis
        res: Tensor = einsum("f->", upper_penalty) + einsum("f->", lower_penalty)

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation
        
        return loss
        
    
    



class DistanceCentroidWithoutGroundTruth(LogBarrierLossWithoutGroundTruth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, probs: Tensor, bounds) -> Tensor:
        assert simplex(probs)

        B, K, *im_shape = probs.shape  # type: ignore

        value: Tensor = soft_dist_centroid(probs[:,self.idc,...])
        lower_b = bounds[:,0, 0]
        upper_b = bounds[:,0, 1]
        lower_b_modif = lower_b.unsqueeze(0)
        lower_b_modif = lower_b_modif.repeat(1, 1, 2)
        upper_b_modif = upper_b.unsqueeze(0)
        upper_b_modif = upper_b_modif.repeat(1, 1, 2)

        upper_z: Tensor = cast(Tensor, (value - upper_b_modif).type(torch.float32)).reshape(B, self.C*2)
        lower_z: Tensor = cast(Tensor, (lower_b_modif - value).type(torch.float32)).reshape(B, self.C*2)

        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)
        assert upper_penalty.numel() == lower_penalty.numel() == upper_z.numel() == lower_z.numel()

        # f for flattened axis
        res: Tensor = einsum("f->", upper_penalty) + einsum("f->", lower_penalty)

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation
        
        return loss
        
    
class DistanceCentroidWithoutGroundTruth2(LogBarrierLossWithoutGroundTruth):
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        #print(f"> Initialized {self.__class__.__name__} with kwargs:")
        #pprint(kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        """
        id: int - Is used to tell if is it the upper or the lower bound
                  0 for lower, 1 for upper
        """
        raise NotImplementedError

    def __call__(self, probs: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part

        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        value: Tensor = cast(Tensor, self.__fn__(probs[:, self.idc, ...]))
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape
        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).reshape(b, self.C * k)
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).reshape(b, self.C * k)
        if(isinstance(filenames,str)):
            len_fn = 1
        else:
            
            len_fn = len(filenames)
        assert len(upper_z) == len(lower_b) == len_fn

        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)
        assert upper_penalty.numel() == lower_penalty.numel() == upper_z.numel() == lower_z.numel()

        # f for flattened axis
        res: Tensor = einsum("f->", upper_penalty) + einsum("f->", lower_penalty)

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss
        
