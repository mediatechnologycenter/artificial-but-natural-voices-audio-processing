# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
aDAPTED FROM https://github.com/gabrielmittag/NISQA/blob/master/nisqa/NISQA_lib.py
"""
import os
import multiprocessing
import copy
import math

import librosa as lb
import numpy as np
import pandas as pd; pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


#%% Models
class NISQA(nn.Module):
    '''
    NISQA: The main speech quality model without speech quality dimension 
    estimation (MOS only). The module loads the submodules for framewise 
    modelling (e.g. CNN), time-dependency modelling (e.g. Self-Attention     
    or LSTM), and pooling (e.g. max-pooling or attention-pooling)                                                  
    '''       
    def __init__(self,
            ms_seg_length=15,
            ms_n_mels=48,
            
            cnn_model='adapt',
            cnn_c_out_1=16, 
            cnn_c_out_2=32,
            cnn_c_out_3=64,
            cnn_kernel_size=3, 
            cnn_dropout=0.2,
            cnn_pool_1=[24,7],
            cnn_pool_2=[12,5],
            cnn_pool_3=[6,3],  
            cnn_fc_out_h=None,     
              
            td='self_att',
            td_sa_d_model=64,
            td_sa_nhead=1,
            td_sa_pos_enc=None,
            td_sa_num_layers=2,
            td_sa_h=64,
            td_sa_dropout=0.1,
            td_lstm_h=128,
            td_lstm_num_layers=1,
            td_lstm_dropout=0,
            td_lstm_bidirectional=True,
            
            td_2='skip',
            td_2_sa_d_model=None,
            td_2_sa_nhead=None,
            td_2_sa_pos_enc=None,
            td_2_sa_num_layers=None,
            td_2_sa_h=None,
            td_2_sa_dropout=None,
            td_2_lstm_h=None,
            td_2_lstm_num_layers=None,
            td_2_lstm_dropout=None,
            td_2_lstm_bidirectional=None,            
            
            pool='att',
            pool_att_h=128,
            pool_att_dropout=0.1,
               
            ):
        super().__init__()
    
        self.name = 'NISQA'
        
        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1, 
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size, 
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,   
            )        
        
        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
            )
        
        self.time_dependency_2 = TimeDependency(
            input_size=self.time_dependency.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
            )        
        
        self.pool = Pooling(
            self.time_dependency_2.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
            )                 

    def forward(self, x, n_wins):
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        x, n_wins = self.time_dependency_2(x, n_wins)
        x = self.pool(x, n_wins)
        return x
    


class NISQA_DIM(nn.Module):
    '''
    NISQA_DIM: The main speech quality model with speech quality dimension 
    estimation (MOS, Noisiness, Coloration, Discontinuity, and Loudness).
    The module loads the submodules for framewise modelling (e.g. CNN),
    time-dependency modelling (e.g. Self-Attention or LSTM), and pooling 
    (e.g. max-pooling or attention-pooling)                                                  
    '''         
    def __init__(self,
            ms_seg_length=15,
            ms_n_mels=48,
            
            cnn_model='adapt',
            cnn_c_out_1=16, 
            cnn_c_out_2=32,
            cnn_c_out_3=64,
            cnn_kernel_size=3, 
            cnn_dropout=0.2,
            cnn_pool_1=[24,7],
            cnn_pool_2=[12,5],
            cnn_pool_3=[6,3],  
            cnn_fc_out_h=None,     
              
            td='self_att',
            td_sa_d_model=64,
            td_sa_nhead=1,
            td_sa_pos_enc=None,
            td_sa_num_layers=2,
            td_sa_h=64,
            td_sa_dropout=0.1,
            td_lstm_h=128,
            td_lstm_num_layers=1,
            td_lstm_dropout=0,
            td_lstm_bidirectional=True,
            
            td_2='skip',
            td_2_sa_d_model=None,
            td_2_sa_nhead=None,
            td_2_sa_pos_enc=None,
            td_2_sa_num_layers=None,
            td_2_sa_h=None,
            td_2_sa_dropout=None,
            td_2_lstm_h=None,
            td_2_lstm_num_layers=None,
            td_2_lstm_dropout=None,
            td_2_lstm_bidirectional=None,               

            pool='att',
            pool_att_h=128,
            pool_att_dropout=0.1,
            
            ):
        super().__init__()
    
        self.name = 'NISQA_DIM'
        
        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1, 
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size, 
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,   
            )        
        
        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
            )

        self.time_dependency_2 = TimeDependency(
            input_size=self.time_dependency.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
            )     

        pool = Pooling(
            self.time_dependency.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
            )         
        
        self.pool_layers = self._get_clones(pool, 5)
        
    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])        

    def forward(self, x, n_wins):
        
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        x, n_wins = self.time_dependency_2(x, n_wins)
        out = [mod(x, n_wins) for mod in self.pool_layers]
        out = torch.cat(out, dim=1)

        return out


    
class NISQA_DE(nn.Module):
    '''
    NISQA: The main speech quality model for double-ended prediction.
    The module loads the submodules for framewise modelling (e.g. CNN), 
    time-dependency modelling (e.g. Self-Attention or LSTM), time-alignment, 
    feature fusion and pooling (e.g. max-pooling or attention-pooling)                                                  
    '''         
    def __init__(self,
            ms_seg_length=15,
            ms_n_mels=48,
            
            cnn_model='adapt',
            cnn_c_out_1=16, 
            cnn_c_out_2=32,
            cnn_c_out_3=64,
            cnn_kernel_size=3, 
            cnn_dropout=0.2,
            cnn_pool_1=[24,7],
            cnn_pool_2=[12,5],
            cnn_pool_3=[6,3],  
            cnn_fc_out_h=None,     
              
            td='self_att',
            td_sa_d_model=64,
            td_sa_nhead=1,
            td_sa_pos_enc=None,
            td_sa_num_layers=2,
            td_sa_h=64,
            td_sa_dropout=0.1,
            td_lstm_h=128,
            td_lstm_num_layers=1,
            td_lstm_dropout=0,
            td_lstm_bidirectional=True,
            
            td_2='skip',
            td_2_sa_d_model=None,
            td_2_sa_nhead=None,
            td_2_sa_pos_enc=None,
            td_2_sa_num_layers=None,
            td_2_sa_h=None,
            td_2_sa_dropout=None,
            td_2_lstm_h=None,
            td_2_lstm_num_layers=None,
            td_2_lstm_dropout=None,
            td_2_lstm_bidirectional=None,               
            
            pool='att',
            pool_att_h=128,
            pool_att_dropout=0.1,
            
            de_align = 'dot',
            de_align_apply = 'hard',
            de_fuse_dim = None,
            de_fuse = True,         
               
            ):
        
        super().__init__()
    
        self.name = 'NISQA_DE'
        
        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1, 
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size, 
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,   
            )        
        
        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
            )
        
        self.align = Alignment(
             de_align, 
             de_align_apply,
             q_dim=self.time_dependency.fan_out,
             y_dim=self.time_dependency.fan_out,
            )
                
        self.fuse = Fusion(
            in_feat=self.time_dependency.fan_out,
            fuse_dim=de_fuse_dim, 
            fuse=de_fuse,
            )             
        
        self.time_dependency_2 = TimeDependency(
            input_size=self.fuse.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
            )                
        
        self.pool = Pooling(
            self.time_dependency_2.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
            )                 
        
    def _split_ref_deg(self, x, n_wins):
        (x, y) = torch.chunk(x, 2, dim=2)        
        (n_wins_x, n_wins_y) = torch.chunk(n_wins, 2, dim=1)
        n_wins_x = n_wins_x.view(-1)
        n_wins_y = n_wins_y.view(-1)       
        return x, y, n_wins_x, n_wins_y 

    def forward(self, x, n_wins):
        
        x, y, n_wins_x, n_wins_y = self._split_ref_deg(x, n_wins)
        
        x = self.cnn(x, n_wins_x)
        y = self.cnn(y, n_wins_y)
        
        x, n_wins_x = self.time_dependency(x, n_wins_x)
        y, n_wins_y = self.time_dependency(y, n_wins_y)
        
        y = self.align(x, y, n_wins_y)
        
        x = self.fuse(x, y)
        
        x, n_wins_x = self.time_dependency_2(x, n_wins_x)
        
        x = self.pool(x, n_wins_x)
        
        return x
    
    
#%% Framewise
class Framewise(nn.Module):
    '''
    Framewise: The main framewise module. It loads either a CNN or feed-forward
    network for framewise modelling of the Mel-spec segments. This module can
    also be skipped by loading the SkipCNN module. There are two CNN modules
    available. AdaptCNN with adaptive maxpooling and the StandardCNN module.
    However, they could also be replaced with new modules, such as PyTorch 
    implementations of ResNet or Alexnet.                                                 
    '''         
    def __init__(
        self, 
        cnn_model,
        ms_seg_length=15,
        ms_n_mels=48,
        c_out_1=16, 
        c_out_2=32,
        c_out_3=64,
        kernel_size=3, 
        dropout=0.2,
        pool_1=[24,7],
        pool_2=[12,5],
        pool_3=[6,3],
        fc_out_h=None,        
        ):
        super().__init__()
        
        if cnn_model=='adapt':
            self.model = AdaptCNN(
                input_channels=1,
                c_out_1=c_out_1, 
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size, 
                dropout=dropout,
                pool_1=pool_1,
                pool_2=pool_2,
                pool_3=pool_3,
                fc_out_h=fc_out_h,
                )
        elif cnn_model=='standard':
            assert ms_n_mels == 48, "ms_n_mels is {} and should be 48, use adaptive model or change ms_n_mels".format(ms_n_mels)
            assert ms_seg_length == 15, "ms_seg_len is {} should be 15, use adaptive model or change ms_seg_len".format(ms_seg_length)
            assert ((kernel_size == 3) or (kernel_size == (3,3))), "cnn_kernel_size is {} should be 3, use adaptive model or change cnn_kernel_size".format(kernel_size)
            self.model = StandardCNN(
                input_channels=1,
                c_out_1=c_out_1, 
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size, 
                dropout=dropout,
                fc_out_h=fc_out_h,
                )       
        elif cnn_model=='dff':
            self.model = DFF(ms_seg_length, ms_n_mels, dropout, fc_out_h)            
        elif (cnn_model is None) or (cnn_model=='skip'):
            self.model = SkipCNN(ms_seg_length, ms_n_mels, fc_out_h)
        else:
            raise NotImplementedError('Framwise model not available')                        
        
    def forward(self, x, n_wins):
        (bs, length, channels, height, width) = x.shape
        x_packed = pack_padded_sequence(
                x,
                n_wins.cpu(),
                batch_first=True,
                enforce_sorted=False
                )     
        x = self.model(x_packed.data) 
        x = x_packed._replace(data=x)                
        x, _ = pad_packed_sequence(
            x, 
            batch_first=True, 
            padding_value=0.0,
            total_length=n_wins.max())
        return x    

class SkipCNN(nn.Module):
    '''
    SkipCNN: Can be used to skip the framewise modelling stage and directly
    apply an LSTM or Self-Attention network.                                              
    '''        
    def __init__(
        self, 
        cnn_seg_length,
        ms_n_mels,
        fc_out_h
        ):
        super().__init__()

        self.name = 'SkipCNN'
        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length*ms_n_mels
        self.bn = nn.BatchNorm2d( 1 )
        
        if fc_out_h is not None:
            self.linear = nn.Linear(self.fan_in, fc_out_h)
            self.fan_out = fc_out_h
        else:
            self.linear = nn.Identity()
            self.fan_out = self.fan_in
        
    def forward(self, x):
        x = self.bn(x)
        x = x.view(-1, self.fan_in)
        x = self.linear(x)
        return x    
    
class DFF(nn.Module):
    '''
    DFF: Deep Feed-Forward network that was used as baseline framwise model as
    comparision to the CNN.
    '''        
    def __init__(self, 
                 cnn_seg_length,
                 ms_n_mels,
                 dropout,
                 fc_out_h=4096,
                 ):
        super().__init__()
        self.name = 'DFF'

        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h
        self.fan_out = fc_out_h
        
        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length*ms_n_mels
        
        self.lin1 = nn.Linear(self.fan_in, self.fc_out_h)
        self.lin2 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin3 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin4 = nn.Linear(self.fc_out_h, self.fc_out_h)
        
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d( self.fc_out_h )
        self.bn3 = nn.BatchNorm1d( self.fc_out_h )
        self.bn4 = nn.BatchNorm1d( self.fc_out_h )
        self.bn5 = nn.BatchNorm1d( self.fc_out_h )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        x = self.bn1(x)
        x = x.view(-1, self.fan_in)
        
        x = F.relu( self.bn2( self.lin1(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn3( self.lin2(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.lin3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn5( self.lin4(x) ) )
                
        return x
        
    
class AdaptCNN(nn.Module):
    '''
    AdaptCNN: CNN with adaptive maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module is more flexible
    than the StandardCNN that requires a fixed input dimension of 48x15.
    '''            
    def __init__(self, 
                 input_channels,
                 c_out_1, 
                 c_out_2,
                 c_out_3,
                 kernel_size, 
                 dropout,
                 pool_1,
                 pool_2,
                 pool_3,
                 fc_out_h=20,
                 ):
        super().__init__()
        self.name = 'CNN_adapt'

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.pool_3 = pool_3
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.dropout = nn.Dropout2d(p=self.dropout_rate)
        
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
            
        # Set kernel width of last conv layer to last pool width to 
        # downsample width to one.
        self.kernel_size_last = (self.kernel_size[0], self.pool_3[1])
            
        # kernel_size[1]=1 can be used for seg_length=1 -> corresponds to 
        # 1D conv layer, no width padding needed.
        if self.kernel_size[1] == 1:
            self.cnn_pad = (1,0)
        else:
            self.cnn_pad = (1,1)   
            
        self.conv1 = nn.Conv2d(
                self.input_channels,
                self.c_out_1,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn1 = nn.BatchNorm2d( self.conv1.out_channels )

        self.conv2 = nn.Conv2d(
                self.conv1.out_channels,
                self.c_out_2,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn2 = nn.BatchNorm2d( self.conv2.out_channels )

        self.conv3 = nn.Conv2d(
                self.conv2.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn3 = nn.BatchNorm2d( self.conv3.out_channels )

        self.conv4 = nn.Conv2d(
                self.conv3.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn4 = nn.BatchNorm2d( self.conv4.out_channels )

        self.conv5 = nn.Conv2d(
                self.conv4.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn5 = nn.BatchNorm2d( self.conv5.out_channels )

        self.conv6 = nn.Conv2d(
                self.conv5.out_channels,
                self.c_out_3,
                self.kernel_size_last,
                padding = (1,0))

        self.bn6 = nn.BatchNorm2d( self.conv6.out_channels )
        
        if self.fc_out_h:
            self.fc = nn.Linear(self.conv6.out_channels * self.pool_3[0], self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.pool_3[0])

    def forward(self, x):
        
        x = F.relu( self.bn1( self.conv1(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_1))

        x = F.relu( self.bn2( self.conv2(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_2))
        
        x = self.dropout(x)
        x = F.relu( self.bn3( self.conv3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.conv4(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_3))

        x = self.dropout(x)
        x = F.relu( self.bn5( self.conv5(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn6( self.conv6(x) ) )
        x = x.view(-1, self.conv6.out_channels * self.pool_3[0])
        
        if self.fc_out_h:
            x = self.fc( x ) 
        return x

class StandardCNN(nn.Module):
    '''
    StandardCNN: CNN with fixed maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module requires a fixed
    input dimension of 48x15.
    '''           
    def __init__(
        self, 
        input_channels, 
        c_out_1, 
        c_out_2, 
        c_out_3, 
        kernel_size, 
        dropout, 
        fc_out_h=None
        ):
        super().__init__()

        self.name = 'CNN_standard'

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_size = 2
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.output_width = 2 # input width 15 pooled 3 times
        self.output_height = 6 # input height 48 pooled 3 times

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        self.pool_first = nn.MaxPool2d(
                self.pool_size,
                stride = self.pool_size,
                padding = (0,1))

        self.pool = nn.MaxPool2d(
                self.pool_size,
                stride = self.pool_size,
                padding = 0)

        self.conv1 = nn.Conv2d(
                self.input_channels,
                self.c_out_1,
                self.kernel_size,
                padding = 1)

        self.bn1 = nn.BatchNorm2d( self.conv1.out_channels )

        self.conv2 = nn.Conv2d(
                self.conv1.out_channels,
                self.c_out_2,
                self.kernel_size,
                padding = 1)

        self.bn2 = nn.BatchNorm2d( self.conv2.out_channels )


        self.conv3 = nn.Conv2d(
                self.conv2.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn3 = nn.BatchNorm2d( self.conv3.out_channels )

        self.conv4 = nn.Conv2d(
                self.conv3.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn4 = nn.BatchNorm2d( self.conv4.out_channels )

        self.conv5 = nn.Conv2d(
                self.conv4.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn5 = nn.BatchNorm2d( self.conv5.out_channels )

        self.conv6 = nn.Conv2d(
                self.conv5.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)
        
        self.bn6 = nn.BatchNorm2d( self.conv6.out_channels )

        if self.fc_out_h:
            self.fc_out = nn.Linear(self.conv6.out_channels * self.output_height * self.output_width, self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.output_height * self.output_width)

    def forward(self, x):
        
        x = F.relu( self.bn1( self.conv1(x) ) )
        x = self.pool_first( x )

        x = F.relu( self.bn2( self.conv2(x) ) )
        x = self.pool( x )

        x = self.dropout(x)
        x = F.relu( self.bn3( self.conv3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.conv4(x) ) )
        x = self.pool( x )
        
        x = self.dropout(x)
        x = F.relu( self.bn5( self.conv5(x) ) )
        x = self.dropout(x)
        
        x = F.relu( self.bn6( self.conv6(x) ) )
                
        x = x.view(-1, self.conv6.out_channels * self.output_height * self.output_width) 
                            
        if self.fc_out_h:
            x = self.fc_out( x )

        return x
    
#%% Time Dependency
class TimeDependency(nn.Module):
    '''
    TimeDependency: The main time-dependency module. It loads either an LSTM 
    or self-attention network for time-dependency modelling of the framewise 
    features. This module can also be skipped.                                              
    '''          
    def __init__(self,
                 input_size,
                 td='self_att',
                 sa_d_model=512,
                 sa_nhead=8,
                 sa_pos_enc=None,
                 sa_num_layers=6,
                 sa_h=2048,
                 sa_dropout=0.1,
                 lstm_h=128,
                 lstm_num_layers=1,
                 lstm_dropout=0,
                 lstm_bidirectional=True,
                 ):
        super().__init__()
        
        if td=='self_att':
            self.model = SelfAttention(
                input_size=input_size,
                d_model=sa_d_model,
                nhead=sa_nhead,
                pos_enc=sa_pos_enc,
                num_layers=sa_num_layers,
                sa_h=sa_h,
                dropout=sa_dropout,
                activation="relu"
                )
            self.fan_out = sa_d_model
            
        elif td=='lstm':
            self.model = LSTM(
                 input_size,
                 lstm_h=lstm_h,
                 num_layers=lstm_num_layers,
                 dropout=lstm_dropout,
                 bidirectional=lstm_bidirectional,
                 )  
            self.fan_out = self.model.fan_out
            
        elif (td is None) or (td=='skip'):
            self.model = self._skip
            self.fan_out = input_size
        else:
            raise NotImplementedError('Time dependency option not available')    
            
    def _skip(self, x, n_wins):
        return x, n_wins

    def forward(self, x, n_wins):
        x, n_wins = self.model(x, n_wins)
        return x, n_wins
                
class LSTM(nn.Module):
    '''
    LSTM: The main LSTM module that can be used as a time-dependency model.                                            
    '''           
    def __init__(self,
                 input_size,
                 lstm_h=128,
                 num_layers=1,
                 dropout=0.1,
                 bidirectional=True
                 ):
        super().__init__()
        
        self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = lstm_h,
                num_layers = num_layers,
                dropout = dropout,
                batch_first = True,
                bidirectional = bidirectional
                )      
            
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1                 
        self.fan_out = num_directions*lstm_h

    def forward(self, x, n_wins):
        
        x = pack_padded_sequence(
                x,
                n_wins.cpu(),
                batch_first=True,
                enforce_sorted=False
                )             
        
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        
        x, _ = pad_packed_sequence(
            x, 
            batch_first=True, 
            padding_value=0.0,
            total_length=n_wins.max())          
  
        return x, n_wins

class SelfAttention(nn.Module):
    '''
    SelfAttention: The main SelfAttention module that can be used as a
    time-dependency model.                                            
    '''         
    def __init__(self,
                 input_size,
                 d_model=512,
                 nhead=8,
                 pool_size=3,
                 pos_enc=None,
                 num_layers=6,
                 sa_h=2048,
                 dropout=0.1,
                 activation="relu"
                 ):
        super().__init__()
        
        encoder_layer = SelfAttentionLayer(d_model, nhead, pool_size, sa_h, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.linear = nn.Linear(input_size, d_model)
        
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead      
        
        if pos_enc:
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoder = nn.Identity()
            
        self._reset_parameters()
        
    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, n_wins=None):            
        src = self.linear(src)
        output = src.transpose(1,0)
        output = self.norm1(output)
        output = self.pos_encoder(output)
        
        for mod in self.layers:
            output, n_wins = mod(output, n_wins=n_wins)
        return output.transpose(1,0), n_wins

class SelfAttentionLayer(nn.Module):
    '''
    SelfAttentionLayer: The SelfAttentionLayer that is used by the
    SelfAttention module.                                            
    '''          
    def __init__(self, d_model, nhead, pool_size=1, sa_h=2048, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, sa_h)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(sa_h, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = self._get_activation_fn(activation)
        
    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu                
        
    def forward(self, src, n_wins=None):
        
        if n_wins is not None:
            mask = ~((torch.arange(src.shape[0])[None, :]).to(src.device) < n_wins[:, None].to(torch.long).to(src.device))
        else:
            mask = None
        
        src2 = self.self_attn(src, src, src, key_padding_mask=mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src, n_wins
    
class PositionalEncoding(nn.Module):
    '''
    PositionalEncoding: PositionalEncoding taken from the PyTorch Transformer
    tutorial. Can be applied to the SelfAttention module. However, it did not 
    improve the results in previous experiments.                          
    '''       
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)    

#%% Pooling
class Pooling(nn.Module):
    '''
    Pooling: Main Pooling module. It can load either attention-pooling, average
    pooling, maxpooling, or last-step pooling. In case of bidirectional LSTMs
    last-step-bi pooling should be used instead of last-step pooling.
    '''      
    def __init__(self,
                 d_input,
                 output_size=1,
                 pool='att',
                 att_h=None,
                 att_dropout=0,
                 ):
        super().__init__()
        
        if pool=='att':
            if att_h is None:
                self.model = PoolAtt(d_input, output_size)
            else:
                self.model = PoolAttFF(d_input, output_size, h=att_h, dropout=att_dropout)
        elif pool=='last_step_bi':
            self.model = PoolLastStepBi(d_input, output_size)      
        elif pool=='last_step':
            self.model = PoolLastStep(d_input, output_size)                  
        elif pool=='max':
            self.model = PoolMax(d_input, output_size)  
        elif pool=='avg':
            self.model = PoolAvg(d_input, output_size)              
        else:
            raise NotImplementedError('Pool option not available')                     

    def forward(self, x, n_wins):
        return self.model(x, n_wins)
    
class PoolLastStepBi(nn.Module):
    '''
    PoolLastStepBi: last step pooling for the case of bidirectional LSTM
    '''       
    def __init__(self, input_size, output_size):
        super().__init__() 
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x, n_wins=None):    
        x = x.view(x.shape[0], n_wins.max(), 2, x.shape[-1]//2)
        x = torch.cat(
            (x[torch.arange(x.shape[0]), n_wins.type(torch.long)-1, 0, :],
            x[:,0,1,:]),
            dim=1
            )
        x = self.linear(x)
        return x    
    
class PoolLastStep(nn.Module):
    '''
    PoolLastStep: last step pooling can be applied to any one-directional 
    sequence.
    '''      
    def __init__(self, input_size, output_size):
        super().__init__() 
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x, n_wins=None):    
        x = x[torch.arange(x.shape[0]), n_wins.type(torch.long)-1]
        x = self.linear(x)
        return x        

class PoolAtt(torch.nn.Module):
    '''
    PoolAtt: Attention-Pooling module.
    '''          
    def __init__(self, d_input, output_size):
        super().__init__()
        
        self.linear1 = nn.Linear(d_input, 1)
        self.linear2 = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):
                
        att = self.linear1(x)
        
        att = att.transpose(2,1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")          
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        
        x = self.linear2(x)
            
        return x
    
class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, d_input, output_size, h, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_input, h)
        self.linear2 = nn.Linear(h, 1)
        
        self.linear3 = nn.Linear(d_input, output_size)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, n_wins):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")          
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        
        x = self.linear3(x)
        
        return x    

class PoolAvg(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''          
    def __init__(self, d_input, output_size):
        super().__init__()
        
        self.linear = nn.Linear(d_input, output_size)
        
    def forward(self, x, n_wins):
                
        mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, 0)

        x = torch.div(x.sum(1), n_wins.unsqueeze(1))   
            
        x = self.linear(x)
        
        return x
    
class PoolMax(torch.nn.Module):
    '''
    PoolMax: Max-pooling that consideres masked time-steps.
    '''        
    def __init__(self, d_input, output_size):
        super().__init__()
        
        self.linear = nn.Linear(d_input, output_size)
        
    def forward(self, x, n_wins):
                
        mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, float("-Inf"))

        x = x.max(1)[0]
        
        x = self.linear(x)
            
        return x    
    
#%% Alignment
class Alignment(torch.nn.Module):
    '''
    Alignment: Alignment module for the double-ended NISQA_DE model. It 
    supports five different alignment mechanisms.
    '''       
    def __init__(self,
                 att_method,
                 apply_att_method,
                 q_dim=None,
                 y_dim=None,
                 ):
        super().__init__()
    
        # Attention method --------------------------------------------------------
        if att_method=='bahd':
            self.att = AttBahdanau(
                    q_dim=q_dim,
                    y_dim=y_dim) 
            
        elif att_method=='luong':
            self.att = AttLuong(
                    q_dim=q_dim, 
                    y_dim=y_dim) 
            
        elif att_method=='dot':
            self.att = AttDot()
            
        elif att_method=='cosine':
            self.att = AttCosine()            

        elif att_method=='distance':
            self.att = AttDistance()
            
        elif (att_method=='none') or (att_method is None):
            self.att = None
        else:
            raise NotImplementedError    
        
        # Apply method ----------------------------------------------------------
        if apply_att_method=='soft':
            self.apply_att = ApplySoftAttention() 
        elif apply_att_method=='hard':
            self.apply_att = ApplyHardAttention() 
        else:
            raise NotImplementedError            
            
    def _mask_attention(self, att, y, n_wins):       
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = mask.unsqueeze(1).expand_as(att)
        att[~mask] = float("-Inf")    
        
    def forward(self, query, y, n_wins_y):        
        if self.att is not None:
            att_score, sim = self.att(query, y)     
            self._mask_attention(att_score, y, n_wins_y)
            att_score = F.softmax(att_score, dim=2)
            y = self.apply_att(y, att_score) 
        return y        

class AttDot(torch.nn.Module):
    '''
    AttDot: Dot attention that can be used by the Alignment module.
    '''       
    def __init__(self):
        super().__init__()
    def forward(self, query, y):
        att = torch.bmm(query, y.transpose(2,1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim
    
class AttCosine(torch.nn.Module):
    '''
    AttCosine: Cosine attention that can be used by the Alignment module.
    '''          
    def __init__(self):
        super().__init__()
        self.pdist = nn.CosineSimilarity(dim=3)
    def forward(self, query, y):
        att = self.pdist(query.unsqueeze(2), y.unsqueeze(1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim    
    
class AttDistance(torch.nn.Module):
    '''
    AttDistance: Distance attention that can be used by the Alignment module.
    '''        
    def __init__(self, dist_norm=1, weight_norm=1):
        super().__init__()
        self.dist_norm = dist_norm
        self.weight_norm = weight_norm
    def forward(self, query, y):
        att = (query.unsqueeze(1)-y.unsqueeze(2)).abs().pow(self.dist_norm)
        att = att.mean(dim=3).pow(self.weight_norm)
        att = - att.transpose(2,1)
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim
    
class AttBahdanau(torch.nn.Module):
    '''
    AttBahdanau: Attention according to Bahdanau that can be used by the 
    Alignment module.
    ''' 
    def __init__(self, q_dim, y_dim, att_dim=128):
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.att_dim = att_dim
        self.Wq = nn.Linear(self.q_dim, self.att_dim)
        self.Wy = nn.Linear(self.y_dim, self.att_dim)
        self.v = nn.Linear(self.att_dim, 1)
    def forward(self, query, y):
        att = torch.tanh( self.Wq(query).unsqueeze(1) + self.Wy(y).unsqueeze(2) )
        att = self.v(att).squeeze(3).transpose(2,1)
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim

class AttLuong(torch.nn.Module):
    '''
    AttLuong: Attention according to Luong that can be used by the 
    Alignment module.
    '''     
    def __init__(self, q_dim, y_dim):
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.W = nn.Linear(self.y_dim, self.q_dim)
    def forward(self, query, y):
        att = torch.bmm(query, self.W(y).transpose(2,1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim

class ApplyHardAttention(torch.nn.Module):
    '''
    ApplyHardAttention: Apply hard attention for the purpose of time-alignment.
    '''       
    def __init__(self):
        super().__init__()
    def forward(self, y, att):        
        self.idx = att.argmax(2)
        y = y[torch.arange(y.shape[0]).unsqueeze(-1), self.idx]        
        return y    
    
class ApplySoftAttention(torch.nn.Module):
    '''
    ApplySoftAttention: Apply soft attention for the purpose of time-alignment.
    '''        
    def __init__(self):
        super().__init__()
    def forward(self, y, att):        
        y = torch.bmm(att, y)       
        return y     
    
class Fusion(torch.nn.Module):
    '''
    Fusion: Used by the double-ended NISQA_DE model and used to fuse the
    degraded and reference features.
    '''      
    def __init__(self, fuse_dim=None, in_feat=None, fuse=None):
        super().__init__()
        self.fuse_dim = fuse_dim
        self.fuse = fuse

        if self.fuse=='x/y/-':
            self.fan_out = 3*in_feat        
        elif self.fuse=='+/-':
             self.fan_out = 2*in_feat                     
        elif self.fuse=='x/y':
            self.fan_out = 2*in_feat
        else:
            raise NotImplementedError         
            
        if self.fuse_dim:
            self.lin_fusion = nn.Linear(self.fan_out, self.fuse_dim)
            self.fan_out = fuse_dim       
                        
    def forward(self, x, y):
        
        if self.fuse=='x/y/-':
            x = torch.cat((x, y, x-y), 2)
        elif self.fuse=='+/-':
            x = torch.cat((x+y, x-y), 2)     
        elif self.fuse=='x/y':
            x = torch.cat((x, y), 2)   
        else:
            raise NotImplementedError           
            
        if self.fuse_dim:
            x = self.lin_fusion(x)
            
        return x
        
#%% Evaluation 
def predict_mos(model, ds, bs, dev, num_workers=0):    
    '''
    predict_mos: predicts MOS of the given dataset with given model. Used for
    NISQA and NISQA_DE model.
    '''       
    dl = DataLoader(ds,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=False,
                    num_workers=num_workers)
    model.to(dev)
    model.eval()
    with torch.no_grad():
        y_hat_list = [ [model(xb.to(dev), n_wins.to(dev)).cpu().numpy(), yb.cpu().numpy()] for xb, yb, (idx, n_wins) in dl]
    yy = np.concatenate( y_hat_list, axis=1 )
    y_hat = yy[0,:,0].reshape(-1,1)
    mos_pred = y_hat.astype(dtype=float)
    return mos_pred

def predict_dim(model, ds, bs, dev, num_workers=0):     
    '''
    predict_dim: predicts MOS and dimensions of the given dataset with given 
    model. Used for NISQA_DIM model.
    '''
    dl = DataLoader(ds,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=False,
                    num_workers=num_workers)
    model.to(dev)
    model.eval()
    with torch.no_grad():
        y_hat_list = [ [model(xb.to(dev), n_wins.to(dev)).cpu().numpy(), yb.cpu().numpy()] for xb, yb, (idx, n_wins) in dl]
    yy = np.concatenate( y_hat_list, axis=1 )
    
    y_hat = yy[0,:,:]
    
    pred = {'mos_pred' : y_hat[:,0].reshape(-1,1),
    'noi_pred' : y_hat[:,1].reshape(-1,1),
    'dis_pred' : y_hat[:,2].reshape(-1,1),
    'col_pred' : y_hat[:,3].reshape(-1,1),
    'loud_pred' : y_hat[:,4].reshape(-1,1),
    }
    return pred


#%% Dataset

class SpeechQualityDataset(Dataset):
    '''
    Dataset for Speech Quality Model.
    '''  
    def __init__(
        self,
        signals,
        seg_length=15,
        max_length=None,
        to_memory=False,
        to_memory_workers=0,
        transform=None,
        seg_hop_length=1,
        ms_n_fft = 1024,
        ms_hop_length = 80,
        ms_win_length = 170,
        ms_n_mels=32,
        ms_sr=48e3,
        ms_fmax=16e3,
        dim=False,
        ):

        self.signals = signals
        self.seg_length = seg_length
        self.seg_hop_length = seg_hop_length
        self.max_length = max_length
        self.transform = transform
        self.to_memory_workers = to_memory_workers
        self.ms_n_fft = ms_n_fft
        self.ms_hop_length = ms_hop_length
        self.ms_win_length = ms_win_length
        self.ms_n_mels = ms_n_mels
        self.ms_sr = ms_sr
        self.ms_fmax = ms_fmax
        self.dim = dim

        # if True load all specs to memory
        self.to_memory = False
        if to_memory:
            self._to_memory()
            
    def _to_memory_multi_helper(self, idx):
        return [self._load_spec(i) for i in idx]
    
    def _to_memory(self):
        if self.to_memory_workers==0:
            self.mem_list = [self._load_spec(idx) for idx in tqdm(range(len(self)))]
        else: 
            buffer_size = 128
            idx = np.arange(len(self))
            n_bufs = int(len(idx)/buffer_size) 
            idx = idx[:buffer_size*n_bufs].reshape(-1,buffer_size).tolist() + idx[buffer_size*n_bufs:].reshape(1,-1).tolist()  
            pool = multiprocessing.Pool(processes=self.to_memory_workers)
            mem_list = []
            for out in tqdm(pool.imap(self._to_memory_multi_helper, idx), total=len(idx)):
                mem_list = mem_list + out
            self.mem_list = mem_list
            pool.terminate()
            pool.join()    
        self.to_memory=True 

    def _load_spec(self, index):
        
            # Load spec    
            y = self.signals[index]
            spec = get_librosa_melspec(
                y,
                sr = self.ms_sr,
                n_fft=self.ms_n_fft,
                hop_length=self.ms_hop_length,
                win_length=self.ms_win_length,
                n_mels=self.ms_n_mels,
                fmax=self.ms_fmax,
                )       
            return spec
            
    def __getitem__(self, index):
        assert isinstance(index, int), 'index must be integer (no slice)'

        if self.to_memory:
            spec = self.mem_list[index]
        else:
            spec = self._load_spec(index)
        
        # Apply transformation if given
        if self.transform:
            spec = self.transform(spec)      
            
        # Segment specs
        if self.seg_length is not None:
            x_spec_seg, n_wins = segment_specs(
                spec,
                self.seg_length,
                self.seg_hop_length,
                self.max_length)
                        
        else:
            x_spec_seg = spec
            n_wins = spec.shape[1]
            if self.max_length is not None:
                x_padded = np.zeros((x_spec_seg.shape[0], self.max_length))
                x_padded[:,:n_wins] = x_spec_seg
                x_spec_seg = np.expand_dims(x_padded.transpose(1,0), axis=(1, 3))      
                if not torch.is_tensor(x_spec_seg):
                    x_spec_seg = torch.tensor(x_spec_seg, dtype=torch.float)                  
               

        # Get MOS (apply NaN in case of prediction only mode)
        if self.dim:

            y = np.full((5,1), np.nan).reshape(-1).astype('float32')
        else:
            y = np.full(1, np.nan).reshape(-1).astype('float32') 

        return x_spec_seg, y, (index, n_wins)

    def __len__(self):
        return len(self.signals)

#%% Spectrograms
def segment_specs(x, seg_length, seg_hop=1, max_length=None):
    '''
    Segment a spectrogram into "seg_length" wide spectrogram segments.
    Instead of using only the frequency bin of the current time step, 
    the neighboring bins are included as input to the CNN. For example 
    for a seg_length of 7, the previous 3 and the follwing 3 frequency 
    bins are included.

    A spectrogram with input size [H x W] will be segmented to:
    [W-(seg_length-1) x C x H x seg_length], where W is the width of the 
    original mel-spec (corresponding to the length of the speech signal),
    H is the height of the mel-spec (corresponding to the number of mel bands),
    C is the number of CNN input Channels (always one in our case).
    '''      
    if seg_length % 2 == 0:
        raise ValueError('seg_length must be odd! (seg_lenth={})'.format(seg_length))
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    n_wins = x.shape[1]-(seg_length-1)
    
    # broadcast magic to segment melspec
    idx1 = torch.arange(seg_length)
    idx2 = torch.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose(1,0)[idx3,:].unsqueeze(1).transpose(3,2)
        
    if seg_hop>1:
        x = x[::seg_hop,:]
        n_wins = int(np.ceil(n_wins/seg_hop))
        
    if max_length is not None:
        if max_length < n_wins:
            raise ValueError('n_wins {} > max_length {} --- {}. Increase max window length ms_max_segments!'.format(n_wins, max_length))
        x_padded = torch.zeros((max_length, x.shape[1], x.shape[2], x.shape[3]))
        x_padded[:n_wins,:] = x
        x = x_padded
                
    return x, np.array(n_wins)

def get_librosa_melspec(
    y,
    sr=48e3,
    n_fft=1024, 
    hop_length=80, 
    win_length=170,
    n_mels=32,
    fmax=16e3,
    ):
    '''
    Calculate mel-spectrograms with Librosa.
    '''    
    # Calc spec
    
    hop_length = int(sr * hop_length)
    win_length = int(sr * win_length)

    S = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=1.0,
    
        n_mels=n_mels,
        fmin=0.0,
        fmax=fmax,
        htk=False,
        norm='slaney',
        )

    spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
    return spec



