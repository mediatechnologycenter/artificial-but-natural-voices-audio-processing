from typing import List, Union
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics import ScaleInvariantSignalDistortionRatio, ScaleInvariantSignalNoiseRatio
import torch
import torchaudio.functional as F
import naturstimmen.utils.nisqa.NISQA as NL

import numpy as np
import os

def pesq(preds: torch.Tensor, target: torch.Tensor, preds_sr:int, target_sr:int = None):
    """
    More info: https://torchmetrics.readthedocs.io/en/stable/audio/perceptual_evaluation_speech_quality.html
    """
    PESQ_SR = 16000 # Must be 16k or 8k

    if not target_sr:
        target_sr = preds_sr
    
    # Resampling to PESQ_SR
    if not target_sr == PESQ_SR:
        target = F.resample(target, target_sr, PESQ_SR)
    if not preds_sr == PESQ_SR:
        preds = F.resample(preds, preds_sr, PESQ_SR)

    wb_pesq = PerceptualEvaluationSpeechQuality(PESQ_SR, 'wb')
    return wb_pesq(preds, target)

def si_sdr(preds: torch.Tensor, target: torch.Tensor):
    """
    The SI-SDR value is in general considered an overall measure of how good a source sound.
    More info: https://torchmetrics.readthedocs.io/en/stable/audio/scale_invariant_signal_distortion_ratio.html
    """
    return ScaleInvariantSignalDistortionRatio()(preds, target)

def si_snr(preds: torch.Tensor, target: torch.Tensor):
    """
    Calculates Scale-invariant signal-to-noise ratio (SI-SNR) metric for evaluating quality of audio.
    More info: https://torchmetrics.readthedocs.io/en/stable/audio/scale_invariant_signal_noise_ratio.html
    """
    return ScaleInvariantSignalNoiseRatio()(preds, target)

def stoi(preds: torch.Tensor, target: torch.Tensor, sampling_rate: int):
    """
    Intelligibility measure which is highly correlated with the intelligibility of degraded speech signals
    More info: https://torchmetrics.readthedocs.io/en/stable/audio/short_time_objective_intelligibility.html
    """
    return ShortTimeObjectiveIntelligibility(sampling_rate)(preds, target)

def nisqa(
    signals: List[np.ndarray],
    sr: int,
    batch_size: int = 1, 
    device: str = "cuda",
    num_workers: int = 0,
    pretrained_model: str = "nisqa",
):
    """
    https://github.com/gabrielmittag/NISQA

    pretrained_models:
    - nisqa: speech quality prediction. Besides overall speech quality, provides predictions for the quality dimensions Noisiness, Coloration, Discontinuity, and Loudness.
    - nisqa_tts:  can be used to estimate the Naturalness of synthetic speech
    - nisqa_mos_only: Overall Quality only (for finetuning/transfer learning)
    """
    AVAILABLE_MODELS = ["nisqa", "nisqa_tts", "nisqa_mos_only"]
    assert pretrained_model in AVAILABLE_MODELS, f"The pretrained_model is not in {AVAILABLE_MODELS=}"

    args = {
        "batch_size": batch_size,
        "device": device,
        "num_workers": num_workers,
        "pretrained_model": os.path.join(os.path.dirname(os.path.abspath(__file__)),"nisqa",f"{pretrained_model}.tar"),
        "sr": sr 
    }

    #Librosa input must be an np.ndarray
    assert type(signals) == list
    assert type(signals[0]) == np.ndarray
    
    nisqa = Nisqa(signals, args)
    return nisqa.predict(), nisqa

def mcd():
    """
    Mel cepstral distortion (MCD) is a measure of how different two sequences of mel cepstra are.
    https://github.com/MattShannon/mcd
    """
    pass



class Nisqa(object):
    '''
    NISQA: Main class that loads the model and the datasets. Contains
    the training loop, prediction, and evaluation function.      
    Adapted from https://github.com/gabrielmittag/NISQA/blob/master/nisqa/NISQA_model.py                                         
    '''      
    def __init__(self, signals, args):
        
        self.signals: torch.Tensor = signals
        self.args = args
        
            
        self.runinfos = {}       
        self.dev = args["device"]
        self._loadModel()
        
        self.dataset = NL.SpeechQualityDataset(
            signals=signals,
            seg_length=self.args['ms_seg_length'],
            max_length=self.args['ms_max_segments'],
            to_memory=None,
            to_memory_workers=None,
            transform=None,
            seg_hop_length=self.args['ms_seg_hop_length'],
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels=self.args['ms_n_mels'],
            ms_sr=self.args['sr'],
            ms_fmax=self.args['ms_fmax'],
            dim=self.args['dim'],
        )
            
    def predict(self):
        # print('---> Predicting ...')        
        
        if self.args['dim']==True:
            pred = NL.predict_dim(
                model = self.model, 
                ds = self.dataset, 
                bs = self.args['batch_size'],
                dev = self.dev,
                num_workers=self.args['num_workers'])
        else:
            pred = NL.predict_mos(
                self.model, 
                self.dataset, 
                self.args['batch_size'],
                self.dev,
                num_workers=self.args['num_workers'])                 
                    
        return pred
  
                
    def _loadModel(self):    
        '''
        Loads the Pytorch models with given input arguments.
        '''   
        # if True overwrite input arguments from pretrained model
        if self.args['pretrained_model']:
            if os.path.isabs(self.args['pretrained_model']):
                model_path = os.path.join(self.args['pretrained_model'])
            else:
                model_path = os.path.join(os.getcwd(), self.args['pretrained_model'])
            checkpoint = torch.load(model_path, map_location=self.dev)
            
            # update checkpoint arguments with new arguments
            checkpoint['args'].update(self.args)
            self.args = checkpoint['args']
            
        if self.args['model']=='NISQA_DIM':
            self.args['dim'] = True
            self.args['csv_mos_train'] = None # column names hardcoded for dim models
            self.args['csv_mos_val'] = None  
        else:
            self.args['dim'] = False
            
        if self.args['model']=='NISQA_DE':
            self.args['double_ended'] = True
        else:
            self.args['double_ended'] = False     
            self.args['csv_ref'] = None

        # Load Model
        self.model_args = {
            
            'ms_seg_length': self.args['ms_seg_length'],
            'ms_n_mels': self.args['ms_n_mels'],
            
            'cnn_model': self.args['cnn_model'],
            'cnn_c_out_1': self.args['cnn_c_out_1'],
            'cnn_c_out_2': self.args['cnn_c_out_2'],
            'cnn_c_out_3': self.args['cnn_c_out_3'],
            'cnn_kernel_size': self.args['cnn_kernel_size'],
            'cnn_dropout': self.args['cnn_dropout'],
            'cnn_pool_1': self.args['cnn_pool_1'],
            'cnn_pool_2': self.args['cnn_pool_2'],
            'cnn_pool_3': self.args['cnn_pool_3'],
            'cnn_fc_out_h': self.args['cnn_fc_out_h'],
            
            'td': self.args['td'],
            'td_sa_d_model': self.args['td_sa_d_model'],
            'td_sa_nhead': self.args['td_sa_nhead'],
            'td_sa_pos_enc': self.args['td_sa_pos_enc'],
            'td_sa_num_layers': self.args['td_sa_num_layers'],
            'td_sa_h': self.args['td_sa_h'],
            'td_sa_dropout': self.args['td_sa_dropout'],
            'td_lstm_h': self.args['td_lstm_h'],
            'td_lstm_num_layers': self.args['td_lstm_num_layers'],
            'td_lstm_dropout': self.args['td_lstm_dropout'],
            'td_lstm_bidirectional': self.args['td_lstm_bidirectional'],
            
            'td_2': self.args['td_2'],
            'td_2_sa_d_model': self.args['td_2_sa_d_model'],
            'td_2_sa_nhead': self.args['td_2_sa_nhead'],
            'td_2_sa_pos_enc': self.args['td_2_sa_pos_enc'],
            'td_2_sa_num_layers': self.args['td_2_sa_num_layers'],
            'td_2_sa_h': self.args['td_2_sa_h'],
            'td_2_sa_dropout': self.args['td_2_sa_dropout'],
            'td_2_lstm_h': self.args['td_2_lstm_h'],
            'td_2_lstm_num_layers': self.args['td_2_lstm_num_layers'],
            'td_2_lstm_dropout': self.args['td_2_lstm_dropout'],
            'td_2_lstm_bidirectional': self.args['td_2_lstm_bidirectional'],                
            
            'pool': self.args['pool'],
            'pool_att_h': self.args['pool_att_h'],
            'pool_att_dropout': self.args['pool_att_dropout'],
            }
            
        if self.args['double_ended']:
            self.model_args.update({
                'de_align': self.args['de_align'],
                'de_align_apply': self.args['de_align_apply'],
                'de_fuse_dim': self.args['de_fuse_dim'],
                'de_fuse': self.args['de_fuse'],        
                })
                      
        # print('Model architecture: ' + self.args['model'])
        if self.args['model']=='NISQA':
            self.model = NL.NISQA(**self.model_args)     
        elif self.args['model']=='NISQA_DIM':
            self.model = NL.NISQA_DIM(**self.model_args)     
        elif self.args['model']=='NISQA_DE':
            self.model = NL.NISQA_DE(**self.model_args)     
        else:
            raise NotImplementedError('Model not available')                        
        
        # Load weights if pretrained model is used ------------------------------------
        if self.args['pretrained_model']:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            # print('Loaded pretrained model from ' + self.args['pretrained_model'])
            if missing_keys:
                print('missing_keys:')
                print(missing_keys)
            if unexpected_keys:
                print('unexpected_keys:')
                print(unexpected_keys)        
            
    def _getDevice(self):
        '''
        Train on GPU if available.
        '''         
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")
    
        if "tr_device" in self.args:
            if self.args['tr_device']=='cpu':
                self.dev = torch.device("cpu")
            elif self.args['tr_device']=='cuda':
                self.dev = torch.device("cuda")
        print('Device: {}'.format(self.dev))
        
        if "tr_parallel" in self.args:
            if (self.dev==torch.device("cpu")) and self.args['tr_parallel']==True:
                self.args['tr_parallel']==False 
                print('Using CPU -> tr_parallel set to False')



if __name__ == "__main__":
    from naturstimmen.utils.audio import load_audio
    import glob
    paths = glob.glob("/media/data-storage/ThorstenVoice-Dataset_2022.10/wavs/*.wav")
    MAX_N = 5
    
    # Librosa works only with numpy arrays
    audios = [load_audio(wav)[0].squeeze(0).numpy() for wav in paths[:MAX_N]]
    _, sr = load_audio(paths[0])

    AVAILABLE_MODELS = ["nisqa", "nisqa_tts", "nisqa_mos_only"]
    for model in AVAILABLE_MODELS:
        preds = nisqa(signals=audios, sr=sr, pretrained_model=model)
        print(preds)
    