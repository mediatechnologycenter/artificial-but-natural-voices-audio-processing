eval "$(conda shell.bash hook)"

INPUTDATAPATH=$1
DATASETNAME=$2
OUTPUTPATH=$3

conda activate demucs
python preprocess.py $INPUTDATAPATH $DATASETNAME $OUTPUTPATH 'split_audio'
python preprocess.py $INPUTDATAPATH $DATASETNAME $OUTPUTPATH 'track_separation'
conda deactivate

conda activate denoiser
python preprocess.py $INPUTDATAPATH $DATASETNAME $OUTPUTPATH 'denoising'
conda deactivate

conda activate demucs
python preprocess.py $INPUTDATAPATH $DATASETNAME $OUTPUTPATH 'diarization'
conda deactivate

conda activate tts
python preprocess.py $INPUTDATAPATH $DATASETNAME $OUTPUTPATH 'speaker_embedding'
conda deactivate
