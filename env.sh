conda create -n lut python=3.10 -y
conda activate lut
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r .\requirements_inference.txt -c torch==2.6.0+cu124