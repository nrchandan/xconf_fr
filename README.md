# Setup instructions

## Torch: http://torch.ch/docs/getting-started.html
* `./clean.sh` (in the torch folder)
* `./install-deps` (in the torch folder)
* `./install.sh` (in the torch folder)
* Add install/bin to PATH
  - `export PATH="~/XConf_Nov18/torch/install/bin:$PATH"`
* `luarocks install dpnn`
* `for NAME in dpnn nn optim optnet csvigo cutorch cunn fblualib torchx tds; do luarocks install $NAME; done`

## Conda and python packages
* `xcode-select --install`
* `brew install cmake`
* Download Miniconda from here (https://conda.io/miniconda.html)
* `conda create -n xconf python==3.5`
* IMPORTANT: `conda activate xconf`
* VERIFY: `conda env list`
* `pip install -r requirements.txt`

## Verify and run
* VIDEO CAPTURE: `python 1_capture_video.py`
* DETECT FACES: `python 2_detect_faces.py`
* TRAIN: `python 6_test_classifier.py train --training_data ./TrainingData`
* TEST: `python 6_test_classifier.py test --testing_data 0`
