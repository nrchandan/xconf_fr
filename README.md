# Setup instructions *macOS-only*

## Assumptions
* you have `brew` set up
* you have `zsh` set up
* you have xcode command line tools; if not run `xcode-select --install`

## Installation
* Run `./install.sh`
* Includes setting up miniconda and torch binaries

## Uninstall
* Run `./clean.sh`
* Remove `conda` and `torch` references from `~/.zshrc` manually

## Verify and run
* Enter the virtual env: `conda activate xconf`
* VIDEO CAPTURE: `python 1_capture_video.py`
* DETECT FACES: `python 2_detect_faces.py`
* TRAIN
  - Create a folder in TrainingData with your name
  - Place your selfies in the folder
  - Run `python 5_train_classifier.py`
* TEST
  - Coming soon
  - `python 6_test_classifier.py test --testing_data 0`
