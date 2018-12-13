#!/usr/bin/env bash

set -v
if ! which cmake > /dev/null; then
  brew install cmake
fi
if ! which wget > /dev/null; then
  brew install wget
fi
if ! ls miniconda.sh > /dev/null; then
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ./miniconda.sh
fi
sh miniconda.sh -p /usr/local/share/miniconda3/ -b -f
tar -C /usr/local/share -xvf torch.tgz
./activate.sh
luarocks install dpnn
conda create -y -n xconf python==3.5
conda activate xconf
pip install -r requirements.txt
conda deactivate
set +v
