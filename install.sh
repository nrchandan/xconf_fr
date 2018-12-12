#!/usr/bin/env bash

set -v

if ! which zsh > /dev/null; then
	brew install zsh
fi
if ! ls miniconda.sh > /dev/null; then
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ./miniconda.sh
fi
sh miniconda.sh -p /usr/local/share/miniconda3/ -b -f
echo 'export PATH="/usr/local/share/miniconda3/bin:$PATH"' >> ~/.zshrc
echo '. /usr/local/share/miniconda3/etc/profile.d/conda.sh' >> ~/.zshrc
. ~/.zshrc
conda create -y -n xconf python==3.5
conda activate xconf
pip install -r requirements.txt
conda deactivate
tar -C /usr/local/share -xvf torch.tgz
echo 'export PATH="/usr/local/share/torch/install/bin:$PATH"' >> ~/.zshrc
echo '. /usr/local/share/torch/install/bin/torch-activate' >> ~/.zshrc
. ~/.zshrc
luarocks install dpnn
set +v
