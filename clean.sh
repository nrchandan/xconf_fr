#!/usr/bin/env bash

if ls miniconda.sh > /dev/null 2> /dev/null; then
	rm miniconda.sh
fi
rm -rf /usr/local/share/miniconda3
echo 'Please remove miniconda3 from PATH in your zshrc.'
rm -rf /usr/local/share/torch
echo 'Please remove torch from PATH in your zshrc.'
