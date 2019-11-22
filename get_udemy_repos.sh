#!/bin/bash

if [ -d "machine_learning_examples" ]; then
  cd machine_learning_examples
  git pull
  cd ..
else
  git clone https://github.com/lazyprogrammer/machine_learning_examples.git
fi
