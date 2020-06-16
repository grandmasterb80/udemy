#!/bin/bash

if [ -d "machine_learning_examples" ]; then
  cd machine_learning_examples
  git pull
  cd ..
else
  git clone https://github.com/lazyprogrammer/machine_learning_examples.git
fi

if [ -d "mpc-course-assignments_basis" ]; then
  cd mpc-course-assignments_basis
  git pull
  cd ..
else
  git clone https://github.com/WuStangDan/mpc-course-assignments.git  mpc-course-assignments_basis
fi

