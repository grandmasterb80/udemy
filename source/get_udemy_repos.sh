#!/bin/bash

if [ -d "machine_learning_examples" ]; then
  chmod a-w machine_learning_examples
  cd machine_learning_examples
  git pull
  cd ..
  chmod a-w machine_learning_examples
else
  git clone https://github.com/lazyprogrammer/machine_learning_examples.git
  chmod a-w machine_learning_examples
fi

if [ -d "mpc-course-assignments_basis" ]; then
  chmod a+w mpc-course-assignments_basis
  cd mpc-course-assignments_basis
  git pull
  cd ..
  chmod a-w mpc-course-assignments_basis
else
  git clone https://github.com/WuStangDan/mpc-course-assignments.git  mpc-course-assignments_basis
  chmod a-w mpc-course-assignments_basis
fi

