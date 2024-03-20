#!/bin/bash

./build/input_generator $1 $2 $3 | ./build/benchmarks "$@"