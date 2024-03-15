#!/bin/bash

make build
./build/input_generator $1 $2 $3 | ./build/benchmarks "$@"