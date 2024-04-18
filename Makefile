default_target: all
.PHONY : all test clean tall build

MAKEFLAGS += --no-print-directory

b = 0
d = 0
seed = 0
density = 0
unit = us
full = "true"
mul = "false"
tab = "false"
out_file = "./benchmark/benchmark.json"
M_ARCH ?=

clean:
	rm -rf build

build:
	mkdir -p build
	cmake -S . -B ./build
	cmake --build build -j 4

rebuild:
	$(MAKE) clean
	cmake -DUSE_$(d)=ON -DM_ARCH=$(march) -S . -B ./build
	cmake --build build -j 4

run:
	./build/app

trun:
	./build/tests

alltest:
	$(MAKE) build
	$(MAKE) trun

all:
	$(MAKE) build
	$(MAKE) run

# Specific tests


catchb:
	./build/tests --benchmark-samples=100 benchmarks

catchbuild:
	$(MAKE) build
	$(MAKE) catchb

test:
	$(MAKE) build
	./build/tests $(p)


# make rebuild d="-DUSE_ACCELERATE=ON"