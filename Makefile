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
# M_ARCH = "native"

clean:
	rm -rf build

build:
	mkdir -p build
	cmake -S . -B ./build
	cmake --build build -j 4

rebuild:
	$(MAKE) clean
ifndef SLURM_ENV
	cmake -DUSE_$(d)=ON -DM_ARCH=$(M_ARCH) -S . -B ./build
else
	cmake -DUSE_$(d)=ON -DCPP_ENV=$(SLURM_ENV) -DM_ARCH=$(M_ARCH) -S . -B ./build
endif
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



gen:
	$(MAKE) build
	./build/input_generator --size=$(size) --density=$(density)

b:
	./build/input_generator --size=$(size) --density=$(density) | ./build/benchmarks --size=$(size) --density=$(density) --benchmark_out=$(out_file) --seed=$(seed) -b $(b) -d $(d) --full=$(full) --mul=$(mul) --tab=$(tab) --benchmark_time_unit=$(unit)

bench:
	$(MAKE) build
	$(MAKE) b

catchb:
	./build/tests --benchmark-samples=100 benchmarks

catchbuild:
	$(MAKE) build
	$(MAKE) catchb


test:
	$(MAKE) build
	./build/tests $(p)


# make rebuild d="-DUSE_ACCELERATE=ON"