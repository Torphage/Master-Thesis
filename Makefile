default_target: all
.PHONY : all test clean tall build

MAKEFLAGS += --no-print-directory


clean:
	rm -r build

build:
	mkdir -p build
	cmake -D USE_$(d)=ON -S . -B ./build
	cmake --build build -j 4

rebuild:
	$(MAKE) clean
	cmake -D USE_$(d)=ON -S . -B ./build
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
	./build/input_generator --size=$(size) --density=$(density) | ./build/benchmarks --size=$(size) --density=$(density) --benchmark_out="benchmark.json"

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