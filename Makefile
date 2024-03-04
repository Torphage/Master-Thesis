default_target: all
.PHONY : all test clean tall build

MAKEFLAGS += --no-print-directory


clean:
	rm -r build

build:
	mkdir -p build
	cmake --build build -j 2 

rebuild:
	$(MAKE) clean
	cmake -D USE_$(d)=ON -S . -B ./build
	cmake --build build -j 2

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

bench:
	$(MAKE) build
	./build/tests --benchmark-samples=100 benchmarks

test:
	$(MAKE) build
	./build/tests $(p)


# make rebuild d="-DUSE_ACCELERATE=ON"