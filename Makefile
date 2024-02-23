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
	cmake -S . -B ./build
	cmake --build build -j 2

run:
	./build/app

trun:
	./build/tests

test:
	$(MAKE) build
	$(MAKE) trun

all:
	$(MAKE) build
	$(MAKE) run

# Specific tests

parallel:
	$(MAKE) build
	./build/tests "Parallel"