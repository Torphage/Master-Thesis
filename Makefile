default_target: all
.PHONY : default_target


redownload:
	rm -r build
	mkdir build 
	cmake -S . -B ./build
	cmake --build build -j 2

rebuild:
	cmake --build build -j 2

run:
	./build/app

new:
	mkdir -p build
	cmake -S . -B ./build
	cmake --build build -j 2

build:
	mkdir -p build
	cmake --build build -j 2

all:
	mkdir -p build
	cmake --build build -j 2
	./build/app