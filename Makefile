setup:
	CUDACXX=clang++-18 CXX=clang++-18 cmake -S . -B build -G Ninja

default:
	cmake --build build

run:
	./build/source/cuarena
