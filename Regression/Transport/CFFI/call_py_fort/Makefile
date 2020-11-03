CMAKE=cmake
all: | build
	mkdir -p build
	cd build && $(CMAKE) ..
	${MAKE} -C build

install: all
	${MAKE} -C build install

debug: | build
	mkdir -p build
	cd build && $(CMAKE) .. -DCMAKE_BUILD_TYPE=Debug
	${MAKE} -C build

clean:
	${RM} -r build

check: all
	cd build && ctest

build:
	mkdir -p $@
