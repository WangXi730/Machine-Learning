rm -rf ./build
rm -rf ./bin
mkdir build
mkdir bin
cd build
cmake -DCMAKE_PREFIX_PATH=/root/test/torch_test/test1/3rd/libtorch ..
make
