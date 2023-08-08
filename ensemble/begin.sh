rm -rf ./build
rm -rf ./bin
mkdir build
mkdir bin
cd build
cmake -DCMAKE_PREFIX_PATH=/root/test/torch_test/Machine-Learning/ensemble/3rd/libtorch ..
make
