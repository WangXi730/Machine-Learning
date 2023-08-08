add_library(CART SHARED ./CART.cpp)
add_executable(CART_test ./test.cpp)
#target_link_libraries(CART ${base} ${TORCH_LIBRARIES})
target_include_directories(CART PRIVATE "${TORCH_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${PROJECT_SOURCE_DIR}/bin")

target_link_libraries(CART_test CART base ${TORCH_LIBRARIES})
target_include_directories(CART_test PRIVATE "${TORCH_INCLUDE_DIRS}")
