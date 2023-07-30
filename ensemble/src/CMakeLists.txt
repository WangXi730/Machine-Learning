add_library(base SHARED ./base.cpp)
target_link_libraries(base PRIVATE ${TORCH_LIBRARIES})

add_subdirectory(CART)


