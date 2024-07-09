add_executable(generate_state generate_state.cpp)
target_link_libraries(generate_state faster_rwkv)

add_executable(export_ncnn export_ncnn.cpp)
target_link_libraries(export_ncnn faster_rwkv)

add_executable(eval_text eval_text.cpp)
target_link_libraries(eval_text faster_rwkv)