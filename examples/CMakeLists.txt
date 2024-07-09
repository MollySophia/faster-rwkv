add_executable(chat chat.cpp)
target_link_libraries(chat faster_rwkv)

add_executable(abc_music abc_music.cpp)
target_link_libraries(abc_music faster_rwkv)

add_executable(midi_music midi_music.cpp)
target_link_libraries(midi_music faster_rwkv)