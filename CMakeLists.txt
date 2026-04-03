cmake_minimum_required(VERSION 3.16)
project(cmas_phase_field2 LANGUAGES CXX)

find_package(deal.II 9.3 REQUIRED HINTS /usr/share/cmake/deal.II /usr/lib/x86_64-linux-gnu/cmake/deal.II)
deal_ii_initialize_cached_variables()

add_executable(cmas_phase_field2 cmas_phase_field2.cc)
deal_ii_setup_target(cmas_phase_field2)
