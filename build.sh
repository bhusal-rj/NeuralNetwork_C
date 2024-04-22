#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lm -lglfw -ldl -lpthread"

clang $CFLAGS -o adder_gen adder_gen.c nn.c $LIBS
clang $CFLAGS -o xor_gen xor_gen.c nn.c $LIBS
clang $CFLAGS -o gym gym.c nn.c $LIBS
