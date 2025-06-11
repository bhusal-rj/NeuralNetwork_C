# Neural Network in C

A neural network implementation in C with visualization capabilities using Raylib.

## Project Structure

- `nn.c` / `nn.h` - Core neural network implementation
- `adder_gen.c` - Training program for addition problems
- `xor_gen.c` - Training program for XOR logic gate
- `vis.c` - Neural network visualizer using Raylib
- `sv.h` - Support/utility header
- `data.txt` - Training data
- `xor.arch` - XOR network architecture file
- `xor.mat` - XOR network weights/matrix file

## Dependencies

- **Raylib** - For visualization (graphics library)
- **GLFW** - Window management
- **Standard C libraries** - math, threading support

## Building

Make sure you have Raylib and its dependencies installed, then run:

```bash
./build.sh
```

This will compile three executables:
- `adder_gen` - Generates/trains neural network for addition
- `xor_gen` - Generates/trains neural network for XOR gate
- `vis` - Visualizes trained neural networks

## Usage

### Training XOR Network
```bash
./xor_gen
```

### Training Adder Network
```bash
./adder_gen
```

### Visualizing Networks
```bash
./vis xor.arch xor.mat
```

The visualizer takes two arguments:
1. Architecture file (`.arch`) - defines network structure
2. Matrix file (`.mat`) - contains trained weights

## Features

- Custom neural network implementation from scratch
- Support for different network architectures
- Real-time visualization of network structure and weights
- Training examples for logical operations (XOR) and arithmetic (addition)