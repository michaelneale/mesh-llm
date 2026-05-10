# Hardware Support

Mesh pools the machines you already have. A node can contribute GPU, Apple Silicon, or CPU capacity depending on the installed release and local hardware.

## macOS

Apple Silicon machines use the local Metal-capable runtime when available.

## Linux

Linux nodes can use CUDA, ROCm, Vulkan, or CPU builds depending on hardware and release flavor.

## Windows

Windows support is planned. For now, use WSL2 and follow the Linux path.

## Planning capacity

For single-machine serving, choose models that fit on one device. For larger models, use mesh-ready catalog entries with layer packages so work can be placed across multiple machines.

