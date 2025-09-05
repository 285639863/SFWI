# SFWI

## Copyright statements

This project uses `OpenMPI` for distributed-memory parallelism via MPI.
OpenMPI is an open-source implementation of the Message Passing Interface, licensed under the New BSD License.

This project uses the `NVIDIA CUDA Toolkit` for GPU programming.
CUDA is provided by NVIDIA under the NVIDIA Software License Agreement.

This project uses `segy.h` from the open-source Seismic Unix (SU) package
developed by the Colorado School of Mines, which is released under a BSD-style license.

---

## Project Structure

- `sfwi-stage1`: SFWI source code of stage1
- `sfwi-stage2`: SFWI source code of stage2
- `example-1`: Data of the first numerical example
- `example-2`: Data of the second numerical example
---

## Build Instructions

Before compiling, make sure you have the following:

- **CUDA Toolkit** 
- **MPI** 

### Compile

```bash
cd /SFWI/sfwi-stage1
make
cd /SFWI/sfwi-stage2
make
````

This produces the executable `SFWI1` and `SFWI2`.

---

## Run Instructions

To run the program:

The commands to run the program are provided in the `run.sh` scripts located in `sfwi-stage1` and `sfwi-stage2`.

---
