# Compilation

## Altera OpenCL

```
make ALTERA_LINUX64=1 BOARD=board-name
```

For emulation, set EMU to 1:

``` 
make ALTERA_LINUX64=1 BOARD=board-name EMU=1
```

## AMD OpenCL SDK

```
make AMD_LINUX64=1
```

## Other platforms

See Makefile. There are basically only two files: nw.cc and
nw_kernel.cl. nw.cc uses OpenCL, so it requires a C++ compiler with
OpenCL header files and libraries.

# Executon

Just run the compiled binary as:

```
./nw
```

The program should valiate the result and display whether the result
is correct or not. It also saves the result in output_itemsets.txt.


# Bugs

Specifying "volatile" to input_itemsets solves the problem. The cache system of the FPGA seems to have some problems.
