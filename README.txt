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

It should create file output_itemsets.txt. Validate the result by
comparing the file with reference_output_itemsets.txt.
