ifdef AMD_LINUX64
	INC = -I$(AMDAPPSDKROOT)/include
	LIB = -L$(AMDAPPSDKROOT)/lib/x86_64 -lOpenCL
	CXXFLAGS = -g -Wall
	LDFLAGS =
all: nw
endif

ifdef ALTERA_LINUX64
	INC = $(shell aocl compile-config) -DALTERA_CL
	LIB = $(shell aocl link-config) -lOpenCL
	CXXFLAGS = -g -Wall
	LDFLAGS =
	AOC_CFLAGS = -g -v --report --board $(BOARD)
	ifdef EMU
		AOC_CFLAGS += -march=emulator
	endif
all: nw nw_kernel.aocx
endif

.PHONY: nw
nw: nw.cc
	$(CXX) nw.cc $(INC) $(LIB) $(CXXFLAGS) $(LDFLAGS) -o nw

ifdef VOLATILE
	AOC_CFLAGS += -DVOLATILE=volatile
endif

nw_kernel.aocx: nw_kernel.cl
	aoc nw_kernel.cl $(AOC_CFLAGS)

#nw_kernel_volatile.aocx: nw_kernel_volatile.cl
#	aoc nw_kernel_volatile.cl $(AOC_CFLAGS)

clean:
	$(RM) -rf nw *.o *.aocx *.aoco nw_kernel
