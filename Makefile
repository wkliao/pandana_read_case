MPICC    = mpicc
HDF5_DIR = $(HOME)/.local/hdf5/1.12.0

OPTS     = -g -O0 -Wall
CPPFLAGS = -I. -I$(HDF5_DIR)/include
CFLAGS   = $(OPTS) $(CPPFLAGS)
LDFLAGS  = -L$(HDF5_DIR)/lib
LIBS     = -ldl -lz -lhdf5

all: pandana_read print_dsets libpandana.a pandana_benchmark

pandana_lib.o: pandana_lib.c pandana_lib.h
	$(MPICC) $(CFLAGS) -c -o $@ $<

libpandana.a: pandana_lib.o
	ar cru libpandana.a pandana_lib.o
	ranlib libpandana.a

pandana_benchmark: pandana_benchmark.c pandana_lib.c pandana_lib.h
	$(MPICC) -DPANDANA_BENCHMARK $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

pandana_read: pandana_read.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

print_dsets: print_dsets.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o core* libpandana.a
	rm -rf pandana_read print_dsets pandana_benchmark

