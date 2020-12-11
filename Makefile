MPICC    = mpicc
HDF5_DIR = $(HOME)/HDF5/1.12.0

OPTS     = -g -O0
CPPFLAGS = -I$(HDF5_DIR)/include
CFLAGS   = $(OPTS) $(CPPFLAGS)
LDFLAGS  = -L$(HDF5_DIR)/lib
LIBS     = -ldl -lz -lhdf5

all: pandana_read print_dsets

pandana_read: pandana_read.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

print_dsets: print_dsets.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o core*
	rm -rf pandana_read print_dsets

