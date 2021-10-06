AM_CPPFLAGS = -I$(top_srcdir)
ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libpandana_lib.la
libpandana_lib_la_SOURCES = 	pandana_lib.c \
							pandana_lib.h

bin_PROGRAMS = pandana_read print_dsets pandana_benchmark
pandana_read_SOURCES = pandana_read.c 
print_dsets_SOURCES = print_dsets.c 
pandana_benchmark_SOURCES = pandana_benchmark.c 
pandana_benchmark_LDADD = libpandana_lib.la

CLEANFILES = core core.* *.o