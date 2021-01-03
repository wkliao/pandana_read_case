# Case Studies

---
## PandAna Parallel Read Kernel Benchmark

**pandana_benchmark.c** is a MPI-based  C program developed to study the performance
of parallel read from NOvA file produced by **ph5_concat**. It uses the same
data partitioning pattern as [PandAna](https://bitbucket.org/mpaterno/pandana),
a Python package that can be used to select NOvA events of interest.

### Data Partitioning Pattern
PandAna's read data partitioning pattern divides the event IDs exclusively into
contiguous ranges of event IDs evenly among all processes. The implementation
includes the followings.
1. Find the total number of unique event IDs. This is essentially the length of
   dataset **'/spill/evt.seq'** whose contents are 0, 1, 2, 3, ...., N-1, if
   its length is N.
2. Calculate the partitioning boundaries, so each process is responsible for a
   exclusive and contiguous range of event IDs. If N is not divisible by P, the
   number of processes, then the remainder IDs are assigned to the processes of
   lower ranks.
3. Note this calculation does not require reading the contents of
   **'/spill/evt.seq'**, but only inquires the length of **'/spill/evt.seq'**.
   ```
   my_count = N / nprocs;
   my_start = my_count * rank;
   if (rank < N % nprocs) {
       my_start += rank;
       my_count++;
   }
   else {
       my_start += N % nprocs;
   }
   my_end = my_start + my_count - 1;
   ```
4. Each process is responsible for the range from 'my_start' to 'my_end' inclusively.

### Parallel reads
Each group, **G**, in the concatenated file contains dataset 'evt.seq' whose values 
correspond to '/spill/evt.seq' and can be used to find the data element ranges of
all other datasets in the same group, to be read by all processes.
   * Note all datasets in the same group share the number of rows, i.e. the
     size of first dimension.
   * The contents of **'/G/evt.seq'** are monotonically non-decreasing. It is
     possible to have repeated event IDs in consecutive elements.

Parallel reads consist of the following steps.
1. Read **'evt.seq'** datasets in all groups and calculate array index ranges
   (boundaries) responsible by individual process. This can be done in 5
   options.
   * Option 0. Root process reads the entire '/G/evt.seq' and then broadcasts 
     to the remaining processes. All processes use the contents of
     '/G/evt.seq' to calculate their responsible index ranges. 
   * Option 1. All processes collectively read the whole '/G/evt.seq'. All
     processes calculate their own responsible index ranges.
   * Option 2. Only root process reads '/G/evt.seq'. Root calculates
     responsible index ranges for all processes, and calls MPI_Scatter to 
     scatter the boundaries of ranges (start and end) to all other processes.
   * Option 3. Distribute reads of evt.seq datasets among processes. Each
     process got one or more evt.seq dataset assigned makes a single MPI
     collective read call to read all asigned evt.seq datasets, calculates
     boundaries for all other processes, and MPI scatters the boundaries.
   * Option 4. Root process makes POSIX read calls to read all chunks of
     evt.seq, one dataset at a time, decompress, calculate the boundaries
     of all processes, and MPI scatter boundaries.
2. Calculate the responsible index ranges by checking the contents of 
   **'/G/evt.seq'** of a given group **'G'** to find the starting and ending
   indices that point to range of event IDs fall into its responsible range.
   * Two binary searches should be used, one to search for starting index and
     the other for ending index. This avoid sequentially checking the array
     contents.
3. All processes read the requested datasets in group G collectively, using
   the starting and ending indices (hyperslab), one dataset at a time.
   * Note reading index ranges are not overlapping among all processes.

### Data partitioning strategies
In general, data partitioning determines the parallel I/O performance. Evenly
distributing the I/O amount among the available MPI processes is commonly used
for achieving the best performance. However, the same principle may not hold
for compressed data, due to compressed data is stored in "chunks". Note the
following HDF5 implementation and requirements for reading compressed datasets.
* HDF5 datasets are divided into chunks, which are compressed individually. The
  chunk dimension sizes are user-tunable.
* To fulfill a read request, all compressed chunks containing all or partial
  data of the request must be first read from the file and decompressed, so the
  requested data can be retrieved/copied into user buffers.
* For partial chunk access, the entire chunk must be read before it can be
  decompressed. Decompression cannot perform on partial chunk.
* In parallel read operations, it is possible for any two processes reading
  from disjoined portions of a compressed dataset to actually read the same
  chunks.
* As the number of same chunks to be read by multiple processes increases, the
  parallel read performance decreases.
  + When MPI independent I/O is used, the data amount to be transferred from
    the file servers to compute nodes increases as the number of shared chunks
    increases.
  + When MPI collective I/O is used, a subset of MPI processes are selected as
    I/O aggregators, which are the only processes performing reads from the
    file system. The data read from files are later redistributed to all other
    processes. In this case, the data amount transferred from file servers can
    be the same as the requested, if MPI-IO data sieving hint is not enabled.
    Hoever, the data amount for redistribution increases as the number of
    shared chunks increases.

One way to prevent shared-chunk access is to align the data partitioning with
the chunk boundaries. The similar approach has been used in ROMIO's driver to
implement MPI collective write operations for Lustre file system, which aligns
the file domain partitioning with the file stripe boundaries and thus minimizes
the file lock contention. The idea of aligning data partitioning with chunk
boundaries is to assign the whole chunks to only one process, so no chunk is
read by two or more processes. This strategy effectively reduces the data
amount to be transferred from file servers and compute nodes, as well as the
data amount for redistribution in collective reads. However, there are some
exceptions.
* When the number of chunks is small, the degree of read workload balance is
  low. Especially, when the number of chunks is smaller than the number of MPI
  processes, some processes may not have chunks to read.
* When computational workload partitioning pattern does not match with I/O
  partitioning, an additional data redistribution among processes is necessary.

**Implementation of chunk-aligned data partitioning** -- Command-line option
"-m 3" enables this feature. It ignores the global key dataset, e.g.
'/spill/evt.seq' and calculates the data partitioning per group basis using
the local key dataset.
* Requirements:
  + All datasets in the same group of the input file must have the same chunk
    dimensional sizes. This requires the 'ph5_concat' program to change its
    implementation on picking chunk dimension sizes.
    See [8e85e29](https://github.com/NU-CUCIS/ph5concat/commit/8e85e2910680a0d3a60eb8cfaadbe43d019557fa)
* Implementation:
  + For each group, the local key dataset of the group, e.g.
    "/rec.energy.numu/evt.seq", is first read by one of the process and used to
    calculate the chunk-aligned partition boundaries for all processes.
  + Because of the above chunk dimensional size requirement, all datasets in
    the same group share the same chunk dimensional setting.
  + However, PandAna requires the data with the same key values to be assigned
    to the same MPI processes. As such data may appear on two consecutive
    chunks, the chunk-aligned partitioning must move such data to either of the
    two processes sharing the boundary.
  + In our implementation, such data is moved from process 'rank' to 'rank+1'.
    Thus, after the reading phase, there is an MPI send and receive
    communication between any two consecutive processes.
  + At the end, the final data partitioning assigned to process 'rank' includes
    the data received from 'rank-1' and subtracted the data sent to 'rank+1'.
  + If the computational phase of PandAna can be adaptive to use the
    chunk-aligned data partitioning, then no additional data redistribution is
    required. Otherwise, another phase of communication is necessary.

### Data Parallelism vs. Task Parallelism
The parallelization strategies are developed based on the ideas of data
parallelism, task parallelism, and maybe a combinations of the two.
* **Data Parallelism** - All processes read all individual datasets in
  parallel. In this case, all processes opens each dataset in each group, reads
  the contents of the dataset using the responsible event ID ranges, and closes
  the dataset.
* **Group Parallelism** -
  + When the number of processes is smaller than the number of groups, the
    groups are divided among available processes. Each process is reading the
    datasets of only assigned groups.
  + When the number of processes is larger than the number of groups, the
    processes are divided into subsets. Processes in a subset are assigned a
    group and are responsible to read the datasets in that group. In this
    approach, groups are considered as **tasks**. Reading multiple groups can
    occur simultaneously.
* **Dataset Parallelism** - All datasets, except key datasets, in all groups
  are evenly assigned to the MPI processes. One dataset is read entirely by a
  process only. Each process is responsible to read the assigned dataset.
  + Only HDF5 H5Dread() is called.
  + Reading the key datasets is skipped, as there is no need of those datasets.
  + The degree of parallelism is limited to the number of datasets . When
    running more processes than the datasets, some processes may have no
    dataset to read.

### Run Command usage:
  ```
  % ./pandana_benchmark -h
  Usage: ./pandana_benchmark [-h] [-p number] [-s number] [-m number] [-r number] [-l file_name] [-i file_name]
    [-h]           print this command usage message
    [-p number]    performance profiling method (0 or 1)
                   0: report file open, close, read timings (default)
                   1: report number of chunks read per process
    [-s number]    read method for key datasets (0, 1, 2, 3, or 4)
                   0: root process HDF5 reads and broadcasts (default)
                   1: all processes HDF5 read the entire keys collectively
                   2: root process HDF5 reads each key, one at a time,
                      calculates, scatters boundaries to other processes
                   3: distribute key reading among processes, make one MPI
                      collective read to read all asigned keys, and scatter
                      boundaries to other processes
                   4: root POSIX reads all chunks of keys, one dataset at a
                      time, decompress, and scatter boundaries
    [-m number]    read method for other datasets (0, 1, 2, or 3)
                   0: use H5Dread, one dataset at a time (default)
                   1: use MPI_file_read_all, one dataset at a time
                   2: use MPI_file_read_all, all datasets in one group at a
                      time
                   3: use chunk-aligned partitioning and H5Dread to read one
                      dataset at a time. When used, -s argument is ignored.
                      Reading key datasets are distributed using H5Dread, one
                      dataset at a time.
    [-r number]    parallelization method (0 or 1)
                   0: data parallelism - all processes read each dataset in
                      parallel (default)
                   1: group parallelism - processes are divided among groups
                      then data parallelism within each groups
                   2: dataset parallelism - divide all datasets among processes
    [-l file_name] name of file containing dataset names to be read
    [-i file_name] name of input HDF5 file
    *ph5concat version 1.1.0 of March 1, 2020.
  ```

### Example Run:
A sample input file named 'dset.txt' is provided in this folder which includes
names of a list of datasets to be read from the concatenated HDF5 file.
Example run and output:
  ```
  % mpiexec -n 4 ./pandana_benchmark -l dset.txt -i nd_165_files_with_evtseq.h5 -p1

  Number of MPI processes = 4
  Input dataset name file 'dset.txt'
  Input concatenated HDF5 file 'nd_165_files_with_evtseq.h5'
  Number of groups   to read = 15
  Number of datasets to read = 123
  MAX/MIN no. datasets per group = 13 / 5
  Read key   datasets method: root process H5Dread and broadcasts
  Read other datasets method: H5Dread, one dataset at a time
  Parallelization method: data parallelism (all processes read individual datasets in parallel)
  ----------------------------------------------------
  MAX and MIN among all 4 processes
  MAX read amount: 468.02 MiB (0.46 GiB)
  MIN read amount: 224.62 MiB (0.22 GiB)
  MAX time: open=0.00 key=0.71 datasets=0.93 close=0.00 inflate=0.00 TOTAL=1.65
  MIN time: open=0.00 key=0.71 datasets=0.93 close=0.00 inflate=0.00 TOTAL=1.65
  ----------------------------------------------------
  Number of unique IDs (size of /spill/evt.seq)=410679
  Read amount MAX=4.77 MiB MIN=0.39 MiB (per dataset, per process)
  Amount of evt.seq datasets  262.81 MiB = 0.26 GiB (compressed  12.18 MiB = 0.01 GiB)
  Amount of  other  datasets  924.93 MiB = 0.90 GiB (compressed 181.23 MiB = 0.18 GiB)
  Sum amount of all datasets 1187.74 MiB = 1.16 GiB (compressed 193.42 MiB = 0.19 GiB)
  total number of chunks in all 123 datasets (exclude /spill/evt.seq): 1232
  Aggregate number of chunks read by all processes: 1556
          averaged per process: 389.00
          averaged per process per dataset: 3.16
  Out of 1232 chunks, number of chunks read by two or more processes: 320
  Out of 1232 chunks, most shared chunk is read by number of processes: 3
  ----------------------------------------------------
  group                                rec.energy.numu size   118 MiB (zipped   35 MiB) nChunks=122
  group                                        rec.hdr size    61 MiB (zipped    2 MiB) nChunks=64
  group                                rec.sel.contain size    83 MiB (zipped   11 MiB) nChunks=86
  group                                rec.sel.cvn2017 size    66 MiB (zipped    8 MiB) nChunks=68
  group                          rec.sel.cvnProd3Train size    66 MiB (zipped    8 MiB) nChunks=68
  group                                  rec.sel.remid size    66 MiB (zipped    4 MiB) nChunks=68
  group                                        rec.slc size   101 MiB (zipped   21 MiB) nChunks=104
  group                                      rec.spill size    61 MiB (zipped    2 MiB) nChunks=64
  group                                 rec.trk.cosmic size    74 MiB (zipped    2 MiB) nChunks=77
  group                                 rec.trk.kalman size    83 MiB (zipped    3 MiB) nChunks=86
  group                          rec.trk.kalman.tracks size    94 MiB (zipped   18 MiB) nChunks=95
  group                         rec.vtx.elastic.fuzzyk size    54 MiB (zipped    2 MiB) nChunks=59
  group                     rec.vtx.elastic.fuzzyk.png size    92 MiB (zipped    2 MiB) nChunks=97
  group              rec.vtx.elastic.fuzzyk.png.shwlid size   153 MiB (zipped   67 MiB) nChunks=162
  group                                          spill size     9 MiB (zipped    1 MiB) nChunks=12


  rank   0: no. chunks read=494 include evt.seq (max=4 min=1 avg=4.02 among 108 datasets, exclude evt.seq)
  rank   1: no. chunks read=392 include evt.seq (max=6 min=1 avg=3.19 among 108 datasets, exclude evt.seq)
  rank   2: no. chunks read=332 include evt.seq (max=5 min=2 avg=2.70 among 108 datasets, exclude evt.seq)
  rank   3: no. chunks read=338 include evt.seq (max=6 min=1 avg=2.75 among 108 datasets, exclude evt.seq)
  ```
---
