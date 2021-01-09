# Case Studies

---
## PandAna Parallel Read Kernel Benchmark

**pandana_benchmark.c** is an MPI-based  C program developed to study
performance of the I/O kernel of [PandAna](https://bitbucket.org/mpaterno/pandana),
a Python package that is used to analyze the NOvA data by finding events of
interest. PandAna reads an HDF5 produced by [ph5_concat](https://github.com/NU-CUCIS/ph5concat),
which concatenates NOvA data stored in multiple HDF5 files into a single file.
The same data partitioning requirement in PandAna python program is implemented
in C in pandana_benchmark.c.

### Data Partitioning Pattern
PandAna's read data partitioning pattern divides the global event IDs into
contiguous ranges evenly among all processes. This strategy is to ensure an
approximately balanced data workload among all processes. The read operations
include the followings.
1. Find the total number of unique event IDs. This is essentially the length of
   dataset **'/spill/evt.seq'** whose contents are integers of values 0, 1, 2,
   3, ...., N-1, if its length is N.
2. Calculate the global partitioning boundaries, such that each process is
   responsible for a disjoined and contiguous range of event IDs. If N is not
   divisible by P, the number of processes, then the remainder IDs are assigned
   to the processes with lower MPI ranks.
3. Note this calculation, shown below, does not require reading the contents of
   dataset **'/spill/evt.seq'**, but only the length of **'/spill/evt.seq'**.
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
4. Each process is responsible for the global event IDs of range from
   'my_start' to 'my_end' inclusively.

### Parallel reads
Once the global event ID ranges for all processes have been calculated, they
are used to determine what data to be read for datasets in each group.  Each
group, **G**, in the concatenated file contains dataset 'evt.seq' whose values
correspond to '/spill/evt.seq' and are used to find the read ranges for all
other datasets in the group. Note the followings.
* All datasets in the same group share the number of rows, i.e. the size of
     first dimension.
* The contents of **'/G/evt.seq'** are monotonically non-decreasing. It is
  possible to have repeated event IDs in consecutive elements.
* Each dataset is read by all processes in parallel.  Data partitioning is
* based on the global event IDs of ranges, i.e. each process reads only the
  rows with event IDs falling into its own global event IDs range.
* To find the range in a dataset that is assigned to a reading process, a
  binary search is required to find the indices of array elements whose values
  are **'my_start'** and **'my_end'**.
  + lower bound: the first array index 'i' such that 'dataset[i] == my_start'
  + upper bound: the last array index 'j' such that 'dataset[j] == my_end'
  + Once i and j have been found, the process uses them to read a subarray
    starting from index 'i' till 'j' ('i' and 'j' can be used to construct an
    HDF5 hyperslab).

Parallel read is done one group at a time, which consists of the following
steps.
1. Read dataset **'evt.seq'** of group 'G' and calculate lower and upper bounds
   for individual processes. There are 5 implementations.
   * Option 0. Root process reads the entire '/G/evt.seq' and broadcasts it to
     the remaining processes. Then, all processes use the contents of
     '/G/evt.seq' to find its lower and upper bounds.
   * Option 1. All processes collectively read the whole '/G/evt.seq'. All
     processes independently calculate their own lower and upper bounds.
   * Option 2. Root reads '/G/evt.seq', calculates lower and upper bounds for
     all processes, and calls MPI_Scatter to scatter the bounds to all other
     processes.
   * Option 3. Distribute reads of all 'evt.seq' datasets among processes. All
     processes assigned with one or more 'evt.seq' datasets make a single MPI
     collective read call to read 'evt.seq' datasets, calculates lower and
     upper bounds for all other processes, and MPI scatters them.
   * Option 4. Similar to option 3, reading key datasets ('evt.seq') is
     distributed among processes, but POSIX read calls are used to read and
     chunks of key datasets, one chunk at a time. Then, readers decpmpress the
     chunks and calculate the lower and upper bounds of all processes, and MPI
     scatter them.
2. Binary search for the lower and upper bound array indices in
   **'/G/evt.seq'** for each process.
   * Two binary searches should be used, one to search for starting index and
     the other for ending index. Sequentially checking the array contents
     should be avoided.
   * Note the reading lower and upper bound index ranges are not overlapped
     among processes.
3. All processes read each datasets in group G collectively, using the lower
   and upper bounds (through an HDF5 hyperslab). This can be done one dataset
   at a time (using H5Dread) or all datasets at a time (using MPI-IO).

### Data partitioning strategies
In general, data partitioning determines the parallel I/O performance. Evenly
distributing the I/O amount among the available MPI processes is commonly
considered a strategy for achieving the best performance. However, the same
principle may not hold for compressed data, due to data chunking is used. Note
the following HDF5 implementation and requirements for storing and reading
compressed datasets.
* In HDF5, a compression enabled dataset is divided into chunks, which are
  compressed individually. The chunk dimension sizes are user-tunable.
* To fulfill a read request, all compressed chunks containing all or partial
  data of the request must be first read from the file and decompressed, so the
  requested data can be retrieved into user buffers.
* For partial chunk access, the entire chunk must be read before it can be
  decompressed. Decompression cannot perform on partial chunk.
* In parallel read operations, it is possible for any two processes reading
  from disjoined portions in byte range of a compressed dataset to actually
  read the same chunks.
* As the number of same chunks to be read by multiple processes increases, the
  parallel read performance can decrease.
  + When MPI independent I/O is used, the amount of data to be transferred from
    the file servers to compute nodes increases as the number of shared chunks
    increases.
  + When MPI collective I/O is used, a subset of MPI processes are selected as
    I/O aggregators, which are the only processes performing reads from the
    file system. The data read from files are later redistributed from
    aggregators to non-aggregators. In this case, the amount of data
    transferred from file servers is the same as the requested, if MPI-IO data
    sieving hint is not enabled. Hoever, the data amount in the redistribution
    phase increases as the number of shared chunks increases.

One way to avoid shared-chunk access is to align the data partitioning with
the chunk boundaries. The similar approach has been used in ROMIO's driver to
implement MPI collective write operations for Lustre and GPFS file systems,
which aligns the file domain partitioning with the file stripe boundaries and
thus minimizes the file lock contention. The idea of such alignment strategy is
to assign individual chunks entirely to only one process, so no chunk is read
by more than one process. This strategy effectively ensures the data amount to
be transferred from file servers to compute nodes and data redistribution the
same as the read request, However, there are some exceptions.
* When the number of chunks is small, the degree of read workload balance is
  low. Especially, when the number of chunks is smaller than the number of MPI
  processes, some processes may not have chunks to read.
* When computational workload partitioning pattern does not match with the
  chunk-aligned data partitioning, an additional data redistribution among
  processes is necessary.

**Implementation of chunk-aligned data partitioning** -- Command-line option
"-m 3" enables this feature. See command usage below. This option ignores the
global key dataset, e.g. '/spill/evt.seq', and calculates the data partitioning
per group basis using the local key dataset, i.e. '/G/evt.seq'.
* Requirement for input file:
  + All datasets in the same group of the input file must have the same chunk
    dimensional sizes. This requires the 'ph5_concat' program to change its
    implementation for picking the chunk dimension sizes. Commit
    [8e85e29](https://github.com/NU-CUCIS/ph5concat/commit/8e85e2910680a0d3a60eb8cfaadbe43d019557fa)
    changes from the 1-MiB based chunk size to 256 K elementa based.
* Implementation:
  + For each group 'G', the local key dataset of the group, e.g. '/G/evt.seq",
    is first read by root process which uses it to calculate the chunk-aligned
    lower and upper bounds for all processes.
  + Because of the above requirement on chunk dimension size for all datasets
    in the same group, the aligned lower and upper bounds can be calculated
    only once and then used to read all datasets.
  + However, PandAna requires the data with the same key values to be assigned
    to the same MPI processes. As there can be multiple 'rows' with the same
    global key ID, such rows may appear on two consecutive chunks. In this
    case, those rows must be moved to either of the two processes.
  + In our implementation, such data is moved from process 'rank' to 'rank+1'.
    Thus, after the reading phase, there is an MPI send and receive
    communication between any two consecutive processes.
  + The final data partitioning array index range assigned to process 'rank'
    includes the data received from 'rank-1' and subtracted the data sent to
    'rank+1'.
  + If the computational phase of PandAna can be adaptive to use the above
    chunk-aligned data partitioning, then no additional data redistribution is
    required. Otherwise, another phase of communication is necessary.

### Data Parallelism vs. Task Parallelism
The parallelization strategies can be data parallelism, task parallelism, and
a combinations of the two.
* **Data Parallelism** - All processes read individual datasets in parallel.
  + Each process opens a dataset, reads a subarray of the dataset using the
    assigned responsible event ID ranges, and closes the dataset.
  + The degree of read workload balance depends on the dataset size and the
    number of MPI processes. A good performance is expected when datasets are
    large enough to be partitioned among processes.
* **Group Parallelism** - A group is considered as a task.
  + When the number of processes is smaller than the number of groups, the
    groups are divided among available processes. Each process is reading the
    datasets of only assigned groups.
  + When the number of processes is larger than the number of groups, the
    processes are divided into subsets, with split MPI communicators. Processes
    in a subset are assigned a group and are responsible to read the datasets
    in that group. In this approach, groups are considered as **tasks**. Within
    a group, the data parallelism is ued. Reading multiple groups can occur
    simultaneously.
* **Dataset Parallelism** - All datasets, except key datasets, in all groups
  are evenly assigned to the MPI processes. One dataset is read entirely by a
  process only. Each process is responsible to read the assigned datasets.
  + Only HDF5 H5Dread() is called.
  + Reading the key datasets is skipped, as there is no use of those datasets.
  + The degree of parallelism is limited to the number of datasets . When the
    number of MPI processes is more than the datasets, some processes have no
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
    [-m number]    read method for other datasets (0, 1, 2, 3, or 4)
                   0: use H5Dread, one dataset at a time (default)
                   1: use MPI_file_read_all, one dataset at a time
                   2: use MPI_file_read_all, all datasets in one group at a
                      time
                   3: use chunk-aligned partitioning and H5Dread to read one
                      dataset at a time. When set, option -s is ignored.
                      Reading key datasets are distributed using H5Dread, one
                      dataset at a time.
                   4: use chunk-aligned partitioning and MPI-IO to read all
                      datasets in a group. When used, -s argument is ignored.
                      Reading key datasets are distributed among processes and
                      MPI_File_read_all to read all assigned key dataset.
    [-r number]    parallelization method (0 or 1)
                   0: data parallelism - all processes read each dataset in
                      parallel (default)
                   1: group parallelism - processes are divided among groups
                      then data parallelism within each groups
                   2: dataset parallelism - divide all datasets of all groups
                      among processes. When set, options -s and -m are ignored.
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
