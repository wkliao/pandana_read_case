# Case Study - PandAna Parallel Read Kernel
---

This folder contains MPI C programs developed to study performance of the I/O
kernel of [PandAna](https://bitbucket.org/mpaterno/pandana), a Python package
that is used to analyze the NOvA data by finding events of interest. PandAna
reads an HDF5 produced by [ph5_concat](https://github.com/NU-CUCIS/ph5concat),
a parallel program that concatenates NOvA data stored in multiple HDF5 files
(usually one event per file) into a single file. The same data partitioning
used in PandAna python program is implemented in C. The main program is
[pandana_benchmark.c](./pandana_benchmark.c).

## Data Partitioning Pattern
PandAna's read data partitioning pattern divides the global unique IDs into
contiguous ranges evenly among all processes. This strategy is to ensure an
approximately balanced data workload (ranges of global unique IDs) among all
processes. The read operations consist of the following steps.
1. Find the total number of unique global unique IDs. This is essentially the
   length of the global key dataset (in this case study the default is dataset
   **'/spill/evt.seq'**) whose contents are integral values 0, 1, 2, 3, ....,
   N-1, if its length is N.
2. Calculate the partitioning boundaries of global unique IDs, such that each
   process is assigned a disjoined and contiguous range of IDs. If N is not
   divisible by P, the number of processes, then the remainder IDs are assigned
   to the processes with lower MPI ranks.
3. Note this calculation, as shown below, does not require reading the contents
   of global key dataset, but only its length.
   * `N` is the number of unique global IDs (length of global key dataset).
   * `nprocs` is the number of MPI processes.
   * `rank` is the MPI rank of a process.
   * `my_start` is the starting ID of the range assigned to process `rank`.
   * `my_count` is the number of IDs of the assigned range to process `rank`.
   * `my_end` is the end ID of the assigned range to process `rank`.
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
4. Each process is responsible for the global IDs of range from 'my_start' to
   'my_end' inclusively.

## Parallel reads
Once the global unique ID ranges for all processes have been calculated, they
are used to determine what subarray of a dataset to be read by a process. In
PandAna, an HDF5 group is referred to as a 'table'. Each group, **G**, in the
concatenated file contains a "key" dataset, referred as the local key dataset
(in this case study, the default is dataset '/G/evt.seq') whose values
correspond to the global unique IDs. The local key dataset is used to calculate
the reading ranges, i.e. subarrays, of all processes for other datasets in the
same group. Note the followings.
* All datasets in the same group share the number of rows, i.e. size of the
  most significant dimension.
* The contents of local key dataset are monotonically non-decreasing. It is
  possible to have repeated IDs in consecutive elements.
* Each dataset is read in parallel. Data partitioning among all processes is
  based on the partitioned global unique IDs, i.e. each process reads only the
  rows corresponding to the local key dataset whose IDs fall into its assigned
  global ID range.
* To find the array index range for a reading process, a binary search is
  required. The starting array index (referred as the lower bound) is obtained
  by searching for **'my_start'** in the local key dataset. Similarly, the end
  array index (referred as the upper bound) is obtained by searching for
  **'my_end'**.
  + lower bound: the first array index 'i' such that 'dataset[i] == my_start'
  + upper bound: the last array index 'j' such that 'dataset[j] == my_end'
  + Once i and j have been found, the subarray to be read by process rank
    starts from index 'i' and ends in 'j' ('i' and 'j' can be used to construct
    an HDF5 hyperslab).

Parallel read is done for one group at a time, which consists of the following
steps.
1. Read the local key dataset of group 'G' and calculate lower and upper bounds
   for individual processes. There are 5 implementations.
   * Option 0. Root process reads the entire local key dataset and broadcasts
     it to the remaining processes. Then, all processes use the contents of
     local key dataset to find its own lower and upper bounds.
   * Option 1. All processes collectively read the whole local key dataset. All
     processes independently calculate their own lower and upper bounds.
   * Option 2. Root reads local key dataset, calculates lower and upper bounds
     for all processes, and then calls MPI_Scatter to scatter the bounds to all
     other processes.
   * Option 3. Distribute reads of all local key datasets of all groups among
     processes. All processes that are assigned with one or more local key
     datasets make a single MPI collective read call to read all local key
     datasets, calculates lower and upper bounds for all processes, and MPI
     scatters them.
   * Option 4. Similar to option 3, reading local key datasets is distributed
     among processes, but POSIX read calls are used to read chunks of key
     datasets, one chunk at a time. Then, readers decpmpress the chunks and
     calculate the lower and upper bounds for all processes, and MPI scatter
     them.
2. Binary search for the lower and upper bound array indices in the local key
   dataset for each process.
   * Two binary searches should be used, one to search for starting index and
     the other for end index. Sequentially checking the array contents should
     be avoided.
   * Note lower and upper bound index ranges among processes are not
     overlapped.
3. All processes collectively read each datasets in group G, using the lower
   and upper bounds (through an HDF5 hyperslab). This can be done one dataset
   at a time (using H5Dread) or all datasets at a time (using MPI-IO).

## Study of Data Partitioning Strategies
In general, data partitioning patterns determine the parallel I/O performance.
Evenly distributing the I/O amount among the available MPI processes is usually
a strategy to achieve the best performance. However, the same idea does not
necessarily result in a good parallel read performance, due to the use of data
chunking in HDF5. This case study demonstrates such problems. The remaining of
this section explains the causes by pointing out the HDF5 designs for
compression-enabled datasets and hence its requirements for storing and reading
them.
* In HDF5, a compression-enabled dataset is divided into chunks, where each is
  compressed independently from others. The chunk dimension sizes are set by
  users.
* To fulfill a read request, all compressed chunks containing all or partial
  data of the request must be first read from the file into the requesting
  process's memory, so they can be decompressed first and then the parts of
  requested data can be copied into user read buffers.
* For partial chunk access, the entire chunk must be read into a temporary
  buffer before it can be decompressed. Decompression cannot perform on s
  partial chunk.
* In a parallel read operation, it is possible for two processes reading from
  disjoined parts of a compressed dataset to read the same chunks, if their
  read requests fall in the same chunk.
* As the number of shared chunks to be read by multiple processes increases,
  the parallel read performance decreases.
  + For MPI independent reads, the amount of data to be transferred from the
    file servers to compute nodes increases as the number of shared chunks
    increases.
  + For MPI collective reads, the two-phase I/O strategy is used, where a
    subset of MPI processes are selected as I/O aggregators, which are the only
    processes performing reads from the file system. The data read from files
    by the aggregators are later redistributed from aggregators to other
    processes. In this case, the amount of data transferred from file servers
    to aggregators stays the same as the requested, if MPI-IO data sieving hint
    is not enabled. However, the data amount in the redistribution phase
    increases when the number of shared chunks increases.

One way to avoid shared-chunk access is to align the data partitioning with
the chunk boundaries. The similar approach has been used in ROMIO's driver to
implement MPI collective I/O operations for Lustre and GPFS file systems, which
aligns the partitioned file domains with the file stripe boundaries. This
strategy minimizes the file lock contentions which allows to achieve much
better performance. The same idea of such alignment strategy can be used to
assign chunks to processes, so no chunk is read by more than one process. This
strategy effectively ensures the data amount to be transferred from file
servers to compute nodes and data redistribution the same as the read request
amount, However, it may perform suboptimally for the following scenarios.
* When the number of chunks of a dataset is small, the balance degree of read
  workload becomes low. Especially, when the number of chunks is smaller than
  the number of MPI processes, some processes may not have chunks to read.
* When the partitioning pattern of computational workload, e.g. PandAana
  filtering operation, does not match with the chunk-aligned data partitioning,
  an additional data redistribution among processes is necessary.

**Implementation of chunk-aligned data partitioning** -- Command-line option
"-m 3" enables this feature. See command usage below. This option ignores the
global key dataset and calculates the data partitioning per group basis using
the local key dataset. The reading array index ranges can be different from one
group to another.
* Requirement for input file:
  + All datasets in the same group must have the same chunk dimensional sizes.
    This requires the 'ph5_concat' program to change its implementation for
    picking the chunk dimension sizes. Commit
    [8e85e29](https://github.com/NU-CUCIS/ph5concat/commit/8e85e2910680a0d3a60eb8cfaadbe43d019557fa)
    changes from the 1-MiB based chunk size to 256-K element based.
* Implementation:
  + For each group 'G', the local key dataset of the group is first read by
    root process which uses it to calculate the chunk-aligned lower and upper
    bounds for all processes.
  + Because of the above requirement on the same chunk dimension sizes for all
    datasets in the same group, the aligned lower and upper bounds can be
    calculated only once and used to read all local datasets.
  + However, PandAna requires the data with the same key values to be assigned
    to the same MPI processes. As there may be contiguous 'rows' with the same
    global key ID that are stored across two consecutive chunks. In this case,
    when the two chunks are read by two processes, those rows must be moved to
    either of the two processes.
  + In our implementation, such data is moved from process 'rank' to 'rank+1'.
    Thus, after the reading phase, there is an MPI send and receive
    communication between any two consecutive processes.
  + Because of such data "shifting" communication, the lower and upper bounds
    of array index range assigned to process 'rank' must be adjusted to include
    the data received from 'rank-1' and subtracted the data sent to 'rank+1'.
  + If the computational phase of PandAna can be adaptive to make use of the
    above chunk-aligned data partitioning, then no further data redistribution
    is required. Otherwise, another phase of communication is necessary to
    achieve the desired partitioning.

### Data Parallelism vs. Task Parallelism
The parallelization strategies can be data parallelism, task parallelism, or
a combinations of the two.
* **Data Parallelism** - All processes read individual datasets in parallel.
  + Each process opens a dataset, reads a subarray of the dataset using the
    assigned responsible ranges of global IDs, and closes the dataset.
  + The degree of read workload balance depends on the dataset size and the
    number of MPI processes. A good performance is expected when datasets are
    large enough such that each process is assigned a sufficiently large data
    to read.
* **Group Parallelism** - Reading datasets in a group is considered as a task.
  + When the number of processes is smaller than the number of groups, the
    groups are divided among available processes. Each process is reading the
    datasets only in the assigned groups.
  + When the number of processes is larger than the number of groups, the
    processes are divided into subsets disjointly. Each subset is identified
    by an MPI communicator split from the MPI communicator MPI_COMM_WORLD.
    Processes in a subset are assigned to a group and are responsible to read
    the datasets in that group. In this approach, groups are considered as
    **tasks**. Within a group, the data parallelism is ued. Reading multiple
    groups can occur simultaneously.
* **Dataset Parallelism** - All datasets of all groups are evenly assigned to
  the MPI processes. One dataset is read entirely by a process only. Each
  process reads only the assigned datasets.
  + The degree of parallelism is limited to the number of datasets . When the
    number of MPI processes is more than the datasets, some processes have no
    dataset to read.
  + After reading, a data redistribution is required to make each process
    possess the dataset subarrays within its assigned global ID ranges.
  + Reading the key datasets is required in order to calculate the amount of
    data to be redistributed from the dataset readers to non-readers.
  + Redistribution is necessary because PandAna's filtering operations are
    applied to multiple datasets element-wisely. For example, the following
    python code fragment is shown in one of PandAna papers.
    ```
    def kNueSecondAnaContainment( tables ):
        df = tables['sel_nuecosrej']
        return (df.distallpngtop    > 63.0) & \
               (df.distallpngbottom > 12.0) & \
               (df. distallpngeast  > 12.0) & \
               (df.distallpngwest   > 12.0) & \
               (df. distallpngfront > 18.0) & \
               (df.distallpngback   > 18.0)
    ```
  + **Currently, only the dataset reading part is implemented. The
    redistribution part has not been added.**

### Run Command usage:
  ```
  % ./pandana_benchmark -h
  Usage: ./pandana_benchmark [-h] [-p number] [-s number] [-m number] [-r number] [-l fname] [-i fname]
    [-h]        print this command usage message
    [-p number] performance profiling method (0 or 1)
                0: report file open, close, read timings (default)
                1: report number of chunks read per process
    [-s number] read method for key datasets (0, 1, 2, 3, or 4)
                0: root process HDF5 reads all keys and broadcasts, all
                   processes calculate their own boundaries
                1: all processes HDF5 read all keys collectively, calculate
                   their own boundaries
                2: root process HDF5 reads keys, calculates and scatters
                   boundaries to other processes
                3: distribute key reading among processes, read all assigned
                   keys using one MPI collective read, calculate and scatter
                   boundaries to other processes (default)
                4: distribute key reading among processes, read all assigned
                   keys using POSIX read, calculate and scatter boundaries to
                   other processes
    [-m number] read method for other datasets (0, 1, 2, or 3)
                0: collective H5Dread, one dataset at a time
                1: MPI_file_read_all, one dataset at a time
                2: MPI_file_read_all, all datasets in one group at a time
                   (default)
                3: use chunk-aligned partitioning and collective H5Dread one
                   dataset at a time. When set, option -s is ignored. Reading
                   key datasets are distributed among processes, independent
                   H5Dread, one dataset at a time.
                4: use chunk-aligned partitioning and one MPI_File_read_all to
                   read all datasets in a group. When set, option -s argument
                   is ignored. Reading key datasets are distributed among
                   processes, one MPI_File_read_all to read all assigned key
                   dataset.
    [-r number] parallelization method (0, 1, or 2)
                0: data parallelism - all processes read each dataset in
                   parallel (default)
                1: group parallelism - processes are divided into groups, then
                   data parallelism is used within each group
                2: dataset parallelism - divide all datasets of all groups
                   among processes. When set, options -s and -m are ignored.
    [-l fname]  name of file containing dataset names to be read
    [-i fname]  name of input HDF5 file
    *ph5concat version 1.1.0 of March 1, 2020.
  ```

### Example Run:
A sample input file named 'dset.txt' is provided in this folder which includes
names of a list of datasets to be read from the concatenated HDF5 file.
Example run and output:
  ```
  % mpiexec -n 4 ./pandana_benchmark -l dset.txt -i nd_165_files_with_evtseq.h5 -p 1
  Number of MPI processes = 4
  Input dataset name file 'dset.txt'
  Input concatenated HDF5 file 'nd_165_files_with_evtseq.h5'
  Number of groups   to read = 15
  Number of datasets to read = 123
  MAX/MIN no. datasets per group = 13 / 5
  Read key   datasets method: Distributed MPI collective read, decompress, and scatters boundaries
  Read other datasets method: MPI collective read and decompress, all datasets in one group at a time
  Parallelization method: data parallelism (all processes read individual datasets in parallel)
  ----------------------------------------------------
  MAX and MIN among all 4 processes
  MAX read amount: 314.25 MiB (0.31 GiB)
  MIN read amount: 275.65 MiB (0.27 GiB)
  MAX time: open=0.00 key=0.19 datasets=1.15 close=0.00 inflate=0.87 TOTAL=1.35
  MIN time: open=0.00 key=0.19 datasets=1.14 close=0.00 inflate=0.59 TOTAL=1.35
  ----------------------------------------------------
  Number of unique IDs (size of /spill/evt.seq)=410679
  Read amount MAX=4.77 MiB MIN=0.39 MiB (per dataset, per process)
  Amount of key   datasets 262.81 MiB = 0.26 GiB (compressed 12.24 MiB = 0.01 GiB)
  Amount of other datasets 924.93 MiB = 0.90 GiB (compressed 181.22 MiB = 0.18 GiB)
  Sum amount of all datasets 1187.74 MiB = 1.16 GiB (compressed 193.46 MiB = 0.19 GiB)
  total number of chunks in all 123 datasets (exclude /spill/evt.seq): 1150
  Aggregate number of chunks read by all processes: 1474
          averaged per process: 368.50
          averaged per process per dataset: 3.00
  Out of 1150 chunks, number of chunks read by two or more processes: 320
  Out of 1150 chunks, most shared chunk is read by number of processes: 3
  ----------------------------------------------------
  group                                rec.energy.numu size   118 MiB (zipped   35 MiB) nChunks=117
  group                                        rec.hdr size    61 MiB (zipped    2 MiB) nChunks=63
  group                                rec.sel.contain size    83 MiB (zipped   11 MiB) nChunks=81
  group                                rec.sel.cvn2017 size    66 MiB (zipped    8 MiB) nChunks=63
  group                          rec.sel.cvnProd3Train size    66 MiB (zipped    8 MiB) nChunks=63
  group                                  rec.sel.remid size    66 MiB (zipped    4 MiB) nChunks=63
  group                                        rec.slc size   101 MiB (zipped   21 MiB) nChunks=99
  group                                      rec.spill size    61 MiB (zipped    2 MiB) nChunks=63
  group                                 rec.trk.cosmic size    74 MiB (zipped    2 MiB) nChunks=63
  group                                 rec.trk.kalman size    83 MiB (zipped    3 MiB) nChunks=72
  group                          rec.trk.kalman.tracks size    94 MiB (zipped   18 MiB) nChunks=90
  group                         rec.vtx.elastic.fuzzyk size    54 MiB (zipped    2 MiB) nChunks=56
  group                     rec.vtx.elastic.fuzzyk.png size    92 MiB (zipped    2 MiB) nChunks=91
  group              rec.vtx.elastic.fuzzyk.png.shwlid size   153 MiB (zipped   67 MiB) nChunks=156
  group                                          spill size     9 MiB (zipped    1 MiB) nChunks=10
  
  
  rank   0: no. chunks read=273 include key evt.seq (max=3 min=1 avg=1.93 among 108 datasets, exclude key evt.seq)
  rank   1: no. chunks read=442 include key evt.seq (max=4 min=1 avg=3.30 among 108 datasets, exclude key evt.seq)
  rank   2: no. chunks read=381 include key evt.seq (max=4 min=2 avg=2.80 among 108 datasets, exclude key evt.seq)
  rank   3: no. chunks read=378 include key evt.seq (max=5 min=1 avg=2.85 among 108 datasets, exclude key evt.seq)
  ```
---

References:
1. M. Paterno, J. Kowalkowski, and S. Sehrish.
   ["Parallel Event Selection on HPC Systems"](https://doi.org/10.1051/epjconf/201921404059),
   in the International Conference on Computing in High Energy and Nuclear
   Physics (CHEP), 2018.
2. S. Sehrish, J. Kowalkowski, M. Paterno, and C. Green. 2017. "Python and
   HPC for High Energy Physics Data Analyses". In PyHPC'17: the 7th Workshop on
   Python for High-Performance and Scientific Computing, November 12-17, 2017.
   https://doi.org/10.1145/3149869.3149877




