# Timing breakdown of end-to-end HDF5 read
## Experiment settings
* Use Cori debug queue, Haswell nodes
* Number of processes: 128
* Lustre stripe count: 32
* Lustre stripe size: 1 MiB
* HDF5 version: 1.10.6
* File: ND 1951 file (Path: /global/cscratch1/sd/wkliao/FS_1M_32/NOvA_ND_1951.h5caf_*.h5)
* Name of file containing dataset names to be read: dset.txt (https://github.com/wkliao/pandana_read_case/blob/master/dset.txt)

| datasets | before compression (MiB) | after compression (MiB) |
| :------: | :----------------------: | :---------------------: |
|"evt.seq" datasets | 2068.72 | 124.04 |
| other datasets | 7264.16 | 1462.62 |
|total size | 9332.88 | 1586.66 |
## Program
* pandana_read.c: https://github.com/NU-CUCIS/ph5concat/blob/master/case_study/pandana_read.c
* I/O methods to read "evt.seq" datasets: root process reads evt.seq and scatters boundaries.
* I/O methods to read other datasets: use H5Dread.

## End-to-end HDF5 read time (14.58 sec):

    When reading "evt.seq" datasets (7.05 sec):
        call H5Dopen (1.24 sec)
        call H5Dread (H5Dio.c) (5.81 sec):
            call H5D__read (H5Dio.c) (5.56 sec):
                call H5D__ioinfo_adjust (adjusting the operation's I/O information) (0.91 sec)
                call multi_read function (4.65 sec)
                
    When reading other datasets (7.53 sec):
        call H5Dopen (0.61 sec)
        call H5Dread (H5Dio.c) (6.92 sec):
            call H5D__read (H5Dio.c) (6.86 sec):
                call H5D__ioinfo_adjust (adjusting the operation's I/O information) (2.43 sec)
                call H5D__chunk_collective_read (H5Dmpio.c) (4.43 sec):
                    call H5D__chunk_collective_io  (4.43 sec):
                        call H5D__multi_chunk_filtered_collective_io (4.43 sec):
                            call  H5D__construct_filtered_io_info_list (building a list of selected chunks in the collective io operation) (1.60 sec)
                            for all chunks of this process:
                                call H5D__filtered_collective_chunk_entry_io (sum: 2.83 sec):
                                    call H5F_block_read (read compressed chunks) (sum: 2.27 sec)
                                    call H5Z_pipeline (chunk decompression) (sum: 0.55 sec)
