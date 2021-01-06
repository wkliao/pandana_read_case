/*
 * Copyright (C) 2020, Northwestern University and Fermi National Accelerator Laboratory
 * See COPYRIGHT notice in top-level directory.
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>    /* strtok() */
#include <errno.h>     /* errno */
#include <sys/types.h> /* open(), lseek() */
#include <unistd.h>    /* open(), lseek(), read(), close(), getopt() */
#include <fcntl.h>     /* open() */
#include <limits.h>    /* INT_MAX */
#include <assert.h>    /* assert() */

#include <mpi.h>
#include "hdf5.h"
#include "zlib.h"      /* inflateInit(), inflate(), inflateEnd() */
#include "pandana_lib.h"

#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif

#define CHECK_ERROR(err, msg) { \
    fprintf(stderr,"Error at line %d, function %s, file %s: %s\n", \
            __LINE__, __func__,__FILE__, msg); \
    assert(0); \
}

static int
inq_dset_meta(hid_t, size_t*, hsize_t*, hsize_t*);

#ifdef PANDANA_BENCHMARK
static size_t
binary_search_min(long long, int64_t*, size_t);
static size_t
binary_search_max(long long, int64_t*, size_t);

static int seq_opt;
static int dset_opt;
static int posix_fd;
static double inflate_t;

#define NUM_TIMERS 5
static double timings[5];

void
set_options(int seq_read_opt, int dset_read_opt)
{
    seq_opt = seq_read_opt;
    dset_opt = dset_read_opt;
}

void
init_timers(void)
{
    int i;
    for (i=0; i<6; i++) timings[i] = 0.0;
}
void
get_timings(double t[NUM_TIMERS])
{
    int i;
    for (i=0; i<NUM_TIMERS; i++) t[i] = timings[i];
}

/*----< hdf5_read_keys() >---------------------------------------------------*/
/* Using H5Dread to read key datasets. It includes 3 user-selectable methods.
 * seq_opt == 0: root reads the whole key dataset and broadcasts it. Then each
 *               process calculates the array index range (lower and upper
 *               bound indices) responsible by itself.
 * seq_opt == 1: All processes collectively read the key dataset, and then each
 *               calculates the array index range (lower and upper bound
 *               indices) responsible by itself.
 * seq_opt == 2: root reads key dataset and calculates the array index range
 *               (lower and upper bound indices) responsible by all processes
 */
static ssize_t
hdf5_read_keys(MPI_Comm   comm,       /* MPI communicator */
               hid_t      fd,         /* HDF5 file ID */
               int        nGroups,    /* number of key datasets */
               char     **key_names,  /* [nGroups] key dataset names */
               long long  numIDs,     /* number of globally unique IDs */
               size_t    *lowers,     /* OUT: [nGroups] */
               size_t    *uppers)     /* OUT: [nGroups] inclusive bound */
{
    int g, j, k, nprocs, rank;
    herr_t err;
    ssize_t read_len=0;
    long long lower_upper[2];

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    /* starts[rank] and ends[rank] store the starting and ending global unique
     * IDs that are responsible by process rank
     */
    long long *starts, *ends;
    starts = (long long*) malloc(nprocs * 2 * sizeof(long long));
    if (starts == NULL) CHECK_ERROR(-1, "malloc");
    ends = starts + nprocs;

    /* calculate the range of unique IDs responsible by all process and store
     * them in starts[nprocs] and ends[nprocs] */
    pandana_inq_ranges(nprocs, rank, numIDs, starts, ends);

    if (seq_opt == 0) {
        for (g=0; g<nGroups; g++) {
            /* root reads the whole key dataset and broadcasts it. Then each
             * process calculates the array index range (lower and upper bound
             * indices) responsible by itself.
             */
            hid_t dset = H5Dopen2(fd, key_names[g], H5P_DEFAULT);
            if (dset < 0) CHECK_ERROR(seq, "H5Dopen2");

            hsize_t dims[2], chunk_dims[2];
            size_t dtype_size;
            inq_dset_meta(dset, &dtype_size, dims, chunk_dims);
            if (dims[1] != 1) CHECK_ERROR(err, "dims[1] != 1");
            size_t buf_len = dims[0] * dtype_size;

            /* rank 0 reads the whole key dataset and broadcasts it */
            int64_t *seqBuf = (int64_t*) malloc(buf_len);
            if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");
            if (rank == 0) {
                err = H5Dread(dset, H5T_STD_I64LE, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, seqBuf);
                if (err < 0) CHECK_ERROR(err, "H5Dread");
                read_len += buf_len;
            }
            err = H5Dclose(dset);
            if (err < 0) CHECK_ERROR(err, "H5Dclose");

            MPI_Bcast(seqBuf, dims[0], MPI_LONG_LONG, 0, comm);
            /* find the array index range from 'lower' to 'upper' that falls
             * into this process's partition domain.
             */
            lowers[g] = binary_search_min(starts[rank], seqBuf, dims[0]);
            uppers[g] = binary_search_max(ends[rank],   seqBuf, dims[0]);

            free(seqBuf);
        }
    }
    else if (seq_opt == 1) {
        /* set MPI-IO collective transfer mode */
        hid_t xfer_plist;
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        if (xfer_plist < 0) CHECK_ERROR(err, "H5Pcreate");
        err = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        if (err < 0) CHECK_ERROR(err, "H5Pset_dxpl_mpio");

        for (g=0; g<nGroups; g++) {
            /* All processes collectively read the key dataset, and then each
             * calculates the array index range (lower and upper bound indices)
             * responsible by itself.
             */
            hid_t dset = H5Dopen2(fd, key_names[g], H5P_DEFAULT);
            if (dset < 0) CHECK_ERROR(seq, "H5Dopen2");

            hsize_t dims[2], chunk_dims[2];
            size_t dtype_size;
            inq_dset_meta(dset, &dtype_size, dims, chunk_dims);
            if (dims[1] != 1) CHECK_ERROR(err, "dims[1] != 1");
            size_t buf_len = dims[0] * dtype_size;
            int64_t *seqBuf = (int64_t*) malloc(buf_len);
            if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");

            err = H5Dread(dset, H5T_STD_I64LE, H5S_ALL, H5S_ALL, xfer_plist,
                          seqBuf);
            /* all processes independently reads the whole dataset is even
             * worse.
             * err = H5Dread(dset, H5T_STD_I64LE, H5S_ALL, H5S_ALL,
             *               H5P_DEFAULT, seqBuf);
             */
            if (err < 0) CHECK_ERROR(err, "H5Dread");
            read_len += buf_len;

            /* find the array index range from 'lower' to 'upper' that falls
             * into this process's partition domain.
             */
            lowers[g] = binary_search_min(starts[rank], seqBuf, dims[0]);
            uppers[g] = binary_search_max(ends[rank],   seqBuf, dims[0]);

            free(seqBuf);
            err = H5Dclose(dset);
            if (err < 0) CHECK_ERROR(err, "H5Dclose");
        }
        err = H5Pclose(xfer_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");
    }
    else if (seq_opt == 2) {
        long long *bounds;
        if (rank == 0) {
            bounds = (long long*) malloc(nprocs * 2 * sizeof(long long));
            if (bounds == NULL) CHECK_ERROR(-1, "malloc");
        }
        for (g=0; g<nGroups; g++) {
            if (rank == 0) {
                /* root reads key dataset and calculates the array index range
                 * (lower and upper bound indices) responsible by all processes
                 */
                hid_t dset = H5Dopen2(fd, key_names[g], H5P_DEFAULT);
                if (dset < 0) CHECK_ERROR(seq, "H5Dopen2");

                hsize_t dims[2], chunk_dims[2];
                size_t dtype_size;
                inq_dset_meta(dset, &dtype_size, dims, chunk_dims);
                if (dims[1] != 1) CHECK_ERROR(err, "dims[1] != 1");
                size_t buf_len = dims[0] * dtype_size;

                int64_t *seqBuf = (int64_t*) malloc(buf_len);
                if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");
                err = H5Dread(dset, H5T_STD_I64LE, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, seqBuf);
                if (err < 0) CHECK_ERROR(err, "H5Dread");
                read_len += buf_len;

                err = H5Dclose(dset);
                if (err < 0) CHECK_ERROR(err, "H5Dclose");

                for (j=0, k=0; j<nprocs; j++) {
                    bounds[k++] = binary_search_min(starts[j], seqBuf, dims[0]);
                    bounds[k++] = binary_search_max(ends[j],   seqBuf, dims[0]);
                }
                free(seqBuf);
            }
            /* root scatters the lower and upper bounds to other processes */
            MPI_Scatter(bounds, 2, MPI_LONG_LONG, lower_upper, 2,
                        MPI_LONG_LONG, 0, comm);
            lowers[g] = lower_upper[0];
            uppers[g] = lower_upper[1];
        }
        if (rank == 0) free(bounds);
    }
    free(starts);

    return read_len;
}

#endif

/*----< pandana_inq_ranges() >-----------------------------------------------*/
/* Given the number of unique IDs from 0 to (numIDs-1), partition the IDs among
 * nprocs processes evenly and disjointly such that process rank is assigned
 * the range from start[rank] to ends[rank], inclusively.
 */
int
pandana_inq_ranges(int        nprocs, /* number of MPI processes */
                   int        rank,   /* MPI rank of this process */
                   long long  numIDs, /* no. unique IDs */
                   long long *starts, /* OUT */
                   long long *ends)   /* OUT */
{
    long long j, my_count;

    assert(nprocs > 0);

    /* evenly divide range of (0, 1, ... numIDs-1). If not divisible, assign
     * the extra to the lower ranks.
     */
    my_count = numIDs / nprocs;
    for (j=0; j<numIDs % nprocs; j++) {
        starts[j] = my_count * j + j;
        ends[j] = starts[j] + my_count;
    }
    for (; j<nprocs; j++) {
        starts[j] = my_count * j + numIDs % nprocs;
        ends[j] = starts[j] + my_count - 1;
    }

    return 1;
}

/* Data structure for running qsort() */
typedef struct {
    MPI_Aint block_dsp;
    int      block_len;
    int      block_idx;
} off_len;

/* comparison function used by qsort() */
static int
off_compare(const void *a, const void *b)
{
    if (((off_len*)a)->block_dsp > ((off_len*)b)->block_dsp) return  1;
    if (((off_len*)a)->block_dsp < ((off_len*)b)->block_dsp) return -1;
    return 0;
}

/*----< pandana_posix_read_dset() >------------------------------------------*/
/* Call POSIX read to read an entire dataset, decompress it, and copy to user
 * buffer. The dataset must be in 2D, chunked, compressed, with only the first
 * dimension, most significant one, chunked.
 */
ssize_t
pandana_posix_read_dset(hid_t  dset,      /* HDF5 dataset ID */
                        int    posix_fd,  /* POSIX file descriptor */
                        void  *buf)       /* OUT: user buffer */
{
    int j;
    size_t dtype_size;
    ssize_t read_len=0;
    herr_t err;
    hsize_t dims[2], chunk_dims[2];

    /* inquire dimension sizes of dset */
    inq_dset_meta(dset, &dtype_size, dims, chunk_dims);
    if (chunk_dims[1] != dims[1]) CHECK_ERROR(-1, "chunk_dims[1] == dims[1]");

    /* find the number of chunks to be read by this process */
    size_t nChunks = dims[0] / chunk_dims[0];
    if (dims[0] % chunk_dims[0]) nChunks++;

    read_len += dims[0] * dims[1] * dtype_size;

    size_t chunk_size = chunk_dims[0] * dtype_size;
    unsigned char *zipBuf;
    zipBuf = (unsigned char*) malloc(chunk_size);
    if (zipBuf == NULL) CHECK_ERROR(-1, "malloc");

    /* the last chunk may be of size less than chunk_dims[0] */
    size_t last_chunk_len = dims[0] % chunk_dims[0];
    if (last_chunk_len == 0) last_chunk_len = chunk_dims[0];
    last_chunk_len *= dtype_size;

    /* read compressed chunks, one at a time, into zipBuf, decompress it into
     * buf
     */
    hsize_t offset[2]={0,0};
    unsigned char *buf_ptr = buf;
    for (j=0; j<nChunks; j++) {
        haddr_t addr;
        hsize_t size;

        err = H5Dget_chunk_info_by_coord(dset, offset, NULL, &addr, &size);
        if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");

        lseek(posix_fd, addr, SEEK_SET);
        ssize_t len = read(posix_fd, zipBuf, size);
        if (len != size) CHECK_ERROR(-1, "read len != size");

#ifdef PANDANA_BENCHMARK
        double timing = MPI_Wtime();
#endif
        int ret;
        z_stream z_strm;
        z_strm.zalloc    = Z_NULL;
        z_strm.zfree     = Z_NULL;
        z_strm.opaque    = Z_NULL;
        z_strm.avail_in  = size;
        z_strm.next_in   = zipBuf;
        z_strm.avail_out = (j == nChunks-1) ? last_chunk_len : chunk_size;
        z_strm.next_out  = buf_ptr;
        ret = inflateInit(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateInit");
        ret = inflate(&z_strm, Z_SYNC_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) CHECK_ERROR(-1, "inflate");
        ret = inflateEnd(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateEnd");

        offset[0] += chunk_dims[0];
        buf_ptr += chunk_size;
#ifdef PANDANA_BENCHMARK
        inflate_t += MPI_Wtime() - timing;
#endif
    }
    free(zipBuf);

    return read_len;
}

/*----< pandana_mpi_read_dsets() >-------------------------------------------*/
/* Use a single MPI collective read to read multiple datasets (whole dataset),
 * decompress, and copy the decompressed chunks to user buffers, buf[].
 * Returned value is the sum of dataset sizes, not the amount of compressed
 * chunks. This is a collective call.
 */
ssize_t
pandana_mpi_read_dsets(MPI_Comm    comm,     /* MPI communicator */
                       hid_t       fd,       /* HDF5 file ID */
                       MPI_File    fh,       /* MPI file handler */
                       int         nDatasets,/* number of datasets */
                       hid_t      *dsets,    /* IN:  [nDatasets] dataset IDs */
                       void      **buf)      /* OUT: [nDatasets] read buffers */
{
    int c, d, j, k, mpi_err;
    herr_t err;
    hsize_t all_nChunks=0, max_chunk_size=0, **dims, **chunk_dims, **size;
    haddr_t **addr;
    size_t *nChunks, *dtype_size;
    ssize_t zip_len=0, read_len=0;

    if (nDatasets == 0) {
        /* This process has nothing to read, it must still participate the MPI
         * collective calls to MPI_File_set_view() and MPI_File_read_all()
         */
        mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native",
                                    MPI_INFO_NULL);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_set_view");
        mpi_err = MPI_File_read_all(fh, NULL, 0, MPI_BYTE, MPI_STATUS_IGNORE);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_read_all");
        return 1;
    }

    /* save dataset dims and chunk_dims for later use */
    dims = (hsize_t**) malloc(nDatasets * 2 * sizeof(hsize_t*));
    if (dims == NULL) CHECK_ERROR(-1, "malloc");
    chunk_dims = dims + nDatasets;
    dims[0] = (hsize_t*) malloc(nDatasets * 4 * sizeof(hsize_t));
    if (dims[0] == NULL) CHECK_ERROR(-1, "malloc");
    chunk_dims[0] = dims[0] + nDatasets * 2;
    for (d=1; d<nDatasets; d++) {
        dims[d]       =       dims[d-1] + 2;
        chunk_dims[d] = chunk_dims[d-1] + 2;
    }

    /* save dataset dims and chunk_dims for later use */
    nChunks = (size_t*) malloc(nDatasets * 2 * sizeof(size_t));
    if (nChunks == NULL) CHECK_ERROR(-1, "malloc");
    dtype_size = nChunks + nDatasets;

    /* save file offsets and sizes of individual chunks for later use */
    addr = (haddr_t**) malloc(nDatasets * sizeof(haddr_t*));
    if (addr == NULL) CHECK_ERROR(-1, "malloc");
    size = (hsize_t**) malloc(nDatasets * sizeof(hsize_t*));
    if (size == NULL) CHECK_ERROR(-1, "malloc");

    /* collect sizes of dimensions and chunk dimensions of all datasets.
     * Also calculate number of chunks and inquire their file offsets and
     * compressed sizes
     */
    for (d=0; d<nDatasets; d++) {
        inq_dset_meta(dsets[d], &dtype_size[d], dims[d], chunk_dims[d]);

        read_len += dims[d][0] * dims[d][1] * dtype_size[d];

        /* track the max chunk size */
        size_t chunk_size = chunk_dims[d][0] * chunk_dims[d][1] * dtype_size[d];
        max_chunk_size = MAX(max_chunk_size, chunk_size);

        /* find the number of chunks to be read by this process */
        nChunks[d] = dims[d][0] / chunk_dims[d][0];
        if (dims[d][0] % chunk_dims[d][0]) nChunks[d]++;

        /* accumulate number of chunks across all datasets */
        all_nChunks += nChunks[d];

        /* collect offsets and sizes of individual chunks */
        addr[d] = (haddr_t*) malloc(nChunks[d] * sizeof(haddr_t));
        if (addr[d] == NULL) CHECK_ERROR(-1, "malloc");
        size[d] = (hsize_t*) malloc(nChunks[d] * sizeof(hsize_t));
        if (size[d] == NULL) CHECK_ERROR(-1, "malloc");
        hsize_t offset[2]={0,0};
        for (j=0; j<nChunks[d]; j++) {
            err = H5Dget_chunk_info_by_coord(dsets[d], offset, NULL,
                                             &addr[d][j], &size[d][j]);
            if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");
            zip_len += size[d][j];
            offset[0] += chunk_dims[d][0];
        }
    }

    /* Note file offsets of chunks may not follow the increasing order of
     * chunk IDs read by this process. We must sort the offsets before
     * creating the MPI derived file type. First, we construct an array of
     * displacement-length-index objects for such sorting.
     */
    off_len *disp_indx = (off_len*) malloc(all_nChunks * sizeof(off_len));
    if (disp_indx == NULL) CHECK_ERROR(-1, "malloc");

    int is_mono_nondecr = 1;
    k = 0;
    for (d=0; d<nDatasets; d++) {
        for (j=0; j<nChunks[d]; j++) {
            disp_indx[k].block_dsp = (MPI_Aint)addr[d][j];  /* chunk's file offset */
            disp_indx[k].block_len = (int)size[d][j];       /* compressed chunk size */
            disp_indx[k].block_idx = j*nDatasets + d;       /* unique ID for this chunk */
            if (k > 0 && disp_indx[k].block_dsp < disp_indx[k-1].block_dsp)
                is_mono_nondecr = 0;
            k++;
        }
        free(addr[d]);
        free(size[d]);
    }
    free(addr);
    free(size);
    if (k != all_nChunks) CHECK_ERROR(-1, "k != all_nChunks");

    /* Sort chunk offsets into an increasing order, as MPI-IO requires that
     * for file view.
     * According to MPI standard Chapter 13.3, file displacements of filetype
     * must be non-negative and in a monotonically nondecreasing order. If the
     * file is opened for writing, neither the etype nor the filetype is
     * permitted to contain overlapping regions.
     */
    if (!is_mono_nondecr)
        qsort(disp_indx, all_nChunks, sizeof(off_len), off_compare);

    /* allocate array_of_blocklengths[] and array_of_displacements[] */
    MPI_Aint *chunk_dsps = (MPI_Aint*) malloc(all_nChunks * sizeof(MPI_Aint));
    if (chunk_dsps == NULL) CHECK_ERROR(-1, "malloc");
    int *chunk_lens = (int*) malloc(all_nChunks * sizeof(int));
    if (chunk_lens == NULL) CHECK_ERROR(-1, "malloc");
    for (j=0; j<all_nChunks; j++) {
        chunk_lens[j] = disp_indx[j].block_len;
        chunk_dsps[j] = disp_indx[j].block_dsp;
    }

    /* create the filetype */
    MPI_Datatype ftype;
    mpi_err = MPI_Type_create_hindexed(all_nChunks, chunk_lens, chunk_dsps,
                                       MPI_BYTE, &ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_create_hindexed");
    mpi_err = MPI_Type_commit(&ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_commit");

    free(chunk_lens);
    free(chunk_dsps);

    /* set the file view, a collective MPI-IO call */
    mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_set_view");

    mpi_err = MPI_Type_free(&ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_free");

    /* allocate a buffer for reading the compressed chunks all at once */
    unsigned char *zipBuf, *zipBuf_ptr;
    zipBuf = (unsigned char*) malloc(zip_len);
    if (zipBuf == NULL) CHECK_ERROR(-1, "malloc");
    zipBuf_ptr = zipBuf;

    /* collective read */
    mpi_err = MPI_File_read_all(fh, zipBuf, zip_len, MPI_BYTE, MPI_STATUS_IGNORE);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_read_all");

    /* decompress individual chunks into buf[] */
    unsigned char *chunkBuf = (unsigned char*) malloc(max_chunk_size);
    if (chunkBuf == NULL) CHECK_ERROR(-1, "malloc");
#ifdef PANDANA_BENCHMARK
    double timing = MPI_Wtime();
#endif
    for (j=0; j<all_nChunks; j++) {
        int ret;
        z_stream z_strm;
        z_strm.zalloc    = Z_NULL;
        z_strm.zfree     = Z_NULL;
        z_strm.opaque    = Z_NULL;
        z_strm.avail_in  = disp_indx[j].block_len;
        z_strm.next_in   = zipBuf_ptr;
        z_strm.avail_out = max_chunk_size;
        z_strm.next_out  = chunkBuf;
        ret = inflateInit(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateInit");
        ret = inflate(&z_strm, Z_SYNC_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) CHECK_ERROR(-1, "inflate");
        ret = inflateEnd(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateEnd");

        /* copy requested data to user buffer */
        d = disp_indx[j].block_idx % nDatasets;  /* dataset ID */
        c = disp_indx[j].block_idx / nDatasets;  /* chunk ID of dataset d */
        size_t len = chunk_dims[d][0] * dtype_size[d];
        size_t off = len * c;
        size_t last_len = dims[d][0] % chunk_dims[d][0];
        if (c == nChunks[d] - 1 && last_len > 0)
            /* last chunk size may not be a whole chunk */
            len = last_len * dtype_size[d];
        memcpy((unsigned char*)buf[d] + off, chunkBuf, len);
        zipBuf_ptr += disp_indx[j].block_len;
    }
#ifdef PANDANA_BENCHMARK
    inflate_t += MPI_Wtime() - timing;
#endif
    free(chunkBuf);
    free(zipBuf);
    free(disp_indx);
    free(nChunks);
    free(dims[0]);
    free(dims);

    return read_len;
}

/*----< pandana_hdf5_read_dset() >-------------------------------------------*/
/* Call H5Dread to read a whole dataset from file.
 */
ssize_t
pandana_hdf5_read_dset(hid_t  fd,    /* HDF5 file descriptor */
                       hid_t  dset,  /* dataset ID */
                       void  *buf)   /* user read buffer */
{
    herr_t  err;
    hid_t   dtype;
    hsize_t dims[2];
    size_t dtype_size;

    /* inquire dimension sizes of dset */
    inq_dset_meta(dset, &dtype_size, dims, NULL);

    /* inquire data type and size */
    dtype = H5Dget_type(dset);
    if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");

    /* read from file into buf */
    err = H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    if (err < 0) CHECK_ERROR(err, "H5Dread");

    err = H5Tclose(dtype);
    if (err < 0) CHECK_ERROR(err, "H5Tclose");

    return (dims[0] * dims[1] * dtype_size);
}

/*----< pandana_hdf5_read_subarray() >---------------------------------------*/
/* Call H5Dread to read ONE subarray of ONE dataset from file, The subarray
 * starts from index 'lower' to index 'upper'
 */
ssize_t
pandana_hdf5_read_subarray(hid_t    fd,         /* HDF5 file descriptor */
                           hid_t    dset,       /* dataset ID */
                           hsize_t  lower,      /* array index lower bound */
                           hsize_t  upper,      /* array index upper bound */
                           hid_t    xfer_plist, /* data transfer property */
                           void    *buf)        /* user read buffer */
{
    herr_t  err;
    hid_t   fspace, mspace, dtype;
    int     ndims;
    hsize_t start[2], count[2], one[2]={1,1}, dims[2];
    size_t dtype_size;
    ssize_t read_len;

    /* inquire dimension sizes of dset */
    fspace = H5Dget_space(dset);
    if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
    ndims = H5Sget_simple_extent_ndims(fspace);
    if (ndims != 2) CHECK_ERROR(-1, "H5Sget_simple_extent_ndims");
    err = H5Sget_simple_extent_dims(fspace, dims, NULL);
    if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");

    /* set subarray/hyperslab access */
    start[0] = lower;
    start[1] = 0;
    count[0] = upper - lower + 1;
    count[1] = dims[1];
    err = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, one, count);
    if (err < 0) CHECK_ERROR(err, "H5Sselect_hyperslab");

    mspace = H5Screate_simple(2, count, NULL);
    if (mspace < 0) CHECK_ERROR(mspace, "H5Screate_simple");

    /* inquire data type and size */
    dtype = H5Dget_type(dset);
    if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
    dtype_size = H5Tget_size(dtype);
    if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");

    /* calculate read size in bytes */
    read_len = count[0] * count[1] * dtype_size;

    /* read from file into buf */
    err = H5Dread(dset, dtype, mspace, fspace, xfer_plist, buf);
    if (err < 0) CHECK_ERROR(err, "H5Dread");

    err = H5Tclose(dtype);
    if (err < 0) CHECK_ERROR(err, "H5Tclose");
    err = H5Sclose(mspace);
    if (err < 0) CHECK_ERROR(err, "H5Sclose");
    err = H5Sclose(fspace);
    if (err < 0) CHECK_ERROR(err, "H5Sclose");

    return read_len;
}

/*----< pandana_hdf5_read_keys_align() >-------------------------------------*/
/* Distribute reading of key datasets among available processes and call
 * H5Dread to read the assigned key datasets in whole. All processes calculate
 * its array index boundaries into lowers[] and uppers[].  Processes with key
 * datasets assigned calculate the numbers of elements for each process rank to
 * be received from its immediately previous process into nRecvs[] and sent to
 * successive process into nSends[]. nRecvs[] is the number of elements read by
 * the immediately previous process that have the same key value of the first
 * element read by this process. In this case, we move those elements from
 * previous process to this process.  nSends[] is the number of elements to be
 * sent to the immediately successive process.  This strategy lets array
 * elements with the same key values to be partitioned into the same processes.
 * nRecvs[] and nSends[] are then MPI scatter-ed to all other processes.
 */
ssize_t
pandana_hdf5_read_keys_align(MPI_Comm   comm,
                             hid_t      fd,      /* HDF5 file ID */
                             int        nKeys,   /* number of key datasets */
                             char     **keyNames,/* IN:  [nKeys] dataset names */
                             size_t    *lowers,  /* OUT: [nKeys] read start array index */
                             size_t    *uppers,  /* OUT: [nKeys] read end   array index */
                             int       *nRecvs,  /* OUT: [nKeys] no. elements to recv */
                             int       *nSends)  /* OUT: [nKeys] no. elements to recv */
{
    int g, j, k, ndims, chunk_ndims, nChunks, my_startChunk, my_nChunks;
    int nprocs, rank, start, count, end, **recv_send, my_nKeys, my_startKey;
    herr_t err;
    hid_t fspace, chunk_plist;
    hsize_t dims[2], chunk_dims[2];
    ssize_t read_len=0;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    /* partition read workload among processes. When nprocs is larger than
     * nKeys, some processes have no data to read, but still participate
     * calls to MPI_Scatter
     */
    my_nKeys = nKeys / nprocs;
    my_startKey = my_nKeys * rank;
    if (rank < nKeys % nprocs) {
        my_startKey += rank;
        my_nKeys++;
    }
    else
        my_startKey += nKeys % nprocs;

    /* only processes got assigned read the key dataset(s) and calculate the
     * send/recv number of elements to/from the neighbors.
     */
    if (my_nKeys > 0) {
        recv_send = (int**) malloc(my_nKeys * sizeof(int*));
        if (recv_send == NULL) CHECK_ERROR(-1, "malloc");
        recv_send[0] = (int*) malloc(my_nKeys * nprocs * 2 * sizeof(int));
        if (recv_send[0] == NULL) CHECK_ERROR(-1, "malloc");
        for (g=1; g<my_nKeys; g++)
            recv_send[g] = recv_send[g-1] + nprocs * 2;
    }

    for (g=0; g<nKeys; g++) {
        /* open dataset */
        hid_t dset = H5Dopen2(fd, keyNames[g], H5P_DEFAULT);
        if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

        /* inquire dimension size and chunk dimension sizes of key dataset */
        fspace = H5Dget_space(dset);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        ndims = H5Sget_simple_extent_ndims(fspace);
        if (ndims != 2) CHECK_ERROR(-1, "H5Sget_simple_extent_ndims");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        if (dims[1] != 1) CHECK_ERROR(err, "dims[1] != 1");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");

        /* inquire chunk size along each dimension of key dataset */
        chunk_plist = H5Dget_create_plist(dset);
        if (chunk_plist < 0) CHECK_ERROR(err, "H5Dget_create_plist");
        chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        if (chunk_ndims != 2) CHECK_ERROR(-1, "chunk_ndims != 2");
        if (chunk_dims[1] != 1) CHECK_ERROR(-1, "chunk_dims[1] != 1");
        err = H5Pclose(chunk_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");

        nChunks = dims[0] / chunk_dims[0];
        if (dims[0] % chunk_dims[0]) nChunks++;

        /* calculate chunks assigned to this process */
        my_nChunks = nChunks / nprocs;
        my_startChunk = my_nChunks * rank;
        if (rank < nChunks % nprocs) {
            my_startChunk += rank;
            my_nChunks++;
        }
        else
            my_startChunk += nChunks % nprocs;

        /* calculate array start index and array element count */
        if (my_nChunks == 0)
            lowers[g] = uppers[g] = 0;
        else {
            lowers[g] = my_startChunk * chunk_dims[0];
            uppers[g] = (my_startChunk + my_nChunks) * chunk_dims[0] - 1;
            uppers[g] = MIN(uppers[g], dims[0] - 1); /* last chunk may not be a full chunk */
        }

// printf("g=%2d nChunks=%2d my_nChunks=%2d my_startChunk=%2d lowers=%zd upper=%zd chunk_dims[0]=%lld key=%s\n", g,nChunks,my_nChunks,my_startChunk,lowers[g],uppers[g],chunk_dims[0],keyNames[g]);
// if(rank==0||rank==nprocs-1)  printf("%3d: g=%2d my_nKeys=%d nChunks=%2d my_nChunks=%2d my_startChunk=%2d key=%s\n", rank,g,my_nKeys,nChunks,my_nChunks,my_startChunk,keyNames[g]);
// if (rank==0)printf("g=%2d nChunks=%3d key=%s\n",g,nChunks,keyNames[g]);

        /* keys are read concurrently by multiple processes. Each root reads
         * the entire key datasets, calculates send/receive amount between 2
         * consecutive processes
         */
        if (g >= my_startKey && g < my_startKey + my_nKeys) {
            int local_g = g - my_startKey;

            /* inquire data type and size */
            hid_t dtype = H5Dget_type(dset);
            if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
            size_t dtype_size = H5Tget_size(dtype);
            if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");

            int64_t *keyBuf = (int64_t*) malloc(dims[0] * dtype_size);
            if (keyBuf == NULL) CHECK_ERROR(-1, "malloc");
            read_len += dims[0] * dtype_size;

            /* read the entire key dataset from file into keyBuf */
            err = H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, keyBuf);
            if (err < 0) CHECK_ERROR(err, "H5Dread");
            err = H5Tclose(dtype);
            if (err < 0) CHECK_ERROR(err, "H5Tclose");

            count = nChunks / nprocs;
            for (j=0; j<nChunks % nprocs; j++) {
                start = count * j + j;
                end = start + count + 1; /* exclusive chunk ID */
                start *= chunk_dims[0];
                end   *= chunk_dims[0];
                end    = MIN(end, dims[0]); /* last chunk may not be a full chunk */

                if (j == 0)
                    recv_send[local_g][2*j] = 0; /* recv nothing from rank-1 */
                else {
                    k = start;
                    while (k > 0 && keyBuf[start] == keyBuf[k-1]) k--;
                    recv_send[local_g][2*j] = start - k; /* to receive from rank-1 */
                    if (start-k >= (count+1)*chunk_dims[0])
                        printf("ERROR: number of same key value %"PRId64" is more than chunk size=%lld\n",
                               keyBuf[start], chunk_dims[0]);
                }
                if (j < nprocs - 1 && j < nChunks - 1) {
                    k = end;
                    while (k > 0 && keyBuf[end] == keyBuf[k-1]) k--;
                    recv_send[local_g][2*j+1] = end - k; /* to send to rank+1 */
                    if (end-k >= (count+1)*chunk_dims[0])
                        printf("ERROR: number of same key value %"PRId64" is more than chunk size=%lld\n",
                               keyBuf[end], chunk_dims[0]);
                }
                else /* send nothing to rank+1 */
                    recv_send[local_g][2*j+1] = 0;
            }
            for (; j<nprocs; j++) {
                if (j >= nChunks) { /* in case nChunks < nprocs */
                    recv_send[local_g][2*j] = 0;
                    recv_send[local_g][2*j+1] = 0;
                    continue;
                }
                start = count * j + nChunks % nprocs;
                end = start + count; /* exclusive chunk ID */
                start *= chunk_dims[0];
                end   *= chunk_dims[0];
                end    = MIN(end, dims[0]); /* last chunk may not be a full chunk */

                if (j == 0)
                    recv_send[local_g][2*j] = 0; /* recv nothing from rank-1 */
                else {
                    k = start;
                    while (k > 0 && keyBuf[start] == keyBuf[k-1]) k--;
                    recv_send[local_g][2*j] = start - k; /* to recv from rank-1 */
                    if (start-k >= count*chunk_dims[0])
                        printf("ERROR: number of same key value %"PRId64" is more than chunk size=%lld\n",
                               keyBuf[start], chunk_dims[0]);
                }
                if (j == nprocs - 1)
                    recv_send[local_g][2*j+1] = 0; /* send nothing to rank+1 */
                else {
                    k = end;
                    while (k > 0 && keyBuf[end] == keyBuf[k-1]) k--;
                    recv_send[local_g][2*j+1] = end - k; /* to send to rank+1 */
                    if (end-k >= count*chunk_dims[0])
                        printf("ERROR: number of same key value %"PRId64" is more than chunk size=%lld\n",
                               keyBuf[end], chunk_dims[0]);
                }
            }
            free(keyBuf);
        }
        herr_t err = H5Dclose(dset);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");
    }

    /* nRecvs and nSends are calculated from nKeys number of key datasets,
     * which are read by nKeys number of processes in parallel.  Thus, there
     * will be nKeys calls to MPI_Scatter from nKeys number of roots. We now
     * calculate rank of root process for each MPI_Scatter. Each time
     * MPI_Scatter is called, all processes must agree on the root rank.
     */
    int root = 0;
    int rem = nKeys % nprocs;
    count = nKeys / nprocs;
    if (rem) count++;
    rem *= count;
    for (g=0,j=0; g<nKeys; g++,j++) {
        if (g > 0) { /* check if need to increment root */
            if (g == rem) { j -= rem; count--; }
            if (j % count == 0) root++;
        }
        /* only root process allocates buffers for recv_send */
        void *scatter_buf = (root == rank) ? recv_send[g-my_startKey] : NULL;
        int nElems[2];
        MPI_Scatter(scatter_buf, 2, MPI_INT, nElems, 2, MPI_INT, root, comm);
        nRecvs[g] = nElems[0]; /* number of elements to be received from rank-1 */
        nSends[g] = nElems[1]; /* number of elements to be sent     to   rank+1 */
    }
    if (my_nKeys > 0) {
        free(recv_send[0]);
        free(recv_send);
    }

    return read_len;
}

/*----< pandana_hdf5_read_subarrays_align() >--------------------------------------*/
/* Call H5Dread to read subarrays of MULTIPLE datasets, one subarray at a time.
 * The hyperslab is from lower bound to upper bound, which are aligned with the
 * chunk boundaries. Once data is read, array elements at the beginning and end
 * are exchanged between this process and the immediately previous process. The
 * same for this process and immediately successive process.
 */
ssize_t
pandana_hdf5_read_subarrays_align(MPI_Comm     comm,
                                  hid_t        fd,         /* HDF5 file ID */
                                  int          nDatasets,  /* number of datasets */
                                  const hid_t *dsets,      /* [nDatasets] dataset IDs */
                                  size_t       lower,      /* read start array index */
                                  size_t       upper,      /* read end   array index */
                                  int          nRecvs,     /* no. elements recv from rank-1 */
                                  int          nSends,     /* no. elements sent to rank+1 */
                                  hid_t        xfer_plist, /* data transfer property */
                                  void       **buf)        /* [nDatasets] read buffer */
{
    int d, nreqs, nprocs, rank;
    herr_t err;
    ssize_t read_len=0;
    MPI_Request *req;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    req = (MPI_Request*) malloc(nDatasets * 2 * sizeof(MPI_Request));
    nreqs = 0;

    /* iterate all datasets and read subarrays */
    for (d=0; d<nDatasets; d++) {
        size_t dtype_size;
        hsize_t one[2]={1,1}, starts[2], counts[2], dims[2];

        /* inquire dimension sizes of dsets[d] */
        hid_t fspace = H5Dget_space(dsets[d]);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        int ndims = H5Sget_simple_extent_ndims(fspace);
        if (ndims != 2) CHECK_ERROR(-1, "H5Sget_simple_extent_ndims");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");

        /* set subarray/hyperslab access */
        starts[0] = lower;
        starts[1] = 0;
        counts[0] = upper - lower + 1;
        counts[1] = dims[1];
        err = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, starts, NULL, one, counts);
        if (err < 0) CHECK_ERROR(err, "H5Sselect_hyperslab");
        hid_t mspace = H5Screate_simple(2, counts, NULL);
        if (mspace < 0) CHECK_ERROR(mspace, "H5Screate_simple");

        /* get data type and size */
        hid_t dtype = H5Dget_type(dsets[d]);
        if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
        dtype_size = H5Tget_size(dtype);
        if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");

        /* calculate read size in bytes */
        dtype_size *= counts[1];
        read_len += counts[0] * dtype_size;

        unsigned char *buf_ptr = (unsigned char*)buf[d] + nRecvs * dtype_size;

        /* collectively read dataset d's contents */
        err = H5Dread(dsets[d], dtype, mspace, fspace, xfer_plist, buf_ptr);
        if (err < 0) CHECK_ERROR(err, "H5Dread");

        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");
        err = H5Sclose(mspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");

        assert(nSends>=0);
        assert(nRecvs>=0);

        /* receive from rank-1 */
        if (nRecvs > 0)
            MPI_Irecv(buf[d], nRecvs*dtype_size, MPI_BYTE, rank-1, rank-1, comm, &req[nreqs++]);

        /* send to rank+1 */
        buf_ptr = buf[d] + (nRecvs + counts[0] - nSends) * dtype_size;
        if (nSends > 0)
            MPI_Isend(buf_ptr, nSends*dtype_size, MPI_BYTE, rank+1, rank, comm, &req[nreqs++]);
    }

    MPI_Waitall(nreqs, req, MPI_STATUSES_IGNORE);
    free(req);

    return read_len;
}

/*----< pandana_hdf5_read_subarrays() >--------------------------------------*/
/* Call H5Dread to read subarrays of MULTIPLE datasets, one subarray at a time.
 * The hyperslab is required to be the same for all datasets.
 */
ssize_t
pandana_hdf5_read_subarrays(hid_t        fd,         /* HDF5 file ID */
                            int          nDatasets,  /* number of datasets */
                            const hid_t *dsets,      /* [nDatasets] dataset IDs */
                            hsize_t      lower,      /* lower bound */
                            hsize_t      upper,      /* upper bound (inclusive) */
                            hid_t        xfer_plist, /* data transfer property */
                            void       **buf)        /* [nDatasets] read buffer */
{
    int d ;
    herr_t err;
    ssize_t read_len=0;

    /* iterate all datasets and read subarrays */
    for (d=0; d<nDatasets; d++) {
        hsize_t one[2]={1,1}, start[2], count[2], dims[2], chunk_dims[2];

        /* inquire dimension sizes of dsets[d] */
        hid_t fspace = H5Dget_space(dsets[d]);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        int ndims = H5Sget_simple_extent_ndims(fspace);
        if (ndims != 2) CHECK_ERROR(-1, "H5Sget_simple_extent_ndims");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");

        /* inquire chunk size along each dimension */
        hid_t chunk_plist = H5Dget_create_plist(dsets[d]);
        if (chunk_plist < 0) CHECK_ERROR(err, "H5Dget_create_plist");
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        if (chunk_ndims != 2) CHECK_ERROR(-1, "chunk_ndims != 2");
        if (chunk_dims[1] != dims[1]) CHECK_ERROR(-1, "chunk_dims[1] != dims[1]");
        err = H5Pclose(chunk_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");

        /* set subarray/hyperslab access */
        start[0] = lower;
        start[1] = 0;
        count[0] = upper - lower + 1;
        count[1] = dims[1];
        err = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, one,
                                  count);
        if (err < 0) CHECK_ERROR(err, "H5Sselect_hyperslab");
        hid_t mspace = H5Screate_simple(2, count, NULL);
        if (mspace < 0) CHECK_ERROR(mspace, "H5Screate_simple");

        /* get data type and size */
        hid_t dtype = H5Dget_type(dsets[d]);
        if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
        size_t dtype_size = H5Tget_size(dtype);
        if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");

        /* calculate read size in bytes */
        read_len += count[0] * count[1] * dtype_size;

        /* collectively read dataset d's contents */
        err = H5Dread(dsets[d], dtype, mspace, fspace, xfer_plist, buf[d]);
        if (err < 0) CHECK_ERROR(err, "H5Dread");

        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");
        err = H5Sclose(mspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");
    }

    return read_len;
}

/*----< pandana_mpi_read_subarray() >----------------------------------------*/
/* Call MPI_File_read_all to read a subarray of a dataset. It reads all raw
 * chunks intersecting the subarray, decompress the raw chunks, and copy the
 * requested data to user buffer.
 */
ssize_t
pandana_mpi_read_subarray(hid_t          fd,    /* HDF5 file descriptor */
                          MPI_File       fh,    /* MPI file handler */
                          const hid_t    dset,  /* dataset ID */
                          hsize_t        lower, /* array index lower bound */
                          hsize_t        upper, /* array index upper bound */
                          unsigned char *buf)   /* user read buffer */
{
    int j, mpi_err;
    size_t dtype_size;
    herr_t  err;
    hsize_t dims[2], chunk_dims[2];
    ssize_t zip_len=0, read_len = 0;

    /* inquire dimension sizes of dset */
    inq_dset_meta(dset, &dtype_size, dims, chunk_dims);

    read_len = (upper - lower + 1) * dims[1] * dtype_size;

    /* find the number of chunks to be read by this process */
    hsize_t nChunks = (upper / chunk_dims[0]) - (lower / chunk_dims[0]) + 1;

    /* Note file offsets of chunks may not follow the increasing order of chunk
     * IDs read by this process. We must sort the offsets before creating a
     * file type. Construct an array of off-len-indx for such sorting.
     */
    off_len *disp_indx = (off_len*) malloc(nChunks * sizeof(off_len));
    if (disp_indx == NULL) CHECK_ERROR(-1, "malloc");

    /* calculate the logical position of chunk's first element. See
     * https://hdf5.io/develop/group___h5_d.html#ga408a49c6ec59c5b65ce4c791f8d26cb0
     */
    int is_mono_nondecr = 1;
    hsize_t offset[2];
    offset[0] = (lower / chunk_dims[0]) * chunk_dims[0];
    offset[1] = 0;
    for (j=0; j<nChunks; j++) {
        hsize_t size;
        haddr_t addr;
        err = H5Dget_chunk_info_by_coord(dset, offset, NULL, &addr, &size);
       if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");

        disp_indx[j].block_dsp = (MPI_Aint)addr;  /* chunk's file offset */
        disp_indx[j].block_len = (int)size;       /* compressed chunk size */
        disp_indx[j].block_idx = j;               /* chunk ID to be read by this process */
        if (j > 0 && disp_indx[j].block_dsp < disp_indx[j-1].block_dsp)
            is_mono_nondecr = 0;
        zip_len += size;
        offset[0] += chunk_dims[0];
    }

    /* Sort chunk offsets into an increasing order, as required by MPI-IO when
     * setting the file view.
     * According to MPI standard Chapter 13.3, file displacements of
     * filetype must be non-negative and in a monotonically nondecreasing
     * order. If the file is opened for writing, neither the etype nor the
     * filetype is permitted to contain overlapping regions.
     */
    if (!is_mono_nondecr)
        qsort(disp_indx, nChunks, sizeof(off_len), off_compare);

    /* allocate array_of_blocklengths[] and array_of_displacements[] */
    MPI_Aint *chunk_dsps = (MPI_Aint*) malloc(nChunks * sizeof(MPI_Aint));
    if (chunk_dsps == NULL) CHECK_ERROR(-1, "malloc");
    int *chunk_lens = (int*) malloc(nChunks * sizeof(int));
    if (chunk_lens == NULL) CHECK_ERROR(-1, "malloc");
    for (j=0; j<nChunks; j++) {
        chunk_lens[j] = disp_indx[j].block_len;
        chunk_dsps[j] = disp_indx[j].block_dsp;
    }

    /* construct the filetype */
    MPI_Datatype ftype;
    mpi_err = MPI_Type_create_hindexed(nChunks, chunk_lens, chunk_dsps,
                                       MPI_BYTE, &ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_create_hindexed");
    mpi_err = MPI_Type_commit(&ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_commit");

    free(chunk_lens);
    free(chunk_dsps);

    /* set the file view, which is a collective MPI-IO call */
    mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, ftype, "native",
                                MPI_INFO_NULL);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_set_view");

    mpi_err = MPI_Type_free(&ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_free");

    /* allocate buffer for reading all the compressed chunks at once */
    unsigned char *zipBuf, *zipBuf_ptr;
    zipBuf = (unsigned char*) malloc(zip_len);
    if (zipBuf == NULL) CHECK_ERROR(-1, "malloc");
    zipBuf_ptr = zipBuf;

    /* collective read */
    mpi_err = MPI_File_read_all(fh, zipBuf, zip_len, MPI_BYTE, MPI_STATUS_IGNORE);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_read_all");

    /* decompress each chunk into buf */
    size_t whole_chunk_size = chunk_dims[0] * chunk_dims[1] * dtype_size;
    unsigned char *whole_chunk = (unsigned char*) malloc(whole_chunk_size);
    if (whole_chunk == NULL) CHECK_ERROR(-1, "malloc");
#ifdef PANDANA_BENCHMARK
    double timing = MPI_Wtime();
#endif
    for (j=0; j<nChunks; j++) {
        int ret;
        z_stream z_strm;
        z_strm.zalloc    = Z_NULL;
        z_strm.zfree     = Z_NULL;
        z_strm.opaque    = Z_NULL;
        z_strm.avail_in  = disp_indx[j].block_len;
        z_strm.next_in   = zipBuf_ptr;
        z_strm.avail_out = whole_chunk_size;
        z_strm.next_out  = whole_chunk;
        ret = inflateInit(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateInit");
        ret = inflate(&z_strm, Z_SYNC_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) CHECK_ERROR(-1, "inflate");
        ret = inflateEnd(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateEnd");

        /* copy requested data to user buffer */
        if (disp_indx[j].block_idx == 0) { /* first chunk */
            size_t off = lower % chunk_dims[0];
            size_t len;
            if (nChunks == 1)
                len = upper - lower + 1;
            else
                len = chunk_dims[0] - off;
            len *= chunk_dims[1] * dtype_size;
            off *= chunk_dims[1] * dtype_size;
            memcpy(buf, whole_chunk + off, len);
        }
        else if (disp_indx[j].block_idx == nChunks - 1) { /* last chunk */
            size_t len = (upper+1) % chunk_dims[0];
            if (len == 0) len = chunk_dims[0];
            size_t off = upper + 1 - len - lower;
            off *= chunk_dims[1] * dtype_size;
            len *= chunk_dims[1] * dtype_size;
            memcpy(buf + off, whole_chunk, len);
        }
        else { /* middle chunk, copy the full chunk */
            size_t off = chunk_dims[0] - lower % chunk_dims[0];
            off += (disp_indx[j].block_idx - 1) * chunk_dims[0];
            off *= chunk_dims[1] * dtype_size;
            memcpy(buf + off, whole_chunk, whole_chunk_size);
        }
        zipBuf_ptr += disp_indx[j].block_len;
    }
#ifdef PANDANA_BENCHMARK
    inflate_t += MPI_Wtime() - timing;
#endif
    free(whole_chunk);
    free(zipBuf);
    free(disp_indx);

    return read_len;
}

/*----< pandana_mpi_read_subarrays() >---------------------------------------*/
/* Call MPI_File_read_all() to read subarrays of multiple datasets. One
 * subarray read per dataset. For each subarray read, all raw chunks
 * intersecting a subarray are read in a single call of MPI_File_read_all() and
 * decompressed into user buffers
 */
ssize_t
pandana_mpi_read_subarrays(hid_t        fd,        /* HDF5 file descriptor */
                          MPI_File      fh,        /* MPI file handler */
                          int           nDatasets, /* number of datasets */
                          const hid_t  *dsets,     /* [nDatasets] dataset IDs */
                          hsize_t       lower,     /* array index lower bound */
                          hsize_t       upper,     /* array index upper bound */
                          void        **buf)       /* [nDatasets] read buffer */
{
    int j, d, mpi_err;
    herr_t  err;
    ssize_t zip_len, read_len=0;
    unsigned char **buf_ptr = (unsigned char**) buf;

    for (d=0; d<nDatasets; d++) {
        size_t dtype_size;
        hsize_t dims[2], chunk_dims[2];

        /* find metadata of all the chunks of this dataset */
        inq_dset_meta(dsets[d], &dtype_size, dims, chunk_dims);

        /* calculate buffer read size in bytes */
        read_len += (upper - lower + 1) * dims[1] * dtype_size;

        /* find the number of chunks to be read by this process */
        hsize_t nChunks = (upper / chunk_dims[0]) - (lower / chunk_dims[0]) + 1;

        /* Note file offsets of chunks may not follow the increasing order of
         * chunk IDs read by this process. We must sort the offsets before
         * creating a file type. Construct an array of off-len-indx for such
         * sorting.
         */
        off_len *disp_indx = (off_len*) malloc(nChunks * sizeof(off_len));
        if (disp_indx == NULL) CHECK_ERROR(-1, "malloc");

        /* calculate the logical position of chunk's first element. See
         * https://hdf5.io/develop/group___h5_d.html#ga408a49c6ec59c5b65ce4c791f8d26cb0
         */
        int is_mono_nondecr = 1;
        hsize_t offset[2];
        offset[0] = (lower / chunk_dims[0]) * chunk_dims[0];
        offset[1] = 0;
        zip_len = 0;
        for (j=0; j<nChunks; j++) {
            hsize_t size;
            haddr_t addr;
            err = H5Dget_chunk_info_by_coord(dsets[d], offset, NULL, &addr, &size);
            if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");
            disp_indx[j].block_dsp = (MPI_Aint)addr;  /* chunk's file offset */
            disp_indx[j].block_len = (int)size;       /* compressed chunk size */
            disp_indx[j].block_idx = j;               /* chunk ID to be read by this process */
            if (j > 0 && disp_indx[j].block_dsp < disp_indx[j-1].block_dsp)
                is_mono_nondecr = 0;
            zip_len += size;
            offset[0] += chunk_dims[0];
        }

        /* Sort chunk offsets into an increasing order, as MPI-IO requires that
         * for file view.
         * According to MPI standard Chapter 13.3, file displacements of
         * filetype must be non-negative and in a monotonically nondecreasing
         * order. If the file is opened for writing, neither the etype nor the
         * filetype is permitted to contain overlapping regions.
         */
        if (!is_mono_nondecr)
            qsort(disp_indx, nChunks, sizeof(off_len), off_compare);

        /* allocate array_of_blocklengths[] and array_of_displacements[] */
        MPI_Aint *chunk_dsps = (MPI_Aint*) malloc(nChunks * sizeof(MPI_Aint));
        if (chunk_dsps == NULL) CHECK_ERROR(-1, "malloc");
        int *chunk_lens = (int*) malloc(nChunks * sizeof(int));
        if (chunk_lens == NULL) CHECK_ERROR(-1, "malloc");
        for (j=0; j<nChunks; j++) {
            chunk_lens[j] = disp_indx[j].block_len;
            chunk_dsps[j] = disp_indx[j].block_dsp;
        }

        /* create the filetype */
        MPI_Datatype ftype;
        mpi_err = MPI_Type_create_hindexed(nChunks, chunk_lens, chunk_dsps,
                                           MPI_BYTE, &ftype);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_create_hindexed");
        mpi_err = MPI_Type_commit(&ftype);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_commit");

        free(chunk_lens);
        free(chunk_dsps);

        /* set the file view, a collective MPI-IO call */
        mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, ftype, "native",
                                    MPI_INFO_NULL);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_set_view");

        mpi_err = MPI_Type_free(&ftype);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_free");

        /* allocate buffer for reading the compressed chunks all at once */
        unsigned char *zipBuf, *zipBuf_ptr;
        zipBuf = (unsigned char*) malloc(zip_len);
        if (zipBuf == NULL) CHECK_ERROR(-1, "malloc");
        zipBuf_ptr = zipBuf;

        /* collective read */
        mpi_err = MPI_File_read_all(fh, zipBuf, zip_len, MPI_BYTE, MPI_STATUS_IGNORE);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_read_all");

        /* decompress each chunk into buf[d] */
        size_t whole_chunk_size = chunk_dims[0] * chunk_dims[1] * dtype_size;
        unsigned char *whole_chunk = (unsigned char*) malloc(whole_chunk_size);
        if (whole_chunk == NULL) CHECK_ERROR(-1, "malloc");
#ifdef PANDANA_BENCHMARK
        double timing = MPI_Wtime();
#endif
        for (j=0; j<nChunks; j++) {
            int ret;
            z_stream z_strm;
            z_strm.zalloc    = Z_NULL;
            z_strm.zfree     = Z_NULL;
            z_strm.opaque    = Z_NULL;
            z_strm.avail_in  = disp_indx[j].block_len;
            z_strm.next_in   = zipBuf_ptr;
            z_strm.avail_out = whole_chunk_size;
            z_strm.next_out  = whole_chunk;
            ret = inflateInit(&z_strm);
            if (ret != Z_OK) CHECK_ERROR(-1, "inflateInit");
            ret = inflate(&z_strm, Z_SYNC_FLUSH);
            if (ret != Z_OK && ret != Z_STREAM_END) CHECK_ERROR(-1, "inflate");
            ret = inflateEnd(&z_strm);
            if (ret != Z_OK) CHECK_ERROR(-1, "inflateEnd");

            /* copy requested data to user buffer */
            if (disp_indx[j].block_idx == 0) { /* first chunk */
                size_t off = lower % chunk_dims[0];
                size_t len;
                if (nChunks == 1)
                    len = upper - lower + 1;
                else
                    len = chunk_dims[0] - off;
                len *= chunk_dims[1] * dtype_size;
                off *= chunk_dims[1] * dtype_size;
                memcpy(buf_ptr[d], whole_chunk + off, len);
            }
            else if (disp_indx[j].block_idx == nChunks - 1) { /* last chunk */
                size_t len = (upper+1) % chunk_dims[0];
                if (len == 0) len = chunk_dims[0];
                size_t off = upper + 1 - len - lower;
                off *= chunk_dims[1] * dtype_size;
                len *= chunk_dims[1] * dtype_size;
                memcpy(buf_ptr[d] + off, whole_chunk, len);
            }
            else { /* middle chunk, copy the full chunk */
                size_t off = chunk_dims[0] - lower % chunk_dims[0];
                off += (disp_indx[j].block_idx - 1) * chunk_dims[0];
                off *= chunk_dims[1] * dtype_size;
                memcpy(buf_ptr[d] + off, whole_chunk, whole_chunk_size);
            }
            zipBuf_ptr += disp_indx[j].block_len;
        }
#ifdef PANDANA_BENCHMARK
        inflate_t += MPI_Wtime() - timing;
#endif
        free(whole_chunk);
        free(zipBuf);
        free(disp_indx);
    }
    return read_len;
}

/*----< pandana_mpi_read_subarrays_aggr() >-----------------------------------*/
/* A single call to MPI_File_read_all() to read subarrays of multiple datasets.
 * All raw chunks containing data of subarray requests are read into memory
 * buffers, decompressed, and copied to user buffers.
 */
ssize_t
pandana_mpi_read_subarrays_aggr(hid_t         fd,        /* HDF5 file descriptor */
                                MPI_File      fh,        /* MPI file handler */
                                int           nDatasets, /* number of datasets */
                                const hid_t  *dsets,     /* [nDatasets] dataset IDs */
                                hsize_t       lower,     /* lower bound */
                                hsize_t       upper,     /* upper bound (inclusive) */
                                void        **buf)       /* [nDatasets] read buffer */
{
    int d, j, k, mpi_err;
    herr_t  err;
    hsize_t **size;
    haddr_t **addr;
    size_t all_nChunks=0, *nChunks, max_chunk_size=0, *dtype_size;
    ssize_t zip_len=0, read_len=0;
    unsigned char **buf_ptr = (unsigned char**) buf;

    dtype_size = (size_t*) malloc(nDatasets * 2 * sizeof(size_t));
    if (dtype_size == NULL) CHECK_ERROR(-1, "malloc");
    nChunks = dtype_size + nDatasets;

    hsize_t **dims;       /* [nDatasets][2] */
    hsize_t **chunk_dims; /* [nDatasets][2] */
    dims = (hsize_t**) malloc(nDatasets * 2 * sizeof(hsize_t*));
    if (dims == NULL) CHECK_ERROR(-1, "malloc");
    chunk_dims = dims + nDatasets;
    dims[0] = (hsize_t*) malloc(nDatasets * 4 * sizeof(hsize_t));
    if (dims[0] == NULL) CHECK_ERROR(-1, "malloc");
    chunk_dims[0] = dims[0] + nDatasets * 2;
    for (d=1;  d<nDatasets; d++) {
              dims[d] =       dims[d-1] + 2;
        chunk_dims[d] = chunk_dims[d-1] + 2;
    }

    /* allocate space to save file offsets and sizes of individual chunks */
    addr = (haddr_t**) malloc(nDatasets * sizeof(haddr_t*));
    if (addr == NULL) CHECK_ERROR(-1, "malloc");
    size = (hsize_t**) malloc(nDatasets * sizeof(hsize_t*));
    if (size == NULL) CHECK_ERROR(-1, "malloc");

    /* collect metadata of all chunks of all datasets, including number of
     * chunks and their file offsets and compressed sizes
     */
    for (d=0; d<nDatasets; d++) {

        inq_dset_meta(dsets[d], &dtype_size[d], dims[d], chunk_dims[d]);

        if (chunk_dims[d][1] != dims[d][1])
            CHECK_ERROR(-1, "chunk_dims[d][1] != dims[d][1]");

        /* find the max chunk size among all datasets */
        size_t buf_len = chunk_dims[d][0] * chunk_dims[d][1] * dtype_size[d];
        max_chunk_size = MAX(max_chunk_size, buf_len);

        /* calculate buffer read size in bytes */
        read_len += (upper - lower + 1) * dims[d][1] * dtype_size[d];

        /* find the number of chunks to be read by this process */
        nChunks[d] = (upper / chunk_dims[d][0]) - (lower / chunk_dims[d][0]) + 1;
        all_nChunks += nChunks[d];

        /* collect offsets of all chunks of this dataset */
        addr[d] = (haddr_t*) malloc(nChunks[d] * sizeof(haddr_t));
        if (addr[d] == NULL) CHECK_ERROR(-1, "malloc");
        size[d] = (hsize_t*) malloc(nChunks[d] * sizeof(hsize_t));
        if (size[d] == NULL) CHECK_ERROR(-1, "malloc");
        hsize_t offset[2];
        offset[0] = (lower / chunk_dims[d][0]) * chunk_dims[d][0];
        offset[1] = 0;
        for (j=0; j<nChunks[d]; j++) {
            err = H5Dget_chunk_info_by_coord(dsets[d], offset, NULL,
                                             &addr[d][j], &size[d][j]);
            if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");
            zip_len += size[d][j];
            offset[0] += chunk_dims[d][0];
        }
    }

    /* Note file offsets of chunks may not follow the increasing order of chunk
     * IDs read by this process. We must sort the chunk offsets before creating
     * a file type. Construct an array of offset-length-index for such sorting.
     */
    off_len *disp_indx = (off_len*) malloc(all_nChunks * sizeof(off_len));
    if (disp_indx == NULL) CHECK_ERROR(-1, "malloc");

    int is_mono_nondecr = 1;
    k = 0;
    for (d=0; d<nDatasets; d++) {
        for (j=0; j<nChunks[d]; j++) {
            disp_indx[k].block_dsp = (MPI_Aint)addr[d][j]; /* chunk's file offset */
            disp_indx[k].block_len = (int)size[d][j];      /* compressed chunk size */
            disp_indx[k].block_idx = j*nDatasets + d;      /* unique ID for this chunk */
            if (k > 0 && disp_indx[k].block_dsp < disp_indx[k-1].block_dsp)
                is_mono_nondecr = 0; /* decreasing order found */
            k++;
        }
        free(addr[d]);
        free(size[d]);
    }
    free(addr);
    free(size);
    if (k != all_nChunks) CHECK_ERROR(-1, "k != all_nChunks");

    /* Sort chunk offsets into an increasing order, as MPI-IO requires that
     * for file view. According to MPI standard Chapter 13.3, file
     * displacements of filetype must be non-negative and in a monotonically
     * nondecreasing order. If the file is opened for writing, neither the
     * etype nor the filetype is permitted to contain overlapping regions.
     */
    if (!is_mono_nondecr)
        qsort(disp_indx, all_nChunks, sizeof(off_len), off_compare);

    /* allocate array_of_blocklengths[] and array_of_displacements[] */
    MPI_Aint *chunk_dsps = (MPI_Aint*) malloc(all_nChunks * sizeof(MPI_Aint));
    if (chunk_dsps == NULL) CHECK_ERROR(-1, "malloc");
    int *chunk_lens = (int*) malloc(all_nChunks * sizeof(int));
    if (chunk_lens == NULL) CHECK_ERROR(-1, "malloc");
    for (j=0; j<all_nChunks; j++) {
        chunk_lens[j] = disp_indx[j].block_len;
        chunk_dsps[j] = disp_indx[j].block_dsp;
    }

    /* create the filetype */
    MPI_Datatype ftype;
    mpi_err = MPI_Type_create_hindexed(all_nChunks, chunk_lens, chunk_dsps,
                                       MPI_BYTE, &ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_create_hindexed");
    mpi_err = MPI_Type_commit(&ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_commit");

    free(chunk_lens);
    free(chunk_dsps);

    /* set the file view, a collective MPI-IO call */
    mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_set_view");

    mpi_err = MPI_Type_free(&ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_free");

    /* allocate buffer for reading the compressed chunks all at once */
    unsigned char *zipBuf, *zipBuf_ptr;
    zipBuf = (unsigned char*) malloc(zip_len);
    if (zipBuf == NULL) CHECK_ERROR(-1, "malloc");
    zipBuf_ptr = zipBuf;

    /* collective read */
    mpi_err = MPI_File_read_all(fh, zipBuf, zip_len, MPI_BYTE, MPI_STATUS_IGNORE);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_read_all");

    /* decompress individual chunks into whole_chunk and copy to buf[d] */
    unsigned char *whole_chunk = (unsigned char*) malloc(max_chunk_size);
    if (whole_chunk == NULL) CHECK_ERROR(-1, "malloc");
#ifdef PANDANA_BENCHMARK
        double timing = MPI_Wtime();
#endif
    for (j=0; j<all_nChunks; j++) {
        int ret;
        z_stream z_strm;
        z_strm.zalloc    = Z_NULL;
        z_strm.zfree     = Z_NULL;
        z_strm.opaque    = Z_NULL;
        z_strm.avail_in  = disp_indx[j].block_len;
        z_strm.next_in   = zipBuf_ptr;
        z_strm.avail_out = max_chunk_size;
        z_strm.next_out  = whole_chunk;
        ret = inflateInit(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateInit");
        ret = inflate(&z_strm, Z_SYNC_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) CHECK_ERROR(-1, "inflate");
        ret = inflateEnd(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateEnd");

        /* copy requested data to user buffer */
        d = disp_indx[j].block_idx % nDatasets;  /* dataset ID */
        k = disp_indx[j].block_idx / nDatasets;  /* chunk ID of dataset d */
        /* copy requested data to user buffer */
        if (k == 0) { /* dataset d's first chunk */
            size_t off = lower % chunk_dims[d][0];
            size_t len;
            if (nChunks[d] == 1)
                len = upper - lower + 1;
            else
                len = chunk_dims[d][0] - off;
            len *= dtype_size[d];
            off *= dtype_size[d];
            memcpy(buf_ptr[d], whole_chunk + off, len);
        }
        else if (k == nChunks[d] - 1) { /* dataset d's last chunk */
            size_t len = (upper+1) % chunk_dims[d][0];
            if (len == 0) len = chunk_dims[d][0];
            size_t off = upper + 1 - len - lower;
            off *= dtype_size[d];
            len *= dtype_size[d];
            memcpy(buf_ptr[d] + off, whole_chunk, len);
        }
        else { /* middle chunk, copy the full chunk */
            size_t len = chunk_dims[d][0] * dtype_size[d];
            size_t off = chunk_dims[d][0] - lower % chunk_dims[d][0];
            off += (k - 1) * chunk_dims[d][0];
            off *= dtype_size[d];
            memcpy(buf_ptr[d] + off, whole_chunk, len);
        }
        zipBuf_ptr += disp_indx[j].block_len;
    }
#ifdef PANDANA_BENCHMARK
    inflate_t += MPI_Wtime() - timing;
#endif
    free(whole_chunk);
    free(zipBuf);
    free(disp_indx);
    free(dtype_size);
    free(dims[0]);
    free(dims);

    return read_len;
}

#ifdef PANDANA_BENCHMARK
/* return the smallest index i, such that base[i] >= key */
static size_t
binary_search_min(long long  key,
                  int64_t   *base,
                  size_t     nmemb)
{
    size_t low=0, high=nmemb;
    while (low != high) {
        size_t mid = (low + high) / 2;
        if (base[mid] < key)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

/* return the largest index i, such that base[i] <= key */
static size_t
binary_search_max(long long  key,
                  int64_t   *base,
                  size_t     nmemb)
{
    size_t low=0, high=nmemb;
    while (low != high) {
        size_t mid = (low + high) / 2;
        if (base[mid] <= key)
            low = mid + 1;
        else
            high = mid;
    }
    return (low-1);
}
#endif

/*----< inq_dset_meta() >----------------------------------------------------*/
static int
inq_dset_meta(hid_t    dset,        /* HDF5 dataset ID */
              size_t  *dtype_size,  /* OUT datatype size in bytes */
              hsize_t *dims,        /* OUT [2] dimension sizes */
              hsize_t *chunk_dims)  /* OUT [2] chunk dimension sizes */
{
    herr_t err;

    /* inquire dimension sizes of dset */
    if (dims != NULL) {
        hid_t fspace = H5Dget_space(dset);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");
    }

    /* inquire chunk size along each dimension */
    if (chunk_dims != NULL) {
        hid_t chunk_plist = H5Dget_create_plist(dset);
        if (chunk_plist < 0) CHECK_ERROR(chunk_plist, "H5Dget_create_plist");
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        if (chunk_ndims != 2) CHECK_ERROR(-1, "H5Pget_chunk: chunk_ndims != 2");
        err = H5Pclose(chunk_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");
    }

    /* get data type and size */
    if (dtype_size != NULL) {
        hid_t dtype = H5Dget_type(dset);
        if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
        *dtype_size = H5Tget_size(dtype);
        if (*dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");;
        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");
    }
    return 1;
}

/*----< pandana_read_keys() >------------------------------------------------*/
/* Read multiple key datasets, one per group, and use them to calculate the
 * lower and upper boundaries responsible by each process into lowers[] and
 * uppers[]
 */
ssize_t
pandana_read_keys(MPI_Comm   comm,       /* MPI communicator */
                  hid_t      fd,         /* HDF5 file ID */
                  MPI_File   fh,         /* MPI file handler */
                  int        nGroups,    /* number of key datasets */
                  char     **key_names,  /* [nGroups] key dataset names */
                  long long  numIDs,     /* number of globally unique IDs */
                  size_t    *lowers,     /* OUT: [nGroups] */
                  size_t    *uppers)     /* OUT: [nGroups] inclusive bound */
{
    int j, g, k, p, nprocs, rank, my_startGrp, my_nGroups;
    ssize_t read_len;
    long long **bounds, lower_upper[2];

#ifdef PANDANA_BENCHMARK
if (seq_opt < 3)
    return hdf5_read_keys(comm, fd, nGroups, key_names, numIDs, lowers, uppers);
#endif

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    /* As each of the nGroups groups contains one key dataset, we assign
     * nGroups key datasets among MIN(nprocs, nGroups) number of processes, so
     * they can be read in parallel. The processes got assigned make a single
     * call to MPI collective read to read all the assigned key datasets,
     * decompress, calculate boundaries of all processes, and MPI scatter them
     * to other processes. Note these are done in parallel, i.e. workload of
     * reads and computation is distributed among MIN(nprocs, nGroups) number
     * of processes.
     */

    /* starts[rank] and ends[rank] store the starting and ending event IDs that
     * are responsible by process rank
     */
    long long *starts, *ends;
    starts = (long long*) malloc(nprocs * 2 * sizeof(long long));
    if (starts == NULL) CHECK_ERROR(-1, "malloc");
    ends = starts + nprocs;

    /* calculate the range of unique IDs responsible by all process and store
     * them in starts[nprocs] and ends[nprocs] */
    pandana_inq_ranges(nprocs, rank, numIDs, starts, ends);

    /* partition read workload among processes. When nprocs is larger than
     * nGroups, some processes have no data to read, but still participate
     * calls to MPI_Scatter
     */
    my_nGroups = nGroups / nprocs;
    my_startGrp = my_nGroups * rank;
    if (rank < nGroups % nprocs) {
        my_startGrp += rank;
        my_nGroups++;
    }
    else
        my_startGrp += nGroups % nprocs;

    /* only processes got assigned read the key dataset(s) and calculate the
     * lower and upper bounds
     */
    if (my_nGroups > 0) {
        bounds = (long long**) malloc(my_nGroups * sizeof(long long*));
        if (bounds == NULL) CHECK_ERROR(-1, "malloc");
        bounds[0] = (long long*) malloc(my_nGroups * nprocs * 2 * sizeof(long long));
        if (bounds[0] == NULL) CHECK_ERROR(-1, "malloc");
        for (g=1; g<my_nGroups; g++)
            bounds[g] = bounds[g-1] + nprocs * 2;
    }
    key_names += my_startGrp;

#ifdef PANDANA_BENCHMARK
if (seq_opt == 3) {
#endif
    /* allocate arrays of dataset IDs, dimension 0 sizes, read buffers */
    hid_t *dsets = (hid_t*) malloc(my_nGroups*sizeof(hid_t));
    if (dsets == NULL) CHECK_ERROR(-1, "malloc");
    hsize_t *dim0 = (hsize_t*) malloc(my_nGroups * sizeof(hsize_t));
    if (dim0 == NULL) CHECK_ERROR(-1, "malloc");
    void **seqBuf = (void**) malloc(my_nGroups*sizeof(void*));
    if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");

    for (g=0; g<my_nGroups; g++) {
        /* open dataset */
        dsets[g] = H5Dopen2(fd, key_names[g], H5P_DEFAULT);
        if (dsets[g] < 0) CHECK_ERROR(dsets[g], "H5Dopen2");
        hsize_t dims[2];
        size_t dtype_size;
        inq_dset_meta(dsets[g], &dtype_size, dims, NULL);
        if (dims[1] != 1) CHECK_ERROR(err, "dims[1] != 1");
        dim0[g] = dims[0];
        size_t buf_len = dims[0] * dtype_size;
        /* allocate read buffers */
        seqBuf[g] = (void*) malloc(buf_len);
        if (seqBuf[g] == NULL) CHECK_ERROR(-1, "malloc");
    }

    /* collectively read all key datasets into seqBuf[] */
    read_len = pandana_mpi_read_dsets(comm, fd, fh, my_nGroups, dsets, seqBuf);

    /* calculate lower and upper bounds for all processes */
    for (g=0; g<my_nGroups; g++) {
        /* Assume starts[p] always >= starts[p-1] */
        for(k = 0; k < dim0[g]; k++){
            while(((int64_t*)seqBuf[g])[k] >= starts[p]){
                bounds[g][p << 1] = k;  /* Start of process p */
                p++;    /* Now looking for start of the next process */
            }
        }
        for(;p < nprocs; p++){ /* k may run out first */
            bounds[g][p << 1] = k;
        }
        for(p = 1; p < nprocs; p++){ /* (end of of p - 1) = (start of p) - 1 */
            bounds[g][(p << 1) - 1] = bounds[g][p << 1] - 1;  
        }
        /* Assume end of rank (np - 1) is always the last element */
        bounds[g][(p << 1) - 1] = dim0[g];

        free(seqBuf[g]);
        herr_t err = H5Dclose(dsets[g]);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");
    }
    free(seqBuf);
    free(dim0);
    free(dsets);
#ifdef PANDANA_BENCHMARK
}
else if (seq_opt == 4) {
    read_len = 0;
    for (g=0; g<my_nGroups; g++) {
        herr_t err;
        /* open dataset */
        hid_t dset = H5Dopen2(fd, key_names[g], H5P_DEFAULT);
        if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");
        hsize_t dims[2];
        size_t dtype_size;
        inq_dset_meta(dset, &dtype_size, dims, NULL);
        if (dims[1] != 1) CHECK_ERROR(err, "dims[1] != 1");
        size_t buf_len = dims[0] * dtype_size;
        /* allocate read buffers */
        void *seqBuf = (void*) malloc(buf_len);
        if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");

        /* pandana_posix_read_dset() is an independent call */
        read_len += pandana_posix_read_dset(dset, posix_fd, seqBuf);

        /* calculate lower and upper bounds for all processes */
        /* Assume starts[p] always >= starts[p-1] */
        for(k = 0; k < dims[0]; k++){
            while(((int64_t*)seqBuf)[k] >= starts[p]){
                bounds[g][p << 1] = k;  /* Start of process p */
                p++;    /* Now looking for start of the next process */
            }
        }
        for(;p < nprocs; p++){ /* k may run out first */
            bounds[g][p << 1] = k;
        }
        for(p = 1; p < nprocs; p++){ /* (end of of p - 1) = (start of p) - 1 */
            bounds[g][(p << 1) - 1] = bounds[g][p << 1] - 1;  
        }
        /* Assume end of rank (np - 1) is always the last element */
        bounds[g][(p << 1) - 1] = dims[0];

        free(seqBuf);
        err = H5Dclose(dset);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");
    }
}
#endif

    /* Lower and upper bounds are calculated from nGroups number of key
     * datasets, which are read by nGroups number of processes in parallel.
     * Thus, there will be nGroups calls to MPI_Scatter from nGroups number of
     * roots. We now calculate rank of root process for each MPI_Scatter. Each
     * time MPI_Scatter is called, all processes must agree on the root rank.
     */
    int root = 0;
    int rem = nGroups % nprocs;
    int count = nGroups / nprocs;
    if (rem) count++;
    rem *= count;
    for (g=0,j=0; g<nGroups; g++,j++) {
        if (g > 0) { /* check if need to increment root */
            if (g == rem) { j -= rem; count--; }
            if (j % count == 0) root++;
        }
        /* only root process allocates buffers for bounds */
        void *scatter_buf = (root == rank) ? bounds[g-my_startGrp] : NULL;
        MPI_Scatter(scatter_buf, 2, MPI_LONG_LONG, lower_upper, 2,
                    MPI_LONG_LONG, root, comm);
        lowers[g] = lower_upper[0];
        uppers[g] = lower_upper[1];
    }
    if (my_nGroups > 0) {
        free(bounds[0]);
        free(bounds);
    }
    /* free allocated memory space */
    if (starts != NULL) free(starts);

    return read_len;
}

/*----< pandana_data_parallelism() >-----------------------------------------*/
ssize_t
pandana_data_parallelism(MPI_Comm    comm,     /* MPI communicator */
                         const char *infile,   /* input HDF5 file name */
                         int         nGroups,  /* number of groups */
                         NOvA_group *groups,   /* array of group objects */
                         long long   numIDs)   /* number of unique IDs */
{
    herr_t  err;
    hid_t   fd, fapl_id;
    int d, g, nprocs, rank, mpi_err;
    ssize_t read_len=0;
    size_t *lowers, *uppers;
    MPI_File fh;
#ifdef PANDANA_BENCHMARK
    int *nRecvs, *nSends;
    double open_t, read_seq_t, read_dset_t, close_t;
    inflate_t=0.0;
#endif

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* open file to get file ID of HDF5, MPI, POSIX ------------------------*/
#ifdef PANDANA_BENCHMARK
    MPI_Barrier(comm);
    open_t = MPI_Wtime();
#define ENABLE_PARALLEL_H5FOPEN
#endif

#ifdef ENABLE_PARALLEL_H5FOPEN
    /* create file access property list and add MPI communicator */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl_id < 0) CHECK_ERROR(fapl_id, "H5Pcreate");
    /* MPI-IO collectively open input file for reading */
    err = H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
    if (err < 0) CHECK_ERROR(err, "H5Pset_fapl_mpio");
#else
    /* Independent open input file for reading */
    fapl_id = H5P_DEFAULT;
#endif
    fd = H5Fopen(infile, H5F_ACC_RDONLY, fapl_id);
    if (fd < 0) {
        fprintf(stderr,"%d: Error: fail to open file %s (%s)\n",
                rank,  infile, strerror(errno));
        fflush(stderr);
        return -1;
    }
#ifdef ENABLE_PARALLEL_H5FOPEN
    err = H5Pclose(fapl_id);
    if (err < 0) CHECK_ERROR(err, "H5Pclose");
#endif

    /* set MPI-IO hints and open input file using MPI-IO */
    MPI_Info info = MPI_INFO_NULL;
    mpi_err = MPI_Info_create(&info);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Info_create");
    mpi_err = MPI_Info_set(info, "romio_cb_read", "enable");
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Info_set");
    mpi_err = MPI_File_open(comm, infile, MPI_MODE_RDONLY, info, &fh);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_open");
    mpi_err = MPI_Info_free(&info);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Info_free");
#ifdef ENABLE_PARALLEL_H5FOPEN
    if (seq_opt == 4) {
        posix_fd = open(infile, O_RDONLY);
        if (posix_fd < 0) CHECK_ERROR(-1, "open");
    }
#endif
#ifdef PANDANA_BENCHMARK
    open_t = MPI_Wtime() - open_t;

    /* Read key datasets ----------------------------------------------------*/
    MPI_Barrier(comm);
    read_seq_t = MPI_Wtime();
#endif

    /* calculate this process's responsible array index range (lower and upper
     * inclusive bounds) for each group */
    lowers = (size_t*) malloc(nGroups * 2 * sizeof(size_t));
    if (lowers == NULL) CHECK_ERROR(-1, "malloc");
    uppers = lowers + nGroups;

    /* construct a list of key dataset names, one per group */
    char **key_names = (char**) malloc(nGroups * sizeof(char*));
    for (g=0; g<nGroups; g++)
        key_names[g] = groups[g].dset_names[0];

#ifdef PANDANA_BENCHMARK
    if (dset_opt == 3) {
        nRecvs = (int*) malloc(nGroups * 2 * sizeof(int));
        nSends = nRecvs + nGroups;
        read_len += pandana_hdf5_read_keys_align(comm, fd, nGroups, key_names,
                    lowers, uppers, nRecvs, nSends);
    }
    else
#endif
    {
        /* calculate this process's lowers[] and uppers[] for all groups */
        read_len += pandana_read_keys(comm, fd, fh, nGroups, key_names, numIDs,
                                      lowers, uppers);
    }
    free(key_names);

#ifdef PANDANA_BENCHMARK
    read_seq_t = MPI_Wtime() - read_seq_t;

    /* read the remaining datasets ------------------------------------------*/
    MPI_Barrier(comm);
    read_dset_t = MPI_Wtime();

    hid_t xfer_plist;
    if (dset_opt == 0 || dset_opt == 3) {
        /* set MPI-IO collective transfer mode */
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        if (xfer_plist < 0) CHECK_ERROR(err, "H5Pcreate");
        err = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        if (err < 0) CHECK_ERROR(err, "H5Pset_dxpl_mpio");
    }
#endif

    /* Read the remaining datasets by iterating all groups */
    for (g=0; g<nGroups; g++) {

        /* open datasets, inquire sizes, in order to allocate read buffers */
        int nDatasets = groups[g].nDatasets-1;
        hid_t *dsets = (hid_t*) malloc(nDatasets * sizeof(hid_t));
        if (dsets == NULL) CHECK_ERROR(-1, "malloc");
        void **buf = (void**) malloc(nDatasets * sizeof(void*));
        if (buf == NULL) CHECK_ERROR(-1, "malloc");
        for (d=0; d<nDatasets; d++) {
            /* open dataset */
            dsets[d] = H5Dopen2(fd, groups[g].dset_names[d+1], H5P_DEFAULT);
            if (dsets[d] < 0) CHECK_ERROR(dset, "H5Dopen2");
            size_t dtype_size;
            hsize_t dims[2];
            inq_dset_meta(dsets[d], &dtype_size, dims, NULL);
            size_t buf_len;
            buf_len = uppers[g] - lowers[g] + 1;
#ifdef PANDANA_BENCHMARK
            /* chunk-aligned method may receive additional elements from rank-1 */
            if (dset_opt == 3) buf_len += nRecvs[g];
#endif
            buf_len *= dims[1] * dtype_size;
            buf[d] = (void*) malloc(buf_len);
            if (buf[d] == NULL) CHECK_ERROR(-1, "malloc");
        }

#ifdef PANDANA_BENCHMARK
        if (dset_opt == 0) {
            /* read datasets using H5Dread() */
            read_len += pandana_hdf5_read_subarrays(fd, nDatasets, dsets,
                        lowers[g], uppers[g], xfer_plist, buf);
        }
        else if (dset_opt == 1) {
            read_len += pandana_mpi_read_subarrays(fd, fh, nDatasets, dsets,
                        lowers[g], uppers[g], buf);
        }
        else if (dset_opt == 3) {
            read_len += pandana_hdf5_read_subarrays_align(comm, fd, nDatasets,
                        dsets, lowers[g], uppers[g], nRecvs[g], nSends[g],
                        xfer_plist, buf);
            /* Updated data partitioning: this process is assigned array index
             * range from (lowers[g] - nRecvs[g]) till (uppers[g] - nSends[g])
             */
            lowers[g] -= nRecvs[g];
            uppers[g] -= nSends[g];
        }
        else if (dset_opt == 2)
#endif
        {
            /* read datasets using MPI-IO, all datasets in group g at once */
            read_len += pandana_mpi_read_subarrays_aggr(fd, fh, nDatasets,
                        dsets, lowers[g], uppers[g], buf);
        }

        /* This is where PandAna performs computation to identify events of
         * interest from the read buffers. Note this process is assigned array
         * index range from lowers[g] till uppers[g] inclusively.
         */

        for (d=0; d<nDatasets; d++) {
            free(buf[d]);
            err = H5Dclose(dsets[d]);
            if (err < 0) CHECK_ERROR(err, "H5Dclose");
        }
        free(buf);
        free(dsets);
    }
    free(lowers);

#ifdef PANDANA_BENCHMARK
    if (dset_opt == 3) free(nRecvs);

    if (dset_opt == 0) {
        err = H5Pclose(xfer_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");
    }
    read_dset_t = MPI_Wtime() - read_dset_t;

    /* close input file -----------------------------------------------------*/
    MPI_Barrier(comm);
    close_t = MPI_Wtime();
#endif
    err = H5Fclose(fd);
    if (err < 0) CHECK_ERROR(err, "H5Fclose");

    mpi_err = MPI_File_close(&fh);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_close");

#ifdef PANDANA_BENCHMARK
    if (seq_opt == 4) close(posix_fd);
    close_t = MPI_Wtime() - close_t;

    timings[0] = open_t;
    timings[1] = read_seq_t;
    timings[2] = read_dset_t;
    timings[3] = close_t;
    timings[4] = inflate_t;
#endif

    return read_len;
}

/*----< pandana_group_parallelism() >----------------------------------------*/
/* Divide processes into subsets. Reading datasets in a group is done by only
 * a subset of processes. When the number of processes is more than the groups,
 * each group is read by a subset of processes. When less, one process reads
 * more than one group.
 */
ssize_t
pandana_group_parallelism(MPI_Comm    comm,     /* MPI communicator */
                          const char *infile,   /* input HDF5 file name */
                          int         nGroups,  /* number of groups */
                          NOvA_group *groups,   /* array of group objects */
                          long long   numIDs)   /* number of unique IDs */
{
    int nprocs, rank, my_startGrp, my_nGroups;
    ssize_t read_len;
    MPI_Comm grp_comm;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (nprocs <= nGroups) {
        /* When number of processes is less than number groups, each process is
         * assigned one or more groups
         */
        grp_comm = MPI_COMM_SELF;
        my_nGroups = nGroups / nprocs;
        my_startGrp = my_nGroups * rank;
        if (rank < nGroups % nprocs) {
            my_startGrp += rank;
            my_nGroups++;
        }
        else
            my_startGrp += nGroups % nprocs;
    } else {
        /* When number of processes is more than number groups, each group is
         * assigned one or more processes
         */
        my_nGroups = 1;
        int grp_len = nprocs / nGroups;
        int grp_rem = nprocs % nGroups;
        if (rank < grp_rem * (grp_len+1))
            my_startGrp = rank / (grp_len+1);
        else
            my_startGrp = (rank - (grp_rem * (grp_len+1))) / grp_len + grp_rem;

        /* split MPI_COMM_WORLD into groups, so an MPI process joins only one
         * group
         */
        MPI_Comm_split(MPI_COMM_WORLD, my_startGrp, rank, &grp_comm);
    }

    groups += my_startGrp;

    /* within a group, data parallelism method is used */
    read_len = pandana_data_parallelism(grp_comm, infile, my_nGroups, groups,
                                        numIDs);

    if (nprocs > nGroups)
        MPI_Comm_free(&grp_comm);

    return read_len;
}

/*----< pandana_datset_parallelism() >---------------------------------------*/
/* Divide all datasets among all processes. Each process reads the entire
 * datasets assigned.
 */
ssize_t
pandana_dataset_parallelism(MPI_Comm    comm,     /* MPI communicator */
                            const char *infile,   /* input HDF5 file name */
                            int         nGroups,  /* number of groups */
                            NOvA_group *groups)   /* array of group objects */
{
    int j, d, g, k, nprocs, rank, nDatasets, my_startDset, my_nDatasets;
    ssize_t read_len=0;
    herr_t err;

#ifdef PANDANA_BENCHMARK
    double open_t, read_dset_t, close_t;

    MPI_Barrier(comm);
    open_t = MPI_Wtime();
#endif
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* Independent open input file for reading */
    hid_t fd = H5Fopen(infile, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fd < 0) {
        fprintf(stderr,"%d: Error %s: fail to open file %s (%s)\n",
                rank, __func__, infile, strerror(errno));
        fflush(stderr);
        return -1;
    }
#ifdef PANDANA_BENCHMARK
    open_t = MPI_Wtime() - open_t;

    /* Read key datasets ----------------------------------------------------*/
    MPI_Barrier(comm);
    read_dset_t = MPI_Wtime();
#endif

    /* calculate total number of datasets (except key datasets) */
    nDatasets = 0;
    for (g=0; g<nGroups; g++)
        nDatasets += groups[g].nDatasets - 1;

    /* Divide all datasets evenly among processes. Calculate the starting index
     * and the number of assigned datasets for this process.
     */
    my_nDatasets = nDatasets / nprocs;
    my_startDset = my_nDatasets * rank;
    if (rank < nDatasets % nprocs) {
        my_startDset += rank;
        my_nDatasets++;
    }
    else
        my_startDset += nDatasets % nprocs;

    /* construct a list of key dataset names */
    char **key_names = (char**) malloc(my_nDatasets * sizeof(char*));
    j = 0;
    k = 0;
    for (g=0; g<nGroups; g++)
        for (d=1; d<groups[g].nDatasets; d++) { /* skip key dataset */
            if (j >= my_startDset && j < my_startDset + my_nDatasets)
                key_names[k++] = groups[g].dset_names[d];
            j++;
        }

    /* allocate an array of buffer pointers */
    void **buf = (void**) malloc(my_nDatasets * sizeof(void*));

    for (d=0; d<my_nDatasets; d++) {
        hid_t dset = H5Dopen2(fd, key_names[d], H5P_DEFAULT);
        if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

        /* inquire dimension sizes of dset */
        hsize_t dims[2];
        hid_t fspace = H5Dget_space(dset);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");

        /* get data type and size */
        hid_t dtype = H5Dget_type(dset);
        if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
        size_t dtype_size = H5Tget_size(dtype);
        if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");;

        /* allocate read buffers */
        size_t buf_len = dims[0] * dims[1] * dtype_size;
        buf[d] = (void*) malloc(buf_len);
        if (buf[d] == NULL) CHECK_ERROR(-1, "malloc");
        read_len += buf_len;

        /* read from file into buf */
        err = H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf[d]);
        if (err < 0) CHECK_ERROR(err, "H5Dread");

        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");

        err = H5Dclose(dset);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");
    }

    /* This is where PandAna performs computation to identify events of
     * interest from the read buffers
     */

    // printf("%d %s my_startDset=%d my_nDatasets=%d read_len=%.2f MiB key[0]=%s\n",rank,__func__,my_startDset,my_nDatasets,(float)read_len/1048576.0,key_names[0]);

    /* free read buffers */
    for (d=0; d<my_nDatasets; d++) free(buf[d]);
    free(buf);
    free(key_names);

#ifdef PANDANA_BENCHMARK
    read_dset_t = MPI_Wtime() - read_dset_t;

    /* close input file -----------------------------------------------------*/
    MPI_Barrier(comm);
    close_t = MPI_Wtime();
#endif
    err = H5Fclose(fd);
    if (err < 0) CHECK_ERROR(err, "H5Fclose");
#ifdef PANDANA_BENCHMARK
    close_t = MPI_Wtime() - close_t;

    timings[0] = open_t;
    timings[1] = 0.0; /* read_seq_t */
    timings[2] = read_dset_t;
    timings[3] = close_t;
    timings[4] = 0.0; /* inflate_t */
#endif

    return read_len;
}
