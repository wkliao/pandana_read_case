/*
 * Copyright (C) 2020, Northwestern University and Fermi National Accelerator Laboratory
 * See COPYRIGHT notice in top-level directory.
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h> /* open(), lseek() */
#include <unistd.h>    /* open(), lseek(), read(), close(), getopt() */
#include <fcntl.h>     /* open() */
#include <limits.h>    /* INT_MAX */
#include <assert.h>
#include <mpi.h>
#include "hdf5.h"
#include "zlib.h"

#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif

#define LINE_SIZE 256

#define CHECK_ERROR(err, msg) { \
    fprintf(stderr,"Error at line %d, function %s, file %s: %s\n", \
            __LINE__, __func__,__FILE__, msg); \
    assert(0); \
}

static int verbose, debug;

typedef struct {
    int       nDatasets;  /* number of datasets in this group */
    char    **dset_names; /* [nDatasets] string names of datasets */

    void    **buf;        /* [nDatasets] read buffers */

    hid_t    *dsets;      /* [nDatasets] HDF5 dataset IDs */
    hsize_t **dims;       /* [nDatasets][2] dimension sizes */
    hsize_t **chunk_dims; /* [nDatasets][2] chunk dimension sizes */
    size_t   *dtype_size; /* [nDatasets] size of data type of datasets */
    size_t   *nChunks;    /* [nDatasets] number of chunks in each dataset */
    double   *read_t;     /* [nDatasets] read timings */
} NOvA_group;

/*----< read_dataset_names() >-----------------------------------------------*/
/* read listfile to retrieve names of all datasets and collect metadata:
 * number of groups, number of datasets in each group, index of group "/spill"
 */
static int
read_dataset_names(int          rank,
                   const char  *listfile,   /* file contains a list of names */
                   NOvA_group **gList,      /* OUT */
                   int         *spill_idx)  /* OUT */
{
    FILE *fptr;
    int d, j, g, len, nGroups, nDatasets, nDsetGrp, maxDsetGrp;
    char line[LINE_SIZE], gname[LINE_SIZE], name[LINE_SIZE], *cur_gname;

    *spill_idx = -1;
    fptr = fopen(listfile, "r");
    if (fptr == NULL) {
        fprintf(stderr,"%d: Error: fail to open file %s (%s)\n",
               rank,listfile,strerror(errno));
        return -1;
    }

    /* check number of datasets listed in the file */
    nGroups = 0;     /* number of groups */
    maxDsetGrp = 0;  /* max number of datasets among all groups */
    nDsetGrp = 0;    /* number of dataset in the current group */
    gname[0] = '\0';
    while (fgets(line, LINE_SIZE, fptr) != NULL) {
        if (line[0] == '\n') continue;   /* skip empty line */
        if (line[0] == '#') continue;    /* skip comment line */
        len = strlen(line);
        if (line[len-1] == '\n') line[--len] = '\0';
        while (len > 0 && line[len-1] == ' ') line[--len] = '\0';
        if (len == 0) continue; /* skip blank line */

        /* retrieve group name */
        cur_gname = strtok(line, "/");
        if (strcmp(gname, cur_gname)) { /* entering a new group */
            strcpy(gname, cur_gname);
            if (nDsetGrp > maxDsetGrp) maxDsetGrp = nDsetGrp;
            nDsetGrp = 0;
            nGroups++;
        }
        nDsetGrp++;
    }
    if (nDsetGrp > maxDsetGrp) maxDsetGrp = nDsetGrp;

    /* allocate an array of group objects */
    *gList = (NOvA_group*) malloc(nGroups * sizeof(NOvA_group));
    if (*gList == NULL) CHECK_ERROR(-1, "malloc");

    /* rewind the file, this time read the dataset names */
    rewind(fptr);

    nGroups = -1;
    nDatasets = 0;
    gname[0] = '\0';
    while (fgets(line, LINE_SIZE, fptr) != NULL) {
        if (line[0] == '\n') continue;   /* skip empty line */
        if (line[0] == '#') continue;    /* skip comment line */
        len = strlen(line);
        if (line[len-1] == '\n') line[--len] = '\0';
        while (len > 0 && line[len-1] == ' ') line[--len] = '\0';
        if (len == 0) continue; /* skip blank line */

        /* now len is the true string length of line */

        /* retrieve group name */
        strcpy(name, line);
        cur_gname = strtok(name, "/");
        if (strcmp(gname, cur_gname)) { /* entering a new group */
            strcpy(gname, cur_gname);
            if (nGroups >= 0) (*gList)[nGroups].nDatasets = nDatasets;
            nGroups++;
            nDatasets = 0;
            if (!strcmp(gname, "spill")) *spill_idx = nGroups;
            /* allocate space to store names of datasets in this group */
            (*gList)[nGroups].dset_names = (char**) malloc(maxDsetGrp * sizeof(char*));
            if ((*gList)[nGroups].dset_names == NULL) CHECK_ERROR(-1, "malloc");
        }

        if (!strcmp(strtok(NULL, "/"), "evt.seq")) {
            /* add dataset evt.seq to first in the group */
            for (j=nDatasets; j>0; j--)
                (*gList)[nGroups].dset_names[j] = (*gList)[nGroups].dset_names[j-1];
            (*gList)[nGroups].dset_names[0] = (char*) malloc(len + 1);
            if ((*gList)[nGroups].dset_names[0] == NULL) CHECK_ERROR(-1, "malloc");
            strcpy((*gList)[nGroups].dset_names[0], line);
        }
        else {
            (*gList)[nGroups].dset_names[nDatasets] = (char*) malloc(len + 1);
            if ((*gList)[nGroups].dset_names[nDatasets] == NULL) CHECK_ERROR(-1, "malloc");
            strcpy((*gList)[nGroups].dset_names[nDatasets], line);
        }
        nDatasets++;
    }
    fclose(fptr);
    (*gList)[nGroups].nDatasets = nDatasets;
    nGroups++;
     if (*spill_idx < 0) CHECK_ERROR(-1, "*spill_idx < 0");

    /* check if dataset evt.seq is missing */
    for (g=0; g<nGroups; g++) {
        /* evt.seq is set to be the first dataset of each group */
        strcpy(name, (*gList)[g].dset_names[0]);
        strtok(name, "/");
        if (strcmp(strtok(NULL, "/"), "evt.seq")) {
            printf("Error: group[g=%d] %s contains no evt.seq\n",g, name);
            return -1;
        }
    }
    if (debug && rank == 0) {
        for (d=0; d<(*gList)[g].nDatasets; d++)
            printf("nGroups=%d group[%d] nDatasets=%d name[%d] %s\n",
                   nGroups, g, (*gList)[g].nDatasets, d, (*gList)[g].dset_names[d]);
    }

    return nGroups;
}

#ifdef PREFETCH_METADATA
/*----< collect_metadata() >-------------------------------------------------*/
/* collect metadata of all datasets of all groups in one place holder.
 * Note this turns out to be slower than opening dataset right before reading
 * it.
 */
int
collect_metadata(hid_t       fd,
                 NOvA_group *groups,
                 int         nGroups)
{
    int g, d;

    for (g=0; g<nGroups; g++) {
        for (d=0; d<groups[g].nDatasets; d++) {
            herr_t err;
            hid_t dset;
            hsize_t *dims       = groups[g].dims[d];
            hsize_t *chunk_dims = groups[g].chunk_dims[d];

            /* open dataset */
            dset = H5Dopen2(fd, groups[g].dset_names[d], H5P_DEFAULT);
            if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

            /* inquire dimension sizes of dset */
            hid_t fspace = H5Dget_space(dset);
            if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
            err = H5Sget_simple_extent_dims(fspace, dims, NULL);
            if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
            err = H5Sclose(fspace);
            if (err < 0) CHECK_ERROR(err, "H5Sclose");

            /* inquire chunk size along each dimension */
            hid_t chunk_plist = H5Dget_create_plist(dset);
            if (chunk_plist < 0) CHECK_ERROR(chunk_plist, "H5Dget_create_plist");
            int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
            if (chunk_ndims != 2) CHECK_ERROR(-1, "H5Pget_chunk: chunk_ndims != 2");
            err = H5Pclose(chunk_plist);
            if (err < 0) CHECK_ERROR(err, "H5Pclose");

            /* get data type and size */
            hid_t dtype = H5Dget_type(dset);
            if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
            groups[g].dtype_size[d] = H5Tget_size(dtype);
            if (groups[g].dtype_size[d] == 0) CHECK_ERROR(-1, "H5Tget_size");;
            err = H5Tclose(dtype);
            if (err < 0) CHECK_ERROR(err, "H5Tclose");

            /* find the number of chunks of dset */
            size_t nChunks[2];
            nChunks[0] = groups[g].dims[d][0] / groups[g].chunk_dims[d][0];
            if (groups[g].dims[d][0] % groups[g].chunk_dims[d][0]) nChunks[0]++;
            nChunks[1] = groups[g].dims[d][1] / groups[g].chunk_dims[d][1];
            if (groups[g].dims[d][1] % groups[g].chunk_dims[d][1]) nChunks[1]++;
            groups[g].nChunks[d] = nChunks[0] * nChunks[1];

            groups[g].dsets[d] = dset;
        }
    }
    return 1;
}
#endif

/*----< inq_num_unique_IDs() >-----------------------------------------------*/
/* Inquire the number of globally unique IDs, for example the size of
 * /spill/evt.seq. Dataset /spill/evt.seq stores a list of unique event IDs of
 * data in the input HDF5. The event IDs are in an increasing order from 0 with
 * increment of 1. Thus the size of /spill/evt.seq is the number of unique
 * event IDs.
 */
static long long
inq_num_unique_IDs(hid_t       fd,
                   const char *key) /* name of key dataset */
{
    herr_t err;
    int rank;
    long long nEvts;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        /* Only root opens the key dataset, e.g. '/spill/evt.seq' */
        hid_t seq = H5Dopen2(fd, key, H5P_DEFAULT);
        if (seq < 0) CHECK_ERROR(seq, "H5Dopen2");

        hsize_t dims[2];

        /* inquire dimension size of '/spill/evt.seq' */
        hid_t fspace = H5Dget_space(seq);
        if (fspace < 0) CHECK_ERROR(seq, "H5Dget_space");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        /* 2nd dimension of key, e.g. /spill/evt.seq, is always 1 */
        if (dims[1] != 1) CHECK_ERROR(-1, "dims[1] != 1");

        nEvts = dims[0];

        if (verbose)
            printf("Size of key %s is %lld\n", key, nEvts);

        if (debug) {
            /* Note contents of key, e.g. /spill/evt.seq, start with 0 and
             * increment by 1, so there is no need to read the contents of it.
             * Check the contents only in debug mode.
             */
            /* get data type and size */
            hid_t dtype = H5Dget_type(seq);
            if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");

            if (H5Tequal(dtype, H5T_STD_I64LE) <= 0) CHECK_ERROR(-1, "dtype != H5T_STD_I64LE");

            size_t dtype_size = H5Tget_size(dtype);
            if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");
            err = H5Tclose(dtype);
            if (err < 0) CHECK_ERROR(err, "H5Tclose");

            int64_t v, *seqBuf;
            seqBuf = (int64_t*) malloc(nEvts * sizeof(int64_t));
            if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");

            err = H5Dread(seq, H5T_STD_I64LE, fspace, fspace, H5P_DEFAULT,
                          seqBuf);
            if (err < 0) CHECK_ERROR(err, "H5Dread");

            for (v=0; v<nEvts; v++)
                if (seqBuf[v] != v) {
                    printf("Error: %s[%"PRId64"] expect %"PRId64" but got %"PRId64"\n",
                           key, v, v, seqBuf[v]);

                    if (seqBuf[v] != v) CHECK_ERROR(-1, "seqBuf[v] != v");
                }
            free(seqBuf);
        }
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");
        err = H5Dclose(seq);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");
    }

    /* MPI_Bcast() makes this function a collective */
    MPI_Bcast(&nEvts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    return nEvts;
}

/*----< calculate_starts_ends() >--------------------------------------------*/
/* The event IDs are evenly partitioned among all processes to balance the
 * computational workload. This function calculates the starting and ending
 * indices for each process and stores them in starts[rank] and ends[rank],
 * the range of event IDs responsible by process rank.
 */
static int
calculate_starts_ends(int      nprocs,
                      int      rank,
                      hsize_t  nEvts,  /* no. unique evt IDs */
                      hsize_t *starts, /* OUT */
                      hsize_t *ends)   /* OUT */
{
    hsize_t j, my_count;

    /* calculate the range of event IDs assigned to all processes. For process
     * of rank 'rank', its responsible range is from starts[rank] to
     * ends[rank].
     */
    my_count = nEvts / nprocs;
    for (j=0; j<nEvts % nprocs; j++) {
        starts[j] = my_count * j + j;
        ends[j] = starts[j] + my_count;
    }
    for (; j<nprocs; j++) {
        starts[j] = my_count * j + nEvts % nprocs;
        ends[j] = starts[j] + my_count - 1;
    }

    return 1;
}

/* return the smallest index i, such that base[i] >= key */
static size_t
binary_search_min(int64_t   key,
                  int64_t *base,
                  size_t   nmemb)
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
binary_search_max(int64_t  key,
                  int64_t *base,
                  size_t   nmemb)
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

typedef struct {
    MPI_Aint block_dsp;
    int      block_len;
    int      block_idx;
} off_len;

static int
off_compare(const void *a, const void *b)
{
    if (((off_len*)a)->block_dsp > ((off_len*)b)->block_dsp) return  1;
    if (((off_len*)a)->block_dsp < ((off_len*)b)->block_dsp) return -1;
    return 0;
}

/*----< posix_read_dataset_by_ID() >-----------------------------------------*/
/* Call POSIX read to read an entire dataset, decompress it, and copy to user
 * buffer
 */
ssize_t
posix_read_dataset_by_ID(hid_t   dset,       /* HDF5 dataset ID */
                         int     posix_fd,   /* POSIX file descriptor */
                         void   *buf,        /* OUT: user buffer */
                         double *read_t,     /* OUT */
                         double *inflate_t)  /* OUT */
{
    int j;
    ssize_t read_len=0;
    herr_t err;
    hsize_t dims[2], chunk_dims[2];

    /* inquire dimension sizes of dset */
    hid_t fspace = H5Dget_space(dset);
    if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
    err = H5Sget_simple_extent_dims(fspace, dims, NULL);
    if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
    err = H5Sclose(fspace);
    if (err < 0) CHECK_ERROR(err, "H5Sclose");

    /* inquire chunk size along each dimension */
    hid_t chunk_plist = H5Dget_create_plist(dset);
    if (chunk_plist < 0) CHECK_ERROR(chunk_plist, "H5Dget_create_plist");
    int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
    if (chunk_ndims != 2) CHECK_ERROR(-1, "chunk_ndims != 2");
    if (chunk_dims[1] != dims[1]) CHECK_ERROR(-1, "chunk_dims[1] == dims[1]");
    err = H5Pclose(chunk_plist);
    if (err < 0) CHECK_ERROR(err, "H5Pclose");

    hid_t dtype = H5Dget_type(dset);
    if (dtype < 0) CHECK_ERROR(err, "H5Dget_type");
    size_t dtype_size = H5Tget_size(dtype);
    if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");
    err = H5Tclose(dtype);
    if (err < 0) CHECK_ERROR(err, "H5Tclose");

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

    /* read compressed chunks, one at a time, into zipBuf, decompress it
     * into buf
     */
    hsize_t offset[2]={0,0};
    unsigned char *buf_ptr = buf;
    for (j=0; j<nChunks; j++) {
        haddr_t addr;
        hsize_t size;
        double timing;

        err = H5Dget_chunk_info_by_coord(dset, offset, NULL, &addr, &size);
        if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");

        if (read_t != NULL) timing = MPI_Wtime();
        lseek(posix_fd, addr, SEEK_SET);
        ssize_t len = read(posix_fd, zipBuf, size);
        if (len != size) CHECK_ERROR(-1, "read len != size");

        if (read_t != NULL) *read_t += MPI_Wtime() - timing;

        if (inflate_t != NULL) timing = MPI_Wtime();
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
        if (inflate_t != NULL) *inflate_t += MPI_Wtime() - timing;
    }
    free(zipBuf);

    return read_len;
}

/*----< mpi_read_datasets_by_IDs() >-----------------------------------------*/
/* Use a single MPI collective read to read multiple datasets (entire dataset),
 * decompress, and copy the requested part to user buffers, buf[]. Returned
 * value is the sum or datasets read, not a true read amount, as the datasets
 * are compressed.
 */
ssize_t
mpi_read_datasets_by_IDs(MPI_Comm    comm,
                         hid_t       fd,
                         MPI_File    fh,
                         int         nDatasets,
                         hid_t      *dsets,      /* IN:  [nDatasets] */
                         void      **buf,        /* OUT: [nDatasets] */
                         double     *read_t,     /* OUT */
                         double     *inflate_t)  /* OUT */
{
    int c, d, j, k, mpi_err;
    herr_t err;
    hsize_t all_nChunks=0, max_chunk_size=0, **dims, **chunk_dims, **size;
    haddr_t **addr;
    size_t *nChunks, *dtype_size;
    ssize_t zip_len=0, read_len=0;
    double timing;

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
        /* inquire dimension sizes of dsets[d] */
        hid_t fspace = H5Dget_space(dsets[d]);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        err = H5Sget_simple_extent_dims(fspace, dims[d], NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");

        /* inquire chunk size along each dimension */
        hid_t chunk_plist = H5Dget_create_plist(dsets[d]);
        if (chunk_plist < 0) CHECK_ERROR(err, "H5Dget_create_plist");
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims[d]);
        if (chunk_ndims != 2) CHECK_ERROR(-1, "H5Pget_chunk chunk_ndims != 2");
        err = H5Pclose(chunk_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");

        hid_t dtype = H5Dget_type(dsets[d]);
        if (dtype < 0) CHECK_ERROR(err, "H5Dget_type");
        dtype_size[d] = H5Tget_size(dtype);
        if (dtype_size[d] == 0) CHECK_ERROR(-1, "H5Tget_size");
        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");

        read_len += dims[d][0] * dims[d][1] * dtype_size[d];

        /* track the max chunk size */
        size_t chunk_size = chunk_dims[d][0] * chunk_dims[d][1] * dtype_size[d];
        max_chunk_size = MAX(max_chunk_size, chunk_size);

        /* find the number of chunks to be read by this process */
        nChunks[d] = dims[d][0] / chunk_dims[d][0];
        if (dims[d][0] % chunk_dims[d][0]) nChunks[d]++;

        /* accumulate number of chunks across all groups */
        all_nChunks += nChunks[d];

        /* collect offsets and sizes of individual chunks */
        addr[d] = (haddr_t*) malloc(nChunks[d] * sizeof(haddr_t));
        if (addr[d] == NULL) CHECK_ERROR(-1, "malloc");
        size[d] = (hsize_t*) malloc(nChunks[d] * sizeof(hsize_t));
        if (size[d] == NULL) CHECK_ERROR(-1, "malloc");
        hsize_t offset[2]={0,0};
        for (j=0; j<nChunks[d]; j++) {
            err = H5Dget_chunk_info_by_coord(dsets[d], offset, NULL, &addr[d][j],
                                             &size[d][j]);
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

    if (read_t != NULL) timing = MPI_Wtime();
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
    if (read_t != NULL) *read_t += MPI_Wtime() - timing;

    /* decompress individual chunks into buf[] */
    if (inflate_t != NULL) timing = MPI_Wtime();
    unsigned char *chunkBuf = (unsigned char*) malloc(max_chunk_size);
    if (chunkBuf == NULL) CHECK_ERROR(-1, "malloc");
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
        if (c == nChunks[d] - 1 && last_len > 0) /* last chunk size may not be a whole chunk */
            len = last_len * dtype_size[d];
        memcpy((unsigned char*)buf[d] + off, chunkBuf, len);
        zipBuf_ptr += disp_indx[j].block_len;
    }
    free(chunkBuf);
    free(zipBuf);
    free(disp_indx);
    free(nChunks);
    free(dims[0]);
    free(dims);
    if (inflate_t != NULL) *inflate_t += MPI_Wtime() - timing;

    return read_len;
}

/*----< read_evt_seq_aggr_all() >--------------------------------------------*/
/* Use a single MPI collective read to read all evt.seq datasets of all groups,
 * decompress, and calculate the array index boundaries responsible by all
 * processes.
 */
static ssize_t
read_evt_seq_aggr_all(hid_t           fd,
                      MPI_File        fh,
                      NOvA_group     *groups,
                      int             nGroups,
                      int             nprocs,
                      int             rank,
                      int             spill_idx,
                      const hsize_t  *starts,     /* IN:  [nprocs] */
                      const hsize_t  *ends,       /* IN:  [nprocs] */
                      long long     **bounds,     /* OUT: [nGroups][nprocs*2] */
                      double         *inflate_t)  /* OUT */
{
    int g, d, j, k, mpi_err;
    herr_t err;
    hsize_t nChunks=0, max_chunk_dim=0, **size;
    haddr_t **addr;
    size_t dtype_size=8;
    ssize_t zip_len=0, read_len=0;
    double timing;

    if (nGroups == 0 || (nGroups == 1 && spill_idx == 0)) {
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

    /* save file offsets and sizes of individual chunks for later use */
    addr = (haddr_t**) malloc(nGroups * sizeof(haddr_t*));
    if (addr == NULL) CHECK_ERROR(-1, "malloc");
    size = (hsize_t**) malloc(nGroups * sizeof(hsize_t*));
    if (size == NULL) CHECK_ERROR(-1, "malloc");

    /* collect sizes of dimensions and chunk dimensions of evt.seq datasets in
     * all groups. Also inquire number of chunks and their file offsets and
     * compressed sizes
     */
    for (g=0; g<nGroups; g++) {
        hsize_t *dims       = groups[g].dims[0];
        hsize_t *chunk_dims = groups[g].chunk_dims[0];

        /* skip dataset evt.seq in group /spill */
        if (g == spill_idx) continue;

        hid_t seq = H5Dopen2(fd, groups[g].dset_names[0], H5P_DEFAULT);
        if (seq < 0) CHECK_ERROR(seq, "H5Dopen2");

        /* inquire dimension sizes of dset */
        hid_t fspace = H5Dget_space(seq);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        if (dims[1] != 1) CHECK_ERROR(err, "dims[1] != 1");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");

        /* inquire chunk size along each dimension */
        hid_t chunk_plist = H5Dget_create_plist(seq);
        if (chunk_plist < 0) CHECK_ERROR(err, "H5Dget_create_plist");
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        if (chunk_ndims != 2) CHECK_ERROR(-1, "chunk_ndims != 2");
        if (chunk_dims[1] != 1) CHECK_ERROR(-1, "chunk_dims[1] != 1");
        err = H5Pclose(chunk_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");
        max_chunk_dim = MAX(max_chunk_dim, chunk_dims[0]);

        /* evt.seq data type is aways H5T_STD_I64LE and of size 8 bytes */
        hid_t dtype = H5Dget_type(seq);
        if (H5Tequal(dtype, H5T_STD_I64LE) <= 0) CHECK_ERROR(-1, "dtype != H5T_STD_I64LE");
        dtype_size = H5Tget_size(dtype);
        if (dtype_size != 8) CHECK_ERROR(-1, "evt.seq dtype_size != 8");
        groups[g].dtype_size[0] = dtype_size;
        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");
        groups[g].buf[0] = (void*) malloc(dims[0] * dtype_size);
        if (groups[g].buf[0] == NULL) CHECK_ERROR(-1, "malloc");

        read_len += dims[0] * dims[1] * dtype_size;

        /* find the number of chunks to be read by this process */
        groups[g].nChunks[0] = dims[0] / chunk_dims[0];
        if (dims[0] % chunk_dims[0]) groups[g].nChunks[0]++;

        /* accumulate number of chunks across all groups */
        nChunks += groups[g].nChunks[0];

        /* collect offsets and sizes of individual chunks of evt.seq */
        addr[g] = (haddr_t*) malloc(groups[g].nChunks[0] * sizeof(haddr_t));
        if (addr[g] == NULL) CHECK_ERROR(-1, "malloc");
        size[g] = (hsize_t*) malloc(groups[g].nChunks[0] * sizeof(hsize_t));
        if (size[g] == NULL) CHECK_ERROR(-1, "malloc");
        hsize_t offset[2]={0,0};
        for (j=0; j<groups[g].nChunks[0]; j++) {
            err = H5Dget_chunk_info_by_coord(seq, offset, NULL, &addr[g][j],
                                             &size[g][j]);
            if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");
            zip_len += size[g][j];
            offset[0] += chunk_dims[0];
        }
        err = H5Dclose(seq);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");
    }

    /* Note file offsets of chunks may not follow the increasing order of
     * chunk IDs read by this process. We must sort the offsets before
     * creating the MPI derived file type. First, we construct an array of
     * displacement-length-index objects for such sorting.
     */
    off_len *disp_indx = (off_len*) malloc(nChunks * sizeof(off_len));
    if (disp_indx == NULL) CHECK_ERROR(-1, "malloc");

    int is_mono_nondecr = 1;
    k = 0;
    for (g=0; g<nGroups; g++) {
        if (g == spill_idx) continue;
        for (j=0; j<groups[g].nChunks[0]; j++) {
            disp_indx[k].block_dsp = (MPI_Aint)addr[g][j];  /* chunk's file offset */
            disp_indx[k].block_len = (int)size[g][j];       /* compressed chunk size */
            disp_indx[k].block_idx = j*nGroups + g;         /* unique ID for this chunk */
            if (k > 0 && disp_indx[k].block_dsp < disp_indx[k-1].block_dsp)
                is_mono_nondecr = 0;
            k++;
        }
        free(addr[g]);
        free(size[g]);
    }
    free(addr);
    free(size);
    if (k != nChunks) CHECK_ERROR(-1, "k != nChunks");

    /* Sort chunk offsets into an increasing order, as MPI-IO requires that
     * for file view.
     * According to MPI standard Chapter 13.3, file displacements of filetype
     * must be non-negative and in a monotonically nondecreasing order. If the
     * file is opened for writing, neither the etype nor the filetype is
     * permitted to contain overlapping regions.
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
    mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_set_view");

    mpi_err = MPI_Type_free(&ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_free");

    /* allocate a buffer for reading the compressed chunks all at once */
    unsigned char *chunk_buf, *chunk_buf_ptr;
    chunk_buf = (unsigned char*) malloc(zip_len);
    if (chunk_buf == NULL) CHECK_ERROR(-1, "malloc");
    chunk_buf_ptr = chunk_buf;

    /* collective read */
    mpi_err = MPI_File_read_all(fh, chunk_buf, zip_len, MPI_BYTE, MPI_STATUS_IGNORE);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_read_all");

    /* decompress individual chunks into groups[g].buf[0] */
    if (inflate_t != NULL) timing = MPI_Wtime();
    size_t whole_chunk_size = max_chunk_dim * dtype_size;
    unsigned char *whole_chunk = (unsigned char*) malloc(whole_chunk_size);
    if (whole_chunk == NULL) CHECK_ERROR(-1, "malloc");
    for (j=0; j<nChunks; j++) {
        int ret;
        z_stream z_strm;
        z_strm.zalloc    = Z_NULL;
        z_strm.zfree     = Z_NULL;
        z_strm.opaque    = Z_NULL;
        z_strm.avail_in  = disp_indx[j].block_len;
        z_strm.next_in   = chunk_buf_ptr;
        z_strm.avail_out = whole_chunk_size;
        z_strm.next_out  = whole_chunk;
        ret = inflateInit(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateInit");
        ret = inflate(&z_strm, Z_SYNC_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) CHECK_ERROR(-1, "inflate");
        ret = inflateEnd(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateEnd");

        /* copy requested data to user buffer */
        g = disp_indx[j].block_idx % nGroups;  /* group ID */
        d = disp_indx[j].block_idx / nGroups;  /* chunk ID of evt.seq in group g */
        size_t len = groups[g].chunk_dims[0][0] * dtype_size;
        size_t off = len * d;
        size_t last_len = groups[g].dims[0][0] % groups[g].chunk_dims[0][0];
        if (d == groups[g].nChunks[0] - 1 && last_len > 0) /* last chunk size may not be a whole chunk */
            len = last_len * dtype_size;
        memcpy(groups[g].buf[0] + off, whole_chunk, len);
        chunk_buf_ptr += disp_indx[j].block_len;
    }
    free(whole_chunk);
    free(chunk_buf);
    free(disp_indx);
    if (inflate_t != NULL) *inflate_t += MPI_Wtime() - timing;

    /* calculate lower and upper boundaries, responsible array index ranges,
     * for all processes
     */
    for (g=0; g<nGroups; g++) {
        if (g == spill_idx) continue;
        for (j=0, k=0; j<nprocs; j++) {
            bounds[g][k++] = binary_search_min(starts[j], groups[g].buf[0],
                                               groups[g].dims[0][0]);
            bounds[g][k++] = binary_search_max(ends[j],   groups[g].buf[0],
                                               groups[g].dims[0][0]);
        }
        free(groups[g].buf[0]);
        groups[g].buf[0] = NULL;
    }
    return read_len;
}

/*----< read_evt_seq() >-----------------------------------------------------*/
/* Read evt.seq dataset in order to calculate the lower and upper boundaries
 * responsible by each process in to lowers[] and uppers[]
 */
static ssize_t
read_evt_seq(MPI_Comm       comm,
             hid_t          fd,
             MPI_File       fh,
             int            posix_fd,
             NOvA_group    *groups,
             int            nGroups,
             int            profile,
             int            seq_opt,
             hid_t          xfer_plist,
             int            spill_idx,
             const hsize_t *starts,     /* IN:  [nprocs] */
             const hsize_t *ends,       /* IN:  [nprocs] */
             size_t        *lowers,     /* OUT: [nprocs] */
             size_t        *uppers,     /* OUT: [nprocs] */
             double        *inflate_t)  /* OUT: */
{
    int g, j, k, nprocs, rank;
    herr_t err;
    hid_t seq;
    ssize_t read_len=0;
    long long lower_upper[2];
    int64_t *seqBuf=NULL;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (seq_opt == 3 || seq_opt == 4) {
        /* As there are nGroups number of groups, each contains one evt.seq
         * dataset, assign nGroups evt.seq datasets among MIN(nprocs, nGroups)
         * number of processes, so they can be read in parallel. For seq_opt 3,
         * the processes got assigned make a single call to MPI collective read
         * to read all the assigned evt.seq datasets, decompress, calculate
         * boundaries of all processes, and scatter them to other processes.
         * For seq_opt 4, POSIX read is used to read the dataset chunks. Note
         * these are done in parallel, i.e. workload of reads and computation
         * is distributed among MIN(nprocs, nGroups) number of processes.
         */
        long long **bounds;
        int my_startGrp, my_nGroups;

        /* partition read workload among nGroups processes. When nprocs is
         * larger than nGroups, some processes have no data to read, but
         * participate calls to MPI_Scatter
         */
        my_nGroups = nGroups / nprocs;
        my_startGrp = my_nGroups * rank;
        if (rank < nGroups % nprocs) {
            my_startGrp += rank;
            my_nGroups++;
        }
        else
            my_startGrp += nGroups % nprocs;

        if (debug)
            printf("%s nGroups=%d spill_idx=%d my_startGrp=%d my_nGroups=%d\n",
                   __func__,nGroups,spill_idx,my_startGrp,my_nGroups);

        /* only processes got assigned read the evt.seq and calculate the lower
         * and upper bounds */
        if (my_nGroups > 0) {
            bounds = (long long**) malloc(my_nGroups * sizeof(long long*));
            if (bounds == NULL) CHECK_ERROR(-1, "malloc");
            bounds[0] = (long long*) malloc(my_nGroups * nprocs * 2 * sizeof(long long));
            if (bounds[0] == NULL) CHECK_ERROR(-1, "malloc");
            for (g=1; g<my_nGroups; g++)
                bounds[g] = bounds[g-1] + nprocs * 2;
        }
        groups += my_startGrp;

        if (seq_opt == 3) {
            /* read_evt_seq_aggr_all is a collective call.  Calling this
             * function makes more sense than mpi_read_datasets_by_IDs(), as
             * the contents of evt.seq matters only when calculating the upper
             * and lower bounds. They are useless thereafter. If there are more
             * than one evt.seq assigned, the read buffer can be reused, hence
             * save memory space.
             */
            read_len += read_evt_seq_aggr_all(fd, fh, groups, my_nGroups, nprocs,
                                              rank, spill_idx-my_startGrp, starts,
                                              ends, bounds, inflate_t);
        }
        else if (seq_opt == 4) {
            /* POSIX read to read evt.seq datasets */
            for (g=0; g<my_nGroups; g++) {
                if (g == spill_idx-my_startGrp) continue;

                hsize_t dims[2];
                hid_t dset = H5Dopen2(fd, groups[g].dset_names[0], H5P_DEFAULT);
                if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

                /* inquire dimension sizes of dset */
                hid_t fspace = H5Dget_space(dset);
                if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
                err = H5Sget_simple_extent_dims(fspace, dims, NULL);
                if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
                if (dims[1] != 1) CHECK_ERROR(err, "dims[1] != 1");
                err = H5Sclose(fspace);
                if (err < 0) CHECK_ERROR(err, "H5Sclose");

                /* evt.seq data type is aways H5T_STD_I64LE and of size 8 bytes */
                hid_t dtype = H5Dget_type(dset);
                if (H5Tequal(dtype, H5T_STD_I64LE) <= 0) CHECK_ERROR(-1, "dtype != H5T_STD_I64LE");
                size_t dtype_size = H5Tget_size(dtype);
                if (dtype_size != 8) CHECK_ERROR(-1, "evt.seq dtype_size != 8");
                err = H5Tclose(dtype);
                if (err < 0) CHECK_ERROR(err, "H5Tclose");

                void *seqBuf = (void*) malloc(dims[0] * dtype_size);
                if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");

                /* posix_read_dataset_by_ID() is an independent call */
                read_len += posix_read_dataset_by_ID(dset, posix_fd, seqBuf,
                                                     &groups[g].read_t[0],
                                                     inflate_t);
                for (k=0, j=0; j<nprocs; j++) {
                    bounds[g][k++] = binary_search_min(starts[j], seqBuf, dims[0]);
                    bounds[g][k++] = binary_search_max(ends[j],   seqBuf, dims[0]);
                }
                free(seqBuf);
                err = H5Dclose(dset);
                if (err < 0) CHECK_ERROR(err, "H5Dclose");
            }
        }

        /* bounds calculated from nGroups number of evt.seq are read by nGroups
         * number of processes in parallel. There will be nGroups calls to
         * MPI_Scatter from nGroups number of roots. Now calculate rank of root
         * process for each MPI_Scatter. Each time MPI_Scatter is called, all
         * processes must agree on the root rank.
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
            if (g == spill_idx) { /* no need to read /spill/evt.seq */
                lowers[g] = starts[rank];
                uppers[g] = ends[rank];
                continue;
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
        return read_len;
    }

    /* for other options, evt.seq datasets are read one at a time, so a single
     * array of bounds is enough */
    long long *bounds;

    /* seq_opt == 0, 1, or 2 */
    if (seq_opt == 2 && rank == 0) {
        bounds = (long long*) malloc(nprocs * 2 * sizeof(long long));
        if (bounds == NULL) CHECK_ERROR(-1, "malloc");
    }

    for (g=0; g<nGroups; g++) {
        /* read dataset evt.seq, the first dataset in the group, and calculate
         * the event range (lower and upper bound indices) responsible by this
         * process
         */
        if (g == spill_idx) {
            /* no need to read /spill/evt.seq */
            lowers[g] = starts[rank];
            uppers[g] = ends[rank];
            continue;
        }

        /* open dataset 'evt.seq', first in the group */
        seq = H5Dopen2(fd, groups[g].dset_names[0], H5P_DEFAULT);
        if (seq < 0) CHECK_ERROR(seq, "H5Dopen2");

        /* inquire dimension sizes of 'evt.seq' in this group */
        hsize_t *dims = groups[g].dims[0];
        hid_t fspace = H5Dget_space(seq);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        hsize_t ndims = H5Sget_simple_extent_ndims(fspace);
        if (ndims != 2) CHECK_ERROR(-1, "H5Sget_simple_extent_ndims");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        if (dims[1] != 1) CHECK_ERROR(err, "dims[1] != 1");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");

        /* inquire chunk size along each dimension */
        hsize_t *chunk_dims = groups[g].chunk_dims[0];
        hid_t chunk_plist = H5Dget_create_plist(seq);
        if (chunk_plist < 0) CHECK_ERROR(err, "H5Dget_create_plist");
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        if (chunk_ndims != 2) CHECK_ERROR(-1, "chunk_ndims != 2");
        if (chunk_dims[1] != 1) CHECK_ERROR(-1, "chunk_dims[1] != 1");
        err = H5Pclose(chunk_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");

        /* data type of evt.seq is 64-bit integer */
        hid_t dtype = H5Dget_type(seq);
        if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
        size_t dtype_size = H5Tget_size(dtype);
        if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");
        groups[g].dtype_size[0] = dtype_size;
        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");

        size_t buf_len = dims[0] * dims[1] * dtype_size;
        if (seq_opt == 2) {
            /* rank 0 reads evt.seq, calculates lower and upper bounds for all
             * processes and scatters
             */
            if (rank == 0) {
                seqBuf = (int64_t*) malloc(buf_len);
                if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");
                if (profile == 2) groups[g].read_t[0]=MPI_Wtime();
                err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, seqBuf);
                if (err < 0) CHECK_ERROR(err, "H5Dread");
                if (profile == 2) groups[g].read_t[0]=MPI_Wtime()-groups[g].read_t[0];
                read_len += buf_len;

                for (j=0, k=0; j<nprocs; j++) {
                    bounds[k++] = binary_search_min(starts[j], seqBuf, dims[0]);
                    bounds[k++] = binary_search_max(ends[j],   seqBuf, dims[0]);
                }
            }
            MPI_Scatter(bounds, 2, MPI_LONG_LONG, lower_upper, 2, MPI_LONG_LONG, 0, comm);
            lowers[g] = lower_upper[0];
            uppers[g] = lower_upper[1];
        }
        else {
            if (seq_opt == 0) {
                /* rank 0 reads evt.seq and broadcasts it */
                seqBuf = (int64_t*) malloc(buf_len);
                if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");
                if (rank == 0) {
                    if (profile == 2) groups[g].read_t[0]=MPI_Wtime();
                    err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, seqBuf);
                    if (err < 0) CHECK_ERROR(err, "H5Dread");
                    if (profile == 2) groups[g].read_t[0]=MPI_Wtime()-groups[g].read_t[0];
                    read_len += buf_len;
                }
                MPI_Bcast(seqBuf, dims[0], MPI_LONG_LONG, 0, comm);
            }
            else {
                /* collective-read-the-whole-dataset is bad */
                seqBuf = (int64_t*) malloc(buf_len);
                if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");
                if (profile == 2) groups[g].read_t[0]=MPI_Wtime();
                err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL, xfer_plist, seqBuf);
                /* all-processes-read-the-whole-dataset is even worse */
                // err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, seqBuf);
                if (err < 0) CHECK_ERROR(err, "H5Dread");
                if (profile == 2) groups[g].read_t[0]=MPI_Wtime()-groups[g].read_t[0];
                read_len += buf_len;
            }
            /* find the array index range from 'lower' to 'upper' that falls into
             * this process's partition domain.
             */
            lowers[g] = binary_search_min(starts[rank], seqBuf, dims[0]);
            uppers[g] = binary_search_max(ends[rank],   seqBuf, dims[0]);
        }

        err = H5Dclose(seq);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");

        if (debug) {
            if (rank == 0)
                printf("%s[0]=%"PRId64" seq[dims[0]-1=%zd]=%"PRId64"\n",
                       groups[g].dset_names[0],seqBuf[0],(size_t)dims[0]-1,seqBuf[dims[0]-1]);

            printf("%d: %s starts[rank]=%zd ends[rank]=%zd lower=%zd upper=%zd\n",
                   rank, groups[g].dset_names[0], (size_t)starts[rank],(size_t)ends[rank], lowers[g], uppers[g]);
        }

        if (seqBuf != NULL) free(seqBuf);
    }
    if (seq_opt == 2 && rank == 0) free(bounds);

    return read_len;
}

/*----< hdf5_read_subarray() >-----------------------------------------------*/
/* call H5Dread to read ONE subarray of ONE dataset from file, The subarray
 * starts from index 'lower' to index 'upper'
 */
ssize_t
hdf5_read_subarray(hid_t       fd,         /* HDF5 file descriptor */
                   const char *dset_name,  /* dataset name */
                   hsize_t     lower,      /* array index lower bound */
                   hsize_t     upper,      /* array index upper bound */
                   hid_t       xfer_plist, /* data transfer property */
                   void       *buf)        /* user read buffer */
{
    herr_t  err;
    hid_t   dset, fspace, mspace, dtype;
    int     ndims;
    hsize_t start[2], count[2], one[2]={1,1}, dims[2];
    size_t dtype_size;
    ssize_t read_len;

    /* open dataset */
    dset = H5Dopen2(fd, dset_name, H5P_DEFAULT);
    if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

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

    /* calculate read size in bytes and allocate read buffers */
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
    err = H5Dclose(dset);
    if (err < 0) CHECK_ERROR(err, "H5Dclose");

    return read_len;
}

/*----< hdf5_read_subarrays() >----------------------------------------------*/
/* call H5Dread to read subarrays of MULTIPLE datasets, one subarray at a time
 */
static ssize_t
hdf5_read_subarrays(hid_t       fd,
                    int         rank,
                    NOvA_group *group,
                    size_t      lower,
                    size_t      upper,
                    int         profile,
                    hid_t       xfer_plist)
{
    int d ;
    herr_t err;
    ssize_t read_len=0;

    /* iterate all the remaining datasets in this group */
    for (d=1; d<group->nDatasets; d++) {
        hsize_t one[2]={1,1}, start[2], count[2];

        if (verbose && rank == 0) printf("dataset %s\n", group->dset_names[d]);

        /* open dataset */
        hid_t dset = H5Dopen2(fd, group->dset_names[d], H5P_DEFAULT);
        if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

        /* inquire dimension sizes of dset */
        hsize_t *dims = group->dims[d];
        hid_t fspace = H5Dget_space(dset);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        int ndims = H5Sget_simple_extent_ndims(fspace);
        if (ndims != 2) CHECK_ERROR(-1, "H5Sget_simple_extent_ndims");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");

        /* inquire chunk size along each dimension */
        hsize_t *chunk_dims = group->chunk_dims[d];
        hid_t chunk_plist = H5Dget_create_plist(dset);
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
        hid_t dtype = H5Dget_type(dset);
        if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
        group->dtype_size[d] = H5Tget_size(dtype);
        if (group->dtype_size[d] == 0) CHECK_ERROR(-1, "H5Tget_size");

        /* calculate read size in bytes and allocate read buffers */
        size_t subarray_len = count[0] * count[1] * group->dtype_size[d];
        group->buf[d] = (void*) malloc(subarray_len);
        if (group->buf[d] == NULL) CHECK_ERROR(-1, "malloc");
        read_len += subarray_len;

        /* collectively read dataset d's contents */
        if (profile == 2) group->read_t[d]=MPI_Wtime();
        err = H5Dread(dset, dtype, mspace, fspace, xfer_plist, group->buf[d]);
        if (err < 0) CHECK_ERROR(err, "H5Dread");
        if (profile == 2) group->read_t[d]=MPI_Wtime()-group->read_t[d];

        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");
        err = H5Sclose(mspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");
        err = H5Dclose(dset);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");
    }

    return read_len;
}

/*----< mpio_read_subarray() >-----------------------------------------------*/
/* A single call to MPI_File_read_all to read all raw chunks intersecting ONE
 * subarray of ONE dataset, decompress the raw chunks, and copy the requested
 * data to user buffer
 */
ssize_t
mpio_read_subarray(hid_t          fd,         /* HDF5 file descriptor */
                   MPI_File       fh,         /* MPI file handler */
                   const char    *dset_name,  /* dataset name */
                   hsize_t        lower,      /* array index lower bound */
                   hsize_t        upper,      /* array index upper bound */
                   unsigned char *buf,        /* user read buffer */
                   double        *inflate_t)  /* timer to measure inflation */
{
    int j, mpi_err;
    herr_t  err;
    hsize_t dims[2], chunk_dims[2];
    ssize_t zip_len=0, read_len = 0;
    double timing;

    /* open dataset */
    hid_t dset = H5Dopen2(fd, dset_name, H5P_DEFAULT);
    if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

    /* inquire dimension sizes of dset */
    hid_t fspace = H5Dget_space(dset);
    if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
    err = H5Sget_simple_extent_dims(fspace, dims, NULL);
    if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
    int ndims = H5Sget_simple_extent_ndims(fspace);
    if (ndims != 2) CHECK_ERROR(-1, "H5Sget_simple_extent_ndims");
    err = H5Sclose(fspace);
    if (err < 0) CHECK_ERROR(err, "H5Sclose");

    /* inquire chunk size along each dimension */
    hid_t chunk_plist = H5Dget_create_plist(dset);
    if (chunk_plist < 0) CHECK_ERROR(chunk_plist, "H5Dget_create_plist");
    int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
    if (chunk_ndims != 2) CHECK_ERROR(-1, "H5Pget_chunk");
    if (chunk_dims[1] != dims[1]) CHECK_ERROR(-1, "chunk_dims[1] != dims[1]");
    err = H5Pclose(chunk_plist);
    if (err < 0) CHECK_ERROR(err, "H5Pclose");

    /* inquire data type and size */
    hid_t dtype = H5Dget_type(dset);
    if (dtype < 0) CHECK_ERROR(err, "H5Dget_type");
    size_t dtype_size = H5Tget_size(dtype);
    if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");
    err = H5Tclose(dtype);
    if (err < 0) CHECK_ERROR(err, "H5Tclose");

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
    err = H5Dclose(dset);
    if (err < 0) CHECK_ERROR(err, "H5Dclose");

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
    if (inflate_t != NULL)  timing = MPI_Wtime();
    size_t whole_chunk_size = chunk_dims[0] * chunk_dims[1] * dtype_size;
    unsigned char *whole_chunk = (unsigned char*) malloc(whole_chunk_size);
    if (whole_chunk == NULL) CHECK_ERROR(-1, "malloc");
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
    free(whole_chunk);
    free(zipBuf);
    free(disp_indx);
    if (inflate_t != NULL) *inflate_t += MPI_Wtime() - timing;

    return read_len;
}

/*----< mpio_read_subarrays() >----------------------------------------------*/
/* Multiple calls to MPI_File_read_all() to read raw chunks of subarrays of
 * MULTIPLE datasets, one call per dataset, and decompress into user buffers
 */
static ssize_t
mpio_read_subarrays(hid_t       fd,
                    int         rank,
                    NOvA_group *group,
                    size_t      lower,
                    size_t      upper,
                    int         profile,
                    MPI_File    fh,
                    double     *inflate_t)
{
    int j, d, mpi_err;
    herr_t  err;
    hid_t   dset;
    ssize_t buf_len, zip_len, read_len=0;
    unsigned char *whole_chunk;
    double timing;

    for (d=1; d<group->nDatasets; d++) {
        hsize_t *dims       = group->dims[d];
        hsize_t *chunk_dims = group->chunk_dims[d];

        /* open dataset */
        dset = H5Dopen2(fd, group->dset_names[d], H5P_DEFAULT);
        if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

        /* find metadata of all the chunks of this dataset */

        /* inquire dimension sizes of dset */
        hid_t fspace = H5Dget_space(dset);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        int ndims = H5Sget_simple_extent_ndims(fspace);
        if (ndims != 2) CHECK_ERROR(-1, "H5Sget_simple_extent_ndims");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");

        /* inquire chunk size along each dimension */
        hid_t chunk_plist = H5Dget_create_plist(dset);
        if (chunk_plist < 0) CHECK_ERROR(err, "H5Dget_create_plist");
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        if (chunk_ndims != 2) CHECK_ERROR(-1, "chunk_ndims != 2");
        if (chunk_dims[1] != dims[1]) CHECK_ERROR(-1, "chunk_dims[1] != dims[1]");
        err = H5Pclose(chunk_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");

        /* get data type and size */
        hid_t dtype = H5Dget_type(dset);
        if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
        size_t dtype_size = H5Tget_size(dtype);
        if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");
        group->dtype_size[d] = dtype_size;
        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");

        /* calculate buffer read size in bytes and allocate read buffer */
        buf_len = (upper - lower + 1) * dims[1] * dtype_size;
        group->buf[d] = (void*) malloc(buf_len);
        if (group->buf[d] == NULL) CHECK_ERROR(-1, "malloc");
        read_len += buf_len;

        hsize_t nChunks, offset[2];

        /* find the number of chunks to be read by this process */
        nChunks = (upper / chunk_dims[0]) - (lower / chunk_dims[0]) + 1;

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
        offset[0] = (lower / chunk_dims[0]) * chunk_dims[0];
        offset[1] = 0;
        zip_len = 0;
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
        err = H5Dclose(dset);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");

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
        unsigned char *chunk_buf, *chunk_buf_ptr;
        chunk_buf = (unsigned char*) malloc(zip_len);
        if (chunk_buf == NULL) CHECK_ERROR(-1, "malloc");
        chunk_buf_ptr = chunk_buf;

        /* collective read */
        if (profile == 2) group->read_t[d]=MPI_Wtime();
        mpi_err = MPI_File_read_all(fh, chunk_buf, zip_len, MPI_BYTE, MPI_STATUS_IGNORE);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_read_all");
        if (profile == 2) group->read_t[d]=MPI_Wtime()-group->read_t[d];

        /* decompress each chunk into group->buf[d] */
        if (inflate_t != NULL) timing = MPI_Wtime();
        size_t whole_chunk_size = chunk_dims[0] * chunk_dims[1] * dtype_size;
        whole_chunk = (unsigned char*) malloc(whole_chunk_size);
        if (whole_chunk == NULL) CHECK_ERROR(-1, "malloc");
        for (j=0; j<nChunks; j++) {
            int ret;
            z_stream z_strm;
            z_strm.zalloc    = Z_NULL;
            z_strm.zfree     = Z_NULL;
            z_strm.opaque    = Z_NULL;
            z_strm.avail_in  = disp_indx[j].block_len;
            z_strm.next_in   = chunk_buf_ptr;
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
                memcpy(group->buf[d], whole_chunk + off, len);
            }
            else if (disp_indx[j].block_idx == nChunks - 1) { /* last chunk */
                size_t len = (upper+1) % chunk_dims[0];
                if (len == 0) len = chunk_dims[0];
                size_t off = upper + 1 - len - lower;
                off *= chunk_dims[1] * dtype_size;
                len *= chunk_dims[1] * dtype_size;
                memcpy(group->buf[d] + off, whole_chunk, len);
            }
            else { /* middle chunk, copy the full chunk */
                size_t off = chunk_dims[0] - lower % chunk_dims[0];
                off += (disp_indx[j].block_idx - 1) * chunk_dims[0];
                off *= chunk_dims[1] * dtype_size;
                memcpy(group->buf[d] + off, whole_chunk, whole_chunk_size);
            }
            chunk_buf_ptr += disp_indx[j].block_len;
        }
        free(whole_chunk);
        free(chunk_buf);
        free(disp_indx);
        if (inflate_t != NULL) *inflate_t += MPI_Wtime() - timing;
    }
    return read_len;
}

/*----< mpio_read_subarrays_aggr() >-----------------------------------------*/
/* A single call to MPI_File_read_all() to read all raw chunks of subarrays of
 * multiple datasets, and decompress into user buffers
 */
static ssize_t
mpio_read_subarrays_aggr(hid_t       fd,
                         int         rank,
                         NOvA_group *group,
                         size_t      lower,
                         size_t      upper,
                         MPI_File    fh,
                         double     *inflate_t)
{
    int d, j, k, mpi_err;
    herr_t  err;
    hsize_t **size;
    haddr_t **addr;
    size_t nChunks=0, max_chunk_size=0, buf_len, *dtype_size;
    ssize_t read_len=0, zip_len=0;
    double timing;

    dtype_size = group->dtype_size;

    /* allocate space to save file offsets and sizes of individual chunks */
    addr = (haddr_t**) malloc(group->nDatasets * sizeof(haddr_t*));
    if (addr == NULL) CHECK_ERROR(-1, "malloc");
    size = (hsize_t**) malloc(group->nDatasets * sizeof(hsize_t*));
    if (size == NULL) CHECK_ERROR(-1, "malloc");

    /* collect metadata of all chunks of all datasets in group g, including
     * number of chunks and their file offsets and compressed sizes
     */
    for (d=1; d<group->nDatasets; d++) {
        hsize_t *dims       = group->dims[d];
        hsize_t *chunk_dims = group->chunk_dims[d];

        /* open dataset */
        hid_t dset = H5Dopen2(fd, group->dset_names[d], H5P_DEFAULT);
        if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

        /* inquire dimension sizes of dset */
        hid_t fspace = H5Dget_space(dset);
        if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");

        /* inquire chunk size along each dimension */
        hid_t chunk_plist = H5Dget_create_plist(dset);
        if (chunk_plist < 0) CHECK_ERROR(err, "H5Dget_create_plist");
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        if (chunk_ndims != 2) CHECK_ERROR(-1, "chunk_ndims != 2");
        if (chunk_dims[1] != dims[1]) CHECK_ERROR(-1, "chunk_dims[1] != dims[1]");
        err = H5Pclose(chunk_plist);
        if (err < 0) CHECK_ERROR(err, "H5Pclose");

        /* get data type and size */
        hid_t dtype = H5Dget_type(dset);
        if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
        dtype_size[d] = H5Tget_size(dtype);
        if (dtype_size[d] == 0) CHECK_ERROR(-1, "H5Tget_size");
        err = H5Tclose(dtype);
        if (err < 0) CHECK_ERROR(err, "H5Tclose");
        buf_len = chunk_dims[0] * chunk_dims[1] * dtype_size[d];
        max_chunk_size = MAX(max_chunk_size, buf_len);

        /* calculate buffer read size in bytes and allocate read buffer */
        buf_len = (upper - lower + 1) * dims[1] * dtype_size[d];
        group->buf[d] = (void*) malloc(buf_len);
        if (group->buf[d] == NULL) CHECK_ERROR(-1, "malloc");
        read_len += buf_len;

        /* find the number of chunks to be read by this process */
        group->nChunks[d] = (upper / chunk_dims[0]) - (lower / chunk_dims[0]) + 1;
        nChunks += group->nChunks[d];

        /* collect offsets of all chunks of this dataset */
        addr[d] = (haddr_t*) malloc(group->nChunks[d] * sizeof(haddr_t));
        if (addr[d] == NULL) CHECK_ERROR(-1, "malloc");
        size[d] = (hsize_t*) malloc(group->nChunks[d] * sizeof(hsize_t));
        if (size[d] == NULL) CHECK_ERROR(-1, "malloc");
        hsize_t offset[2];
        offset[0] = (lower / chunk_dims[0]) * chunk_dims[0];
        offset[1] = 0;
        for (j=0; j<group->nChunks[d]; j++) {
            err = H5Dget_chunk_info_by_coord(dset, offset, NULL, &addr[d][j],
                                             &size[d][j]);
            if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");
            zip_len += size[d][j];
            offset[0] += chunk_dims[0];
        }
        err = H5Dclose(dset);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");
    }

    /* Note file offsets of chunks may not follow the increasing order of
     * chunk IDs read by this process. We must sort the offsets before
     * creating a file type. Construct an array of off-len-indx for such
     * sorting.
     */
    off_len *disp_indx = (off_len*) malloc(nChunks * sizeof(off_len));
    if (disp_indx == NULL) CHECK_ERROR(-1, "malloc");

    int is_mono_nondecr = 1;
    k = 0;
    for (d=1; d<group->nDatasets; d++) {
        for (j=0; j<group->nChunks[d]; j++) {
            disp_indx[k].block_dsp = (MPI_Aint)addr[d][j];   /* chunk's file offset */
            disp_indx[k].block_len = (int)size[d][j];        /* compressed chunk size */
            disp_indx[k].block_idx = j*group->nDatasets + d; /* unique ID for this chunk */
            if (k > 0 && disp_indx[k].block_dsp < disp_indx[k-1].block_dsp)
                is_mono_nondecr = 0;
            k++;
        }
        free(addr[d]);
        free(size[d]);
    }
    free(addr);
    free(size);
    if (k != nChunks) CHECK_ERROR(-1, "k != nChunks");

    /* Sort chunk offsets into an increasing order, as MPI-IO requires that
     * for file view.
     * According to MPI standard Chapter 13.3, file displacements of filetype
     * must be non-negative and in a monotonically nondecreasing order. If the
     * file is opened for writing, neither the etype nor the filetype is
     * permitted to contain overlapping regions.
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
    mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_set_view");

    mpi_err = MPI_Type_free(&ftype);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Type_free");

    /* allocate buffer for reading the compressed chunks all at once */
    unsigned char *chunk_buf, *chunk_buf_ptr;
    chunk_buf = (unsigned char*) malloc(zip_len);
    if (chunk_buf == NULL) CHECK_ERROR(-1, "malloc");
    chunk_buf_ptr = chunk_buf;

    /* collective read */
    mpi_err = MPI_File_read_all(fh, chunk_buf, zip_len, MPI_BYTE, MPI_STATUS_IGNORE);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_read_all");

    /* decompress individual chunks into group->buf[d] */
    if (inflate_t != NULL) timing = MPI_Wtime();
    unsigned char *whole_chunk = (unsigned char*) malloc(max_chunk_size);
    if (whole_chunk == NULL) CHECK_ERROR(-1, "malloc");
    for (j=0; j<nChunks; j++) {
        int ret;
        z_stream z_strm;
        z_strm.zalloc    = Z_NULL;
        z_strm.zfree     = Z_NULL;
        z_strm.opaque    = Z_NULL;
        z_strm.avail_in  = disp_indx[j].block_len;
        z_strm.next_in   = chunk_buf_ptr;
        z_strm.avail_out = max_chunk_size;
        z_strm.next_out  = whole_chunk;
        ret = inflateInit(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateInit");
        ret = inflate(&z_strm, Z_SYNC_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) CHECK_ERROR(-1, "inflate");
        ret = inflateEnd(&z_strm);
        if (ret != Z_OK) CHECK_ERROR(-1, "inflateEnd");

        /* copy requested data to user buffer */
        hsize_t *chunk_dims;
        d = disp_indx[j].block_idx % group->nDatasets;  /* dataset ID */
        k = disp_indx[j].block_idx / group->nDatasets;  /* chunk ID of dataset d */
        chunk_dims = group->chunk_dims[d];
        /* copy requested data to user buffer */
        if (k == 0) { /* dataset d's first chunk */
            size_t off = lower % chunk_dims[0];
            size_t len;
            if (group->nChunks[d] == 1)
                len = upper - lower + 1;
            else
                len = chunk_dims[0] - off;
            len *= dtype_size[d];
            off *= dtype_size[d];
            memcpy(group->buf[d], whole_chunk + off, len);
        }
        else if (k == group->nChunks[d] - 1) { /* dataset d's last chunk */
            size_t len = (upper+1) % chunk_dims[0];
            if (len == 0) len = chunk_dims[0];
            size_t off = upper + 1 - len - lower;
            off *= dtype_size[d];
            len *= dtype_size[d];
            memcpy(group->buf[d] + off, whole_chunk, len);
        }
        else { /* middle chunk, copy the full chunk */
            size_t len = chunk_dims[0] * dtype_size[d];
            size_t off = chunk_dims[0] - lower % chunk_dims[0];
            off += (k - 1) * chunk_dims[0];
            off *= dtype_size[d];
            memcpy(group->buf[d] + off, whole_chunk, len);
        }
        chunk_buf_ptr += disp_indx[j].block_len;
    }
    free(whole_chunk);
    free(chunk_buf);
    free(disp_indx);
    if (inflate_t != NULL) *inflate_t += MPI_Wtime() - timing;

    return read_len;
}

/*----< data_parallelism() >-------------------------------------------------*/
/* Data parallelism is to have all processes in comm read each individual
 * datasets in parallel. Data partitioning is done by linearly and evenly the
 * unique event keys in /spill/evt.seq among all processes. These partitioning
 * ranges are used to calculate the array index lower and upper boundaries for
 * all processes by checking the dataset evt.seq in each group. The array index
 * lower and upper boundaries are the same for all datasets in the same group,
 * but different from other groups.
 */
static ssize_t
data_parallelism(MPI_Comm    comm,
                 const char *infile,
                 NOvA_group *groups,
                 int         nGroups,
                 int         spill_idx,
                 int         seq_opt,
                 int         dset_opt,
                 int         profile,
                 double     *timings)
{
    herr_t  err;
    hid_t   fd, fapl_id, xfer_plist;
    hsize_t *starts, *ends, nEvtIDs;
    int d, g, nprocs, rank, posix_fd, mpi_err;
    ssize_t read_len=0;
    size_t *lowers, *uppers;
    double open_t, read_seq_t, read_dset_t, close_t, inflate_t=0.0;
    MPI_File fh;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* allocate space to store dataset metadata */
    for (g=0; g<nGroups; g++) {
        size_t nDatasets = groups[g].nDatasets;

        /* allocate array of read buffer pointers */
        groups[g].buf = (void**) malloc(nDatasets * sizeof(void*));
        if (groups[g].buf == NULL) CHECK_ERROR(-1, "malloc");

        /* allocate space to store data type sizes */
        groups[g].dtype_size = (size_t*) calloc(nDatasets, sizeof(size_t));
        if (groups[g].dtype_size == NULL) CHECK_ERROR(-1, "calloc");

        /* allocate space to store dimension sizes */
        groups[g].dims = (hsize_t**) malloc(nDatasets * sizeof(hsize_t*));
        if (groups[g].dims == NULL) CHECK_ERROR(-1, "malloc");
        groups[g].dims[0] = (hsize_t*) malloc(nDatasets * 2 * sizeof(hsize_t));
        if (groups[g].dims[0] == NULL) CHECK_ERROR(-1, "malloc");
        for (d=1; d<nDatasets; d++)
            groups[g].dims[d] = groups[g].dims[d-1] + 2;

        /* allocate space to store chunk dimension sizes */
        groups[g].chunk_dims = (hsize_t**) malloc(nDatasets * sizeof(hsize_t*));
        if (groups[g].chunk_dims == NULL) CHECK_ERROR(-1, "malloc");
        groups[g].chunk_dims[0] = (hsize_t*) malloc(nDatasets * 2 * sizeof(hsize_t));
        if (groups[g].chunk_dims[0] == NULL) CHECK_ERROR(-1, "malloc");
        for (d=1; d<nDatasets; d++)
            groups[g].chunk_dims[d] = groups[g].chunk_dims[d-1] + 2;

        /* allocate space to store number of chunks and initialize to zeros */
        groups[g].nChunks = (size_t*) calloc(nDatasets, sizeof(size_t));
        if (groups[g].nChunks == NULL) CHECK_ERROR(-1, "calloc");
    }

    /* open file to get file ID of HDF5, MPI, POSIX ------------------------*/
    MPI_Barrier(comm);
    open_t = MPI_Wtime();

    /* create file access property list and add MPI communicator */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl_id < 0) CHECK_ERROR(fapl_id, "H5Pcreate");
    err = H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
    if (err < 0) CHECK_ERROR(err, "H5Pset_fapl_mpio");

    /* collectively open input file for reading */
    fd = H5Fopen(infile, H5F_ACC_RDONLY, fapl_id);
    if (fd < 0) {
        fprintf(stderr,"%d: Error: fail to open file %s (%s)\n",
                rank,  infile, strerror(errno));
        fflush(stderr);
        return -1;
    }
    err = H5Pclose(fapl_id);
    if (err < 0) CHECK_ERROR(err, "H5Pclose");

    /* set MPI-IO hints and open input file using MPI-IO */
    if (seq_opt == 3 || dset_opt > 0) {
        MPI_Info info = MPI_INFO_NULL;
        mpi_err = MPI_Info_create(&info);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Info_create");
        mpi_err = MPI_Info_set(info, "romio_cb_read", "enable");
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Info_set");
        mpi_err = MPI_File_open(comm, infile, MPI_MODE_RDONLY, info, &fh);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_open");
        mpi_err = MPI_Info_free(&info);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_Info_free");
    }
    if (seq_opt == 4) {
        posix_fd = open(infile, O_RDONLY);
        if (posix_fd < 0) CHECK_ERROR(-1, "open");
    }
    open_t = MPI_Wtime() - open_t;

    /* Read evt.seq datasets ------------------------------------------------*/
    MPI_Barrier(comm);
    read_seq_t = MPI_Wtime();

    /* this is an MPI_COMM_WORLD collective call */
    nEvtIDs = inq_num_unique_IDs(fd, "/spill/evt.seq");
    if (nEvtIDs < 0) return -1;

    /* starts[rank] and ends[rank] store the starting and ending event IDs that
     * are responsible by process rank
     */
    starts = (hsize_t*) malloc(nprocs * 2 * sizeof(hsize_t));
    if (starts == NULL) CHECK_ERROR(-1, "malloc");
    ends = starts + nprocs;

    /* calculate the range of event IDs responsible by all process and store
     * them in starts[nprocs] and ends[nprocs] */
    calculate_starts_ends(nprocs, rank, nEvtIDs, starts, ends);

    /* set MPI-IO collective transfer mode */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    if (xfer_plist < 0) CHECK_ERROR(err, "H5Pcreate");
    err = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    if (err < 0) CHECK_ERROR(err, "H5Pset_dxpl_mpio");

    /* calculate this process's array index range (lower and upper bounds) for
     * each group */
    lowers = (size_t*) malloc(nGroups * 2 * sizeof(size_t));
    if (lowers == NULL) CHECK_ERROR(-1, "malloc");
    uppers = lowers + nGroups;

    /* iterate all groups to calculate lowers[] and uppers[] */
    read_len += read_evt_seq(comm, fd, fh, posix_fd, groups, nGroups, profile,
                             seq_opt, xfer_plist, spill_idx, starts, ends,
                             lowers, uppers, &inflate_t);
    read_seq_t = MPI_Wtime() - read_seq_t;

    if (debug) {
        int test_seq_opt = (seq_opt == 0) ? 1 : 0;
        size_t *debug_lowers, *debug_uppers;
        debug_lowers = (size_t*) malloc(nGroups * 2 * sizeof(size_t));
        if (debug_lowers == NULL) CHECK_ERROR(-1, "malloc");
        debug_uppers = debug_lowers + nGroups;
        read_evt_seq(comm, fd, fh, posix_fd, groups, nGroups, profile,
                     test_seq_opt, xfer_plist, spill_idx, starts, ends,
                     debug_lowers, debug_uppers, &inflate_t);
        for (g=0; g<nGroups; g++) {
            if (debug_lowers[g] != lowers[g] || debug_uppers[g] != uppers[g]) {
                printf("%d: Error: group %d debug_lowers(%zd) != lowers(%zd) || debug_uppers(%zd) != uppers(%zd)\n",
                       rank, g, debug_lowers[g],lowers[g],debug_uppers[g],uppers[g]);
                if (debug_lowers[g] != lowers[g]) CHECK_ERROR(-1, "debug_lowers[g] != lowers[g]");
                if (debug_uppers[g] != uppers[g]) CHECK_ERROR(-1, "debug_upper[g] != upper[g]");
            }
        }
        free(debug_lowers);
    }

    /* read the remaining datasets ------------------------------------------*/
    MPI_Barrier(comm);
    read_dset_t = MPI_Wtime();

    /* Read the remaining datasets by iterating all groups */
    for (g=0; g<nGroups; g++) {
        if (dset_opt == 0)
            /* read datasets using H5Dread() */
            read_len += hdf5_read_subarrays(fd, rank, groups+g, lowers[g],
                                            uppers[g], profile, xfer_plist);
        else if (dset_opt == 1)
            /* read datasets using MPI-IO, one dataset at a time */
            read_len += mpio_read_subarrays(fd, rank, groups+g, lowers[g],
                                            uppers[g], profile, fh, &inflate_t);
        else if (dset_opt == 2)
            /* read datasets using MPI-IO, all datasets in one group and one
             * group at a time
             */
            read_len += mpio_read_subarrays_aggr(fd, rank, groups+g, lowers[g],
                                                 uppers[g], fh, &inflate_t);

        /* This is where PandAna performs computation to identify events of
         * interest from the read buffers
         */

        /* free read allocated buffers all at once */
        for (d=1; d<groups[g].nDatasets; d++)
            free(groups[g].buf[d]);
    }
    free(lowers);
    err = H5Pclose(xfer_plist);
    if (err < 0) CHECK_ERROR(err, "H5Pclose");

    read_dset_t = MPI_Wtime() - read_dset_t;

    /* close input file -----------------------------------------------------*/
    MPI_Barrier(comm);
    close_t = MPI_Wtime();
    err = H5Fclose(fd);
    if (err < 0) CHECK_ERROR(err, "H5Fclose");

    if (seq_opt == 3 || dset_opt > 0) {
        mpi_err = MPI_File_close(&fh);
        if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_close");
    }
    if (seq_opt == 4) close(posix_fd);
    close_t = MPI_Wtime() - close_t;

    /* free allocated memory space */
    if (starts != NULL) free(starts);

    for (g=0; g<nGroups; g++) {
        free(groups[g].nChunks);
        free(groups[g].chunk_dims[0]);
        free(groups[g].chunk_dims);
        free(groups[g].dims[0]);
        free(groups[g].dims);
        free(groups[g].dtype_size);
        free(groups[g].buf);
    }

    timings[0] = open_t;
    timings[1] = read_seq_t;
    timings[2] = read_dset_t;
    timings[3] = close_t;
    timings[4] = inflate_t;

    return read_len;
}

/*----< group_parallelism() >------------------------------------------------*/
static ssize_t
group_parallelism(MPI_Comm    comm,
                  const char *infile,
                  NOvA_group *groups,
                  int         nGroups,
                  int         spill_idx,
                  int         seq_opt,
                  int         dset_opt,
                  int         profile,
                  double     *timings)
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

        if (debug) {
            int grp_rank, grp_nprocs;
            MPI_Comm_rank(grp_comm, &grp_rank);
            MPI_Comm_size(grp_comm, &grp_nprocs);
            printf("%2d nGroups=%d my_startGrp=%2d my_nGroups=%2d grp_rank=%2d grp_nprocs=%2d\n",
                   rank,nGroups,my_startGrp,my_nGroups,grp_rank,grp_nprocs);
        }
    }

    groups += my_startGrp;
    spill_idx -= my_startGrp;

    /* within a group, data parallelism method is used */
    read_len = data_parallelism(grp_comm, infile, groups, my_nGroups,
                                spill_idx, seq_opt, dset_opt, profile,
                                timings);

    if (nprocs > nGroups)
        MPI_Comm_free(&grp_comm);

    return read_len;
}

/*----< chunk_statistics() >-------------------------------------------------*/
static int
chunk_statistics(MPI_Comm    comm,
                 const char *infile,
                 NOvA_group *groups,
                 int         nGroups,
                 int         seq_opt,
                 int         dset_opt,
                 int         profile,
                 int         spill_idx)
{
    herr_t  err;
    hid_t   fd, dset;
    hsize_t *starts, *ends;

    int g, d, j, k, nprocs, rank, nDatasets;
    int nchunks_shared=0, max_shared_chunks=0;
    int max_nchunks_read=0, min_nchunks_read=INT_MAX, total_nchunks=0;
    long long nEvtIDs, aggr_nchunks_read, my_nchunks_read=0, my_nchunks_read_nokeys=0;
    long long all_dset_size, all_evt_seq_size;
    long long all_dset_size_z, all_evt_seq_size_z;
    long long *bounds, maxRead=0, minRead=LONG_MAX;
    size_t *grp_sizes, *grp_zip_sizes, *grp_nChunks;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (profile == 2) {
        for (g=0; g<nGroups; g++)
            MPI_Allreduce(MPI_IN_PLACE, groups[g].read_t, groups[g].nDatasets,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        grp_sizes = (size_t*) calloc(nGroups * 3, sizeof(size_t));
        grp_zip_sizes = grp_sizes + nGroups;
        grp_nChunks = grp_zip_sizes + nGroups;
        bounds = (long long*) malloc(nprocs * 2 * sizeof(long long));
        if (bounds == NULL) CHECK_ERROR(-1, "malloc");
    }

    /* collect statistics describing chunk contention */
    fd = H5Fopen(infile, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* this is an MPI_COMM_WORLD collective call */
    nEvtIDs = inq_num_unique_IDs(fd, "/spill/evt.seq");
    if (rank == 0)
        printf("Number of unique evt IDs (size of /spill/evt.seq)=%lld\n",nEvtIDs);

    /* starts[rank] and ends[rank] store the starting and ending event IDs that
     * are responsible by process rank
     */
    starts = (hsize_t*) malloc(nprocs * 2 * sizeof(hsize_t));
    if (starts == NULL) CHECK_ERROR(-1, "malloc");
    ends = starts + nprocs;

    /* calculate the range of event IDs responsible by all process and store
     * them in starts[nprocs] and ends[nprocs] */
    calculate_starts_ends(nprocs, rank, nEvtIDs, starts, ends);

    all_dset_size = 0;
    all_evt_seq_size = 0;
    all_dset_size_z = 0;
    all_evt_seq_size_z = 0;
    nDatasets = 0;
    for (g=0; g<nGroups; g++) {
        size_t lower=0, upper=0;

        groups[g].nChunks = (size_t*) malloc(groups[g].nDatasets * sizeof(size_t));

        nDatasets += groups[g].nDatasets;
        if (g == spill_idx) {
            groups[g].nChunks[0] = 0;
            if (rank == 0) {
                for (j=0, k=0; j<nprocs; j++) {
                    bounds[k++] = starts[j];
                    bounds[k++] = ends[j];
                }
            }
            lower = starts[rank];
            upper = ends[rank];
        }
        else {
            /* rank 0 reads evt.seq, calculates lower and upper bounds for all
             * processes and scatters them.
             */
            long long lower_upper[2];

            if (rank == 0) {
                hsize_t dset_size, dims[2], chunk_dims[2];

                /* open dataset 'evt.seq', first in the group */
                hid_t seq = H5Dopen2(fd, groups[g].dset_names[0], H5P_DEFAULT);
                if (seq < 0) CHECK_ERROR(seq, "H5Dopen2");

                /* collect metadata of ect.seq again, as root may not have
                 * collected metadata of all evt.seq
                 */

                /* inquire dimension sizes of dset */
                hid_t fspace = H5Dget_space(seq);
                if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
                err = H5Sget_simple_extent_dims(fspace, dims, NULL);
                if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
                err = H5Sclose(fspace);
                if (err < 0) CHECK_ERROR(err, "H5Sclose");

                /* inquire chunk size along each dimension */
                hid_t chunk_plist = H5Dget_create_plist(seq);
                if (chunk_plist < 0) CHECK_ERROR(err, "H5Dget_create_plist");
                int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
                if (chunk_ndims != 2) CHECK_ERROR(-1, "chunk_ndims != 2");
                /* evt.seq chunk_dims[1] should always be 1 */
                if (chunk_dims[1] != 1) CHECK_ERROR(-1, "chunk_dims[1] != 1");
                err = H5Pclose(chunk_plist);
                if (err < 0) CHECK_ERROR(err, "H5Pclose");

                /* data type of evt.seq is 64-bit integer */
                hid_t dtype = H5Dget_type(seq);
                if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
                size_t dtype_size = H5Tget_size(dtype);
                if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");
                err = H5Tclose(dtype);
                if (err < 0) CHECK_ERROR(err, "H5Tclose");

                dset_size = dims[0] * dims[1] * dtype_size;
                all_evt_seq_size += dset_size;
                grp_sizes[g] += dset_size;

                groups[g].nChunks[0] = dims[0] / chunk_dims[0];
                if (dims[0] % chunk_dims[0]) groups[g].nChunks[0]++;

                my_nchunks_read += groups[g].nChunks[0];
                total_nchunks += groups[g].nChunks[0];
                grp_nChunks[g] += groups[g].nChunks[0];

                /* calculate read sizes of compressed data */
                hsize_t offset[2]={0, 0};
                for (j=0; j<groups[g].nChunks[0]; j++) {
                    haddr_t addr;
                    hsize_t size;
                    err = H5Dget_chunk_info_by_coord(seq, offset, NULL, &addr,
                          &size);
                    if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");
                    all_evt_seq_size_z += size;
                    offset[0] += chunk_dims[0];
                    grp_zip_sizes[g] += size;
                }

                int64_t *seqBuf = (int64_t*) malloc(dset_size);
                if (seqBuf == NULL) CHECK_ERROR(-1, "malloc");
                err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, seqBuf);
                if (err < 0) CHECK_ERROR(err, "H5Dread");

                for (j=0, k=0; j<nprocs; j++) {
                    bounds[k++] = binary_search_min(starts[j], seqBuf, dims[0]);
                    bounds[k++] = binary_search_max(ends[j],   seqBuf, dims[0]);
                }
                free(seqBuf);

                if (seq_opt == 1) nchunks_shared += groups[g].nChunks[0];

                err = H5Dclose(seq);
                if (err < 0) CHECK_ERROR(err, "H5Dclose");
            }

            MPI_Scatter(bounds, 2, MPI_LONG_LONG, lower_upper, 2, MPI_LONG_LONG,
                        0, comm);
            lower = lower_upper[0];
            upper = lower_upper[1];

            long long ll = groups[g].nChunks[0];
            MPI_Bcast(&ll, 1, MPI_LONG_LONG, 0, comm);
            groups[g].nChunks[0] = ll;
            if (rank > 0 && seq_opt == 1)
                my_nchunks_read += groups[g].nChunks[0];
        }

        for (d=1; d<groups[g].nDatasets; d++) {
            hsize_t dims[2], chunk_dims[2];

            /* open dataset */
            dset = H5Dopen2(fd, groups[g].dset_names[d], H5P_DEFAULT);
            if (dset < 0) CHECK_ERROR(dset, "H5Dopen2");

            /* collect metadata of ect.seq again, as this process may not have
             * collected metadata of all evt.seq
             */

            /* inquire dimension sizes of dset */
            hid_t fspace = H5Dget_space(dset);
            if (fspace < 0) CHECK_ERROR(fspace, "H5Dget_space");
            err = H5Sget_simple_extent_dims(fspace, dims, NULL);
            if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
            err = H5Sclose(fspace);
            if (err < 0) CHECK_ERROR(err, "H5Sclose");

            /* inquire chunk size along each dimension */
            hid_t chunk_plist = H5Dget_create_plist(dset);
            if (chunk_plist < 0) CHECK_ERROR(err, "H5Dget_create_plist");
            int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
            if (chunk_ndims != 2) CHECK_ERROR(-1, "chunk_ndims != 2");
            err = H5Pclose(chunk_plist);
            if (err < 0) CHECK_ERROR(err, "H5Pclose");
            /* Note chunk_dims[1] may be larger than 1, but dims[1] is not
             * chunked
             */
            if (chunk_dims[1] != dims[1]) CHECK_ERROR(-1, "chunk_dims[1] != dims[1]");

            /* data type of evt.seq is 64-bit integer */
            hid_t dtype = H5Dget_type(dset);
            if (dtype < 0) CHECK_ERROR(dtype, "H5Dget_type");
            size_t dtype_size = H5Tget_size(dtype);
            if (dtype_size == 0) CHECK_ERROR(-1, "H5Tget_size");
            err = H5Tclose(dtype);
            if (err < 0) CHECK_ERROR(err, "H5Tclose");

            /* calculate number of chunks of this dataset */
            groups[g].nChunks[d] = dims[0] / chunk_dims[0];
            if (dims[0] % chunk_dims[0]) groups[g].nChunks[d]++;
            total_nchunks += groups[g].nChunks[d];

            /* calculate number of chunks read by this process */
            int nchunks = (upper / chunk_dims[0]) - (lower / chunk_dims[0]) + 1;
            my_nchunks_read += nchunks;
            my_nchunks_read_nokeys += nchunks;
            max_nchunks_read = MAX(max_nchunks_read, nchunks);
            min_nchunks_read = MIN(min_nchunks_read, nchunks);

            /* calculate nchunks_shared and max_shared_chunks */
            if (rank == 0) {
                /* bounds[] is populated on root only */
                int prev_chunk_id=-1, max_chunks=1;
                int k=0, chunk_id, prev_chunk=-1, prev_prev_chunk=-1;
                for (j=0; j<nprocs; j++) {
                    size_t read_len = (bounds[k+1] - bounds[k] + 1)
                                    * dims[1] * dtype_size;
                    maxRead = MAX(maxRead, read_len);
                    minRead = MIN(minRead, read_len);
                    chunk_id = bounds[k++] / chunk_dims[0];
                    if (chunk_id == prev_chunk) {
                        if (chunk_id > prev_prev_chunk) /* count only once */
                            nchunks_shared++;
                        prev_prev_chunk = prev_chunk;
                        prev_chunk = chunk_id;
                    }
                    else
                        prev_prev_chunk = -1;
                    prev_chunk = bounds[k++] / chunk_dims[0];

                    if (chunk_id == prev_chunk_id)
                        max_chunks++;
                    else {
                        max_shared_chunks = MAX(max_shared_chunks, max_chunks);
                        max_chunks = 1;
                        prev_chunk_id = prev_chunk;
                    }
                }
                max_shared_chunks = MAX(max_shared_chunks, max_chunks);
                all_dset_size += dims[0] * dims[1] * dtype_size;
                grp_sizes[g] += dims[0] * dims[1] * dtype_size;
                grp_nChunks[g] += groups[g].nChunks[d];

                /* calculate read sizes of compressed data */
                hsize_t offset[2]={0, 0};
                for (j=0; j<groups[g].nChunks[d]; j++) {
                    haddr_t addr;
                    hsize_t size;
                    err = H5Dget_chunk_info_by_coord(dset, offset, NULL,
                                                     &addr, &size);
                    if (err < 0) CHECK_ERROR(err, "H5Dget_chunk_info_by_coord");
                    all_dset_size_z += size;
                    offset[0] += chunk_dims[0];
                    grp_zip_sizes[g] += size;
                }
            }
            err = H5Dclose(dset);
            if (err < 0) CHECK_ERROR(err, "H5Dclose");
        }
    }
    free(starts);
    H5Fclose(fd);
    if (rank == 0) free(bounds);

    MPI_Allreduce(&my_nchunks_read, &aggr_nchunks_read, 1, MPI_LONG_LONG,
                  MPI_SUM, comm);

    if (rank == 0) {
        printf("Read amount MAX=%.2f MiB MIN=%.2f MiB (per dataset, per process)\n",
               (float)maxRead/1048576.0,(float)minRead/1048576.0);
        printf("Amount of evt.seq datasets %.2f MiB = %.2f GiB (compressed %.2f MiB = %.2f GiB)\n",
               (float)all_evt_seq_size/1048576.0, (float)all_evt_seq_size/1073741824.0,
               (float)all_evt_seq_size_z/1048576.0, (float)all_evt_seq_size_z/1073741824.0);
        printf("Amount of  other  datasets %.2f MiB = %.2f GiB (compressed %.2f MiB = %.2f GiB)\n",
               (float)all_dset_size/1048576.0, (float)all_dset_size/1073741824.0,
               (float)all_dset_size_z/1048576.0, (float)all_dset_size_z/1073741824.0);
        all_dset_size += all_evt_seq_size;
        all_dset_size_z += all_evt_seq_size_z;
        printf("Sum amount of all datasets %.2f MiB = %.2f GiB (compressed %.2f MiB = %.2f GiB)\n",
               (float)all_dset_size/1048576.0, (float)all_dset_size/1073741824.0,
               (float)all_dset_size_z/1048576.0, (float)all_dset_size_z/1073741824.0);
        printf("total number of chunks in all %d datasets (exclude /spill/evt.seq): %d\n",
               nDatasets, total_nchunks);
        printf("Aggregate number of chunks read by all processes: %lld\n",
               aggr_nchunks_read);
        printf("        averaged per process: %.2f\n", (float)aggr_nchunks_read/nprocs);
        printf("        averaged per process per dataset: %.2f\n", (float)aggr_nchunks_read/nprocs/nDatasets);
        printf("Out of %d chunks, number of chunks read by two or more processes: %d\n",
               total_nchunks,nchunks_shared);
        printf("Out of %d chunks, most shared chunk is read by number of processes: %d\n",
               total_nchunks,max_shared_chunks);
        printf("----------------------------------------------------\n");

        if (profile == 2 && (seq_opt != 3 || dset_opt != 2)) {
            /* When seq_opt == 2, evt.seq of all groups are read together.
             * When dset_opt == 2, all datasets in a group are read together.
             */
            printf("----------------------------------------------------\n");
            for (j=0, g=0; g<nGroups; g++)
                for (d=0; d<groups[g].nDatasets; d++)
                    printf("dataset[%3d] nchunks: %4zd read time: %.4f sec. (%s)\n",
                           j++, groups[g].nChunks[d], groups[g].read_t[d], groups[g].dset_names[d]);
        }
        if (profile == 1) {
            for (g=0; g<nGroups; g++) {
                char *gname = strtok(groups[g].dset_names[0], "/");
                printf("group %46s size %5zd MiB (zipped %4zd MiB) nChunks=%zd\n",
                       gname,grp_sizes[g]/1048576, grp_zip_sizes[g]/1048576, grp_nChunks[g]);
            }
        }
        free(grp_sizes);

        printf("\n\n");
    }
    fflush(stdout);
    MPI_Barrier(comm);

    if (profile >= 1)
        printf("rank %3d: no. chunks read=%lld include evt.seq (max=%d min=%d avg=%.2f among %d datasets, exclude evt.seq)\n",
               rank, my_nchunks_read, max_nchunks_read, min_nchunks_read,
               (float)my_nchunks_read_nokeys/(float)nDatasets, nDatasets-nGroups);

    for (g=0; g<nGroups; g++)
        free(groups[g].nChunks);

    return 1;
}

/*----< usage() >------------------------------------------------------------*/
static void
usage(char *progname)
{
#define USAGE   "\
  [-h]           print this command usage message\n\
  [-v]           verbose mode (default: off)\n\
  [-d]           debug mode (default: off)\n\
  [-p number]    performance profiling method (0, 1, or 2)\n\
                 0: report file open, close, read timings (default)\n\
                 1: report number of chunks read per process\n\
                 2: report read times for individual datasets\n\
  [-s number]    read method for evt.seq (0, 1, 2, 3, or 4)\n\
                 0: root process reads evt.seq and broadcasts (default)\n\
                 1: all processes read the entire evt.seq collectively\n\
                 2: root process reads each evt.seq, one at a time, and\n\
                    scatters boundaries to other processes\n\
                 3: distribute reads of evt.seq datasets among processes,\n\
                    each assigned process makes one MPI collective read call\n\
                    to read all asigned evt.seq and scatter boundaries\n\
                 4: root POSIX reads all chunks of evt.seq, one dataset at a\n\
                    time, decompress, and scatter boundaries\n\
  [-m number]    read method for other datasets (0, 1, or 2)\n\
                 0: use H5Dread (default)\n\
                 1: use MPI_file_read_all one dataset at a time\n\
                 2: use MPI_file_read_all to read all datasets in one group\n\
                    at a time\n\
  [-r number]    parallelization method (0 or 1)\n\
                 0: data parallelism - all processes read each dataset in\n\
                    parallel (default)\n\
                 1: group parallelism - processes are divided among groups\n\
                    then data parallelism within each groups\n\
                 2: dataset parallelism - divide all datasets among processes\n\
  [-l file_name] name of file containing dataset names to be read\n\
  [-i file_name] name of input HDF5 file\n\
  *ph5concat version _PH5CONCAT_VERSION_ of _PH5CONCAT_RELEASE_DATE_\n"

    printf("Usage: %s [-h|-v|-d] [-p number] [-s number] [-m number] [-r number] [-l file_name] [-i file_name]\n%s\n",
           progname, USAGE);
}

/*----< main() >-------------------------------------------------------------*/
int main(int argc, char **argv)
{
    int seq_opt=0, dset_opt=0, profile=0, spill_idx;
    int c, d, g, nprocs, rank, nGroups, parallelism=0;
    char *listfile=NULL, *infile=NULL;
    double all_t, timings[6], max_t[6], min_t[6];
    long long read_len;
    NOvA_group *groups=NULL;

    verbose = 0;
    debug = 0;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* command-line arguments */
    while ((c = getopt(argc, argv, "hvdr:p:s:m:l:i:")) != -1)
        switch(c) {
            case 'h': if (rank  == 0) usage(argv[0]);
                      goto fn_exit;
            case 'v': verbose = 1;
                      break;
            case 'd': debug = 1;
                      break;
            case 'r': parallelism = atoi(optarg);
                      break;
            case 'p': profile = atoi(optarg);
                      break;
            case 's': seq_opt = atoi(optarg);
                      break;
            case 'm': dset_opt = atoi(optarg);
                      break;
            case 'l': listfile = strdup(optarg);
                      break;
            case 'i': infile = strdup(optarg);
                      break;
            default: break;
        }

    if (listfile == NULL) { /* list file name is mandatory */
        if (rank  == 0) {
            printf("Error: list file is missing\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }
    if (infile == NULL) { /* input file name is mandatory */
        if (rank  == 0) {
            printf("Error: input file is missing\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }
    if (seq_opt < 0 || seq_opt > 4) { /* option for reading evt.seq */
        if (rank  == 0) {
            printf("Error: option -s must be 0, 1, 2, 3, or 4\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }
    if (dset_opt < 0 || dset_opt > 2) { /* option for reading other datasets */
        if (rank  == 0) {
            printf("Error: option -m must be 0, 1, or 2\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }
    if (parallelism < 0 || parallelism > 2) { /* option of parallelization method */
        if (rank  == 0) {
            printf("Error: option -r must be 0, 1, or 2\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }

    /* From file 'listfile', read dataset names, calculate number of datasets,
     * number of groups, maximum number of datasets among groups, find the
     * array index of group /spill
     */
    nGroups = read_dataset_names(rank, listfile, &groups, &spill_idx);
    if (nGroups == -1) goto fn_exit;

    /* print the running parameters and metadata of input file */
    if (rank == 0) {
        int nDatasets, max_nDatasets, min_nDatasets;
        nDatasets = max_nDatasets = min_nDatasets = groups[0].nDatasets;
        for (g=1; g<nGroups; g++) {
            nDatasets += groups[g].nDatasets;
            max_nDatasets = MAX(max_nDatasets, groups[g].nDatasets);
            min_nDatasets = MIN(min_nDatasets, groups[g].nDatasets);
        }

        printf("Number of MPI processes = %d\n", nprocs);
        printf("Input dataset name text file '%s'\n", listfile);
        printf("Input concatenated HDF5 file '%s'\n", infile);
        printf("Number of groups   to read = %d\n", nGroups);
        printf("Number of datasets to read = %d\n", nDatasets);
        printf("MAX/MIN no. datasets per group = %d / %d\n", max_nDatasets, min_nDatasets);
        printf("Read evt.seq    method: ");
        if (seq_opt == 0)
            printf("root process H5Dread and broadcasts\n");
        else if (seq_opt == 1)
            printf("all processes H5Dread the entire evt.seq collectively\n");
        else if (seq_opt == 2)
            printf("root process H5Dread evt.seq and scatters boundaries\n");
        else if (seq_opt == 3)
            printf("Distributed MPI collective read evt.seq datasets, decompress, and scatters boundaries\n");
        else if (seq_opt == 4)
            printf("Distributed POSIX read evt.seq datasets, decompress, and scatters boundaries\n");

        printf("Read datasets   method: ");
        if (dset_opt == 0)
            printf("H5Dread, one dataset at a time\n");
        else if (dset_opt == 1)
            printf("MPI collective read and decompress, one dataset at a time\n");
        else if (dset_opt == 2)
            printf("MPI collective read and decompress, all datasets in one group at a time\n");

        printf("Parallelization method: ");
        if (parallelism == 0)
            printf("data parallelism (all processes read individual datasets in parallel)\n");
        else if (parallelism == 1)
            printf("group parallelism (processes divided into groups, data parallelism in each group)\n");
        else if (parallelism == 2)
            printf("dataset parallelism (divide all datasets among processes)\n");
    }
    fflush(stdout);

    /* allocate read timings for individual datasets */
    for (g=0; g<nGroups; g++) {
        groups[g].read_t = (double*) calloc(groups[g].nDatasets, sizeof(double));
        if (groups[g].read_t == NULL) CHECK_ERROR(-1, "calloc");
    }

    for (d=0; d<6; d++) timings[d] = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    all_t = MPI_Wtime();

    if (parallelism == 0) /* data parallelism */
        read_len = data_parallelism(MPI_COMM_WORLD, infile, groups, nGroups,
                                    spill_idx, seq_opt, dset_opt, profile,
                                    timings);
    else if (parallelism == 1) /* group parallelism */
        read_len = group_parallelism(MPI_COMM_WORLD, infile, groups, nGroups,
                                     spill_idx, seq_opt, dset_opt, profile,
                                     timings);
    else if (parallelism == 2) /* dataset parallelism */
        printf("dataset parallelism has not been implemented yet\n");

    timings[5] = MPI_Wtime() - all_t;

    /* find the max/min timings among all processes.
     *   timings[0] : open_t      -- file open
     *   timings[1] : read_seq_t  -- read evt.seq datasets
     *   timings[2] : read_dset_t -- read other datasets
     *   timings[3] : close_t     -- file close
     *   timings[4] : inflate_t   -- data inflation
     *   timings[5] : all_t       -- end-to-end
     */
    MPI_Reduce(timings, max_t, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(timings, min_t, 6, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    long long max_read_len, min_read_len;
    MPI_Reduce(&read_len, &max_read_len, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&read_len, &min_read_len, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("----------------------------------------------------\n");
        printf("MAX and MIN among all %d processes\n", nprocs);
        printf("MAX read amount: %.2f MiB (%.2f GiB)\n",
               (float)max_read_len/1048576.0, (float)max_read_len/1073741824.0);
        printf("MIN read amount: %.2f MiB (%.2f GiB)\n",(
               float)min_read_len/1048576.0, (float)min_read_len/1073741824.0);
        printf("MAX time: open=%.2f evt.seq=%.2f datasets=%.2f close=%.2f inflate=%.2f TOTAL=%.2f\n",
               max_t[0],max_t[1],max_t[2],max_t[3],max_t[4],max_t[5]);
        printf("MIN time: open=%.2f evt.seq=%.2f datasets=%.2f close=%.2f inflate=%.2f TOTAL=%.2f\n",
               min_t[0],min_t[1],min_t[2],min_t[3],min_t[4],min_t[5]);
        printf("----------------------------------------------------\n");
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (profile > 0)
        chunk_statistics(MPI_COMM_WORLD, infile, groups, nGroups, seq_opt,
                         dset_opt, profile, spill_idx);

fn_exit:
    if (groups != NULL) {
        for (g=0; g<nGroups; g++) {
            for (d=0; d<groups[g].nDatasets; d++)
                free(groups[g].dset_names[d]);
            free(groups[g].dset_names);
            free(groups[g].read_t);
        }
        free(groups);
    }
    if (listfile != NULL) free(listfile);
    if (infile != NULL) free(infile);

    MPI_Finalize();
    return 0;
}
