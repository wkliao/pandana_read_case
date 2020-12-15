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

static int verbose, debug;

typedef struct {
    int       nDatasets;  /* number of datasets in this group */
    char    **dset_names; /* [nDatasets] string names of datasets */
    void    **buf;        /* [nDatasets] read buffers */

    size_t   *dtype_size; /* [nDatasets] size of data type of datasets */
    hsize_t **dims;       /* [nDatasets][2] dimension sizes */
    hsize_t **chunk_dims; /* [nDatasets][2] chunk dimension sizes */
    size_t   *nChunks;    /* [nDatasets] number of chunks in each dataset */
    double   *read_t;     /* [nDatasets] read timings */
} NOvA_group;

/*----< read_dataset_names() >-----------------------------------------------*/
/* read listfile to retrieve names of all datasets and collect metadata:
 * number of groups, number of datasets in each group, index of group "/spill"
 */
static
int read_dataset_names(int          rank,
                       const char  *listfile,
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
        assert(0);
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
    assert(*gList != NULL);

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
            assert((*gList)[nGroups].dset_names != NULL);
        }

        if (!strcmp(strtok(NULL, "/"), "evt.seq")) {
            /* add dataset evt.seq to first in the group */
            for (j=nDatasets; j>0; j--)
                (*gList)[nGroups].dset_names[j] = (*gList)[nGroups].dset_names[j-1];
            (*gList)[nGroups].dset_names[0] = (char*) malloc(len + 1);
            assert((*gList)[nGroups].dset_names[0] != NULL);
            strcpy((*gList)[nGroups].dset_names[0], line);
        }
        else {
            (*gList)[nGroups].dset_names[nDatasets] = (char*) malloc(len + 1);
            assert((*gList)[nGroups].dset_names[nDatasets] != NULL);
            strcpy((*gList)[nGroups].dset_names[nDatasets], line);
        }
        nDatasets++;
    }
    fclose(fptr);
    (*gList)[nGroups].nDatasets = nDatasets;
    nGroups++;
    assert(*spill_idx >= 0);

    if (rank == 0) {
        for (g=0; g<nGroups; g++)
            /* check if the first dataset of each group is evt.seq */
            strcpy(name, (*gList)[g].dset_names[0]);
            strtok(name, "/");
            if (strcmp(strtok(NULL, "/"), "evt.seq")) {
                printf("Error: group[g=%d] %s contains no evt.seq\n",g, name);
                assert(0);
            }
            if (debug) {
                for (d=0; d<(*gList)[g].nDatasets; d++)
                    printf("nGroups=%d group[%d] nDatasets=%d name[%d] %s\n",
                           nGroups, g, (*gList)[g].nDatasets, d, (*gList)[g].dset_names[d]);
            }
    }

    return nGroups;
}

#ifdef INQUIRE_METADATA_FIRST
/*----< collect_metadata() >-------------------------------------------------*/
/* collect metadata of all datasets of all groups in one place.
 * Note this turns out to be slower than opening dataset right before reading
 * it.
 */
static
int collect_metadata(hid_t       fd,
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
            assert(dset >= 0);

            /* inquire dimension sizes of dset */
            hid_t fspace = H5Dget_space(dset); assert(fspace >= 0);
            err = H5Sget_simple_extent_dims(fspace, dims, NULL); assert(err>=0);
            err = H5Sclose(fspace); assert(err >= 0);

            /* inquire chunk size along each dimension */
            hid_t chunk_plist = H5Dget_create_plist(dset);
            assert(chunk_plist >= 0);
            int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
            assert(chunk_ndims == 2);
            err = H5Pclose(chunk_plist); assert(err>=0);

            /* get data type and size */
            hid_t dtype = H5Dget_type(dset); assert(dtype >= 0);
            groups[g].dtype_size[d] = H5Tget_size(dtype); assert(groups[g].dtype_size[d] > 0);
            err = H5Tclose(dtype); assert(err >= 0);

            /* find the number of chunks of dset */
            size_t nChunks[2];
            nChunks[0] = groups[g].dims[d][0] / groups[g].chunk_dims[d][0];
            if (groups[g].dims[d][0] % groups[g].chunk_dims[d][0]) nChunks[0]++;
            nChunks[1] = groups[g].dims[d][1] / groups[g].chunk_dims[d][1];
            if (groups[g].dims[d][1] % groups[g].chunk_dims[d][1]) nChunks[1]++;
            groups[g].nChunks[d] = nChunks[0] * nChunks[1];

            groups[g].dset[d] = dset;
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
static
long long inq_num_unique_IDs(hid_t       fd,
                             const char *key)
{
    herr_t err;
    int rank;
    long long nEvts;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        /* open dataset key, e.g. '/spill/evt.seq' */
        hid_t seq = H5Dopen2(fd, key, H5P_DEFAULT); assert(seq >= 0);

        hsize_t dims[2];

        /* inquire dimension size of '/spill/evt.seq' */
        hid_t fspace = H5Dget_space(seq); assert(fspace >= 0);
        hsize_t ndims = H5Sget_simple_extent_ndims(fspace); assert(ndims == 2);
        err = H5Sget_simple_extent_dims(fspace, dims, NULL); assert(err>=0);
        /* 2nd dimension of key, e.g. /spill/evt.seq, is always 1 */
        assert(dims[1] == 1);

        nEvts = dims[0];

        if (verbose)
            printf("Size of /spill/evt.seq is %lld\n", nEvts);

        if (debug) {
            /* Note contents of key, e.g. /spill/evt.seq, start with 0 and
             * increment by 1, so there is no need to read the contents of it.
             * Check the contents only in debug mode.
             */
            /* get data type and size */
            hid_t dtype = H5Dget_type(seq); assert(dtype >= 0);
            assert(H5Tequal(dtype, H5T_STD_I64LE) > 0);
            size_t dtype_size = H5Tget_size(dtype); assert(dtype_size > 0);
            err = H5Tclose(dtype); assert(err >= 0);

            int64_t v, *seqBuf;
            seqBuf = (int64_t*) malloc(nEvts * sizeof(int64_t));
            assert(seqBuf != NULL);

            err = H5Dread(seq, H5T_STD_I64LE, fspace, fspace, H5P_DEFAULT,
                          seqBuf);
            assert(err >= 0);

            for (v=0; v<nEvts; v++)
                if (seqBuf[v] != v) {
                    printf("Error: %s[%"PRId64"] expect %"PRId64" but got %"PRId64"\n",
                           key, v, v, seqBuf[v]);
                    assert(seqBuf[v] == v);
                }
            free(seqBuf);
        }
        err = H5Sclose(fspace); assert(err >= 0);
        err = H5Dclose(seq); assert(err >= 0);
    }
    MPI_Bcast(&nEvts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    return nEvts;
}

/*----< calculate_starts_ends() >--------------------------------------------*/
/* The event IDs are evenly partitioned among all processes to balance the
 * computational workload. This function calculates the starting and ending
 * indices for each process and stores them in starts[rank] and ends[rank],
 * the range of event IDs responsible by process rank.
 */
static
int calculate_starts_ends(int       nprocs,
                          int       rank,
                          hsize_t   nEvts,  /* no. unique evt IDs */
                          hsize_t  *starts, /* OUT */
                          hsize_t  *ends)   /* OUT */
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
static
size_t binary_search_min(int64_t   key,
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
static
size_t binary_search_max(int64_t  key,
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

/*----< read_dataset_posix() >------------------------------------------------*/
/* Call POSIX read to read a dataset, decompress it, and copy to user buffer */
static
int read_dataset_posix(hid_t       fd,
                       int         posix_fd,
                       NOvA_group *group,
                       int         dset_idx,
                       int         profile,
                       double     *inflate_t)  /* OUT */
{
    int j;
    herr_t err;
    hsize_t offset[2];
    hsize_t *dims = group->dims[dset_idx];
    hsize_t *chunk_dims = group->chunk_dims[dset_idx];

    hid_t dset = H5Dopen2(fd, group->dset_names[dset_idx], H5P_DEFAULT);
    assert(dset >= 0);

    /* inquire dimension sizes of dset */
    hid_t fspace = H5Dget_space(dset); assert(fspace >= 0);
    err = H5Sget_simple_extent_dims(fspace, dims, NULL); assert(err>=0);
    assert(dims[1] == 1);
    err = H5Sclose(fspace); assert(err >= 0);

    /* inquire chunk size along each dimension */
    hid_t chunk_plist = H5Dget_create_plist(dset); assert(chunk_plist >= 0);
    int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
    assert(chunk_ndims == 2);
    assert(chunk_dims[1] == dims[1]);
    err = H5Pclose(chunk_plist); assert(err>=0);

    hid_t dtype = H5Dget_type(dset); assert(dtype >= 0);
    size_t dtype_size = H5Tget_size(dtype); assert(dtype_size > 0);
    group->dtype_size[dset_idx] = dtype_size;
    err = H5Tclose(dtype); assert(err >= 0);
    group->buf[dset_idx] = (void*) malloc(dims[0] * dtype_size);
    assert(group->buf[dset_idx] != NULL);

    /* find the number of chunks to be read by this process */
    group->nChunks[dset_idx] = dims[0] / chunk_dims[0];
    if (dims[0] % chunk_dims[0]) group->nChunks[dset_idx]++;

    /* collect offsets of all chunks */
    size_t chunk_size = chunk_dims[0] * dtype_size;
    unsigned char *chunk_buf;
    chunk_buf = (unsigned char*) malloc(chunk_size);
    assert(chunk_buf != NULL);

    /* the last chunk may be of size less than chunk_dims[0] */
    size_t last_chunk_len = dims[0] % chunk_dims[0];
    if (last_chunk_len == 0) last_chunk_len = chunk_dims[0];
    last_chunk_len *= dtype_size;

    /* read compressed a chunk into chunk_buf at a time and decompress it into
     * group->buf[dset_idx]
     */
    unsigned char *buf_ptr = group->buf[dset_idx];
    offset[0] = 0;
    offset[1] = 0;
    for (j=0; j<group->nChunks[dset_idx]; j++) {
        haddr_t addr;
        hsize_t size;
        ssize_t len;
        double timing;

        err = H5Dget_chunk_info_by_coord(dset, offset, NULL, &addr, &size);
        assert(err>=0);
        if (profile == 2) group->read_t[dset_idx]=MPI_Wtime();
        lseek(posix_fd, addr, SEEK_SET);
        len = read(posix_fd, chunk_buf, size);
        assert(len == size);
        timing = MPI_Wtime();
        if (profile == 2) group->read_t[dset_idx]=timing-group->read_t[dset_idx];

        int ret;
        z_stream z_strm;
        z_strm.zalloc    = Z_NULL;
        z_strm.zfree     = Z_NULL;
        z_strm.opaque    = Z_NULL;
        z_strm.avail_in  = size;
        z_strm.next_in   = chunk_buf;
        z_strm.avail_out = (j == group->nChunks[dset_idx]-1) ? last_chunk_len : chunk_size;
        z_strm.next_out  = buf_ptr;
        ret = inflateInit(&z_strm); assert(ret == Z_OK);
        ret = inflate(&z_strm, Z_SYNC_FLUSH); assert(ret == Z_OK || Z_STREAM_END);
        ret = inflateEnd(&z_strm); assert(ret == Z_OK);

        offset[0] += chunk_dims[0];
        buf_ptr += chunk_size;
        *inflate_t += MPI_Wtime() - timing;
    }
    free(chunk_buf);

    err = H5Dclose(dset); assert(err >= 0);

    return 1;
}

/*----< read_evt_seq_aggr_all() >--------------------------------------------*/
/* Use a single MPI collective read to read all evt.seq datasets of all groups,
 * decompress, and calculate the array index boundaries responsible by all
 * processes.
 */
static
int read_evt_seq_aggr_all(hid_t           fd,
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
    hsize_t nChunks=0, offset[2], max_chunk_dim=0, **size;
    haddr_t **addr;
    size_t dtype_size=8, read_len=0;
    MPI_Status status;

    if (nGroups == 0 || (nGroups == 1 && spill_idx == 0)) {
        /* This process has nothing to read, it must still participate the MPI
         * collective calls to MPI_File_set_view and MPI_File_read_all
         */
        mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native",
                                    MPI_INFO_NULL);
        assert(mpi_err == MPI_SUCCESS);
        mpi_err = MPI_File_read_all(fh, NULL, 0, MPI_BYTE, &status);
        assert(mpi_err == MPI_SUCCESS);
        return 1;
    }

    offset[1] = 0;

    /* save file offsets and sizes of individual chunks for later use */
    addr = (haddr_t**) malloc(nGroups * sizeof(haddr_t*));
    assert(addr != NULL);
    size = (hsize_t**) malloc(nGroups * sizeof(hsize_t*));
    assert(size != NULL);

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
        assert(seq >= 0);

        /* inquire dimension sizes of dset */
        hid_t fspace = H5Dget_space(seq); assert(fspace >= 0);
        err = H5Sget_simple_extent_dims(fspace, dims, NULL); assert(err>=0);
        assert(dims[1] == 1);
        err = H5Sclose(fspace); assert(err >= 0);

        /* inquire chunk size along each dimension */
        hid_t chunk_plist = H5Dget_create_plist(seq); assert(chunk_plist >= 0);
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        assert(chunk_ndims == 2);
        assert(chunk_dims[1] == 1);
        err = H5Pclose(chunk_plist); assert(err>=0);
        max_chunk_dim = MAX(max_chunk_dim, chunk_dims[0]);

        /* evt.seq data type is aways H5T_STD_I64LE and of size 8 bytes */
        hid_t dtype = H5Dget_type(seq);
        assert(H5Tequal(dtype, H5T_STD_I64LE) > 0);
        dtype_size = H5Tget_size(dtype); assert(dtype_size > 0);
        assert(dtype_size == 8); /* evt.seq is always of type int64_t */
        groups[g].dtype_size[0] = dtype_size;
        err = H5Tclose(dtype); assert(err >= 0);
        groups[g].buf[0] = (void*) malloc(dims[0] * dtype_size);
        assert(groups[g].buf[0] != NULL);

        /* find the number of chunks to be read by this process */
        groups[g].nChunks[0] = dims[0] / chunk_dims[0];
        if (dims[0] % chunk_dims[0]) groups[g].nChunks[0]++;

        /* accumulate number of chunks across all groups */
        nChunks += groups[g].nChunks[0];

        /* collect offsets and sizes of individual chunks of evt.seq */
        addr[g] = (haddr_t*) malloc(groups[g].nChunks[0] * sizeof(haddr_t));
        assert(addr[g] != NULL);
        size[g] = (hsize_t*) malloc(groups[g].nChunks[0] * sizeof(hsize_t));
        assert(size[g] != NULL);
        offset[0] = 0;
        for (j=0; j<groups[g].nChunks[0]; j++) {
            err = H5Dget_chunk_info_by_coord(seq, offset, NULL, &addr[g][j],
                                             &size[g][j]); assert(err>=0);
            read_len += size[g][j];
            offset[0] += chunk_dims[0];
        }
        err = H5Dclose(seq); assert(err >= 0);
    }

    /* Note file offsets of chunks may not follow the increasing order of
     * chunk IDs read by this process. We must sort the offsets before
     * creating the MPI derived file type. First, we construct an array of
     * displacement-length-index objects for such sorting.
     */
    off_len *disp_indx = (off_len*) malloc(nChunks * sizeof(off_len));
    assert(disp_indx != NULL);

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
    assert(k == nChunks);

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
    assert(chunk_dsps != NULL);
    int *chunk_lens = (int*) malloc(nChunks * sizeof(int));
    assert(chunk_lens != NULL);
    for (j=0; j<nChunks; j++) {
        chunk_lens[j] = disp_indx[j].block_len;
        chunk_dsps[j] = disp_indx[j].block_dsp;
    }

    /* create the filetype */
    MPI_Datatype ftype;
    mpi_err = MPI_Type_create_hindexed(nChunks, chunk_lens, chunk_dsps,
                                       MPI_BYTE, &ftype);
    assert(mpi_err == MPI_SUCCESS);
    mpi_err = MPI_Type_commit(&ftype);
    assert(mpi_err == MPI_SUCCESS);

    free(chunk_lens);
    free(chunk_dsps);

    /* set the file view, a collective MPI-IO call */
    mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
    assert(mpi_err == MPI_SUCCESS);

    mpi_err = MPI_Type_free(&ftype);
    assert(mpi_err == MPI_SUCCESS);

    /* allocate a buffer for reading the compressed chunks all at once */
    unsigned char *chunk_buf, *chunk_buf_ptr;
    chunk_buf = (unsigned char*) malloc(read_len);
    assert(chunk_buf != NULL);
    chunk_buf_ptr = chunk_buf;

    /* collective read */
    mpi_err = MPI_File_read_all(fh, chunk_buf, read_len, MPI_BYTE, &status);
    assert(mpi_err == MPI_SUCCESS);

    /* decompress individual chunks into groups[g].buf[0] */
    double timing = MPI_Wtime();
    size_t whole_chunk_size = max_chunk_dim * dtype_size;
    unsigned char *whole_chunk = (unsigned char*) malloc(whole_chunk_size);
    assert(whole_chunk != NULL);
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
        ret = inflateInit(&z_strm); assert(ret == Z_OK);
        ret = inflate(&z_strm, Z_SYNC_FLUSH); assert(ret == Z_OK || Z_STREAM_END);
        ret = inflateEnd(&z_strm); assert(ret == Z_OK);

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
    *inflate_t += MPI_Wtime() - timing;

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
    }
    return 1;
}

/*----< read_evt_seq() >-----------------------------------------------------*/
/* Read evt.seq dataset in order to calculate the lower and upperr boundaries
 * responsible by each process in to lowers[] and uppers[]
 */
static
int read_evt_seq(MPI_Comm       comm,
                 hid_t          fd,
                 MPI_File       fh,
                 int            posix_fd,
                 NOvA_group    *groups,
                 int            nGroups,
                 int            profile,
                 int            seq_opt,
                 hid_t          xfer_plist,
                 int            spill_idx,
                 const hsize_t *starts,  /* IN:  [nprocs] */
                 const hsize_t *ends,    /* IN:  [nprocs] */
                 size_t        *lowers,  /* OUT: [nprocs] */
                 size_t        *uppers,  /* OUT: [nprocs] */
                 double        *inflate_t)  /* OUT: */
{
    int g, j, k, nprocs, rank;
    herr_t err;
    hid_t seq;
    long long low_high[2];
    int64_t *seq_buf=NULL;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (seq_opt == 3) {
        /* Assign evt.seq datasets among processes, so they can be read in
         * parallel. The processes got assigned make a single MPI collective
         * read call to read all evt.seq datasets in assigned groups,
         * decompress, calculate boundaries and scatter them to other
         * processes. Note these are done in parallel, i.e. workload is
         * distributed among all processes.
         */
        long long **bounds;
        int my_startGrp, my_nGroups;

        /* partition read workload among all processes */
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

        /* groups[spill_idx].nChunks[0] = 0; */

        if (my_nGroups > 0) {
            bounds = (long long**) malloc(my_nGroups * sizeof(long long*));
            assert(bounds != NULL);
            bounds[0] = (long long*) malloc(my_nGroups * nprocs * 2 * sizeof(long long));
            assert(bounds[0] != NULL);
            for (g=1; g<my_nGroups; g++)
                bounds[g] = bounds[g-1] + nprocs * 2;
        }

        /* read_evt_seq_aggr_all is a collective call */
        err = read_evt_seq_aggr_all(fd, fh, groups+my_startGrp, my_nGroups,
                                    nprocs, rank, spill_idx-my_startGrp,
                                    starts, ends, bounds, inflate_t);

        /* calculate rank of root process for each MPI_Scatter */
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
            void *scatter_buf = (root == rank) ? bounds[g-my_startGrp] : NULL;
            MPI_Scatter(scatter_buf, 2, MPI_LONG_LONG, low_high, 2,
                        MPI_LONG_LONG, root, comm);
            lowers[g] = low_high[0];
            uppers[g] = low_high[1];
        }
        if (my_nGroups > 0) {
            free(bounds[0]);
            free(bounds);
        }
        return 1;
    }

    long long *bounds;

    if (seq_opt == 4) {
        /* Use POSIX read to read individual chunks, decompress them, and
         * scatter boundaries
         */
        if (rank == 0) {
            bounds = (long long*) malloc(nprocs * 2 * sizeof(long long));
            assert(bounds != NULL);
        }

        for (g=0; g<nGroups; g++) {
            /* read dataset evt.seq, the first dataset in the group, and
             * calculate the event range (low and high indices) responsible by
             * this process
             */
            if (g == spill_idx) {
                /* no need to read /spill/evt.seq */
                lowers[g] = starts[rank];
                uppers[g] = ends[rank];
                continue;
            }

            /* root process reads each evt.seq and scatters boundaries */
            if (rank == 0) {
                /* read_dataset_posix() is an independent call */
                err = read_dataset_posix(fd, posix_fd, groups+g, 0, profile,
                                         inflate_t);
                for (j=0, k=0; j<nprocs; j++) {
                    bounds[k++] = binary_search_min(starts[j], groups[g].buf[0],
                                                    groups[g].dims[0][0]);
                    bounds[k++] = binary_search_max(ends[j],   groups[g].buf[0],
                                                    groups[g].dims[0][0]);
                }
                free(groups[g].buf[0]);
            }
            MPI_Scatter(bounds, 2, MPI_LONG_LONG, low_high, 2, MPI_LONG_LONG,
                        0, comm);
            lowers[g] = low_high[0];
            uppers[g] = low_high[1];
        }
        if (rank == 0) free(bounds);
        return 1;
    }

    /* seq_opt == 0, 1, or 2 */
    if (seq_opt == 2 && rank == 0) {
        bounds = (long long*) malloc(nprocs * 2 * sizeof(long long));
        assert(bounds != NULL);
    }

    for (g=0; g<nGroups; g++) {
        /* read dataset evt.seq, the first dataset in the group, and calculate
         * the event range (low and high indices) responsible by this process
         */
        if (g == spill_idx) {
            /* no need to read /spill/evt.seq */
            lowers[g] = starts[rank];
            uppers[g] = ends[rank];
            continue;
        }

        /* open dataset 'evt.seq', first in the group */
        seq = H5Dopen2(fd, groups[g].dset_names[0], H5P_DEFAULT);
        assert(seq >= 0);

        /* inquire dimension sizes of 'evt.seq' in this group */
        hsize_t *dims = groups[g].dims[0];
        hid_t fspace = H5Dget_space(seq); assert(fspace >= 0);
        hsize_t ndims = H5Sget_simple_extent_ndims(fspace); assert(ndims == 2);
        err = H5Sget_simple_extent_dims(fspace, dims, NULL); assert(err>=0);
        assert(dims[1] == 1);
        err = H5Sclose(fspace); assert(err >= 0);

        /* inquire chunk size along each dimension */
        hsize_t *chunk_dims = groups[g].chunk_dims[0];
        hid_t chunk_plist = H5Dget_create_plist(seq); assert(chunk_plist >= 0);
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        assert(chunk_ndims == 2);
        assert(chunk_dims[1] == 1);
        err = H5Pclose(chunk_plist); assert(err>=0);

        /* data type of evt.seq is 64-bit integer */
        hid_t dtype = H5Dget_type(seq); assert(dtype >= 0);
        size_t dtype_size = H5Tget_size(dtype); assert(dtype_size > 0);
        groups[g].dtype_size[0] = dtype_size;
        err = H5Tclose(dtype); assert(err >= 0);

        if (seq_opt == 2) {
            /* rank 0 reads evt.seq, calculates lows and highs for all processes
             * and scatters
             */
            if (rank == 0) {
                seq_buf = (int64_t*) malloc(dims[0] * dims[1] * dtype_size);
                assert(seq_buf != NULL);
                if (profile == 2) groups[g].read_t[0]=MPI_Wtime();
                err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, seq_buf);
                assert(err >= 0);
                if (profile == 2) groups[g].read_t[0]=MPI_Wtime()-groups[g].read_t[0];

                for (j=0, k=0; j<nprocs; j++) {
                    bounds[k++] = binary_search_min(starts[j], seq_buf, dims[0]);
                    bounds[k++] = binary_search_max(ends[j],   seq_buf, dims[0]);
                }
            }
            MPI_Scatter(bounds, 2, MPI_LONG_LONG, low_high, 2, MPI_LONG_LONG, 0, comm);
            lowers[g] = low_high[0];
            uppers[g] = low_high[1];
        }
        else {
            if (seq_opt == 0) {
                /* rank 0 reads evt.seq and broadcasts it */
                seq_buf = (int64_t*) malloc(dims[0] * dims[1] * dtype_size);
                assert(seq_buf != NULL);
                if (rank == 0) {
                    if (profile == 2) groups[g].read_t[0]=MPI_Wtime();
                    err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, seq_buf); assert(err >= 0);
                    if (profile == 2) groups[g].read_t[0]=MPI_Wtime()-groups[g].read_t[0];
                }
                MPI_Bcast(seq_buf, dims[0], MPI_LONG_LONG, 0, comm);
            }
            else {
                /* collective-read-the-whole-dataset is bad */
                seq_buf = (int64_t*) malloc(dims[0] * dims[1] * dtype_size);
                assert(seq_buf != NULL);
                if (profile == 2) groups[g].read_t[0]=MPI_Wtime();
                err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL, xfer_plist, seq_buf);
                /* all-processes-read-the-whole-dataset is even worse */
                // err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, seq_buf);
                assert(err >= 0);
                if (profile == 2) groups[g].read_t[0]=MPI_Wtime()-groups[g].read_t[0];
            }
            /* find the array index range from 'low' to 'high' that falls into
             * this process's partition domain.
             */
            lowers[g] = binary_search_min(starts[rank], seq_buf, dims[0]);
            uppers[g] = binary_search_max(ends[rank],   seq_buf, dims[0]);
        }

        err = H5Dclose(seq); assert(err >= 0);

        if (debug) {
            if (rank == 0)
                printf("%s[0]=%"PRId64" seq[len-1=%zd]=%"PRId64"\n",
                       groups[g].dset_names[0],seq_buf[0],(size_t)dims[0]-1,seq_buf[dims[0]-1]);

            printf("%d: %s starts[rank]=%zd ends[rank]=%zd low=%zd high=%zd\n",
                   rank, groups[g].dset_names[0], (size_t)starts[rank],(size_t)ends[rank], lowers[g], uppers[g]);
        }

        if (seq_buf != NULL) free(seq_buf);
    }
    if (seq_opt == 2 && rank == 0) free(bounds);

    return 1;
}

/*----< read_hdf5() >--------------------------------------------------------*/
/* call H5Dread to read datasets one at a time */
static
int read_hdf5(hid_t       fd,
              int         rank,
              NOvA_group *group,
              int         spill_idx,
              size_t      low,
              size_t      high,
              int         profile,
              hid_t       xfer_plist)
{
    int d;
    herr_t  err;
    hid_t   dset, fspace, mspace, dtype;
    hsize_t start[2], count[2];
    size_t read_len;

    /* iterate all the remaining datasets in this group */
    for (d=1; d<group->nDatasets; d++) {
        hsize_t one[2]={1,1};

        if (verbose && rank == 0) printf("dataset %s\n", group->dset_names[d]);

        /* open dataset */
        dset = H5Dopen2(fd, group->dset_names[d], H5P_DEFAULT);
        assert(dset >= 0);

        /* inquire dimension sizes of dset */
        hsize_t *dims = group->dims[d];
        fspace = H5Dget_space(dset); assert(fspace >= 0);
        hsize_t ndims = H5Sget_simple_extent_ndims(fspace); assert(ndims == 2);
        err = H5Sget_simple_extent_dims(fspace, dims, NULL); assert(err>=0);

        /* inquire chunk size along each dimension */
        hsize_t *chunk_dims = group->chunk_dims[d];
        hid_t chunk_plist = H5Dget_create_plist(dset); assert(chunk_plist >= 0);
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        assert(chunk_ndims == 2);
        assert(chunk_dims[1] == dims[1]);
        err = H5Pclose(chunk_plist); assert(err>=0);

        /* set subarray/hyperslab access */
        start[0] = low;
        start[1] = 0;
        count[0] = high - low + 1;
        count[1] = dims[1];
        err = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, one,
                                  count); assert(err>=0);
        mspace = H5Screate_simple(2, count, NULL); assert(mspace>=0);

        /* get data type and size */
        dtype = H5Dget_type(dset); assert(dtype >= 0);
        group->dtype_size[d] = H5Tget_size(dtype);
        assert(group->dtype_size[d] > 0);

        /* calculate read size in bytes and allocate read buffers */
        read_len = count[0] * count[1] * group->dtype_size[d];
        group->buf[d] = (void*) malloc(read_len);
        assert(group->buf[d] != NULL);

        /* collectively read dataset d's contents */
        if (verbose)
            printf("%d: READ read_len=%zd dataset %s\n",rank,read_len,group->dset_names[d]);

        if (profile == 2) group->read_t[d]=MPI_Wtime();
        err = H5Dread(dset, dtype, mspace, fspace, xfer_plist, group->buf[d]);
        assert(err >= 0);
        if (profile == 2) group->read_t[d]=MPI_Wtime()-group->read_t[d];

        err = H5Tclose(dtype);  assert(err >= 0);
        err = H5Sclose(mspace); assert(err >= 0);
        err = H5Sclose(fspace); assert(err >= 0);
        err = H5Dclose(dset);   assert(err >= 0);
    }

    return 1;
}

/*----< read_mpio() >--------------------------------------------------------*/
/* call MPI_File_read_all to read all raw chunks of each dataset at a time
 * and decompress into user buffers
 */
static
int read_mpio(hid_t       fd,
              int         rank,
              NOvA_group *group,
              int         spill_idx,
              size_t      low,
              size_t      high,
              int         profile,
              MPI_File    fh,
              double     *inflate_t)
{
    int j, d, mpi_err;
    herr_t  err;
    hid_t   dset;
    size_t read_len;
    unsigned char *whole_chunk;
    double timing;
    MPI_Status status;

    for (d=1; d<group->nDatasets; d++) {
        hsize_t *dims       = group->dims[d];
        hsize_t *chunk_dims = group->chunk_dims[d];

        /* open dataset */
        dset = H5Dopen2(fd, group->dset_names[d], H5P_DEFAULT);
        assert(dset >= 0);

        /* find metadata of all the chunks of this dataset */

        /* inquire dimension sizes of dset */
        hid_t fspace = H5Dget_space(dset); assert(fspace >= 0);
        err = H5Sget_simple_extent_dims(fspace, dims, NULL); assert(err>=0);
        err = H5Sclose(fspace); assert(err >= 0);

        /* inquire chunk size along each dimension */
        hid_t chunk_plist = H5Dget_create_plist(dset); assert(chunk_plist >= 0);
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        assert(chunk_ndims == 2);
        assert(chunk_dims[1] == dims[1]);
        err = H5Pclose(chunk_plist); assert(err>=0);

        /* get data type and size */
        hid_t dtype = H5Dget_type(dset); assert(dtype >= 0);
        size_t dtype_size = H5Tget_size(dtype); assert(dtype_size > 0);
        group->dtype_size[d] = dtype_size;
        err = H5Tclose(dtype); assert(err >= 0);

        /* calculate buffer read size in bytes and allocate read buffer */
        read_len = (high - low + 1) * dims[1] * dtype_size;
        group->buf[d] = (void*) malloc(read_len);
        assert(group->buf[d] != NULL);

        hsize_t nChunks, offset[2];

        /* find the number of chunks to be read by this process */
        nChunks = (high / chunk_dims[0]) - (low / chunk_dims[0]) + 1;

        /* Note file offsets of chunks may not follow the increasing order of
         * chunk IDs read by this process. We must sort the offsets before
         * creating a file type. Construct an array of off-len-indx for such
         * sorting.
         */
        off_len *disp_indx = (off_len*) malloc(nChunks * sizeof(off_len));
        assert(disp_indx != NULL);

        /* calculate the logical position of chunk's first element. See
         * https://hdf5.io/develop/group___h5_d.html#ga408a49c6ec59c5b65ce4c791f8d26cb0
         */
        int is_mono_nondecr = 1;
        offset[0] = (low / chunk_dims[0]) * chunk_dims[0];
        offset[1] = 0;
        read_len = 0;
        for (j=0; j<nChunks; j++) {
            hsize_t size;
            haddr_t addr;
            err = H5Dget_chunk_info_by_coord(dset, offset, NULL, &addr, &size); assert(err>=0);
            disp_indx[j].block_dsp = (MPI_Aint)addr;  /* chunk's file offset */
            disp_indx[j].block_len = (int)size;       /* compressed chunk size */
            disp_indx[j].block_idx = j;               /* chunk ID to be read by this process */
            if (j > 0 && disp_indx[j].block_dsp < disp_indx[j-1].block_dsp)
                is_mono_nondecr = 0;
            read_len += size;
            offset[0] += chunk_dims[0];
        }
        err = H5Dclose(dset); assert(err >= 0);

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
        assert(chunk_dsps != NULL);
        int *chunk_lens = (int*) malloc(nChunks * sizeof(int));
        assert(chunk_lens != NULL);
        for (j=0; j<nChunks; j++) {
            chunk_lens[j] = disp_indx[j].block_len;
            chunk_dsps[j] = disp_indx[j].block_dsp;
        }

        /* create the filetype */
        MPI_Datatype ftype;
        mpi_err = MPI_Type_create_hindexed(nChunks, chunk_lens, chunk_dsps,
                                           MPI_BYTE, &ftype);
        assert(mpi_err == MPI_SUCCESS);
        mpi_err = MPI_Type_commit(&ftype);
        assert(mpi_err == MPI_SUCCESS);

        free(chunk_lens);
        free(chunk_dsps);

        /* set the file view, a collective MPI-IO call */
        mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, ftype, "native",
                                    MPI_INFO_NULL);
        assert(mpi_err == MPI_SUCCESS);

        mpi_err = MPI_Type_free(&ftype);
        assert(mpi_err == MPI_SUCCESS);

        /* allocate buffer for reading the compressed chunks all at once */
        unsigned char *chunk_buf, *chunk_buf_ptr;
        chunk_buf = (unsigned char*) malloc(read_len);
        assert(chunk_buf != NULL);
        chunk_buf_ptr = chunk_buf;

        /* collective read */
        if (profile == 2) group->read_t[d]=MPI_Wtime();
        mpi_err = MPI_File_read_all(fh, chunk_buf, read_len, MPI_BYTE, &status);
        assert(mpi_err == MPI_SUCCESS);
        if (profile == 2) group->read_t[d]=MPI_Wtime()-group->read_t[d];

        /* decompress each chunk into group->buf[d] */
        timing = MPI_Wtime();
        size_t whole_chunk_size = chunk_dims[0] * chunk_dims[1] * dtype_size;
        whole_chunk = (unsigned char*) malloc(whole_chunk_size);
        assert(whole_chunk != NULL);
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
            ret = inflateInit(&z_strm); assert(ret == Z_OK);
            ret = inflate(&z_strm, Z_SYNC_FLUSH); assert(ret == Z_OK || Z_STREAM_END);
            ret = inflateEnd(&z_strm); assert(ret == Z_OK);

            /* copy requested data to user buffer */
            if (disp_indx[j].block_idx == 0) { /* first chunk */
                size_t off = low % chunk_dims[0];
                size_t len;
                if (nChunks == 1)
                    len = high - low + 1;
                else
                    len = chunk_dims[0] - off;
                len *= chunk_dims[1] * dtype_size;
                off *= chunk_dims[1] * dtype_size;
                memcpy(group->buf[d], whole_chunk + off, len);
            }
            else if (disp_indx[j].block_idx == nChunks - 1) { /* last chunk */
                size_t len = (high+1) % chunk_dims[0];
                if (len == 0) len = chunk_dims[0];
                size_t off = high + 1 - len - low;
                off *= chunk_dims[1] * dtype_size;
                len *= chunk_dims[1] * dtype_size;
                memcpy(group->buf[d] + off, whole_chunk, len);
            }
            else { /* middle chunk, copy the full chunk */
                size_t off = chunk_dims[0] - low % chunk_dims[0];
                off += (disp_indx[j].block_idx - 1) * chunk_dims[0];
                off *= chunk_dims[1] * dtype_size;
                memcpy(group->buf[d] + off, whole_chunk, whole_chunk_size);
            }
            chunk_buf_ptr += disp_indx[j].block_len;
        }
        free(whole_chunk);
        free(chunk_buf);
        free(disp_indx);
        *inflate_t += MPI_Wtime() - timing;
    }
    return 1;
}

/*----< read_mpio_aggr() >---------------------------------------------------*/
/* A single call to MPI_File_read_all to read all raw chunks of all datasets
 * in a group, one group at a time, and decompress into user buffers
 */
static
int read_mpio_aggr(hid_t       fd,
                   int         rank,
                   NOvA_group *group,
                   size_t      low,
                   size_t      high,
                   MPI_File    fh,
                   double     *inflate_t)
{
    int d, j, k, mpi_err;
    herr_t  err;
    hid_t   dset;
    hsize_t **size;
    haddr_t **addr;
    size_t nChunks, max_chunk_size, read_len, *dtype_size;
    MPI_Status status;

    dtype_size = group->dtype_size;

    /* allocate space to save file offsets and sizes of individual chunks */
    addr = (haddr_t**) malloc(group->nDatasets * sizeof(haddr_t*));
    assert(addr != NULL);
    size = (hsize_t**) malloc(group->nDatasets * sizeof(hsize_t*));
    assert(size != NULL);

    /* collect metadata of all chunks of all datasets in group g, including
     * number of chunks and their file offsets and compressed sizes
     */
    nChunks = 0;
    read_len = 0;
    max_chunk_size = 0;
    for (d=1; d<group->nDatasets; d++) {
        hsize_t *dims       = group->dims[d];
        hsize_t *chunk_dims = group->chunk_dims[d];

        /* open dataset */
        dset = H5Dopen2(fd, group->dset_names[d], H5P_DEFAULT);
        assert(dset >= 0);

        /* inquire dimension sizes of dset */
        hid_t fspace = H5Dget_space(dset); assert(fspace >= 0);
        err = H5Sget_simple_extent_dims(fspace, dims, NULL); assert(err>=0);
        err = H5Sclose(fspace); assert(err >= 0);

        /* inquire chunk size along each dimension */
        hid_t chunk_plist = H5Dget_create_plist(dset); assert(chunk_plist >= 0);
        int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
        assert(chunk_ndims == 2);
        assert(chunk_dims[1] == dims[1]);
        err = H5Pclose(chunk_plist); assert(err>=0);

        /* get data type and size */
        hid_t dtype = H5Dget_type(dset); assert(dtype >= 0);
        dtype_size[d] = H5Tget_size(dtype); assert(dtype_size[d] > 0);
        err = H5Tclose(dtype); assert(err >= 0);
        max_chunk_size = MAX(max_chunk_size, chunk_dims[0] * dtype_size[d]);

        /* calculate buffer read size in bytes and allocate read buffer */
        group->buf[d] = (void*) malloc((high - low + 1) * dims[1] * dtype_size[d]);
        assert(group->buf[d] != NULL);

        /* find the number of chunks to be read by this process */
        group->nChunks[d] = (high / chunk_dims[0]) - (low / chunk_dims[0]) + 1;
        nChunks += group->nChunks[d];

        /* collect offsets of all chunks of this dataset */
        addr[d] = (haddr_t*) malloc(group->nChunks[d] * sizeof(haddr_t));
        assert(addr[d] != NULL);
        size[d] = (hsize_t*) malloc(group->nChunks[d] * sizeof(hsize_t));
        assert(size[d] != NULL);
        hsize_t offset[2]={0, 0};
        for (j=0; j<group->nChunks[d]; j++) {
            err = H5Dget_chunk_info_by_coord(dset, offset, NULL, &addr[d][j],
                                             &size[d][j]); assert(err>=0);
            read_len += size[d][j];
            offset[0] += chunk_dims[0];
        }
        err = H5Dclose(dset); assert(err >= 0);
    }

    /* Note file offsets of chunks may not follow the increasing order of
     * chunk IDs read by this process. We must sort the offsets before
     * creating a file type. Construct an array of off-len-indx for such
     * sorting.
     */
    off_len *disp_indx = (off_len*) malloc(nChunks * sizeof(off_len));
    assert(disp_indx != NULL);

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
    assert(k == nChunks);

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
    assert(chunk_dsps != NULL);
    int *chunk_lens = (int*) malloc(nChunks * sizeof(int));
    assert(chunk_lens != NULL);
    for (j=0; j<nChunks; j++) {
        chunk_lens[j] = disp_indx[j].block_len;
        chunk_dsps[j] = disp_indx[j].block_dsp;
    }

    /* create the filetype */
    MPI_Datatype ftype;
    mpi_err = MPI_Type_create_hindexed(nChunks, chunk_lens, chunk_dsps,
                                       MPI_BYTE, &ftype);
    assert(mpi_err == MPI_SUCCESS);
    mpi_err = MPI_Type_commit(&ftype);
    assert(mpi_err == MPI_SUCCESS);

    free(chunk_lens);
    free(chunk_dsps);

    /* set the file view, a collective MPI-IO call */
    mpi_err = MPI_File_set_view(fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
    assert(mpi_err == MPI_SUCCESS);

    mpi_err = MPI_Type_free(&ftype);
    assert(mpi_err == MPI_SUCCESS);

    /* allocate buffer for reading the compressed chunks all at once */
    unsigned char *chunk_buf, *chunk_buf_ptr;
    chunk_buf = (unsigned char*) malloc(read_len);
    assert(chunk_buf != NULL);
    chunk_buf_ptr = chunk_buf;

    /* collective read */
    mpi_err = MPI_File_read_all(fh, chunk_buf, read_len, MPI_BYTE, &status);
    assert(mpi_err == MPI_SUCCESS);

    /* decompress individual chunks into group->buf[d] */
    double timing = MPI_Wtime();
    unsigned char *whole_chunk = (unsigned char*) malloc(max_chunk_size);
    assert(whole_chunk != NULL);
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
        ret = inflateInit(&z_strm); assert(ret == Z_OK);
        ret = inflate(&z_strm, Z_SYNC_FLUSH); assert(ret == Z_OK || Z_STREAM_END);
        ret = inflateEnd(&z_strm); assert(ret == Z_OK);

        /* copy requested data to user buffer */
        hsize_t *chunk_dims;
        d = disp_indx[j].block_idx % group->nDatasets;  /* dataset ID */
        k = disp_indx[j].block_idx / group->nDatasets;  /* chunk ID of dataset d */
        chunk_dims = group->chunk_dims[d];
        /* copy requested data to user buffer */
        if (k == 0) { /* dataset d's first chunk */
            size_t off = low % chunk_dims[0];
            size_t len;
            if (group->nChunks[d] == 1)
                len = high - low + 1;
            else
                len = chunk_dims[0] - off;
            len *= dtype_size[d];
            off *= dtype_size[d];
            memcpy(group->buf[d], whole_chunk + off, len);
        }
        else if (k == group->nChunks[d] - 1) { /* dataset d's last chunk */
            size_t len = (high+1) % chunk_dims[0];
            if (len == 0) len = chunk_dims[0];
            size_t off = high + 1 - len - low;
            off *= dtype_size[d];
            len *= dtype_size[d];
            memcpy(group->buf[d] + off, whole_chunk, len);
        }
        else { /* middle chunk, copy the full chunk */
            size_t len = chunk_dims[0] * dtype_size[d];
            size_t off = chunk_dims[0] - low % chunk_dims[0];
            off += (k - 1) * chunk_dims[0];
            off *= dtype_size[d];
            memcpy(group->buf[d] + off, whole_chunk, len);
        }
        chunk_buf_ptr += disp_indx[j].block_len;
    }
    free(whole_chunk);
    free(chunk_buf);
    free(disp_indx);
    *inflate_t += MPI_Wtime() - timing;

    return 1;
}

/*----< chunk_statistics() >-------------------------------------------------*/
static
int chunk_statistics(MPI_Comm    comm,
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
    long long nEvtIDs, aggr_nchunks_read, my_nchunks_read=0;
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
        assert(bounds != NULL);
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
    assert(starts != NULL);
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
        size_t low=0, high=0;

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
            low  = starts[rank];
            high = ends[rank];
        }
        else {
            /* rank 0 reads evt.seq, calculates lows and highs for all processes
             * and scatters them.
             */
            long long low_high[2];

            if (rank == 0) {
                hsize_t dset_size, dims[2], chunk_dims[2];

                /* open dataset 'evt.seq', first in the group */
                hid_t seq = H5Dopen2(fd, groups[g].dset_names[0], H5P_DEFAULT);
                assert(seq >= 0);

                /* collect metadata of ect.seq again, as root may not have
                 * collected metadata of all evt.seq
                 */

                /* inquire dimension sizes of dset */
                hid_t fspace = H5Dget_space(seq); assert(fspace >= 0);
                err = H5Sget_simple_extent_dims(fspace, dims, NULL);
                assert(err>=0);
                err = H5Sclose(fspace); assert(err >= 0);

                /* inquire chunk size along each dimension */
                hid_t chunk_plist = H5Dget_create_plist(seq);
                assert(chunk_plist >= 0);
                int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
                assert(chunk_ndims == 2);
                /* evt.seq chunk_dims[1] should always be 1 */
                assert(chunk_dims[1] == 1);
                err = H5Pclose(chunk_plist); assert(err>=0);

                /* data type of evt.seq is 64-bit integer */
                hid_t dtype = H5Dget_type(seq); assert(dtype >= 0);
                size_t dtype_size = H5Tget_size(dtype); assert(dtype_size > 0);
                err = H5Tclose(dtype); assert(err >= 0);

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
                    assert(err>=0);
                    all_evt_seq_size_z += size;
                    offset[0] += chunk_dims[0];
                    grp_zip_sizes[g] += size;
                }

                int64_t *seqBuf = (int64_t*) malloc(dset_size);
                assert(seqBuf != NULL);
                err = H5Dread(seq, H5T_STD_I64LE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, seqBuf);
                assert(err >= 0);

                for (j=0, k=0; j<nprocs; j++) {
                    bounds[k++] = binary_search_min(starts[j], seqBuf, dims[0]);
                    bounds[k++] = binary_search_max(ends[j],   seqBuf, dims[0]);
                }
                free(seqBuf);

                if (seq_opt == 1) nchunks_shared += groups[g].nChunks[0];

                err = H5Dclose(seq); assert(err >= 0);
            }

            MPI_Scatter(bounds, 2, MPI_LONG_LONG, low_high, 2, MPI_LONG_LONG,
                        0, comm);
            low  = low_high[0];
            high = low_high[1];

            MPI_Bcast(&groups[g].nChunks[0], 1, MPI_INT, 0, comm);
            if (rank > 0 && seq_opt == 1)
                my_nchunks_read += groups[g].nChunks[0];
        }

        for (d=1; d<groups[g].nDatasets; d++) {
            hsize_t dims[2], chunk_dims[2];

            /* open dataset */
            dset = H5Dopen2(fd, groups[g].dset_names[d], H5P_DEFAULT);
            assert(dset >= 0);

            /* collect metadata of ect.seq again, as this process may not have
             * collected metadata of all evt.seq
             */

            /* inquire dimension sizes of dset */
            hid_t fspace = H5Dget_space(dset); assert(fspace >= 0);
            err = H5Sget_simple_extent_dims(fspace, dims, NULL); assert(err>=0);
            err = H5Sclose(fspace); assert(err >= 0);

            /* inquire chunk size along each dimension */
            hid_t chunk_plist = H5Dget_create_plist(dset);
            assert(chunk_plist >= 0);
            int chunk_ndims = H5Pget_chunk(chunk_plist, 2, chunk_dims);
            assert(chunk_ndims == 2);
            err = H5Pclose(chunk_plist); assert(err>=0);
            /* Note chunk_dims[1] may be larger than 1, but dims[1] is not
             * chunked
             */
            assert(chunk_dims[1] == dims[1]);

            /* data type of evt.seq is 64-bit integer */
            hid_t dtype = H5Dget_type(dset); assert(dtype >= 0);
            size_t dtype_size = H5Tget_size(dtype); assert(dtype_size > 0);
            err = H5Tclose(dtype); assert(err >= 0);

            /* calculate number of chunks of this dataset */
            groups[g].nChunks[d] = dims[0] / chunk_dims[0];
            if (dims[0] % chunk_dims[0]) groups[g].nChunks[d]++;
            total_nchunks += groups[g].nChunks[d];

            /* calculate number of chunks read by this process */
            int nchunks = (high / chunk_dims[0]) - (low / chunk_dims[0]) + 1;
            my_nchunks_read += nchunks;
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
                    assert(err>=0);
                    all_dset_size_z += size;
                    offset[0] += chunk_dims[0];
                    grp_zip_sizes[g] += size;
                }
            }
            err = H5Dclose(dset); assert(err >= 0);
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
               (float)my_nchunks_read/(float)nDatasets, nDatasets-nGroups);

    for (g=0; g<nGroups; g++)
        free(groups[g].nChunks);

    return 1;
}

/*----< data_parallelism() >-------------------------------------------------*/
static
int data_parallelism(MPI_Comm    comm,
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
        assert(groups[g].buf != NULL);

        /* allocate space to store data type sizes */
        groups[g].dtype_size = (size_t*) calloc(nDatasets, sizeof(size_t));
        assert(groups[g].dtype_size != NULL);

        /* allocate space to store dimension sizes */
        groups[g].dims = (hsize_t**) malloc(nDatasets * sizeof(hsize_t*));
        assert(groups[g].dims != NULL);
        groups[g].dims[0] = (hsize_t*) malloc(nDatasets * 2 * sizeof(hsize_t*));
        assert(groups[g].dims[0] != NULL);
        for (d=1; d<nDatasets; d++)
            groups[g].dims[d] = groups[g].dims[d-1] + 2;

        /* allocate space to store chunk dimension sizes */
        groups[g].chunk_dims = (hsize_t**) malloc(nDatasets * sizeof(hsize_t*));
        assert(groups[g].chunk_dims != NULL);
        groups[g].chunk_dims[0] = (hsize_t*) malloc(nDatasets * 2 * sizeof(hsize_t*));
        assert(groups[g].chunk_dims[0] != NULL);
        for (d=1; d<nDatasets; d++)
            groups[g].chunk_dims[d] = groups[g].chunk_dims[d-1] + 2;

        /* allocate space to store number of chunks and initialize to zeros */
        groups[g].nChunks = (size_t*) calloc(nDatasets, sizeof(size_t));
        assert(groups[g].nChunks != NULL);
    }

    MPI_Barrier(comm);
    open_t = MPI_Wtime();

    /* create file access property list and add MPI communicator */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS); assert(fapl_id >= 0);
    err = H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL); assert(err >= 0);

    /* collectively open input file for reading */
    fd = H5Fopen(infile, H5F_ACC_RDONLY, fapl_id);
    if (fd < 0) {
        fprintf(stderr,"%d: Error: fail to open file %s (%s)\n",
                rank,  infile, strerror(errno));
        fflush(stderr);
        assert(0);
    }
    err = H5Pclose(fapl_id); assert(err >= 0);

    /* set MPI-IO hints and open input file using MPI-IO */
    if (seq_opt == 3 || dset_opt > 0) {
        MPI_Info info = MPI_INFO_NULL;
        mpi_err = MPI_Info_create(&info); assert(mpi_err == MPI_SUCCESS);
        mpi_err = MPI_Info_set(info, "romio_cb_read", "enable");
        assert(mpi_err == MPI_SUCCESS);
        mpi_err = MPI_File_open(comm, infile, MPI_MODE_RDONLY, info, &fh);
        assert(mpi_err == MPI_SUCCESS);
        mpi_err = MPI_Info_free(&info); assert(mpi_err == MPI_SUCCESS);
    }
    if (seq_opt == 4) {
        posix_fd = open(infile, O_RDONLY);
        assert(posix_fd >= 0);
    }
    open_t = MPI_Wtime() - open_t;

    MPI_Barrier(comm);
    read_seq_t = MPI_Wtime();

    /* this is an MPI_COMM_WORLD collective call */
    nEvtIDs = inq_num_unique_IDs(fd, "/spill/evt.seq");

    /* starts[rank] and ends[rank] store the starting and ending event IDs that
     * are responsible by process rank
     */
    starts = (hsize_t*) malloc(nprocs * 2 * sizeof(hsize_t));
    assert(starts != NULL);
    ends = starts + nprocs;

    /* calculate the range of event IDs responsible by all process and store
     * them in starts[nprocs] and ends[nprocs] */
    calculate_starts_ends(nprocs, rank, nEvtIDs, starts, ends);

    /* set MPI-IO collective transfer mode */
    xfer_plist = H5Pcreate(H5P_DATASET_XFER); assert(xfer_plist>=0);
    err = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE); assert(err>=0);

    /* calculate this process's array index range (lower and upper) for each
     * group */
    lowers = (size_t*) malloc(nGroups * 2 * sizeof(size_t));
    assert(lowers != NULL);
    uppers = lowers + nGroups;

    /* iterate all groups to calculate lowers[] and uppers[] */
    read_evt_seq(comm, fd, fh, posix_fd, groups, nGroups, profile, seq_opt,
                 xfer_plist, spill_idx, starts, ends, lowers, uppers,
                 &inflate_t);
    read_seq_t = MPI_Wtime() - read_seq_t;

    if (debug) {
        int test_seq_opt = (seq_opt == 0) ? 1 : 0;
        size_t *debug_lowers, *debug_uppers;
        debug_lowers = (size_t*) malloc(nGroups * 2 * sizeof(size_t));
        assert(debug_lowers != NULL);
        debug_uppers = debug_lowers + nGroups;
        read_evt_seq(comm, fd, fh, posix_fd, groups, nGroups, profile,
                     test_seq_opt, xfer_plist, spill_idx, starts, ends,
                     debug_lowers, debug_uppers, &inflate_t);
        for (g=0; g<nGroups; g++) {
            if (debug_lowers[g] != lowers[g] || debug_uppers[g] != uppers[g]) {
                printf("%d: Error: group %d debug_lowers(%zd) != lowers(%zd) || debug_uppers(%zd) != uppers(%zd)\n",
                       rank, g, debug_lowers[g],lowers[g],debug_uppers[g],uppers[g]);
                assert(debug_lowers[g] == lowers[g]);
                assert(debug_uppers[g] != uppers[g]);
            }
        }
        free(debug_lowers);
    }

    MPI_Barrier(comm);
    read_dset_t = MPI_Wtime();

    /* Read the remaining datasets by iterating all groups */
    for (g=0; g<nGroups; g++) {
        if (dset_opt == 0)
            /* read datasets using H5Dread() */
            err = read_hdf5(fd, rank, groups+g, spill_idx, lowers[g],
                            uppers[g], profile, xfer_plist);
        else if (dset_opt == 1)
            /* read datasets using MPI-IO, one dataset at a time */
            err = read_mpio(fd, rank, groups+g, spill_idx, lowers[g],
                            uppers[g], profile, fh, &inflate_t);
        else if (dset_opt == 2)
            /* read datasets using MPI-IO, all datasets in one group and one
             * group at a time
             */
            err = read_mpio_aggr(fd, rank, groups+g, lowers[g], uppers[g], fh,
                                 &inflate_t);

        /* This is where PandAna performs computation to identify events of
         * interest from the read buffers
         */

        /* free read allocated buffers all at once */
        for (d=1; d<groups[g].nDatasets; d++)
            free(groups[g].buf[d]);
    }
    free(lowers);
    err = H5Pclose(xfer_plist); assert(err>=0);

    read_dset_t = MPI_Wtime() - read_dset_t;

    /* close input file */
    MPI_Barrier(comm);
    close_t = MPI_Wtime();
    err = H5Fclose(fd); assert(err >= 0);

    if (seq_opt == 3 || dset_opt > 0) {
        mpi_err = MPI_File_close(&fh);
        assert(mpi_err == MPI_SUCCESS);
    }
    if (seq_opt == 4) close(posix_fd);
    close_t = MPI_Wtime() - close_t;

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

    return 1;
}

/*----< group_parallelism() >------------------------------------------------*/
static
int group_parallelism(MPI_Comm    comm,
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
    int grp_rank, grp_nprocs;
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

        MPI_Comm_split(MPI_COMM_WORLD, my_startGrp, rank, &grp_comm);

        if (debug) {
            MPI_Comm_rank(grp_comm, &grp_rank);
            MPI_Comm_size(grp_comm, &grp_nprocs);
            printf("%2d nGroups=%d my_startGrp=%2d my_nGroups=%2d grp_rank=%2d grp_nprocs=%2d\n",
                   rank,nGroups,my_startGrp,my_nGroups,grp_rank,grp_nprocs);
        }
    }

    /* within a group, data parallelism method is used */
    data_parallelism(grp_comm, infile, groups + my_startGrp, my_nGroups,
                     spill_idx - my_startGrp, seq_opt, dset_opt, profile,
                     timings);

    if (nprocs > nGroups)
        MPI_Comm_free(&grp_comm);

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
    int c, d, g, nprocs, rank, nGroups, nDatasets, parallelism=0;
    char *listfile=NULL, *infile=NULL;
    double all_t, timings[6], max_t[6], min_t[6];
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
    if (seq_opt < 0 || seq_opt > 4) {
        if (rank  == 0) {
            printf("Error: option -s must be 0, 1, 2, 3, or 4\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }
    if (dset_opt < 0 || dset_opt > 2) {
        if (rank  == 0) {
            printf("Error: option -m must be 0, 1, or 2\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }
    if (parallelism < 0 || parallelism > 2) {
        if (rank  == 0) {
            printf("Error: option -r must be 0, 1, or 2\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }

    if (rank == 0) {
        printf("Number of MPI processes = %d\n", nprocs);
        printf("Input dataset name file '%s'\n", listfile);
        printf("Input concatenated HDF5 file '%s'\n", infile);
    }

    /* read dataset names and get number of datasets, number of groups, maximum
     * number of datasets among groups, find the array index of group /spill
     */
    nGroups = read_dataset_names(rank, listfile, &groups, &spill_idx);

    for (nDatasets=0, g=0; g<nGroups; g++) nDatasets += groups[g].nDatasets;

    /* print the running parameters and metadata of input file */
    if (rank == 0) {
        printf("Number of datasets to read = %d\n", nDatasets);
        printf("Number of groups = %d\n",nGroups);
        if (seq_opt == 0)
            printf("Read evt.seq method: root process H5Dread and broadcasts\n");
        else if (seq_opt == 1)
            printf("Read evt.seq method: all processes H5Dread the entire evt.seq collectively\n");
        else if (seq_opt == 2)
            printf("Read evt.seq method: root process H5Dread evt.seq and scatters boundaries\n");
        else if (seq_opt == 3)
            printf("Read evt.seq method: MPI collective read all evt.seq, decompress, and scatters boundaries\n");
        else if (seq_opt == 4)
            printf("Read evt.seq method: root POSIX reads one chunk at a time, decompress, and scatters boundaries\n");

        if (dset_opt == 0)
            printf("Read datasets method: H5Dread\n");
        else if (dset_opt == 1)
            printf("Read datasets method: MPI collective read and decompress, one dataset at a time\n");
        else if (dset_opt == 2)
            printf("Read datasets method: MPI collective read and decompress, all datasets in one group at a time\n");
        if (parallelism == 0)
            printf("Parallelization: data parallelism (all processes read all datasets)\n");
        else if (parallelism == 1)
            printf("Parallelization: group parallelism (processes divided into groups and data parallelism in each group)\n");
        else if (parallelism == 2)
            printf("Parallelization: dataset parallelism (divide all datasets among processes)\n");
    }
    fflush(stdout);

    for (g=0; g<nGroups; g++) {
        /* allocate space to store read timings for individual datasets */
        groups[g].read_t = (double*) calloc(groups[g].nDatasets, sizeof(double));
        assert(groups[g].read_t != NULL);
    }

    for (d=0; d<6; d++) timings[d] = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    all_t = MPI_Wtime();

    if (parallelism == 0) /* data parallelism */
        data_parallelism(MPI_COMM_WORLD, infile, groups, nGroups, spill_idx,
                         seq_opt, dset_opt, profile, timings);
    else if (parallelism == 1) /* group parallelism */
        group_parallelism(MPI_COMM_WORLD, infile, groups, nGroups, spill_idx,
                          seq_opt, dset_opt, profile, timings);
    else if (parallelism == 2) /* dataset parallelism */
        printf("dataset parallelism has not been implemented yet\n");

    all_t = MPI_Wtime() - all_t;

    /* find the max/min timings among all processes */
    max_t[0] = min_t[0] = timings[0]; /* open_t */
    max_t[1] = min_t[1] = timings[1]; /* read_seq_t */
    max_t[2] = min_t[2] = timings[2]; /* read_dset_t */
    max_t[3] = min_t[3] = timings[3]; /* close_t */
    max_t[4] = min_t[4] = timings[4]; /* inflate_t */
    max_t[5] = min_t[5] = all_t;
    MPI_Allreduce(MPI_IN_PLACE, max_t, 6, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, min_t, 6, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("----------------------------------------------------\n");
        printf("MAX and MIN among all %d processes\n", nprocs);
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
