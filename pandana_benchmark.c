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
#include <unistd.h>    /* getopt() */
#include <assert.h>    /* assert() */

#include <mpi.h>
#include "hdf5.h"
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

#define LINE_SIZE 256

/*----< read_dataset_names() >-----------------------------------------------*/
/* read listfile to retrieve names of all datasets and collect metadata:
 * number of groups, number of datasets in each group.
 */
static int
read_dataset_names(int          rank,
                   const char  *listfile,   /* file contains a list of names */
                   NOvA_group **gList)      /* OUT */
{
    FILE *fptr;
    int j, g, len, nGroups, nDatasets, nDsetGrp, maxDsetGrp;
    char line[LINE_SIZE], gname[LINE_SIZE], name[LINE_SIZE], *cur_gname;

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

    return nGroups;
}

/*----< chunk_statistics() >-------------------------------------------------*/
static int
chunk_statistics(MPI_Comm    comm,
                 const char *infile,
                 NOvA_group *groups,
                 int         nGroups,
                 int         seq_opt)
{
    herr_t  err;
    hid_t   fd, dset;

    int g, d, j, nprocs, rank, all_nDatasets;
    int nchunks_shared=0, max_shared_chunks=0;
    int max_nchunks_read=0, min_nchunks_read=INT_MAX, total_nchunks=0;
    long long aggr_nchunks_read, my_nchunks_read=0, my_nchunks_read_nokeys=0;
    long long all_dset_size, all_evt_seq_size;
    long long all_dset_size_z, all_evt_seq_size_z;
    long long numIDs, maxRead=0, minRead=LONG_MAX;
    size_t *grp_sizes, *grp_zip_sizes, *grp_nChunks, **nChunks;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef PANDANA_BENCHMARK
    extern void set_options(int seq_read_opt, int dset_read_opt);
    set_options(2, 0);

    MPI_File fh = MPI_FILE_NULL;
#else
    MPI_File fh;
    int mpi_err = MPI_File_open(comm, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_open");
#endif

    /* collect statistics describing chunk contention */
    fd = H5Fopen(infile, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Inquire number of globally unique IDs (size of dset_global_ID) */
    dset = H5Dopen2(fd, "/spill/evt.seq", H5P_DEFAULT);
    hid_t fspace = H5Dget_space(dset);
    hsize_t dims[2];
    err = H5Sget_simple_extent_dims(fspace, dims, NULL);
    err = H5Sclose(fspace);
    err = H5Dclose(dset);
    numIDs = dims[0];

    if (rank == 0)
        printf("Number of unique IDs (size of /spill/evt.seq)=%lld\n",numIDs);

    /* calculate this process's responsible array index range (lower and upper
     * bounds) for each group */
    size_t *lowers = (size_t*) malloc(nGroups * 2 * sizeof(size_t));
    if (lowers == NULL) CHECK_ERROR(-1, "malloc");
    size_t *uppers = lowers + nGroups;

    /* construct a list of key dataset names, one per group */
    char **key_names = (char**) malloc(nGroups * sizeof(char*));
    for (g=0; g<nGroups; g++)
        key_names[g] = groups[g].dset_names[0];

    /* calculate this process's lowers[] and uppers[] for all groups */
    pandana_read_keys(MPI_COMM_WORLD, fd, fh, nGroups, key_names,
                      numIDs, lowers, uppers);
    free(key_names);

    long long *bounds = (long long*) malloc(nprocs * 2 * sizeof(long long));

    grp_sizes = (size_t*) calloc(nGroups * 3, sizeof(size_t));
    grp_zip_sizes = grp_sizes + nGroups;
    grp_nChunks = grp_zip_sizes + nGroups;
    nChunks = (size_t**) malloc(nGroups * sizeof(size_t*));

    size_t my_startGrp, my_nGroups;
    my_nGroups = nGroups / nprocs;
    my_startGrp = my_nGroups * rank;
    if (rank < nGroups % nprocs) {
        my_startGrp += rank;
        my_nGroups++;
    }
    else
        my_startGrp += nGroups % nprocs;

    all_dset_size = 0;
    all_evt_seq_size = 0;
    all_dset_size_z = 0;
    all_evt_seq_size_z = 0;
    all_nDatasets = 0;
    for (g=0; g<nGroups; g++) {
        hsize_t dset_size, dims[2], chunk_dims[2];

        nChunks[g] = (size_t*) malloc(groups[g].nDatasets * sizeof(size_t));

        all_nDatasets += groups[g].nDatasets;

        hid_t seq = H5Dopen2(fd, groups[g].dset_names[0], H5P_DEFAULT);
        hid_t fspace = H5Dget_space(seq);
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        err = H5Sclose(fspace);
        hid_t chunk_plist = H5Dget_create_plist(seq);
        H5Pget_chunk(chunk_plist, 2, chunk_dims);
        err = H5Pclose(chunk_plist);
        hid_t dtype = H5Dget_type(seq);
        size_t dtype_size = H5Tget_size(dtype);
        err = H5Tclose(dtype);

        dset_size = dims[0] * dims[1] * dtype_size;
        all_evt_seq_size += dset_size;
        grp_sizes[g] += dset_size;

        nChunks[g][0] = dims[0] / chunk_dims[0];
        if (dims[0] % chunk_dims[0]) nChunks[g][0]++;

        if (seq_opt == 0) {
            if (rank == 0) my_nchunks_read += nChunks[g][0];
        } else if (seq_opt == 1) {
            my_nchunks_read += nChunks[g][0];
        } else if (seq_opt == 2) {
            if (rank == 0) my_nchunks_read += nChunks[g][0];
        } else if (seq_opt == 3) {
            if (g >= my_startGrp && g < my_startGrp + my_nGroups)
                my_nchunks_read += nChunks[g][0];
        } else if (seq_opt == 4) {
            if (rank == 0) my_nchunks_read += nChunks[g][0];
        }

        total_nchunks += nChunks[g][0];
        grp_nChunks[g] += nChunks[g][0];

        /* calculate read sizes of compressed data */
        hsize_t offset[2]={0, 0};
        for (j=0; j<nChunks[g][0]; j++) {
            haddr_t addr;
            hsize_t size;
            err = H5Dget_chunk_info_by_coord(seq, offset, NULL, &addr, &size);
            all_evt_seq_size_z += size;
            offset[0] += chunk_dims[0];
            grp_zip_sizes[g] += size;
        }

        if (seq_opt == 1) nchunks_shared += nChunks[g][0];
        err = H5Dclose(seq);

        long long lower_upper[2];
        lower_upper[0] = lowers[g];
        lower_upper[1] = uppers[g];
        MPI_Gather(lower_upper, 2, MPI_LONG_LONG, bounds, 2, MPI_LONG_LONG, 0, comm);

        for (d=1; d<groups[g].nDatasets; d++) {
            hsize_t dims[2], chunk_dims[2];

            /* open dataset */
            dset = H5Dopen2(fd, groups[g].dset_names[d], H5P_DEFAULT);

            /* collect metadata of ect.seq again, as this process may not have
             * collected metadata of all evt.seq
             */

            /* inquire dimension sizes of dset */
            hid_t fspace = H5Dget_space(dset);
            err = H5Sget_simple_extent_dims(fspace, dims, NULL);
            err = H5Sclose(fspace);

            /* inquire chunk size along each dimension */
            hid_t chunk_plist = H5Dget_create_plist(dset);
            H5Pget_chunk(chunk_plist, 2, chunk_dims);
            err = H5Pclose(chunk_plist);
            /* Note chunk_dims[1] may be larger than 1, but dims[1] is not
             * chunked
             */
            if (chunk_dims[1] != dims[1]) CHECK_ERROR(-1, "chunk_dims[1] != dims[1]");

            /* data type of evt.seq is 64-bit integer */
            hid_t dtype = H5Dget_type(dset);
            size_t dtype_size = H5Tget_size(dtype);
            err = H5Tclose(dtype);

            /* calculate number of chunks of this dataset */
            nChunks[g][d] = dims[0] / chunk_dims[0];
            if (dims[0] % chunk_dims[0]) nChunks[g][d]++;
            total_nchunks += nChunks[g][d];

            /* calculate number of chunks read by this process */
            int nchunks = (uppers[g] / chunk_dims[0]) - (lowers[g] / chunk_dims[0]) + 1;
            my_nchunks_read += nchunks;
            my_nchunks_read_nokeys += nchunks;
            max_nchunks_read = MAX(max_nchunks_read, nchunks);
            min_nchunks_read = MIN(min_nchunks_read, nchunks);

            /* calculate nchunks_shared and max_shared_chunks */
            if (rank == 0) {
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
                grp_nChunks[g] += nChunks[g][d];

                /* calculate read sizes of compressed data */
                hsize_t offset[2]={0, 0};
                for (j=0; j<nChunks[g][d]; j++) {
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
    H5Fclose(fd);
    free(lowers);
    free(bounds);

#ifndef PANDANA_BENCHMARK
    mpi_err = MPI_File_close(&fh);
    if (mpi_err != MPI_SUCCESS) CHECK_ERROR(-1, "MPI_File_close");
#endif

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
               all_nDatasets, total_nchunks);
        printf("Aggregate number of chunks read by all processes: %lld\n",
               aggr_nchunks_read);
        printf("        averaged per process: %.2f\n", (float)aggr_nchunks_read/nprocs);
        printf("        averaged per process per dataset: %.2f\n", (float)aggr_nchunks_read/nprocs/all_nDatasets);
        printf("Out of %d chunks, number of chunks read by two or more processes: %d\n",
               total_nchunks,nchunks_shared);
        printf("Out of %d chunks, most shared chunk is read by number of processes: %d\n",
               total_nchunks,max_shared_chunks);
        printf("----------------------------------------------------\n");

        for (g=0; g<nGroups; g++) {
            char *gname = strtok(groups[g].dset_names[0], "/");
            printf("group %46s size %5zd MiB (zipped %4zd MiB) nChunks=%zd\n",
                   gname,grp_sizes[g]/1048576, grp_zip_sizes[g]/1048576, grp_nChunks[g]);
        }
        printf("\n\n");
    }
    fflush(stdout);
    MPI_Barrier(comm);
    free(grp_sizes);

    printf("rank %3d: no. chunks read=%lld include evt.seq (max=%d min=%d avg=%.2f among %d datasets, exclude evt.seq)\n",
           rank, my_nchunks_read, max_nchunks_read, min_nchunks_read,
           (float)my_nchunks_read_nokeys/(float)all_nDatasets, all_nDatasets-nGroups);

    for (g=0; g<nGroups; g++)
        free(nChunks[g]);
    free(nChunks);

    return 1;
}


/*----< usage() >------------------------------------------------------------*/
static void
usage(char *progname)
{
#define USAGE   "\
  [-h]           print this command usage message\n\
  [-p number]    performance profiling method (0 or 1)\n\
                 0: report file open, close, read timings (default)\n\
                 1: report number of chunks read per process\n\
  [-s number]    read method for key datasets (0, 1, 2, 3, or 4)\n\
                 0: root process HDF5 reads and broadcasts (default)\n\
                 1: all processes HDF5 read the entire keys collectively\n\
                 2: root process HDF5 reads each key, one at a time,\n\
                    calculates, scatters boundaries to other processes\n\
                 3: distribute key reading among processes, make one MPI\n\
                    collective read to read all asigned keys, and scatter\n\
                    boundaries to other processes\n\
                 4: root POSIX reads all chunks of keys, one dataset at a\n\
                    time, decompress, and scatter boundaries\n\
  [-m number]    read method for other datasets (0, 1, 2, or 3)\n\
                 0: use H5Dread, one dataset at a time (default)\n\
                 1: use MPI_file_read_all, one dataset at a time\n\
                 2: use MPI_file_read_all, all datasets in one group at a\n\
                    time\n\
                 3: use chunk-aligned partitioning and H5Dread to read one\n\
                    dataset at a time. When used, -s argument is ignored.\n\
                    Reading key datasets are distributed using H5Dread, one\n\
                    dataset at a time.\n\
  [-r number]    parallelization method (0 or 1)\n\
                 0: data parallelism - all processes read each dataset in\n\
                    parallel (default)\n\
                 1: group parallelism - processes are divided among groups\n\
                    then data parallelism within each groups\n\
                 2: dataset parallelism - divide all datasets among processes\n\
  [-l file_name] name of file containing dataset names to be read\n\
  [-i file_name] name of input HDF5 file\n\
  *ph5concat version _PH5CONCAT_VERSION_ of _PH5CONCAT_RELEASE_DATE_\n"

    printf("Usage: %s [-h] [-p number] [-s number] [-m number] [-r number] [-l file_name] [-i file_name]\n%s\n",
           progname, USAGE);
}

/*----< main() >-------------------------------------------------------------*/
int main(int argc, char **argv)
{
    int seq_opt=0, dset_opt=0, profile=0;
    int c, d, g, nprocs, rank, nGroups, parallelism=0;
    char *listfile=NULL, *infile=NULL;
    double all_t, max_t[6], min_t[6];
    long long read_len, numIDs;
    NOvA_group *groups=NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* command-line arguments */
    while ((c = getopt(argc, argv, "hr:p:s:m:l:i:")) != -1)
        switch(c) {
            case 'h': if (rank  == 0) usage(argv[0]);
                      goto fn_exit;
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
    if (seq_opt < 0 || seq_opt > 4) { /* option for reading keys */
        if (rank  == 0) {
            printf("Error: option -s must be 0, 1, 2, 3, or 4\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }
    if (dset_opt < 0 || dset_opt > 3) { /* option for reading other datasets */
        if (rank  == 0) {
            printf("Error: option -m must be 0, 1, 2, or 3\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }
    if (parallelism < 0 || parallelism > 2) { /* option of parallelization method */
        if (rank  == 0) {
            printf("Error: option -r must be 0 or 1\n");
            usage(argv[0]);
        }
        goto fn_exit;
    }

    /* From file 'listfile', read dataset names, calculate number of datasets,
     * number of groups, maximum number of datasets among groups
     */
    nGroups = read_dataset_names(rank, listfile, &groups);
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
        printf("Read key   datasets method: ");
        if (parallelism == 2)
            printf("skipped\n");
        if (dset_opt == 3)
            printf("Distributed H5Dread and scatters aligned boundaries\n");
        else if (seq_opt == 0)
            printf("root process H5Dread and broadcasts\n");
        else if (seq_opt == 1)
            printf("all processes H5Dread collectively\n");
        else if (seq_opt == 2)
            printf("root process H5Dread and scatters boundaries\n");
        else if (seq_opt == 3)
            printf("Distributed MPI collective read, decompress, and scatters boundaries\n");
        else if (seq_opt == 4)
            printf("Distributed POSIX read, decompress, and scatters boundaries\n");

        printf("Read other datasets method: ");
        if (parallelism == 2)
            printf("Independent H5Dread, one dataset by one process\n");
        else if (dset_opt == 0)
            printf("Collective H5Dread, one dataset at a time\n");
        else if (dset_opt == 1)
            printf("MPI collective read and decompress, one dataset at a time\n");
        else if (dset_opt == 2)
            printf("MPI collective read and decompress, all datasets in one group at a time\n");
        else if (dset_opt == 3)
            printf("Align data partitioning with chunk boundaries and call H5Dread to read in parallel\n");

        printf("Parallelization method: ");
        if (parallelism == 0)
            printf("data parallelism (all processes read individual datasets in parallel)\n");
        else if (parallelism == 1)
            printf("group parallelism (processes divided into groups, data parallelism in each group)\n");
        else if (parallelism == 2)
            printf("dataset parallelism (divide all datasets among processes)\n");
    }
    fflush(stdout);

#ifdef PANDANA_BENCHMARK
    extern void set_options(int seq_read_opt, int dset_read_opt);
    set_options(seq_opt, dset_opt);

    extern void init_timers(void);
    init_timers();
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    all_t = MPI_Wtime();

    char *dset_global_ID = "/spill/evt.seq";

    if (parallelism < 2 && dset_opt != 3) { /* data and group parallelism only */
        /* Inquire number of globally unique IDs (size of dset_global_ID) */
        herr_t err;
        hid_t fd = H5Fopen(infile, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fd < 0) {
            fprintf(stderr,"%d: Error: fail to open file %s (%s)\n",
                    rank,  infile, strerror(errno));
            goto fn_exit;
        }
        hid_t dset = H5Dopen2(fd, dset_global_ID, H5P_DEFAULT);
        if (dset < 0) CHECK_ERROR(seq, "H5Dopen2");
        hid_t fspace = H5Dget_space(dset);
        if (fspace < 0) CHECK_ERROR(dset, "H5Dget_space");
        hsize_t dims[2];
        err = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (err < 0) CHECK_ERROR(err, "H5Sget_simple_extent_dims");
        err = H5Sclose(fspace);
        if (err < 0) CHECK_ERROR(err, "H5Sclose");
        /* 2nd dimension of global ID dataset is always 1 */
        if (dims[1] != 1) CHECK_ERROR(-1, "dims[1] != 1");
        err = H5Dclose(dset);
        if (err < 0) CHECK_ERROR(err, "H5Dclose");
        numIDs = dims[0];
        err = H5Fclose(fd);
        if (err < 0) CHECK_ERROR(err, "H5Fclose");
    }

    if (parallelism == 0) /* data parallelism */
        read_len = pandana_data_parallelism(MPI_COMM_WORLD, infile, nGroups,
                                    groups, numIDs);
    else if (parallelism == 1) /* group parallelism */
        read_len = pandana_group_parallelism(MPI_COMM_WORLD, infile, nGroups,
                                    groups, numIDs);
    else if (parallelism == 2) /* dataset parallelism */
        read_len = pandana_dataset_parallelism(MPI_COMM_WORLD, infile, nGroups,
                                    groups);

#ifdef PANDANA_BENCHMARK
    double timings[6];
    timings[5] = MPI_Wtime() - all_t;

    extern void get_timings(double*);
    get_timings(timings);

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
#else
    all_t = MPI_Wtime() - all_t;
    MPI_Reduce(&all_t, max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&all_t, min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
#endif

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
#ifdef PANDANA_BENCHMARK
        printf("MAX time: open=%.2f key=%.2f datasets=%.2f close=%.2f inflate=%.2f TOTAL=%.2f\n",
               max_t[0],max_t[1],max_t[2],max_t[3],max_t[4],max_t[5]);
        printf("MIN time: open=%.2f key=%.2f datasets=%.2f close=%.2f inflate=%.2f TOTAL=%.2f\n",
               min_t[0],min_t[1],min_t[2],min_t[3],min_t[4],min_t[5]);
#else
        printf("MAX end-to-end time %.2f seconds\n", max_t[0]);
        printf("MIN end-to-end time %.2f seconds\n", min_t[0]);
#endif
        printf("----------------------------------------------------\n");
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (profile)
        chunk_statistics(MPI_COMM_WORLD, infile, groups, nGroups, seq_opt);

fn_exit:
    if (groups != NULL) {
        for (g=0; g<nGroups; g++) {
            for (d=0; d<groups[g].nDatasets; d++)
                free(groups[g].dset_names[d]);
            free(groups[g].dset_names);
        }
        free(groups);
    }
    if (listfile != NULL) free(listfile);
    if (infile != NULL) free(infile);

    MPI_Finalize();
    return 0;
}
