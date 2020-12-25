/*
 * Copyright (C) 2020, Northwestern University and Fermi National Accelerator
 * Laboratory
 * See COPYRIGHT notice in top-level directory.
 */

#include <mpi.h>
#include "hdf5.h"

int
pandana_inq_ranges(int        nprocs, /* number of MPI processes */
                   int        rank,   /* MPI rank of this process */
                   long long  numIDs, /* no. unique IDs */
                   long long *starts, /* OUT */
                   long long *ends);  /* OUT */

ssize_t
pandana_posix_read_dset(hid_t  dset,      /* HDF5 dataset ID */
                        int    posix_fd,  /* POSIX file descriptor */
                        void  *buf);      /* OUT: user buffer */

ssize_t
pandana_hdf5_read_dset(hid_t  fd,    /* HDF5 file descriptor */
                       hid_t  dset,  /* dataset ID */
                       void  *buf);  /* user read buffer */
ssize_t
pandana_hdf5_read_subarray(hid_t    fd,         /* HDF5 file descriptor */
                           hid_t    dset,       /* dataset ID */
                           hsize_t  lower,      /* array index lower bound */
                           hsize_t  upper,      /* array index upper bound */
                           hid_t    xfer_plist, /* data transfer property */
                           void    *buf);       /* user read buffer */
ssize_t
pandana_hdf5_read_subarrays(hid_t        fd,        /* HDF5 file ID */
                            int          nDatasets, /* number of datasets */
                            const hid_t *dsets,     /* [nDatasets] dataset IDs */
                            hsize_t      lower,     /* lower bound */
                            hsize_t      upper,     /* upper bound */
                            hid_t        xfer_plist,/* data transfer property */
                            void       **buf);      /* [nDatasets] user buffer */
ssize_t
pandana_mpi_read_dsets(MPI_Comm   comm,      /* MPI communicator */
                       hid_t      fd,        /* HDF5 file ID */
                       MPI_File   fh,        /* MPI file handler */
                       int        nDatasets, /* number of datasets */
                       hid_t     *dsets,     /* IN:  [nDatasets] dataset IDs */
                       void     **buf);      /* OUT: [nDatasets] read buffers */
ssize_t
pandana_mpi_read_subarray(hid_t          fd,     /* HDF5 file descriptor */
                          MPI_File       fh,     /* MPI file handler */
                          const hid_t    dset,   /* dataset ID */
                          hsize_t        lower,  /* array index lower bound */
                          hsize_t        upper,  /* array index upper bound */
                          unsigned char *buf);   /* user read buffer */
ssize_t
pandana_mpi_read_subarrays(hid_t        fd,       /* HDF5 file descriptor */
                          MPI_File      fh,       /* MPI file handler */
                          int           nDatasets,/* number of datasets */
                          const hid_t  *dsets,    /* [nDatasets] dataset IDs */
                          hsize_t       lower,    /* array index lower bound */
                          hsize_t       upper,    /* array index upper bound */
                          void        **buf);     /* [nDatasets] user buffer */
ssize_t
pandana_mpi_read_subarrays_aggr(hid_t         fd,        /* HDF5 file descriptor */
                                MPI_File      fh,        /* MPI file handler */
                                int           nDatasets, /* number of datasets */
                                const hid_t  *dsets,     /* [nDatasets] dataset IDs */
                                hsize_t       lower,     /* lower bound */
                                hsize_t       upper,     /* upper bound */
                                void        **buf);      /* [nDatasets] user buffer */

ssize_t
pandana_read_keys(MPI_Comm   comm,       /* MPI communicator */
                  hid_t      fd,         /* HDF5 file ID */
                  MPI_File   fh,         /* MPI file handler */
                  int        nGroups,    /* number of key datasets */
                  char     **key_names,  /* [nGroups] key dataset names */
                  long long  numIDs,     /* number of globally unique IDs */
                  size_t    *lowers,     /* OUT: [nGroups] */
                  size_t    *uppers);    /* OUT: [nGroups] */

typedef struct {
    int       nDatasets;  /* number of datasets in this group */
    char    **dset_names; /* [nDatasets] string names of datasets */
} NOvA_group;

ssize_t
pandana_data_parallelism(MPI_Comm    comm,     /* MPI communicator */
                         const char *infile,   /* input HDF5 file name */
                         int         nGroups,  /* number of groups */
                         NOvA_group *groups,   /* array of group objects */
                         long long   numIDs);  /* number of unique IDs */

ssize_t
pandana_group_parallelism(MPI_Comm    comm,     /* MPI communicator */
                          const char *infile,   /* input HDF5 file name */
                          int         nGroups,  /* number of groups */
                          NOvA_group *groups,   /* array of group objects */
                          long long   numIDs);  /* number of unique IDs */

ssize_t
pandana_dataset_parallelism(MPI_Comm    comm,     /* MPI communicator */
                            const char *infile,   /* input HDF5 file name */
                            int         nGroups,  /* number of groups */
                            NOvA_group *groups);  /* array of group objects */

