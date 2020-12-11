/*
 * Copyright (C) 2019, Northwestern University and Fermi National Accelerator Laboratory
 * See COPYRIGHT notice in top-level directory.
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strdup() */
#include <unistd.h> /* getopt() */
#include <assert.h> /* assert() */

#include <hdf5.h>

#define HANDLE_ERROR(msg,name) { \
    printf("Error at line %d: func %s on %s\n",__LINE__,msg,name); \
    err_exit = -1; \
    goto fn_exit; \
}

struct op_data {
    int     non_zero_only;
    int     spill_grp;
    int     print_spill_grp;
    hsize_t upper_bound;
    herr_t  err;
};

hsize_t nGroups;
char **grp_names;
size_t *grp_size;
int     debug;
int     numLargeGrp;
int     numZeroGrp;

/*----< count_size() >----------------------------------------------------------*/
/* call back function used in H5Ovisit() */
static
herr_t count_size(hid_t             loc_id,        /* object ID */
                  const char       *name,          /* object name */
                  const H5O_info_t *info,          /* object metadata */
                  void             *operator_data) /* data passed from caller */
{
    int err_exit=0;
    herr_t err;
    struct op_data *it_op = (struct op_data*)operator_data;

    if (info->type == H5O_TYPE_GROUP) {
        if (!strcmp(name, ".")) { /* ignore root group '.' */
            nGroups = 0;
            return 0;
        }
        if (!strcmp(name, "spill")) it_op->spill_grp = nGroups;

        grp_names[nGroups] = (char*) malloc(strlen(name) + 1);
        strcpy(grp_names[nGroups], name);
        grp_size[nGroups] = 0;
        nGroups++;
    }
    else if (info->type == H5O_TYPE_DATASET) {
        /* check if group name is the same of previous visit */
        char *str;
        char *gname = (char*) malloc(strlen(name) + 1);
        strcpy(gname, name);
        str = strchr(gname, '/');
        *str = '\0';
        assert(strcmp(gname, grp_names[nGroups-1]) == 0);
        free(gname);

        hid_t dset = H5Dopen(loc_id, name, H5P_DEFAULT);
        if (dset < 0) HANDLE_ERROR("H5Dopen", name);

        /* Retrieve number of dimensions */
        hid_t fspace = H5Dget_space(dset);
        if (fspace < 0) HANDLE_ERROR("H5Dget_space", name)

        /* retrieve dimension sizes */
        hsize_t dims[2];
        int ndims = H5Sget_simple_extent_dims(fspace, dims, NULL);
        if (ndims < 0) HANDLE_ERROR("H5Sget_simple_extent_dims", name)
        assert(ndims == 2);

        /* get data type and size */
        hid_t dtype = H5Dget_type(dset); assert(dtype >= 0);
        size_t dtype_size = H5Tget_size(dtype); assert(dtype_size > 0);
        err = H5Tclose(dtype); assert(err >= 0);

        /* accumulate dataset size in this group */
        grp_size[nGroups-1] += dims[0]*dims[1]*dtype_size;

        err = H5Sclose(fspace); assert(err >= 0);
        err = H5Dclose(dset); assert(err >= 0);
    }

fn_exit:
    it_op->err = err_exit;
    return (err_exit == 0) ? 0 : 1;
}

/*----< print_name() >----------------------------------------------------------*/
/* call back function used in H5Ovisit() */
static
herr_t print_names(hid_t             loc_id,        /* object ID */
                   const char       *name,          /* object name */
                   const H5O_info_t *info,          /* object metadata */
                   void             *operator_data) /* data passed from caller */
{
    struct op_data *it_op = (struct op_data*)operator_data;

    if (info->type == H5O_TYPE_GROUP) {
        if (!strcmp(name, ".")) { /* ignore root group '.' */
            nGroups = 0;
            return 0;
        }
        if (grp_size[nGroups] == 0) numZeroGrp++;
        else if (grp_size[nGroups] >= it_op->upper_bound) numLargeGrp++;

        nGroups++;
    }
    else if (!debug && info->type == H5O_TYPE_DATASET) {
        /* check if group name is the same of previous visit */
        char *str;
        char *gname = (char*) malloc(strlen(name) + 1);
        strcpy(gname, name);
        str = strchr(gname, '/');
        *str = '\0';
        assert(strcmp(gname, grp_names[nGroups-1]) == 0);
        free(gname);

        if (it_op->non_zero_only) {
            if (!it_op->print_spill_grp && !strcmp(name, "spill/evt.seq"))
                /* if spill group is too big, print only /spill/evt.seq */
                printf("/%s\n",name);
            else if (grp_size[nGroups-1] > 0 && grp_size[nGroups-1] < it_op->upper_bound)
                printf("/%s\n",name);
        }
        else
            printf("/%s\n",name);
    }

fn_exit:
    it_op->err = 0;
    return 0;
}

/*----< main() >-------------------------------------------------------------*/
int main(int argc, char **argv)
{
    int i, err_exit=0, upper_bound;
    herr_t err;
    hid_t fd_in=-1;
    H5G_info_t grp_info;
    struct op_data it_op;

    debug = 0;
    upper_bound = 64; /* default 64 MiB */

    if (argc < 2 || argc > 3) {
        printf("Usage: %s input_file [upper bound in MiB]\n",argv[0]);
        goto fn_exit;
    }
    if (argc == 3) upper_bound = atoi(argv[2]);
    it_op.upper_bound = 1048576 * upper_bound;
    it_op.non_zero_only = 1;
    it_op.spill_grp = 0;
    it_op.print_spill_grp = 1;

    /* open input file in read-only mode */
    fd_in = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fd_in < 0) HANDLE_ERROR("Can't open input file", argv[1])

    err = H5Gget_info_by_name(fd_in, "/", &grp_info, H5P_DEFAULT);
    if (err < 0) HANDLE_ERROR("H5Gget_info_by_name - root group",  argv[1])

    grp_names = (char**) malloc(grp_info.nlinks * sizeof(char*));
    grp_size = (size_t*) malloc(grp_info.nlinks * sizeof(size_t));;

    numZeroGrp = 0;
    numLargeGrp = 0;
    /* Iterate all objects and count the accumulated dataset sizes of each group */
#define HAS_H5OVISIT3 1
#if defined HAS_H5OVISIT3 && HAS_H5OVISIT3
    err = H5Ovisit3(fd_in, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, count_size, &it_op, H5O_INFO_ALL);
    if (err < 0) HANDLE_ERROR("H5Ovisit3", argv[1])
    if (it_op.err < 0) HANDLE_ERROR("H5Ovisit3", argv[1])
#else
    err = H5Ovisit(fd_in, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, count_size, &it_op);
    if (err < 0) HANDLE_ERROR("H5Ovisit", argv[1])
    if (it_op.err < 0) HANDLE_ERROR("H5Ovisit", argv[1])
#endif
    assert(nGroups == grp_info.nlinks);

    if (grp_size[it_op.spill_grp] >= it_op.upper_bound)
        it_op.print_spill_grp = 0;

    /* Iterate all objects and print dataset names */
#if defined HAS_H5OVISIT3 && HAS_H5OVISIT3
    err = H5Ovisit3(fd_in, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, print_names, &it_op, H5O_INFO_ALL);
    if (err < 0) HANDLE_ERROR("H5Ovisit3", argv[1])
    if (it_op.err < 0) HANDLE_ERROR("H5Ovisit3", argv[1])
#else
    err = H5Ovisit(fd_in, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, print_names, &it_op);
    if (err < 0) HANDLE_ERROR("H5Ovisit", argv[1])
    if (it_op.err < 0) HANDLE_ERROR("H5Ovisit", argv[1])
#endif

    if (debug) {
        printf("number of groups           = %zd\n", nGroups);
        printf("number of zero-size groups = %d\n", numZeroGrp);
        printf("number of large groups     = %d\n", numLargeGrp);
        printf("number of groups size < %d MiB = %d\n",
               upper_bound,nGroups-numZeroGrp-numLargeGrp);
        printf("size of spill group[%d] = %d (%d MiB)\n", it_op.spill_grp,
               grp_size[it_op.spill_grp], grp_size[it_op.spill_grp]/1048576);
    }

    for (i=0; i<nGroups; i++) free(grp_names[i]);
    free(grp_names);
    free(grp_size);

    err = H5Fclose(fd_in);
    if (err < 0) printf("Error at line %d: H5Fclose\n",__LINE__);

fn_exit:
    return (err_exit != 0);
}
