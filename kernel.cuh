__global__ void VectorAdd(int N, double* A, double* B, double* C);

void d1_caller(int N, double* h_I);

void iden_caller(int N, double* h_I);

void kron_caller(int A_rows, int A_cols, int B_rows, int B_cols,
                 double* h_A, double* h_B, double* h_C);

void sum_caller(int N, double* h_A, int* h_non_zero_bool, int* h_row_count);

void scan_caller(int N, int* h_input, int* h_output, int* h_sums, int* h_sums_results);

void warp_max_caller(int N, int* h_input, int* h_output);

void crs_caller(int N, double* h_A, int* h_rpt, int numnz, double* h_val, int* h_col);

void sellc_caller(int N, int* h_row_count, int* h_scan_output, double* h_input_vals, int* h_warpmax_output, int* h_warpscan_output, double* val_output, int* col_input, int* col_output);

__global__ void spmv_crs(int N, const double* val, const int* col, const int    * rpt, const double* x, double* y);

void sellc_multiply_caller(int N, double* h_val, int* h_col, int* h_offset,     int* h_max_nz_per_warp, double* h_x, double* h_y);

void axpy_caller(int M, double alpha, const double* h_x, double* h_y);

double dot_prod(int N, double* d_v1, double* d_v2);

double vector_norm(int N, double* h_v);

void scale_caller(int M, double alpha, const double* h_x, double* h_y);
__global__ void sellc_multiply(int N, const double* __restrict__ val, const int* __restrict__ col, const int* __restrict__ offset, const int* __restrict__ max_nz_per_warp, const double* __restrict__ x, double* __restrict__ y);

double dot_prod_gpu(int N, double* d_v1, double* d_v2, double* d_result);

__global__ void scale_gpu(int N, double alpha, const double* x, double* y);

__global__ void axpy(int N, double alpha, const double* x, double* y);

void full_reduce_caller(int N, double* d_input, double* d_result);

__global__ void vecvec_prod(int N, double* A, double* B, double* C);
