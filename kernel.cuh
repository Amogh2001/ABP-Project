void sum_caller(int N, double* h_A, int* h_non_zero_bool, int* h_row_count);

void scan_caller(int N, int* h_input, int* h_output, int* h_sums, int* h_sums_results);

void warp_max_caller(int N, int* h_input, int* h_output);

__global__ void sellc_assembler(
        int N,
        const int* __restrict__ row_nnz,
        const int* __restrict__ row_start,
        const double* __restrict__ input_vals,
        const int* __restrict__ max_nz_per_warp,
        const int* __restrict__ offset,
        double* __restrict__ val_out);
