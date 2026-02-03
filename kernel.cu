#include <stdio.h>


//#define block_size 32


//============================= LANCZOS SECTION ==========================


__global__ void VectorAdd(int N, double* A, double* B, double* C) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < N) {
                C[i] = A[i] + B[i];
        }

}


__global__ void kron_prod(int A_rows, int A_cols, int B_rows, int B_cols,
                          double* A, double* B, double* C) {
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    
    int C_rows = A_rows * B_rows;
    int C_cols = A_cols * B_cols;
    
    if(Row < C_rows && Col < C_cols) {
        int A_row = Row / B_rows;
        int A_col = Col / B_cols;
        int B_row = Row % B_rows;
        int B_col = Col % B_cols;
        
        C[Row*C_cols + Col] = A[A_row*A_cols + A_col] * B[B_row*B_cols + B_col];
    }
}

__global__ void iden_set(int N, double* I) {

         int gid_x = blockDim.x*blockIdx.x + threadIdx.x;
         int gid_y = blockDim.y*blockIdx.y + threadIdx.y;

         if(gid_x < N && gid_y < N) {

                 if(gid_x == gid_y) {
                         I[gid_y*N + gid_x] = 1.0;

                 }
         }

}


__global__ void delta1_set(int N, double* I) {

         int gid_x = blockDim.x*blockIdx.x + threadIdx.x;
         int gid_y = blockDim.y*blockIdx.y + threadIdx.y;

         if(gid_x < N && gid_y < N) {

                 if(gid_x == gid_y) {
                         I[gid_y*N + gid_x] = -2.0;

                 }

                 if(gid_x == gid_y + 1 || gid_x == gid_y -1) {
                         I[gid_y*N + gid_x] = 1.0;
                 }

         }

}

void d1_caller(int N, double* h_I) {

         //I is N*N
         int block_size = 32;
         double* d_I;

         cudaMalloc(&d_I, N*N*sizeof(double));

         cudaMemset(d_I, 0, N*N*sizeof(double));

         dim3 block_dim(block_size, block_size);
         dim3 grid_dim((N+block_size-1) / block_size,
                         (N+block_size - 1) / block_size);

         delta1_set<<<grid_dim, block_dim>>>(N, d_I);

         cudaMemcpy(h_I, d_I, N*N*sizeof(double), cudaMemcpyDeviceToHost);


         cudaFree(d_I);

}





void iden_caller(int N, double* h_I) {

         //I is N*N

         double* d_I;

         cudaMalloc(&d_I, N*N*sizeof(double));

         cudaMemset(d_I, 0, N*N*sizeof(double));

         int block_size = 32;
         dim3 block_dim(block_size, block_size);
         dim3 grid_dim((N+block_size-1) / block_size,
                         (N+block_size - 1) / block_size);

         iden_set<<<grid_dim, block_dim>>>(N, d_I);

         cudaMemcpy(h_I, d_I, N*N*sizeof(double), cudaMemcpyDeviceToHost);

         cudaFree(d_I);

}




void kron_caller(int A_rows, int A_cols, int B_rows, int B_cols,
                 double* h_A, double* h_B, double* h_C) {
    
    int C_rows = A_rows * B_rows;
    int C_cols = A_cols * B_cols;
    
    int block_size = 32;
    double* d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, A_rows*A_cols*sizeof(double));
    cudaMalloc(&d_B, B_rows*B_cols*sizeof(double));
    cudaMalloc(&d_C, C_rows*C_cols*sizeof(double));
    
    cudaMemcpy(d_A, h_A, A_rows*A_cols*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_rows*B_cols*sizeof(double), cudaMemcpyHostToDevice);
    
    const int numBlocks = 1+((C_rows > C_cols ? C_rows : C_cols)-1)/block_size;
    
    dim3 grid_dim(numBlocks, numBlocks);
    dim3 block_dim(block_size, block_size);
    
    kron_prod<<<grid_dim, block_dim>>>(A_rows, A_cols, B_rows, B_cols, 
                                       d_A, d_B, d_C);
    
    cudaMemcpy(h_C, d_C, C_rows*C_cols*sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



//========================== SUM KERNEL =================================

__global__ void nz_bool_finder(int N, const double* A, int* non_zero_bool) {

        int iglob = blockDim.x*blockIdx.x + threadIdx.x;
        int jglob = blockDim.y*blockIdx.y + threadIdx.y;

        if(iglob < N && jglob < N) {
                if(fabs(A[iglob*N + jglob]) > 1e-8) {
                        non_zero_bool[iglob*N + jglob] = 1;
                }
        }
}

__device__ void warpReduce(volatile int* sdata, int tid) {

        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];

}

__global__ void rowReduce(int N, const int* idata, int* odata) {

        extern __shared__ int sdata[];

        unsigned int tid = threadIdx.x;

        unsigned int i_row = blockIdx.x;

        int sum = 0;

        for(int k = tid; k < N; k += blockDim.x) {

                sum += idata[i_row*N + k];
        }

        sdata[tid] = sum;
        __syncthreads();

        for(unsigned int s = blockDim.x/2; s > 32; s >>= 1) {

                if(tid < s) {
                        sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
        }

        if(tid < 32) {
                warpReduce(sdata, tid);
        }

        if(tid == 0) {
                odata[i_row] = sdata[0];
        }
}

//============================= SUM CALLER ===============================

void sum_caller(int N, double* h_A, int* h_non_zero_bool, int* h_row_count) {

        int block_size = 32;

        double* d_A;
        int* d_non_zero_bool, *d_row_count;

        cudaMalloc(&d_A, N*N*sizeof(double));

        cudaMalloc(&d_row_count, N*sizeof(int));
        cudaMalloc(&d_non_zero_bool, N*N*sizeof(int));

        cudaMemcpy(d_A, h_A, N*N*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemset(d_non_zero_bool, 0, N*N*sizeof(int));

        int num_blocks = (N + block_size - 1) / block_size;
        dim3 grid_size(num_blocks, num_blocks);
        dim3 block_layout(block_size, block_size);

        nz_bool_finder<<<grid_size, block_layout>>>(N, d_A, d_non_zero_bool);
        cudaDeviceSynchronize();

        rowReduce<<<N, 256, 256*sizeof(int)>>>(N, d_non_zero_bool, d_row_count);
        cudaDeviceSynchronize();

        cudaMemcpy(h_row_count, d_row_count, N*sizeof(int), cudaMemcpyDeviceToHost);


        cudaFree(d_A);
        cudaFree(d_non_zero_bool);
        cudaFree(d_row_count);

}

//======================= SCAN KERNEL =============================

__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums) {
        int blockID = blockIdx.x;
        int threadID = threadIdx.x;
        int blockOffset = blockID * n;

        extern __shared__ int temp[];
        temp[2 * threadID] = input[blockOffset + (2 * threadID)];
        temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

        int offset = 1;
        for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
        {
                __syncthreads();
                if (threadID < d)
                {
                        int ai = offset * (2 * threadID + 1) - 1;
                        int bi = offset * (2 * threadID + 2) - 1;
                        temp[bi] += temp[ai];
                }
                offset *= 2;
        }
        __syncthreads();


        if (threadID == 0) {
                sums[blockID] = temp[n - 1];
                temp[n - 1] = 0;
        }

        for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
        {
                offset >>= 1;
                __syncthreads();
                if (threadID < d)
                {
                        int ai = offset * (2 * threadID + 1) - 1;
                        int bi = offset * (2 * threadID + 2) - 1;
                        int t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                }
        }
        __syncthreads();

        output[blockOffset + (2 * threadID)] = temp[2 * threadID];
        output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}


__global__ void add(int *output, int length, int *n) {
        int blockID = blockIdx.x;
        int threadID = threadIdx.x;
        int blockOffset = blockID * length;

        output[blockOffset + (2*threadID)] += n[blockID];
        output[blockOffset + (2*threadID)+1] += n[blockID];
}

void host_scan(int N, int* A, int* sums_result) {

        sums_result[0] = 0;

        for(int i = 1; i < N; ++i) {
                sums_result[i] = sums_result[i-1] + A[i-1];
        }
}

//=============================== SCAN CALLER =========================

void scan_caller(int N, int* h_input, int* h_output, int* h_sums, int* h_sums_results) {


        int block_size = 128;
        int n = block_size * 2;
        //int numBlocks = N / n;
        int numBlocks = (N + n - 1) / n;

        if(numBlocks == 0) numBlocks = 1;

        int sum_blocks = numBlocks;
        int sharedBytes = n*sizeof(int);

        int* d_input, *d_output, *d_sums, *d_sums_results;

        cudaMalloc(&d_input, N*sizeof(int));
        cudaMalloc(&d_output, (N+1)*sizeof(int));
        cudaMalloc(&d_sums, sum_blocks*sizeof(int));
        cudaMalloc(&d_sums_results, (sum_blocks+1)*sizeof(int));

        cudaMemcpy(d_input, h_input, N*sizeof(int), cudaMemcpyHostToDevice);


        prescan_large_unoptimized<<<numBlocks, block_size, sharedBytes>>>(d_output, d_input, n, d_sums);

        cudaDeviceSynchronize();
        /*
        cudaMemcpy(h_sums, d_sums, sum_blocks*sizeof(int), cudaMemcpyDeviceToHost);


        host_scan(sum_blocks, h_sums, h_sums_results);  

        cudaMemcpy(d_sums_results, h_sums_results, sum_blocks*sizeof(int), cudaMemcpyHostToDevice);

        */

        if (sum_blocks > 1) {
                // Recursively scan the block sums on GPU
                int numBlocks2 = (sum_blocks + n - 1) / n;
                if (numBlocks2 == 0) numBlocks2 = 1;
        
                int* d_temp_sums;
                cudaMalloc(&d_temp_sums, numBlocks2*sizeof(int));
        
                prescan_large_unoptimized<<<numBlocks2, block_size, sharedBytes>>>(
                d_sums_results, d_sums, n, d_temp_sums);
                cudaDeviceSynchronize();
        
                // If there are more levels needed, handle recursively
                // (For most cases, sum_blocks is small enough that one level suffices)
        
        cudaFree(d_temp_sums);
        } else {
                // Only one block, no need to scan
                cudaMemset(d_sums_results, 0, sizeof(int));
        }



        add<<<numBlocks, block_size>>>(d_output, n, d_sums_results); 

        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, (N+1)*sizeof(int), cudaMemcpyDeviceToHost); 

        cudaFree(d_input); 
        cudaFree(d_output); 
        cudaFree(d_sums); 
        cudaFree(d_sums_results); 

}

//============================ WARP MAX SECTION ========================

__device__ int warpMax(int value) {
    unsigned mask = 0xffffffff;
    value = max(value, __shfl_down_sync(mask, value, 16));
    value = max(value, __shfl_down_sync(mask, value, 8));
    value = max(value, __shfl_down_sync(mask, value, 4));
    value = max(value, __shfl_down_sync(mask, value, 2));
    value = max(value, __shfl_down_sync(mask, value, 1));
    
    // Broadcast result to all threads in warp (optional but cleaner)
    //value = __shfl_sync(mask, value, 0);
    return value;
}

__global__ void computeWarpMax(const int* rpt, int* warpMaxOut, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    int lane = threadIdx.x % warpSize;
    int warpId = tid / warpSize;  // Global warp ID - this is correct IF warpMaxOut is sized properly
    
    int val = rpt[tid];
    int maxVal = warpMax(val);
    
    if (lane == 0)
        warpMaxOut[warpId] = maxVal;
}

void warp_max_caller(int N, int* h_input, int* h_output) {

        int numWarps = (N + 32 - 1) / 32;

        int* d_input, *d_output;
        cudaMalloc(&d_input, N*sizeof(int));
        cudaMalloc(&d_output, numWarps*sizeof(int));

        cudaMemcpy(d_input, h_input, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, numWarps*sizeof(int));
        int block_size = 128;
        int numBlocks = (N + block_size - 1) / block_size;

        computeWarpMax<<<numBlocks, block_size>>>(d_input, d_output, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, numWarps*sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

}

//=========================== CRS SECTION =========================

__global__ void nz_per_row(int N, double* A, int* rpt, double* val, int* col) {

        int tid = blockDim.x*blockIdx.x + threadIdx.x;
        if(tid >= N) return;

        int start = rpt[tid];
        int end = rpt[tid + 1];
        int out = start;

        for(int i = 0; i < N; i++) {
                if(fabs(A[tid*N + i]) > 1e-8) {
                        if(out >= end) break;
                        val[out] = A[tid*N + i];
                        col[out] = i;
                        out++;
                }
        }
}


void crs_caller(int N, double* h_A, int* h_rpt, int numnz, double* h_val, int* h_col) {

        int* d_rpt, *d_col;
        double* d_A, *d_val;

        cudaMalloc(&d_A, N*N*sizeof(double));
        cudaMalloc(&d_val, numnz*sizeof(double));
        cudaMalloc(&d_rpt, (N+1)*sizeof(int));
        cudaMalloc(&d_col, numnz*sizeof(int));

        cudaMemcpy(d_A, h_A, N*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_val, 0, numnz*sizeof(double));
        cudaMemcpy(d_rpt, h_rpt, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_col, 0, numnz*sizeof(int)); 


        int block_size = 256;
        int numBlocks = (N + block_size - 1) / block_size;

        nz_per_row<<<numBlocks, block_size>>>(N, d_A, d_rpt, d_val, d_col);

        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
                printf("error = %s\n ", cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();

        cudaMemcpy(h_val, d_val, numnz*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_col,d_col, numnz*sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_val);
        cudaFree(d_rpt);
        cudaFree(d_col);

}

//=========================== SELL-C-SIGMA SECTION =================

__global__ void sellc_assembler(
        int N,
        const int* __restrict__ row_nnz,
        const int* __restrict__ row_start,
        const double* __restrict__ input_vals,
        const int* __restrict__ max_nz_per_warp,
        const int* __restrict__ offset,
        double* __restrict__ val_out,
        const int* __restrict__ input_cols,
        int* __restrict__ col_out) {


        int gid = blockIdx.x*blockDim.x + threadIdx.x;
        int warp = gid / 32;
        //int lane = threadIdx.x % 32;
        int lane = gid % 32;
        int row = warp*32+lane;

        if(row>=N) return;

        int max_nz = max_nz_per_warp[warp];
        int write_base = offset[warp];

        for(int k = 0; k < max_nz; k++) {

                int out_index = write_base + k * 32 + lane;

                if(k < row_nnz[row]) {
                        val_out[out_index] = input_vals[row_start[row] + k];
                        col_out[out_index] = input_cols[row_start[row] + k];
                } else {
                        val_out[out_index] = 0.0;
                        col_out[out_index] = -1;
                }
        }
}

void sellc_caller(int N, int* h_row_count, int* h_scan_output, double* h_input_vals, int* h_warpmax_output, int* h_warpscan_output, double* h_val_output, int* h_col_input, int* h_col_output) {


        int numWarps = (N + 31) / 32;
        int block_size = 256;
        int numThreads = numWarps*32;
        int gridSize = (numThreads + block_size - 1) / block_size;

        int* d_row_count, *d_scan_output, *d_warpmax_output, *d_warpscan_output, *d_col_output, *d_col_input;
        double* d_input_vals, *d_val_output;

        cudaMalloc(&d_row_count, N*sizeof(int));
        cudaMalloc(&d_scan_output, (N+1)*sizeof(int));
        cudaMalloc(&d_warpmax_output, numWarps*sizeof(int));
        cudaMalloc(&d_warpscan_output, (numWarps+1)*sizeof(int));

        cudaMemcpy(d_row_count, h_row_count, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scan_output, h_scan_output, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_warpmax_output, h_warpmax_output, numWarps*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_warpscan_output, h_warpscan_output, (numWarps+1)*sizeof(int), cudaMemcpyHostToDevice);

        int num_nonzeros = h_scan_output[N];
        int numnz_sellc = h_warpscan_output[numWarps];

        cudaMalloc(&d_input_vals, num_nonzeros*sizeof(double));
        cudaMalloc(&d_val_output, numnz_sellc*sizeof(double)); 
        cudaMalloc(&d_col_input, num_nonzeros*sizeof(int)); 
        cudaMalloc(&d_col_output, numnz_sellc*sizeof(int)); 

        cudaMemcpy(d_input_vals, h_input_vals, num_nonzeros*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_input, h_col_input, num_nonzeros*sizeof(int), cudaMemcpyHostToDevice);

        sellc_assembler<<<gridSize, block_size>>>(N, d_row_count, d_scan_output, d_input_vals, d_warpmax_output, d_warpscan_output, d_val_output, d_col_input, d_col_output);

        cudaDeviceSynchronize();

        cudaMemcpy(h_val_output, d_val_output, numnz_sellc*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_col_output, d_col_output, numnz_sellc*sizeof(int), cudaMemcpyDeviceToHost);


        cudaFree(d_row_count);
        cudaFree(d_scan_output);
        cudaFree(d_warpmax_output);
        cudaFree(d_warpscan_output);

        cudaFree(d_input_vals);
        cudaFree(d_val_output);
        cudaFree(d_col_output);
        cudaFree(d_col_input);
} 


// ============================ CRS MULTIPLY ==========================

__global__ void spmv_crs(int N, const double* val, const int* col, const int* rpt, const double* x, double* y) {

        int row = blockIdx.x*blockDim.x + threadIdx.x;

        if(row >= N) return; 

        double sum = 0.0;
        int start = rpt[row];
        int end = rpt[row+1];

        for(int i = start; i < end; i++) {

                sum += val[i] * x[col[i]];
        }

        y[row] = sum;

}



// ========================== SELL C SIGMA MULTIPLY ======================


__global__ void sellc_multiply(int N, const double* __restrict__ val, const int* __restrict__ col, const int* __restrict__ offset, const int* __restrict__ max_nz_per_warp, const double* __restrict__ x, double* __restrict__ y) {


        int gid = blockDim.x*blockIdx.x + threadIdx.x;

        if(gid >= N) return;

        int warp = gid / 32;
        int lane = threadIdx.x % 32;

        int nz_per_warp = max_nz_per_warp[warp];
        int warp_offset = offset[warp];

        double sum = 0.0;

        for(int i = 0; i < nz_per_warp; i++) {

                int idx = warp_offset+i * 32 + lane;

                int col_idx = col[idx];
                double val_elem = val[idx];

                if(col_idx >= 0 && col_idx < N) {
                        sum += val_elem * x[col_idx];
                }
        }

        y[gid] = sum;

}


void sellc_multiply_caller(int N, double* h_val, int* h_col, int* h_offset, int* h_max_nz_per_warp, double* h_x, double* h_y) {

        int numWarps = (N + 31) / 32;

        double *d_val, *d_x, *d_y;
        int *d_col, *d_offset, *d_max_nz_per_warp;

        int numnz = h_offset[numWarps];

        cudaMalloc(&d_val, numnz*sizeof(double));
        cudaMalloc(&d_col, numnz*sizeof(int));
        cudaMalloc(&d_offset, (numWarps+1)*sizeof(int));
        cudaMalloc(&d_max_nz_per_warp, numWarps*sizeof(int));
        cudaMalloc(&d_x, N*sizeof(double));
        cudaMalloc(&d_y, N*sizeof(double));

        cudaMemcpy(d_val, h_val, numnz*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, h_col, numnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offset, h_offset, (numWarps+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_nz_per_warp, h_max_nz_per_warp, numWarps*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, N*sizeof(double), cudaMemcpyHostToDevice);

        int block_size = 256;
        int grid_size = (N + block_size -1) / block_size;

        sellc_multiply<<<grid_size, block_size>>>(N, d_val, d_col, d_offset, d_max_nz_per_warp, d_x, d_y);

        cudaDeviceSynchronize();

        cudaMemcpy(h_y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_val);
        cudaFree(d_col);
        cudaFree(d_offset);
        cudaFree(d_max_nz_per_warp);
        cudaFree(d_x);
        cudaFree(d_y);

}

//========================= HELPER FUNCTIONS ========================


__global__ void vecvec_prod(int N, double* A, double* B, double* C) {

        int i = blockDim.x*blockIdx.x + threadIdx.x;

        if(i < N) {
                C[i] = A[i] * B[i];
        }
}

void vecvec_caller(int N, double* h_A, double* h_B, double* h_C) {

        double* d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N*sizeof(double));
        cudaMalloc(&d_B, N*sizeof(double));
        cudaMalloc(&d_C, N*sizeof(double));


        cudaMemcpy(d_A, h_A, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, N*sizeof(double), cudaMemcpyHostToDevice);

        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;

        vecvec_prod<<<grid_size, block_size>>>(N, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, N*sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
}


__device__ void warpReduce(volatile double* sdata, int tid) {
        sdata[tid] += sdata[tid+32];
        sdata[tid] += sdata[tid+16];
        sdata[tid] += sdata[tid+8];
        sdata[tid] += sdata[tid+4];
        sdata[tid] += sdata[tid+2];
        sdata[tid] += sdata[tid+1];
}

__global__ void addReduce(double* g_idata, double* g_odata) {

        extern __shared__ double sdata_red[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
        sdata_red[tid] = g_idata[i] + g_idata[i + blockDim.x];
        __syncthreads();

        for(unsigned int s = blockDim.x/2; s > 32; s>>=1) {
                if(tid < s) {
                        sdata_red[tid] += sdata_red[tid+s];
                        __syncthreads();
                }
        }
        if(tid < 32) warpReduce(sdata_red, tid);
        if(tid == 0) g_odata[blockIdx.x] = sdata_red[0];
}


void full_reduce_caller(int N, double* d_input, double* d_result) {

    int block_size = 256;
    int grid_size = (N + (block_size*2) - 1) / (block_size*2);

    int padded_N = grid_size * block_size * 2;

    double *d_padded, *d_output;

    cudaMalloc(&d_padded, padded_N * sizeof(double));
    cudaMalloc(&d_output, grid_size * sizeof(double));

    cudaMemcpy(d_padded, d_input, N * sizeof(double), cudaMemcpyDeviceToDevice);

    if(padded_N > N) {
        cudaMemset(d_padded+N, 0, (padded_N - N)*sizeof(double));
    }

    // First reduction
    addReduce<<<grid_size, block_size, block_size*sizeof(double)>>>(d_padded, d_output);
    cudaDeviceSynchronize();
    // If multiple blocks, reduce again
    while (grid_size > 1) {
        int prev_grid = grid_size;
        grid_size = (grid_size + (block_size*2) - 1) / (block_size*2);

        int padded_prev = grid_size * block_size * 2;

        double* d_temp;
        cudaMalloc(&d_temp, grid_size * sizeof(double));

        if(padded_prev > prev_grid) {
                cudaMemset(d_output+prev_grid, 0, (padded_prev - prev_grid)*sizeof(double));
        }

        addReduce<<<grid_size, block_size, block_size*sizeof(double)>>>(d_output, d_temp);
        cudaDeviceSynchronize();
        cudaFree(d_output);
        d_output = d_temp;
    }

    cudaMemcpy(d_result, d_output, sizeof(double), cudaMemcpyDeviceToDevice); 

    cudaFree(d_padded);
    cudaFree(d_output);
}



__global__ void axpy(int N, double alpha, const double* x, double* y) {

        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i < N) {
                y[i] = y[i] + alpha * x[i];
        }
}

void axpy_caller(int M, double alpha, const double* h_x, double* h_y) {

        int block_size_axpy = 256;
        int numBlocks = (M + block_size_axpy - 1) / block_size_axpy;

        double* d_x, *d_y;
        cudaMalloc(&d_x, M*sizeof(double));
        cudaMalloc(&d_y, M*sizeof(double));

        cudaMemcpy(d_x, h_x, M*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, M*sizeof(double), cudaMemcpyHostToDevice);

        axpy<<<numBlocks, block_size_axpy>>>(M, alpha, d_x, d_y);
        cudaDeviceSynchronize();

        cudaMemcpy(h_y, d_y, M*sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_x);
        cudaFree(d_y);

}

__global__ void scale_gpu(int N, double alpha, const double* x, double* y) {

        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i < N) {
                y[i] = alpha * x[i];

        }
}

void scale_caller(int M, double alpha, const double* h_x, double* h_y) {

        double* d_x, *d_y;
        cudaMalloc(&d_x, M*sizeof(double));
        cudaMalloc(&d_y, M*sizeof(double));

        cudaMemcpy(d_x, h_x, M*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, M*sizeof(double), cudaMemcpyHostToDevice);

        int block_size_scale = 256;
        int numBlocks_scale = (M * block_size_scale - 1) / block_size_scale;

        scale_gpu<<<numBlocks_scale, block_size_scale>>>(M, alpha, d_x, d_y);
        cudaDeviceSynchronize();


        cudaMemcpy(h_y, d_y, M*sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_x);
        cudaFree(d_y);

}


double vector_norm(int N, double* h_v) {


        double* d_temp, *d_result, *d_v;


        int block_size = 256;
        int grid_size = (N+ (block_size*2) - 1) / (block_size*2);
        int padded_N = grid_size * block_size * 2;

        cudaMalloc(&d_temp, padded_N*sizeof(double));
        cudaMalloc(&d_v, N*sizeof(double));
        cudaMalloc(&d_result, sizeof(double));

        cudaMemcpy(d_v, h_v, N*sizeof(double), cudaMemcpyHostToDevice);
        int numBlocks = (N + block_size - 1)/ block_size;

        vecvec_prod<<<numBlocks, block_size>>>(N, d_v, d_v, d_temp);
        cudaDeviceSynchronize();

        full_reduce_caller(N, d_temp, d_result);

        double h_result;
        cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_temp);
        cudaFree(d_v);
        cudaFree(d_result);

        return sqrt(h_result);

}



double dot_prod(int N, double* h_v1, double* h_v2) {

        double* d_temp, *d_result, *d_v1, *d_v2;
        int block_size = 256;
        int grid_size = (N+ (block_size*2) - 1) / (block_size*2);
        int padded_N = grid_size * block_size * 2;

        cudaMalloc(&d_v1, N*sizeof(double));
        cudaMalloc(&d_v2, N*sizeof(double));
        cudaMalloc(&d_temp, padded_N*sizeof(double));
        cudaMalloc(&d_result, sizeof(double));

        cudaMemcpy(d_v1, h_v1, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2, h_v2, N*sizeof(double), cudaMemcpyHostToDevice);


        int numBlocks = (N + block_size - 1)/ block_size;

        vecvec_prod<<<numBlocks, block_size>>>(N, d_v1, d_v2, d_temp);
        cudaDeviceSynchronize();

        full_reduce_caller(N, d_temp, d_result);

        double h_result;
        cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_temp);
        cudaFree(d_v1);
        cudaFree(d_v2);
        cudaFree(d_result);

        return h_result;

}

double dot_prod_gpu(int N, double* d_v1, double* d_v2, double* d_result) {
        
        double* d_temp;
        cudaMalloc(&d_temp, N*sizeof(double));
        int block_size = 256;
        int numBlocks = (N + block_size - 1) / block_size;

    // Compute element-wise product
        vecvec_prod<<<numBlocks, block_size>>>(N, d_v1, d_v2, d_temp);
        cudaDeviceSynchronize();

    // Reduce to get final dot product
        full_reduce_caller(N, d_temp, d_result);

    // Copy single scalar result back to host
        double h_result;
        cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_temp);

        return h_result;
}
