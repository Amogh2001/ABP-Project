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

	output[blockOffset + threadID] += n[blockID];
	//output[blockOffset + (2*threadID)+1] += n[blockID];
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
	cudaMalloc(&d_sums_results, sum_blocks*sizeof(int));

	cudaMemcpy(d_input, h_input, N*sizeof(int), cudaMemcpyHostToDevice);


	prescan_large_unoptimized<<<numBlocks, block_size, sharedBytes>>>(d_output, d_input, n, d_sums);

	cudaDeviceSynchronize();

	cudaMemcpy(h_sums, d_sums, sum_blocks*sizeof(int), cudaMemcpyDeviceToHost);


	host_scan(sum_blocks, h_sums, h_sums_results);  	

	cudaMemcpy(d_sums_results, h_sums_results, sum_blocks*sizeof(int), cudaMemcpyHostToDevice);

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

//=========================== SELL-C-SIGMA SECTION =================

__global__ void sellc_assembler(
	int N,
	const int* __restrict__ row_nnz,
	const int* __restrict__ row_start,
	const double* __restrict__ input_vals,
	const int* __restrict__ max_nz_per_warp,
	const int* __restrict__ offset,
	double* __restrict__ val_out) {


	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int warp = gid / 32;
	int lane = threadIdx.x % 32;

	int row = warp*32+lane;

	if(row>=N) return;

	int max_nz = max_nz_per_warp[warp];
	int write_base = offset[warp];

	for(int k = 0; k < max_nz; k++) {

		int out_index = write_base + k * 32 + lane;
	
		if(k < row_nnz[row]) {
			val_out[out_index] = input_vals[row_start[row] + k];
		} else {
			val_out[out_index] = 0.0;
		}
	}
}
