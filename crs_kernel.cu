__global__ void nz_per_row(int N, double* A, int* rpt, double* val, int* col) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if(tid >= N) return;

	int start = rpt[tid];
	int end = rpt[tid + 1];
	int out = start;

	for(int i = 0; i < N; i++) {
		if(fabs(A[tid*N + i]) > 0.0f) {
			if(out >= end) return;
			val[out] = A[tid*N + i];
			col[out] = i;
			out++;
		}
	}
}
