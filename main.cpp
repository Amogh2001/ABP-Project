#include <iostream>
#include <stdlib.h>

#include "kernel.cuh"

int main(int argc, char* argv[]) {


	int N = 3200;

	//================== SUM SECTION =======================

	double* h_A; 
	int *h_non_zero_bool, *h_row_count;

	h_A = (double*)malloc(N*N*sizeof(double));
	h_non_zero_bool = (int*)calloc(N*N, sizeof(int));
	h_row_count = (int*)malloc(N*sizeof(int));

	for(unsigned int i = 0; i < N*N; i+=1) {	
		h_A[i] = i * 7.0f;
	}
	
	sum_caller(N, h_A, h_non_zero_bool, h_row_count);


	//================== SCAN SECTION =======================

	//int N = 102400;	
	//int block_size = 512;
	int n = 256;
	//int numBlocks = N / n;
	int  numBlocks = (N + n - 1) / n;

	int sum_blocks = numBlocks;
	int sharedBytes = n*sizeof(int);
	int *h_output, *h_sums, *h_sums_results; 

	//h_input = (int*)malloc(N*sizeof(int));
	h_output = (int*)malloc((N+1)*sizeof(int));
	h_sums = (int*)malloc(sum_blocks*sizeof(int));
	h_sums_results = (int*)calloc(sum_blocks, sizeof(int));

	scan_caller(N, h_row_count, h_output, h_sums, h_sums_results);


	for(int i = 3180; i < 3200; i++) {
		std::cout << "Index = " << i << ", Value = " << h_output[i] << "\n";
	}	

	//========================== WARP MAX SECTION ============

	int numWarps = (N + 32 - 1) / 32;

	int* h_warp_max_output;
	
	h_warp_max_output = (int*)calloc(numWarps, sizeof(int));
	
	warp_max_caller(N, h_row_count, h_warp_max_output);

	
	for(int i = 0; i < 32; ++i) {
		printf("Warp max[%d] = %d\n", i, h_warp_max_output[i]); 
	}


	//================== WARP SCAN SECTION =======================

	int numBlocksWarpScan = (numWarps + n - 1) / n;

	int *h_warpoutput, *h_warpsums, *h_warpsums_results; 


	h_warpoutput = (int*)malloc((numWarps+1)*sizeof(int));
	h_warpsums = (int*)malloc(numBlocksWarpScan*sizeof(int));
	h_warpsums_results = (int*)calloc(numBlocksWarpScan, sizeof(int));

	scan_caller(numWarps, h_warp_max_output, h_warpoutput, h_warpsums, h_warpsums_results);


	for(int i = 0; i < 10; i++) {
		std::cout << "Warp Scan = " << i << ", Value = " << h_warpoutput[i] << "\n";
	}	


	for(int i = 0; i < 32; ++i) {
		printf("C[%d] = %d\n", i, h_row_count[i]); 
	}

	//============= SELL-C-SIGMA SECTION ========================

		


	
	free(h_output); 
	free(h_sums); 
	free(h_sums_results); 

	free(h_warpoutput);	
	free(h_warpsums_results);
	free(h_warpsums);
	free(h_warp_max_output);
	
	free(h_row_count);
	free(h_A);
	free(h_non_zero_bool);
	return 0;
}
