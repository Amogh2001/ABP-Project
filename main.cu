#include <iostream>
#include <stdlib.h>
#include <random>

#include <cusparse.h>

#include "kernel.cuh"
#include <chrono>
#include <sys/time.h>

void crs_matvec_cpu(const int* row_ptr, const int* col, const double* val,
                    const double* vec, double* result, int num_rows) {
    for(int i = 0; i < num_rows; ++i) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        
        double sum = 0.0;
        for(int j = row_start; j < row_end; ++j) {
            sum += val[j] * vec[col[j]];
        }
        result[i] = sum;
    }
}

int main(int argc, char* argv[]) {


         int N = 6;
         std::cout << "N = " << N << "\n";

         auto start1 = std::chrono::high_resolution_clock::now();

         double* h_I = (double*)malloc(N*N*sizeof(double));

         iden_caller(N, h_I);

         double* h_d1 = (double*)malloc(N*N*sizeof(double));

         d1_caller(N, h_d1);
         // term1 = I ⊗ Δ₁  (N×N ⊗ N×N = N²×N²)
         double* term1 = (double*)malloc(N*N*N*N*sizeof(double));
         kron_caller(N, N, N, N, h_I, h_d1, term1);

         // term2 = I ⊗ term1  (N×N ⊗ N²×N² = N³×N³)
         double* term2 = (double*)malloc(N*N*N*N*N*N*sizeof(double));
         kron_caller(N, N, N*N, N*N, h_I, term1, term2);

         // term3 = Δ₁ ⊗ I  (N×N ⊗ N×N = N²×N²)
         double* term3 = (double*)malloc(N*N*N*N*sizeof(double));
         kron_caller(N, N, N, N, h_d1, h_I, term3);

         // term4 = I ⊗ term3  (N×N ⊗ N²×N² = N³×N³)
         double* term4 = (double*)malloc(N*N*N*N*N*N*sizeof(double));
         kron_caller(N, N, N*N, N*N, h_I, term3, term4);

         // term5 = I ⊗ I  (N×N ⊗ N×N = N²×N²)
         double* term5 = (double*)malloc(N*N*N*N*sizeof(double));
         kron_caller(N, N, N, N, h_I, h_I, term5);

         // term6 = Δ₁ ⊗ term5  (N×N ⊗ N²×N² = N³×N³)
         double* term6 = (double*)malloc(N*N*N*N*N*N*sizeof(double));
         kron_caller(N, N, N*N, N*N, h_d1, term5, term6);
         cudaDeviceSynchronize();
         //term6 = d1 kron term5

         auto end1 = std::chrono::high_resolution_clock::now();
         auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
         std::cout << "Time to assemble delta: " << duration1.count() << " microseconds" << std::endl;

         int d3_rows = N*N*N;
         int d3_cols = N*N*N;


         int M = d3_rows*d3_cols;

         auto start2 = std::chrono::high_resolution_clock::now();

         double* h_d3_final = (double*)malloc(M*sizeof(double));

         double* d_d3; 
         cudaMalloc(&d_d3, M*sizeof(double));
         double* d_term6;  
         cudaMalloc(&d_term6, M*sizeof(double));
         double* d_term4;  
         cudaMalloc(&d_term4, M*sizeof(double));
         double* d_term2;  
         cudaMalloc(&d_term2, M*sizeof(double));
         double* d_d3_final;  
         cudaMalloc(&d_d3_final, M*sizeof(double));
         
         cudaMemcpy(d_term6, term6, M*sizeof(double), cudaMemcpyHostToDevice);

         cudaMemcpy(d_term4, term4, M*sizeof(double), cudaMemcpyHostToDevice);
         cudaMemcpy(d_term2, term2, M*sizeof(double), cudaMemcpyHostToDevice);
         auto end2 = std::chrono::high_resolution_clock::now();
         auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
         std::cout << "Time to move delta to device: " << duration2.count() << " microseconds" << std::endl; 
         
         int threadsPerBlock = 256;
         int numBlocks_d3 = (M + threadsPerBlock - 1) / threadsPerBlock;

         auto start3 = std::chrono::high_resolution_clock::now();

         VectorAdd<<<numBlocks_d3, threadsPerBlock>>>(M, d_term6, d_term4, d_d3);
         cudaDeviceSynchronize();
         VectorAdd<<<numBlocks_d3, threadsPerBlock>>>(M, d_term2, d_d3, d_d3_final);
         cudaDeviceSynchronize();

         cudaMemcpy(h_d3_final, d_d3_final, M*sizeof(double), cudaMemcpyDeviceToHost);
         auto end3 = std::chrono::high_resolution_clock::now();
         auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);
         std::cout << "Time to assemble delta_3: " << duration3.count() << " microseconds" << std::endl;
         double scale = 1.0 / ((N+1)*(N+1));
         for(int i = 0; i < N*N*N*N*N*N; ++i) {
                h_d3_final[i] = scale * h_d3_final[i];
         }
         //for(int i = 0; i < 320; i++) {
         //        std::cout << "Final value at index " << i << "= " << h_d3_final[i] << "\n    ";
         //
         //}


         free(h_I);
         free(h_d1);
         free(term1);
         free(term2);
         free(term3);
         free(term4);
         free(term5);
         free(term6);

         cudaFree(d_term6);
         cudaFree(d_term4);
         cudaFree(d_term2);
         cudaFree(d_d3);
         cudaFree(d_d3_final);



         auto start4 = std::chrono::high_resolution_clock::now();
        //================== SUM SECTION =======================

        //double* h_A; 
        int *h_non_zero_bool, *h_row_count;

        //h_A = (double*)malloc(N*N*sizeof(double));
        h_non_zero_bool = (int*)calloc(d3_rows*d3_rows, sizeof(int));
        h_row_count = (int*)malloc(d3_rows*sizeof(int));


        sum_caller(d3_rows, h_d3_final, h_non_zero_bool, h_row_count);


        //================== SCAN SECTION =======================

        //int N = 102400;
        //int block_size = 512;
        int n = 256;
        //int numBlocks = N / n;
        int  numBlocks = (d3_rows + n - 1) / n;

        int sum_blocks = numBlocks;
        //int sharedBytes = n*sizeof(int);
        int *h_output, *h_sums, *h_sums_results; 

        //h_input = (int*)malloc(N*sizeof(int));
        h_output = (int*)malloc((d3_rows+1)*sizeof(int));
        h_sums = (int*)malloc(sum_blocks*sizeof(int));
        h_sums_results = (int*)calloc(sum_blocks, sizeof(int));

        //scan_caller(d3_rows, h_row_count, h_output, h_sums, h_sums_results);
        scan_caller(d3_rows, h_row_count, h_output, NULL, NULL);

        //for(int i = M-10; i < M+2; i++) {
        //      std::cout << "Index = " << i << ", Value = " << h_output[i] << "\n";
        //}

        free(h_sums);
        free(h_sums_results);

        //========================== WARP MAX SECTION ============

        int numWarps = (d3_rows + 32 - 1) / 32;

        int* h_warp_max_output;

        h_warp_max_output = (int*)calloc(numWarps, sizeof(int));

        warp_max_caller(d3_rows, h_row_count, h_warp_max_output);


        //for(int i = 0; i < numWarps; ++i) {
        //      printf("Warp max[%d] = %d\n", i, h_warp_max_output[i]); 
        //}


        //================== WARP SCAN SECTION =======================
/*
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
*/

        int numBlocksWarpScan = (numWarps + n - 1) / n;

        int *h_warpoutput, *h_warpsums, *h_warpsums_results; 
        int *h_warp_storage_size;  // NEW: storage size per warp

        h_warp_storage_size = (int*)malloc(numWarps*sizeof(int));
        h_warpoutput = (int*)malloc((numWarps+1)*sizeof(int));
        h_warpsums = (int*)malloc(numBlocksWarpScan*sizeof(int));
        h_warpsums_results = (int*)calloc(numBlocksWarpScan, sizeof(int));

        // Convert max_nz to storage size (max_nz * 32)
        for(int i = 0; i < numWarps; i++) {
                h_warp_storage_size[i] = h_warp_max_output[i] * 32;
        }

        // Scan the storage sizes, not the max values
        scan_caller(numWarps, h_warp_storage_size, h_warpoutput, h_warpsums, h_warpsums_results);

        free(h_warp_storage_size);  // Clean up temporary array



        //=================== CRS SECTION ========================

        int numnz = h_output[d3_rows];

        std::cout << "numnz = " << numnz << "\n";

        int* h_col;
        double* h_val;

        h_val = (double*)malloc(numnz*sizeof(double));
        h_col = (int*)malloc(numnz*sizeof(int));

        crs_caller(d3_rows, h_d3_final, h_output, numnz, h_val, h_col); 

        //============= SELL-C-SIGMA SECTION ========================

        int numWarps_sellc = (d3_rows + 31) / 32;
        int numnz_sellc = h_warpoutput[numWarps];  // Total non-zeros in SELL-C format
        //printf("SELL-C requires %d elements (CSR has %d)\n", numnz_sellc, numnz);

        double* h_scs_val_output = (double*)malloc(numnz_sellc*sizeof(double));
        int* h_scs_col_output = (int*)malloc(numnz_sellc*sizeof(int));

        sellc_caller(d3_rows, h_row_count, h_output, h_val, h_warp_max_output, h_warpoutput, h_scs_val_output, h_col,h_scs_col_output); 


        //for(int i = 0; i < 32; ++i) {
        //      printf("CRS[%d] = %lf\n", i, h_val[i]); 
        //}

        //for(int i = 0; i < 32; ++i) {
        //      printf("SCS[%d] = %lf\n", i, h_scs_val_output[i]); 
        //}

        //for(int i = 0; i < 32; ++i) {
        //      printf("Col SCS[%d] = %d\n", i, h_scs_col_output[i]); 
        //}
        auto end4 = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end4 - start4);
        //std::cout << "Time to get SELL-C_sigma: " << duration4.count() << " microseconds" << std::endl;
        std::cout << "Time to get CRS: " << duration4.count() << " microseconds" << std::endl;
//============================================================
//=========== LANCZOS ALGORITHM ===============================
//============================================================
        auto start5 = std::chrono::high_resolution_clock::now();
        double* v0 = (double*)malloc(d3_rows*sizeof(double));
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);


        for(int i = 0; i < d3_rows; i++) {
                v0[i] = dist(gen);
        }

        double norm = 0.0;
        for(int i = 0; i < d3_rows; i++) {
                norm += v0[i] * v0[i];
        }

        norm = std::sqrt(norm);

        for(int i = 0; i < d3_rows; i++) {
                v0[i] /= norm;
        }

//======================== SELL C MULTIPLY=====================

        double* v_prev = (double*)calloc(d3_rows, sizeof(double));
        double* v_next = (double*)malloc(d3_rows*sizeof(double));

        double* w = (double*)malloc(d3_rows*sizeof(double));

/*      double* alpha = (double*)malloc((20*N)*sizeof(double));
        double* beta = (double*)malloc(((20*N)+1)*sizeof(double));
        beta[0] = 0.0;
*/
        double* orig_v0 = v0;
        double* orig_vprev = v_prev;
        double* orig_vnext = v_next;
        double* orig_w = w;

//      double* v_curr = v0;

//      int block_size_sellc_mul = 256;
//      int grid_size_sellc_mul = (N + block_size_sellc_mul -1) / block_size_sellc_mul;


/*
        int block_size_axpy = 256;
        int numBlocks_axpy = (M + block_size_axpy - 1) / block_size_axpy;


        //h_warp_max_output = max_nz_per_warp
        //h_warpoutput = offset 
*/


// ============ DEVICE ALLOCATION ============
//      int numWarps_sellc = (d3_rows + 31) / 32;
//      int numnz_sellc = h_warpoutput[numWarps];  // Total non-zeros in SELL-C format

        int block_size_axpy = 256;
        int numBlocks_axpy = (d3_rows + block_size_axpy - 1) / block_size_axpy;

        int block_size_scale = 256;
        int numBlocks_scale = (d3_rows + block_size_scale - 1) / block_size_scale;

        //int block_size_dot = 256;
        //int grid_size_dot = (M+ (block_size_dot*2) - 1) / (block_size_dot*2);
        //int numBlocks_dot = (N + block_size_dot - 1)/ block_size_dot;

        //int padded_N_dot = grid_size_dot * block_size_dot * 2;


// Before: for(int i = 0; i < num_iter; i++) {
double* h_v_curr = (double*)malloc(d3_rows * sizeof(double));
double* h_w = (double*)malloc(d3_rows * sizeof(double));


        //double *d_padded, *d_output;
        double *d_val;
        int *d_col, *d_offset, *d_max_nz_per_warp;

        double *d_v_curr, *d_v_prev, *d_v_next, *d_w, *d_result;

        cudaMalloc(&d_val, numnz_sellc*sizeof(double));
        cudaMalloc(&d_col, numnz_sellc*sizeof(int));
        cudaMalloc(&d_offset, (numWarps_sellc+1)*sizeof(int));
        cudaMalloc(&d_max_nz_per_warp, numWarps_sellc*sizeof(int));

        //cudaMalloc(&d_red_temp, padded_N_dot*sizeof(double));
        cudaMalloc(&d_result, sizeof(double));

        cudaMalloc(&d_v_curr, d3_rows*sizeof(double));
        cudaMalloc(&d_v_prev, d3_rows*sizeof(double));
        cudaMalloc(&d_v_next, d3_rows*sizeof(double));
        cudaMalloc(&d_w, d3_rows*sizeof(double));

        cudaMemcpy(d_val, h_scs_val_output, numnz_sellc*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, h_scs_col_output, numnz_sellc*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offset, h_warpoutput, (numWarps_sellc+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_nz_per_warp, h_warp_max_output, numWarps_sellc*sizeof(int), cudaMemcpyHostToDevice);

// ============ INITIALIZE VECTORS ON DEVICE ============
        cudaMemcpy(d_v_curr, v0, d3_rows*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemset(d_v_prev, 0, d3_rows*sizeof(double));

        int block_size_la = 256;
        int grid_size_la = (d3_rows + block_size_la - 1) / block_size_la;

        int num_iter = 20*N;  // Limit iterations
        double* alpha = (double*)malloc(num_iter*sizeof(double));
        double* beta = (double*)malloc((num_iter+1)*sizeof(double));
        beta[0] = 0.0;

        double total_time_mv = 0.0;
        double total_time_iteration = 0.0;
        int successful_iterations = 0;

        for(int i = 0; i < num_iter; i++) {
                auto start_iteration = std::chrono::high_resolution_clock::now();
                auto start_mv = std::chrono::high_resolution_clock::now();

                cudaMemcpy(h_v_curr, d_v_curr, d3_rows*sizeof(double), cudaMemcpyDeviceToHost);
                crs_matvec_cpu(h_output, h_col, h_val, h_v_curr, h_w, d3_rows);
                cudaMemcpy(d_w, h_w, d3_rows*sizeof(double), cudaMemcpyHostToDevice);


                //sellc_multiply<<<grid_size_la, block_size_la>>>(d3_rows, d_val, d_col, d_offset, d_max_nz_per_warp, d_v_curr, d_w);
                //spmv_crs<<<grid_size_la, block_size_la>>>(d3_rows, d_val, d_col, d_offset, d_v_curr, d_w);
                cudaDeviceSynchronize();
                auto end_mv = std::chrono::high_resolution_clock::now();
                auto duration_mv = std::chrono::duration_cast<std::chrono::microseconds>(end_mv - start_mv);
                total_time_mv += duration_mv.count();
                //std::cout << "Time for multiplication: " << duration_mv.count() << " microseconds" << std::endl;

                if(beta[i] != 0.0) {
                        axpy<<<numBlocks_axpy, block_size_axpy>>>(d3_rows, -beta[i], d_v_prev, d_w);
                        cudaDeviceSynchronize();
                }
                alpha[i] = dot_prod_gpu(d3_rows, d_w, d_v_curr, d_result);  // Need to implement this



                axpy<<<numBlocks_axpy, block_size_axpy>>>(d3_rows, -alpha[i], d_v_curr, d_w);
                cudaDeviceSynchronize();
   
                double norm = dot_prod_gpu(d3_rows, d_w, d_w, d_result);

                beta[i+1] = sqrt(norm); 
    
                if(beta[i+1] < 1e-8) {
                        printf("Lanczos breakdown at iteration %d\n", i);
                        break;
                }
    
                scale_gpu<<<numBlocks_scale, block_size_scale>>>(d3_rows, (1.0/beta[i+1]), d_w, d_v_next);
                cudaDeviceSynchronize();
   
                double* temp = d_v_prev;
                d_v_prev = d_v_curr;
                d_v_curr = d_v_next;
                d_v_next = temp;
    
                auto end_iteration = std::chrono::high_resolution_clock::now();
                auto duration_iteration = std::chrono::duration_cast<std::chrono::microseconds>(end_iteration - start_iteration);
                total_time_iteration += duration_iteration.count();
                successful_iterations++;

    // Optional: print progress
     //         if(i % 10 == 0) {
//                      printf("Iteration %d: alpha=%.6f, beta=%.6f\n", i, alpha[i], beta[i+1]);
  //            }
        }

auto end5 = std::chrono::high_resolution_clock::now();
auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end5 - start5);
std::cout << "Time for Lanczos (total): " << duration5.count() << " microseconds" << std::endl;

double avg_time_mv = total_time_mv / successful_iterations;
double avg_time_iteration = total_time_iteration / successful_iterations;

std::cout << "\n=== Timing Statistics ===\n";
std::cout << "Total Lanczos time: " << duration5.count() << " microseconds" << std::endl;
std::cout << "Number of iterations: " << successful_iterations << std::endl;
std::cout << "Average time per iteration: " << avg_time_iteration << " microseconds" << std::endl;
std::cout << "Average SpMV time per iteration: " << avg_time_mv << " microseconds" << std::endl;
std::cout << "SpMV percentage of iteration time: " << (avg_time_mv / avg_time_iteration * 100) << "%" << std::endl;


        // After the loop, print the tridiagonal matrix structure
printf("\n=== Tridiagonal Matrix (first 10x10) ===\n");
for(int i = 0; i < 10; i++) {
    for(int j = 0; j < 10; j++) {
        if(i == j) printf("%8.4f ", alpha[i]);
        else if(i == j-1) printf("%8.4f ", beta[i+1]);
        else if(i == j+1) printf("%8.4f ", beta[i]);
        else printf("%8.4f ", 0.0);
    }
    printf("\n");
}
//===========================================================



        cudaFree(d_val);
        cudaFree(d_col);
        cudaFree(d_offset);
        cudaFree(d_max_nz_per_warp);
        cudaFree(d_v_curr);
        cudaFree(d_v_prev);
        cudaFree(d_v_next);
        cudaFree(d_w);
        //cudaFree(d_red_temp);
        cudaFree(d_result);

        free(h_output); 
        //free(h_sums); 
        //free(h_sums_results); 

        free(h_warpoutput);
        free(h_warpsums_results);
        free(h_warpsums);
        free(h_warp_max_output);

        free(h_val);
        free(h_col);
        free(orig_v0);
        free(orig_w);
        free(orig_vprev);
        free(orig_vnext);
        //free(v_curr);
        free(alpha);
        free(beta);

        free(h_scs_val_output);
        free(h_scs_col_output);

        free(h_row_count);
        free(h_d3_final);
        free(h_non_zero_bool);
        return 0;
}
