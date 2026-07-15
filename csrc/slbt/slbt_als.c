#define FS(t,i,j) Fs[(t)*I*J + (i)*J + (j)]

#define ALPHA(t,i,k) alpha_out[(t)*I*2 + (i)*2 + (k)]
#define BETA(t,j,k)  beta_out[(t)*J*2 + (j)*2 + (k)]

#include "slbt_als.h"
#include "linalg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h> 

//*------Internally used functions------
//Helper functions
void mat_print_double(int I, int J, double mat[I][J]){ //Function used to print a double matrix
    for(int i=0; i<I; i++){
        for(int j=0; j<J; j++){
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }   
}
void mat_tras_ptr(int I, int J, double** mat, double mat_res[J][I]){ //Function that given a matrix returns its transpose
    for(int i=0; i<I; i++){
        for(int j=0; j<J; j++){
            mat_res[j][i] = mat[i][j];
        }
    }
}
void prod_mat_ptr_vect(int I, int J, double **mat, double arr[J], double arr_res[I]){
    double sum;
    for(int i=0; i<I; i++){
        sum = 0;
        for(int j=0; j<J; j++){
            sum += mat[i][j]*arr[j];
        }
        arr_res[i] = sum;
    }
}
void prod_matrix_ptr(int I, int J, int Z, double** mat1, double mat2[J][Z], double mat_res[I][Z]){ //Given two matrices this function returns their product
    double sum;
    for(int i=0; i<I; i++){
        for(int z=0; z<Z; z++){
            sum = 0.0;
            for(int j=0; j<J; j++){
                sum += ((double)mat1[i][j] * (double)mat2[j][z]);
            }
            mat_res[i][z] = sum;
        }
    }
}
void mat_rand(int I, int J, double mat[I][J]){ //Function to randomly generate a double matrix
    for (int i=0; i<I; i++){
        for (int j=0; j<J; j++){
            mat[i][j] = ((double)rand()/RAND_MAX);
        }
    }
}
void mat_tras(int I, int J, double mat[I][J], double mat_res[J][I]){ //Function that given a matrix returns its transpose
    for(int i=0; i<I; i++){
        for(int j=0; j<J; j++){
            mat_res[j][i] = mat[i][j];
        }
    }
}
void prod_matrix(int I, int J, int Z, double mat1[I][J], double mat2[J][Z], double mat_res[I][Z]){ //Given two matrices this function returns their product
    double sum;
    for(int i=0; i<I; i++){
        for(int z=0; z<Z; z++){
            sum = 0.0;
            for(int j=0; j<J; j++){
                sum += ((double)mat1[i][j] * (double)mat2[j][z]);
            }
            mat_res[i][z] = sum;
        }
    }
}
void prod_mat_vect(int I, int J, double mat[I][J], double arr[J], double arr_res[I]){
    double sum;
    for(int i=0; i<I; i++){
        sum = 0;
        for(int j=0; j<J; j++){
            sum += mat[i][j]*arr[j];
        }
        arr_res[i] = sum;
    }
}
void mat_zero(int I, int J, double mat[I][J]){ //Function to zero a double matrix
    for (int i=0; i<I; i++){
        for (int j=0; j<J; j++){
            mat[i][j] = 0.0;
        }
    }
}
double calc_tr(int I, double mat[I][I]){ //Function that given a matrix returns its trace
    double sum = 0;
    for(int i=0; i<I; i++){
        sum += mat[i][i];
    }
    return sum;
}

//Optimization-related functions
void conv_UA(int I, int J, double mat1[I][J], double mat2[I][J]){ //Function used, given U, to get A 
    double sum;
    for(int i=0; i<I; i++){
        sum = 0.0;
        for(int j=0; j<J; j++){
            sum += exp(mat1[i][j]);
        }
        for(int j=0; j<J; j++){
            mat2[i][j] = exp(mat1[i][j])/sum;
        }
    }
}
void conv_ZB(int I, int J, double mat1[I][J], double mat2[I][J]){ //Function used, given Z, to get B
    double sum;
    for(int j=0; j<J; j++){
        sum = 0.0;
        for(int i=0; i<I; i++){
            sum += exp(mat1[i][j]);
        }
        for(int i=0; i<I; i++){
            mat2[i][j] = exp(mat1[i][j])/sum;
        }
    }
}
void conv_df_dU(int Jb, int K, double mat1[Jb][K], double mat2[Jb][K], double mat_res[Jb][K]){ //Given df_dA this function returns df_dU
    double sum = 0;
    double check;
    for (int i=0; i<Jb; i++){
        for (int j=0; j<K; j++){
            sum = 0;
            for (int k=0; k<K; k++){
                check = 0.0;
                if(j == k){
                    check = 1.0;
                }
                sum += mat1[i][k]*(check*mat2[i][k] - mat2[i][k]*mat2[i][j]);

            }
            mat_res[i][j] = sum;
        }
    }
}
void conv_df_dZ(int Jy, int K, double mat1[Jy][K], double mat2[Jy][K], double mat_res[Jy][K]){ //Given df_dB this function returns df_dZ
    double sum = 0;
    double check;
    for (int i=0; i<Jy; i++){
        for (int j=0; j<K; j++){
            sum = 0;
            for (int k=0; k<K; k++){
                check = 0.0;
                if(i == k){
                    check = 1.0;
                }
                sum += mat1[k][j]*(check*mat2[k][j] - mat2[k][j]*mat2[i][j]);
            }
            mat_res[i][j] = sum;
        }
    }
}
void check_UZ(int J, int K, double mat[J][K]){ //Given U(or Z) this function checks if the rows sum to 1
    double sum;
    for(int i=0; i<J; i++){
        for(int j=0; j<K; j++){
            if(mat[i][j] > 20){
                mat[i][j] = 20;
            }
            if(mat[i][j] < -20){
                mat[i][j] = -20;
            }
        }
    }
}
void update_UZ(int J, int K, double rate, double mat1[J][K], double mat2[J][K]){ //Given U(or Z) and df_dU(or df_dZ) this function updates U(or Z) 
    for(int i=0; i<J; i++){
        for(int j=0; j<K; j++){
            mat1[i][j] = mat1[i][j] - rate*mat2[i][j];
        }
    }
}
void DX_calc(int I, int J, double mat1[I][J], double mat2[J][J]){ //Given the transpose of F this function returns DX
    double sum = 0.0;
    for(int j=0; j<J; j++){
        for(int i=0; i<J; i++){
            mat2[i][j] = 0.0;
        }
    }
    for(int j=0; j<J; j++){
        sum = 0.0;
        for(int i=0; i<I; i++){
            sum += mat1[i][j];
        }
        mat2[j][j] = sum;
    }
}

//*------Out-called from slbt_core.c------
//Unconstrained Simultaneous Latent Budget Analysis
void slba_C_execute(int I, int J, int T, int TA, int TB, const double *Fs, double *alpha_out, double *beta_out){ //Unconstrained Simultaneous Latent Budget Analysis 
    int K = 2,  iter_count=0, MAX_ITER = 1000, stop=0, ta, tb;
    double r = 0.5, d_stop = 0.0004, f_stop=0.05, sum;
    double rA = r / (TA == 1 ? T : 1);
    double rB = r / (TB == 1 ? T : 1);
    double f=0, N=0, f_p, d;

    double A[TA][I][K], B[TB][J][K], At[TA][K][I], Bt[TB][K][J], df_dA[TA][I][K], df_dB[TB][J][K]; 
    double U[TA][I][K], Z[TB][J][K], df_dU[TA][I][K], df_dZ[TB][J][K];

    double F[T][I][J], Ft[T][J][I], DX[T][I][I];

    double FB[I][K], BtB[K][K], ABtB[I][K], DXABtB[I][K]; //Used to calculate df_dA
    double FtA[J][K], DXA[I][K], AtDXA[K][K], BAtDXA[J][K]; //Used to calculate df_dB
    double ABt[I][J], FtABt[J][J], DXABt[I][J], AtDXABt[K][J], BAtDXABt[J][J];

    // Initialization of U and Z, consequently of A and B
    for(int t=0; t<TA; t++){
        mat_rand(I, K, U[t]);
        check_UZ(I, K, U[t]);
    }
    for(int t=0; t<TB; t++){
        mat_rand(J, K, Z[t]);
        check_UZ(J, K, Z[t]);
    }

    // initialization of F and the transpose of F
    for(int t=0; t<T; t++){
        for(int i=0; i<I; i++){
            for(int j=0; j<J; j++){
                Ft[t][j][i] = FS(t,i,j);
                F[t][i][j] = FS(t,i,j);
                N += FS(t,i,j);
            }
        }
    }

    // Initialization of DX
    for(int t=0; t<T; t++){
        for(int i=0; i<I; i++){
            sum = 0.0;
            for(int j=0; j<J; j++){
                sum += F[t][i][j] ;
                DX[t][i][j] = 0.0;
            }
            DX[t][i][i] = sum;
        }
    }

    // Iterative execution begins
    f = N + 1.0; //Just to enter the while loop
    while(stop == 0 && iter_count < MAX_ITER){
        f_p = f;
        f = N;

        for(int t=0; t<TA; t++){
            conv_UA(I, K, U[t], A[t]);
        }
        for(int t=0; t<TB; t++){
            conv_ZB(J, K, Z[t], B[t]);
        }
        
        // Initialization of df_dA and df_dB
        for(int t=0; t<TA; t++){
            for(int k=0; k<K; k++){
                for(int i=0; i<I; i++){
                    df_dA[t][i][k] = 0.0;
                }
            }
        }
        // Initialization of df_dA and df_dB
        for(int t=0; t<TB; t++){
            for(int k=0; k<K; k++){
                for(int j=0; j<J; j++){
                    df_dB[t][j][k] = 0.0;
                }
            }
        }

        for (int t=0; t<T; t++){
            // Homogeneity in the handling of A and B matrices
            ta = (TA == 1) ? 0 : t;
            tb = (TB == 1) ? 0 : t;

            // Calc f
            mat_tras(J, K, B[tb], Bt[tb]);
            prod_matrix(I, K, J, A[ta], Bt[tb], ABt);
            prod_matrix(J, I, J, Ft[t], ABt, FtABt);
            f -= 2.0*calc_tr(J, FtABt);

            mat_tras(I, K, A[ta], At[ta]);
            prod_matrix(I, K, J, A[ta], Bt[tb], ABt);
            prod_matrix(I, I, J, DX[t], ABt, DXABt);
            prod_matrix(K, I, J, At[ta], DXABt, AtDXABt);
            prod_matrix(J, K, J, B[tb], AtDXABt, BAtDXABt);
            f += calc_tr(J, BAtDXABt);

            // df_dA
            prod_matrix(I, J, K, F[t], B[tb], FB);
            prod_matrix(K, J, K, Bt[tb], B[tb], BtB);
            prod_matrix(I, K, K, A[ta], BtB, ABtB);
            prod_matrix(I, I, K, DX[t], ABtB, DXABtB);
            
            for(int i=0; i<I; i++){
                for(int k=0; k<K; k++){
                    df_dA[ta][i][k] += 2.0*DXABtB[i][k] - 2.0*FB[i][k];
                }
            }

            // df_dB
            prod_matrix(J, I, K, Ft[t], A[ta], FtA);
            prod_matrix(I, I, K, DX[t], A[ta], DXA);
            prod_matrix(K, I, K, At[ta], DXA, AtDXA);
            prod_matrix(J, K, K, B[tb], AtDXA, BAtDXA);
            
            for(int j=0; j<J; j++){
                for(int k=0; k<K; k++){
                    df_dB[tb][j][k] += 2.0*BAtDXA[j][k] - 2.0*FtA[j][k];
                }
            }
        }

        // Iteration update and stopping criteria
        iter_count++;
        d = f_p - f;

        stop = 1;
        for(int k=0; k<K; k++){
            for(int t=0; t<TA; t++){
                for(int i=0; i<I; i++){
                    if(fabs(df_dA[t][i][k]) > 0.1){
                        stop = 0;
                    }
                }
            }
            for(int t=0; t<TB; t++){
                for(int j=0; j<J; j++){
                    if(fabs(df_dB[t][j][k]) > 0.1){
                        stop = 0;
                    }
                }
            }
        }

        if((fabs(d))<d_stop||f<f_stop){
            stop = 1;
        }

        if(stop==0){
            // Using df_dA and df_dB, df_dU and df_dZ are calculated and used to update U and Z
            for(int t=0; t<TA; t++){
                conv_df_dU(I, K, df_dA[t], A[t], df_dU[t]);
                update_UZ(I, K, rA, U[t], df_dU[t]); 
                check_UZ(I, K, U[t]);
            }
            for(int t=0; t<TB; t++){
                conv_df_dZ(J, K, df_dB[t], B[t], df_dZ[t]);
                update_UZ(J, K, rB, Z[t], df_dZ[t]);
                check_UZ(J, K, Z[t]);
            }   
        }

        if(f>100000000000000||f<0||isnan(f)){
            for(int t=0; t<TA; t++){
                mat_zero(I, K, U[t]);
            }
            for(int t=0; t<TB; t++){
                mat_zero(J, K, Z[t]);
            }
            stop = 1;
            f = N + 1.0;      
        }
    }

    // Copying results into output arrays
    for(int k=0; k<K; k++){
        for(int t=0; t<TA; t++){
            for(int i=0; i<I; i++){
                ALPHA(t,i,k) = A[t][i][k];
            }
        }
        for(int t=0; t<TB; t++){
            for(int j=0; j<J; j++){
                BETA(t,j,k) = B[t][j][k];
            }
        }
    }

}