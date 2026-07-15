#include <stdlib.h>
#include <stdio.h>
#include "slbt_als.h"

#define Fij(i,j) F[(i)*J + (j)]
#define FnoNij(i,j) F_noN[(i)*J + (j)]
#define FsnoNkij(t,i,j) Fs_noN[(t)*I*J + (i)*J + (j)]
#define Fskij(t,i,j) Fs[(t)*I*J + (i)*J + (j)]

#define ALPHA(t,i,k) out_alpha[(t)*I*2 + (i)*2 + (k)]
#define BETA(t,j,k)  out_beta[(t)*J*2 + (j)*2 + (k)]

#define Ski(k,i) out_S[(k)*I + (i)]

#define FL(t,i,j)  F_left[(t)*I*J + (i)*J + (j)]
#define FR(t,i,j)  F_right[(t)*I*J + (i)*J + (j)]

// Internally used functions
double gini_impurity_complete(int I, int J, const double *F){
    double sumI, sumJ = 0.0;
    double N = 0.0;

    // Calculate N
    for(int i=0; i<I; i++){
        for(int j=0; j<J; j++){
            N += Fij(i,j);
        }
    }

    // Avoid division by zero
    if(N == 0.0){
        return 0.0;
    }

    // Calculate gini impurity
    for(int j=0; j<J; j++){
        sumI = 0.0;
        for(int i=0; i<I; i++){
            sumI += Fij(i,j); 
        }
        sumI /= N;
        sumJ += sumI*sumI;
    }

    // Return gini impurity
    return N*(1.0-sumJ);
}

double gini_impurity_strat_k(int I, int J, const double F[I][J]){
    double pyz;
    double impurity = 0.0;

    for(int j=0; j<J; j++){
        pyz = 0.0;
        for(int i=0; i<I; i++){
            pyz += F[i][j];
        }
        impurity += pyz*pyz;
    }

    return 1-impurity;
}

double gini_impurity_rows(int I, int J, const int S[I], const double *F){ //Function used to evaluate the gini impurity
    double sumI, sumJ = 0.0;
    double N = 0.0;
    
    // Calculate N
    for(int i=0; i<I; i++){
        if(S[i]){
            for(int j=0; j<J; j++){
                N += Fij(i,j);
            }
        }
    }
    
    // Avoid division by zero
    if(N == 0.0){
        return 0.0;
    }

    // Calculate gini impurity
    for(int j=0; j<J; j++){
        sumI = 0.0;
        for(int i=0; i<I; i++){
            if(S[i]){
                sumI += Fij(i,j);
            }
        }
        sumI /= N;
        sumJ += sumI*sumI;
    }

    // Return gini impurity
    return N*(1.0-sumJ);
}

// Out-called from slbt/_utils/criteria.py
double gpi_c(int K, int I, int J, const double *Fs){ //Given F this function returns the gpi
    double tau_yk = 0.0; // tau_Y|K
    double tau_ykx = 0.0; // tau_Y|(K,X)

    double P_k, P_jk, P_ik, P_ijk;

    // Calc tau_yk
    for(int k=0; k<K; k++){
        P_k = 0.0;
        for(int j=0; j<J; j++){
            for(int i=0; i<I; i++){
                P_k += Fskij(k,i,j);
            }
        }
        for(int j=0; j<J; j++){
            P_jk = 0.0;
            for(int i=0; i<I; i++){
                P_jk += Fskij(k,i,j);
            }
            if(P_k != 0.0){
                tau_yk += (P_jk*P_jk)/P_k;
            }
        }
    }

    // Calc tau_ykx
    for(int k=0; k<K; k++){
        for(int i=0; i<I; i++){
            P_ik = 0.0;
            P_ijk = 0.0;
            for(int j=0; j<J; j++){
                P_ik += Fskij(k,i,j);
            }
            for(int j=0; j<J; j++){
                P_ijk += Fskij(k,i,j)*Fskij(k,i,j);
            }
            if(P_ik != 0.0){
                tau_ykx += P_ijk/P_ik;
            }
        }
    }

    return (tau_ykx - tau_yk)/(1 - tau_yk);
}

// Out-called from slbt/_tree/split.py
void slba_c(int K, int KA, int KB, int I, int J, // Input sizes
    const double *Fs_noN, const double *Fs, // Input matrices
    double *out_pi, double *out_S, double *out_alpha, double *out_beta // Outputs
    ){ //Given F this function returns the split none
    // Algo variables
    double parent_tau;
    double left_tau, right_tau;
    double L = 0, R = 0;
    double pi;

    double *F_left  = calloc(K * I * J, sizeof(double));
    double *F_right = calloc(K * I * J, sizeof(double));

    int S[KA][I], S_N[KA][I];
    int all_left=1, all_right=1; 

    // S and S_Negative inizialization
    for(int t=0; t<KA; t++){
        for(int i=0; i<I; i++){
            S[t][i] = 1;
            S_N[t][i] = 0;
        }   
    }
    
    // Parent impurity eval
    parent_tau = gpi_c(K ,I ,J ,Fs_noN);

    // LBA is executed
    slba_C_execute(I, J, K, KA, KB, Fs, out_alpha, out_beta);
    
    // Eval S and S_Negative based on the LBA results
    for(int t=0; t<KA; t++){
        for(int i=0; i<I; i++){
            if(ALPHA(t,i,0) < ALPHA(t,i,1)){
                S[t][i] = 0;
                S_N[t][i] = 1;
            } 
        }
    }

    // Two local matrices F_left and F_right are evaluated in order to calculate the left and right impurity
    for(int t=0; t<K; t++){
        for(int i=0; i<I; i++){
            for(int j=0; j<J; j++){
                FL(t, i, j) = 0.0;
                FR(t, i, j) = 0.0;
            }
        }
    }

    if(KA != K){
        for(int t=0; t<K; t++){
            for(int i=0; i<I; i++){
                for(int j=0; j<J; j++){
                    FL(t, i,j) += (double)FsnoNkij(t,i,j) * S[0][i];
                    FR(t, i,j) += (double)FsnoNkij(t,i,j) * S_N[0][i];
                    L += (double)FsnoNkij(t,i,j) * S[0][i];
                    R += (double)FsnoNkij(t,i,j) * S_N[0][i];
                }
            }
        }
    } else {
        for(int t=0; t<K; t++){
            for(int i=0; i<I; i++){
                for(int j=0; j<J; j++){
                    FL(t, i,j) += (double)FsnoNkij(t,i,j) * S[t][i];
                    FR(t, i,j) += (double)FsnoNkij(t,i,j) * S_N[t][i];
                    L += (double)FsnoNkij(t,i,j) * S[t][i];
                    R += (double)FsnoNkij(t,i,j) * S_N[t][i];
                }
            }
        }
    }
    
    // Check for all left or all right
    for (int t = 0; t < KA; t++) {
        for (int i = 0; i < I; i++) {
            if (S[t][i] == 0) all_left = 0;
            if (S_N[t][i] == 0) all_right = 0;
        }
    }

    // PPI eval
    if(all_left || all_right){
        pi = 0;
    } else {
        left_tau = gpi_c(K, I, J, F_left);
        right_tau = gpi_c(K, I, J, F_right);

        // PPI eval
        pi = parent_tau - (L*left_tau + R*right_tau);
    }

    *out_pi = pi;
    for(int t=0; t<KA; t++){
        for(int i=0; i<I; i++){
            Ski(t,i) = (double)S[t][i];
        }
    }

    free(F_left);
    free(F_right);
}