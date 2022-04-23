/*
Code name: compute_E_parallel.c
Author: Marco Antonio Esquivel Basaldua

Description: 
    Given a map entered as input file, computes E table from evader and pursuer velocities v_e and v_p.
    This version uses openMP parallelisation.

Input arguments:
    - env.txt : Environment map, first row represents map dimensions, and resolution. Map is represeted with 1 for free space and 0 for obstacles
    - visual_Env<env>_res<res>.txt: Visual matrix as output to the code 'compute_visibility.py'

Outputs (to be used in python codes):
    - E.txt : computed E table from dynamic programming using Tekdas and Isler algorithm
    - W.txt : Environment Workspace given in matrix coordinates 
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>

void neigborhood(uint *Mapa, uint pos_in_Mapa, uint *neighbors, uint *total_neighbors, uint *W, uint v, uint n);
uint find_in_workspace(uint *W, uint ser, uint n);

// Global variables
uint Width, Length, n;
uint v_e = 1, v_p = 1;
double env, res;

int main(int argc, char* argv[]){
    // Time excecution starts
    double start, end;
    start = omp_get_wtime();

    ////////// Open and load map file ///////////////////////
    FILE* fin = NULL;
	fin = fopen(argv[1], "r");
	if(!fin)
		printf("no input file's name\n");
    
    fscanf(fin, "%d %d %lf %lf", &Width, &Length, &env, &res);

    uint Map_size = Width * Length;      // Total elements in original map

    uint *Mapa = (uint *)malloc(sizeof(uint) * Map_size);
    uint *W = (uint *)malloc(sizeof(uint)*0);                // Wokspace
    n = 0;                                              // Total espaces in W

    for(uint i=0; i<Width; i++){
        for(uint j=0; j<Length; j++){
            uint index = i*Length + j;

            fscanf(fin, "%d", &Mapa[index]);
             
            if(Mapa[index] == 1){
                n++;
                W = (uint *)realloc(W, n*sizeof(uint));
                W[n-1] = index;
            }/*end if Mapa*/
        }/*end for j*/
    }/*end for i*/
    fclose(fin);
    ///////////////////////////////////////////////////////
    printf("n = %d\n", n);
    printf("E table dimensions = nxn = %d\n", n*n);

    ////// Visibility between cells /////////////////////
    uint *visMatrix = (uint *)malloc(sizeof(int) * (n*n));
    FILE* fin2 = NULL;
	fin2 = fopen(argv[2], "r");
	if(!fin)
		printf("no input file's name\n");

    for(int i=0; i<n*n; i++)
        fscanf(fin, "%d", &visMatrix[i]);
    fclose(fin2);
 
    ////////// Compute table E ////////////////////////
    double inf = INFINITY;
    double *E = (double *)malloc(sizeof(double) * (n*n));
    # pragma omp parallel for
    for(uint i=0; i< n*n; i++)
        E[i] = inf;

    // Fill with 0 where visualization is already lost
    # pragma omp parallel for
    for(uint e=0; e<n; e++){
        # pragma omp parallel for
        for(uint p=0; p<n; p++)
            if(!visMatrix[e*n + p])
                E[e*n + p] = 0.0;
    }
    
    for(uint k=1; k <= n*n; k++){
        # pragma omp parallel for
        for(uint e=0; e<n; e++){
            // find N_e
            uint *N_e = (uint *)malloc(sizeof(uint)*0);                // evader neighborhood
            uint total_N_e = 0;
            neigborhood(Mapa, W[e], N_e, &total_N_e, W, v_e, n);

            # pragma omp parallel for
            for(uint p=0; p<n; p++){

                if(E[e*n + p] == inf){
                    //find N_p
                    uint *N_p = (uint *)malloc(sizeof(uint)*0);                // persuer neighborhood
                    uint total_N_p = 0;
                    neigborhood(Mapa, W[p], N_p, &total_N_p, W, v_p,  n);

                    double *maxes = (double *)malloc(sizeof(double)*total_N_e);
                    for(uint i=0; i<total_N_e; i++){
                        double max = 0.0;

                        for(uint j=0; j<total_N_p; j++){
                            if(E[N_e[i]*n + N_p[j]] > max)
                                max = E[N_e[i]*n + N_p[j]];
                        }
                        maxes[i] = max;
                    }
                    double min = inf;
                    for(uint t=0; t<total_N_e; t++){
                        if(maxes[t] < min)
                            min = maxes[t];
                    }

                    if(min - (k-1) < 1e-1)
                        E[e*n + p] = (double)k;

                    free(maxes);
                    free(N_p);
                }
            }
            free(N_e);
        }
    }
    //////////////////////////////////////////////////////////////////////////////////

    //// Save E table as txt file////////////////////////
    char *file_name = "E.txt";

    FILE* fout = NULL;
    fout = fopen(file_name, "w");
    if(  !fout )  
        printf("Error: No se abrio %s\n" , file_name );

    for(uint i=0; i<n; i++){
        for(uint j=0; j<n; j++){
            fprintf(fout, "%.0lf ", E[i*n + j]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
    ////////////////////////////////////////////////////


    //// Save Workspace as txt file ///////////////////
    char *W_txt = "W.txt";

    fout = fopen(W_txt, "w");
    if(!fout)
        printf("Error: No se abrio %s\n" , W_txt);

    for(uint i=0; i<n; i++){
        uint i_ = W[i]/Length;
        uint j_ = W[i] - i_*Length;
        fprintf(fout, "%d %d\n", i_, j_);
    }
    fclose(fout);
    //////////////////////////////////////////////////

    // Time excecution is displayed
    end = omp_get_wtime(); 
    printf("%.2lf seconds taken in the algorithm\n", end - start);
    printf("%.2lf minutes taken in the algorithm\n", (end - start)/60.0);
    printf("%.2lf hours taken in the algorithm\n", (end - start)/3600.0);
    
    free(Mapa); free(W); free(E);
    return 0;
}

void neigborhood(uint *Mapa, uint pos_in_Mapa, uint *neighbors, uint *total_neighbors, uint *W, uint v, uint n){
    // Convert pos_in_Mapa to matrix coordinates
    uint i = pos_in_Mapa/Length;
    uint j = pos_in_Mapa - i*Length;

    // uint window = 2*v +1;

    // // Find neighborhood
    // uint start[] = {i-v, j-v};

    // for(uint i_=0; i_<window; i_++){
    //     for(uint j_=0; j_<window; j_++){
    //         uint test[] = {start[0]+i_, start[1]+j_};

    //         uint test_converted = test[0]*Length + test[1];

    //         if(test_converted >= 0 && test_converted < Length*Width && test_converted != pos_in_Mapa){
    //             if(dist(test[0],test[1],i,j) <= (double)v + 1e-2 && Mapa[test_converted] == 1 && visual(Mapa_ext,pos_in_Mapa,test_converted)){
    //                 *total_neighbors += 1;
    //                 neighbors = (uint *)realloc(neighbors, *total_neighbors * sizeof(uint));

    //                 neighbors[*total_neighbors-1] = find_in_workspace(W, test_converted, n);
    //             }
    //         }
    //     }
    // }




    // Find neighborhood
    uint x[] = {i,j+1, i-1,j, i,j-1, i+1,j};

    for(uint it=0; it<4; it++){
        uint i_ = x[it*2];
        uint j_ = x[it*2 + 1];

        if(i_ >= 0 && i_ < Width && j_ >= 0 && j_ < Length){
            // Convert i_,j_ to location in vectorized map
            uint pos = i_*Length + j_;
            if(Mapa[pos] == 1){
                *total_neighbors +=1;
                neighbors = (uint *)realloc(neighbors, *total_neighbors * sizeof(uint));

                neighbors[*total_neighbors-1] = find_in_workspace(W, pos, n);
            }
        }/*end if Mapa*/
    }/*end for it*/
}

uint find_in_workspace(uint *W, uint ser, uint n){
    for(uint i=0; i<n; i++)
        if(W[i] == ser)
            return i;

    return -1;
}