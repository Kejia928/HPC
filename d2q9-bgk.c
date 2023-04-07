/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int tot_cells;        /* total number of cell */
  /* MPI param */
  int np;
  int rank;
  int left;
  int right;
  int task_nx;
  int task_ny;
  int startX;
  int endX;
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float* restrict speed0;
  float* restrict speed1;
  float* restrict speed2;
  float* restrict speed3;
  float* restrict speed4;
  float* restrict speed5;
  float* restrict speed6;
  float* restrict speed7;
  float* restrict speed8;
} grid;


/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile,  const char* obstaclefile,
               t_param* params, grid** cells_ptr, grid** tmp_cells_ptr, grid** result_cells, 
               char** obstacles_ptr, float** av_vels_ptr,
               float** sendBuffer, float** recvBuffer,
               float** sendResultBuffer, float** recvResultBuffer);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep( t_param params, grid** cells, grid** tmp_cells, char* obstacles, float* sendBuffer, float* recvBuffer);
int accelerate_flow( t_param params, grid* cells, char* obstacles);
/* compute the next state for the cell */
float calculateNextState( t_param params, grid* cells, grid* tmp_cells, char* obstacles, float* sendBuffer, float* recvBuffer);
int write_values( t_param params, grid* cells, char* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const char* paramfile,  const char* obstaclefile,
             t_param* params, grid** cells_ptr, grid** tmp_cells_ptr, grid** result_cells, 
             char** obstacles_ptr, float** av_vels_ptr,
             float** sendBuffer, float** recvBuffer,
             float** sendResultBuffer, float** recvResultBuffer);

/* Sum all the densities in the grid.
** The total should remain ant from one timestep to the next. */
float total_density( t_param params, grid* cells);

/* compute average velocity */
float av_velocity( t_param params, grid* cells, char* obstacles);

/* calculate Reynolds number */
float calc_reynolds( t_param params, grid* cells, char* obstacles);

/* utility functions */
void die( char* message,  int line,  char* file);
void usage( char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  grid* cells     = NULL;    /* grid containing fluid densities */
  grid* tmp_cells = NULL;    /* scratch space */
  grid* result_cells = NULL; /* final; result grid */
  char*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /*Buffer for MPI*/
  // halo region buffer
  float* sendBuffer = NULL;
  float* recvBuffer = NULL;
  // result buffer
  float* sendResultBuffer = NULL;
  float* recvResultBuffer = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &params.np);
  MPI_Comm_rank(MPI_COMM_WORLD, &params.rank);

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &result_cells, &obstacles, &av_vels, &sendBuffer, &recvBuffer, &sendResultBuffer, &recvResultBuffer);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++){
    float u = timestep(params, &cells, &tmp_cells, obstacles, sendBuffer, recvBuffer);
    //printf("No problem in calculation on %d \n", params.rank);
    if (params.rank == 0){
      if (params.np > 1){ 
        MPI_Reduce(MPI_IN_PLACE, &u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      }
    } else {
      MPI_Reduce(&u, NULL, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    av_vels[tt] = u / (float)params.tot_cells;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
    //printf("No problem in %d round on %d \n", tt, params.rank); 
  }
  //printf("No problem in timestep on %d \n", params.rank);
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 
  if (params.rank != 0){
    /* Send the data to master */
    for (int jj = 0; jj < params.task_ny; jj++){
      for (int ii = 0; ii < params.task_nx; ii++){
        sendResultBuffer[ii + jj*(params.task_nx) + (params.task_ny * params.task_nx * 0)] = cells -> speed0[(ii+1) + jj*(params.task_nx+2)];
        sendResultBuffer[ii + jj*(params.task_nx) + (params.task_ny * params.task_nx * 1)] = cells -> speed1[(ii+1) + jj*(params.task_nx+2)];
        sendResultBuffer[ii + jj*(params.task_nx) + (params.task_ny * params.task_nx * 2)] = cells -> speed2[(ii+1) + jj*(params.task_nx+2)];
        sendResultBuffer[ii + jj*(params.task_nx) + (params.task_ny * params.task_nx * 3)] = cells -> speed3[(ii+1) + jj*(params.task_nx+2)];
        sendResultBuffer[ii + jj*(params.task_nx) + (params.task_ny * params.task_nx * 4)] = cells -> speed4[(ii+1) + jj*(params.task_nx+2)];
        sendResultBuffer[ii + jj*(params.task_nx) + (params.task_ny * params.task_nx * 5)] = cells -> speed5[(ii+1) + jj*(params.task_nx+2)];
        sendResultBuffer[ii + jj*(params.task_nx) + (params.task_ny * params.task_nx * 6)] = cells -> speed6[(ii+1) + jj*(params.task_nx+2)];
        sendResultBuffer[ii + jj*(params.task_nx) + (params.task_ny * params.task_nx * 7)] = cells -> speed7[(ii+1) + jj*(params.task_nx+2)];
        sendResultBuffer[ii + jj*(params.task_nx) + (params.task_ny * params.task_nx * 8)] = cells -> speed8[(ii+1) + jj*(params.task_nx+2)];
      }
    }
    MPI_Send(sendResultBuffer, (params.task_nx * params.task_ny * NSPEEDS), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    //printf("Send result on %d \n", params.rank);
  } else {
    /* Collect cell from master local */
    for (int jj = 0; jj < params.task_ny; jj++){
      for (int ii = 0; ii < params.task_nx; ii++){
        result_cells->speed0[ii + jj*params.nx] = cells -> speed0[(ii+1) + jj*(params.task_nx+2)];
        result_cells->speed1[ii + jj*params.nx] = cells -> speed1[(ii+1) + jj*(params.task_nx+2)];
        result_cells->speed2[ii + jj*params.nx] = cells -> speed2[(ii+1) + jj*(params.task_nx+2)];
        result_cells->speed3[ii + jj*params.nx] = cells -> speed3[(ii+1) + jj*(params.task_nx+2)];
        result_cells->speed4[ii + jj*params.nx] = cells -> speed4[(ii+1) + jj*(params.task_nx+2)];
        result_cells->speed5[ii + jj*params.nx] = cells -> speed5[(ii+1) + jj*(params.task_nx+2)];
        result_cells->speed6[ii + jj*params.nx] = cells -> speed6[(ii+1) + jj*(params.task_nx+2)];
        result_cells->speed7[ii + jj*params.nx] = cells -> speed7[(ii+1) + jj*(params.task_nx+2)];
        result_cells->speed8[ii + jj*params.nx] = cells -> speed8[(ii+1) + jj*(params.task_nx+2)];
      }
    }
    /* Collect cell from other node */
    if (params.np > 1) {
      for (int from = 1; from < params.np; from++){
        /* Calculate the number of columns for each task */
        int task_nx = params.nx/params.np;
        int startX = 0;
        int remainder = params.nx % params.np;
        if(remainder != 0){
          if (from < remainder){
            task_nx += 1;
            startX = from * task_nx;
          } else {
            startX = from * task_nx + remainder;
          }
        } else {
          startX = from * task_nx;
        }
        //printf("startX is %d and task_nx is %d \n", startX, task_nx);
        
        MPI_Recv(recvResultBuffer, (params.task_ny * task_nx * NSPEEDS), MPI_FLOAT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("receive result from %d on %d \n", from, params.rank);
        for (int jj = 0; jj < params.ny; jj++){
          for (int ii = 0; ii < task_nx; ii++){
            result_cells->speed0[ii + startX + jj*params.nx] = recvResultBuffer[ii + jj*(task_nx) + (params.task_ny * task_nx * 0)];
            result_cells->speed1[ii + startX + jj*params.nx] = recvResultBuffer[ii + jj*(task_nx) + (params.task_ny * task_nx * 1)];
            result_cells->speed2[ii + startX + jj*params.nx] = recvResultBuffer[ii + jj*(task_nx) + (params.task_ny * task_nx * 2)];
            result_cells->speed3[ii + startX + jj*params.nx] = recvResultBuffer[ii + jj*(task_nx) + (params.task_ny * task_nx * 3)];
            result_cells->speed4[ii + startX + jj*params.nx] = recvResultBuffer[ii + jj*(task_nx) + (params.task_ny * task_nx * 4)];
            result_cells->speed5[ii + startX + jj*params.nx] = recvResultBuffer[ii + jj*(task_nx) + (params.task_ny * task_nx * 5)];
            result_cells->speed6[ii + startX + jj*params.nx] = recvResultBuffer[ii + jj*(task_nx) + (params.task_ny * task_nx * 6)];
            result_cells->speed7[ii + startX + jj*params.nx] = recvResultBuffer[ii + jj*(task_nx) + (params.task_ny * task_nx * 7)];
            result_cells->speed8[ii + startX + jj*params.nx] = recvResultBuffer[ii + jj*(task_nx) + (params.task_ny * task_nx * 8)];
          }
        }
        //printf("No problem on combination on %d from %d \n", params.rank, from);
      }
    }
  }

  //printf("Finish combination");

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  if(params.rank == 0){
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, result_cells, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    write_values(params, result_cells, obstacles, av_vels);
  }
  MPI_Finalize();
  finalise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &result_cells, &obstacles, &av_vels, &sendBuffer, &recvBuffer, &sendResultBuffer, &recvResultBuffer);
  return EXIT_SUCCESS;
}

float timestep( t_param params, grid** cells, grid** tmp_cells, char* restrict obstacles, float* sendBuffer, float* recvBuffer)
{
  accelerate_flow(params, *cells, obstacles);
  float tot_u = calculateNextState(params, *cells, *tmp_cells, obstacles, sendBuffer, recvBuffer);
  grid *tmp = *tmp_cells;
  *tmp_cells = *cells;
  *cells = tmp;
  return tot_u;
}

int accelerate_flow( t_param params, grid* restrict cells, char* restrict obstacles)
{
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  __assume_aligned(obstacles, 64);
  __assume((params.nx)%16==0);
  __assume((params.ny)%16==0); 
  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = params.task_ny - 2;
  #pragma omp simd
  for (int ii = 1; ii < params.task_nx+1; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + (jj*params.nx+params.startX) - 1]
        && (cells->speed3[ii + jj*(params.task_nx+2)] - w1) > 0.f
        && (cells->speed6[ii + jj*(params.task_nx+2)] - w2) > 0.f
        && (cells->speed7[ii + jj*(params.task_nx+2)] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed1[ii + jj*(params.task_nx+2)] += w1;
      cells->speed5[ii + jj*(params.task_nx+2)] += w2;
      cells->speed8[ii + jj*(params.task_nx+2)] += w2;
      /* decrease 'west-side' densities */
      cells->speed3[ii + jj*(params.task_nx+2)] -= w1;
      cells->speed6[ii + jj*(params.task_nx+2)] -= w2;
      cells->speed7[ii + jj*(params.task_nx+2)] -= w2;
    }
  }
  return EXIT_SUCCESS;
}

float calculateNextState( t_param params, grid* restrict cells, grid* restrict tmp_cells, char* restrict obstacles, float* sendBuffer, float* recvBuffer)
{
  __assume_aligned(cells, 64);
  __assume_aligned(tmp_cells, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);
  __assume_aligned(obstacles, 64);
  __assume((params.nx)%16==0);
  __assume((params.ny)%16==0);

  /* Halo exchange code */
  /* Send to left and receive from right */
  for (int y = 0; y < params.task_ny; y++){
    // send the left coloum
    sendBuffer[0 + (y*NSPEEDS)] = cells->speed0[y * (params.task_nx+2) + 1];
    sendBuffer[1 + (y*NSPEEDS)] = cells->speed1[y * (params.task_nx+2) + 1];
    sendBuffer[2 + (y*NSPEEDS)] = cells->speed2[y * (params.task_nx+2) + 1];
    sendBuffer[3 + (y*NSPEEDS)] = cells->speed3[y * (params.task_nx+2) + 1];
    sendBuffer[4 + (y*NSPEEDS)] = cells->speed4[y * (params.task_nx+2) + 1];
    sendBuffer[5 + (y*NSPEEDS)] = cells->speed5[y * (params.task_nx+2) + 1];
    sendBuffer[6 + (y*NSPEEDS)] = cells->speed6[y * (params.task_nx+2) + 1];
    sendBuffer[7 + (y*NSPEEDS)] = cells->speed7[y * (params.task_nx+2) + 1];
    sendBuffer[8 + (y*NSPEEDS)] = cells->speed8[y * (params.task_nx+2) + 1];
  }
  MPI_Sendrecv(sendBuffer, (params.ny*NSPEEDS), MPI_FLOAT, params.left, 0, recvBuffer, (params.ny*NSPEEDS), MPI_FLOAT, params.right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (int y = 0; y < params.task_ny; y++){
    cells->speed0[(y+1) * (params.task_nx+2) - 1] = recvBuffer[0 + (y*NSPEEDS)];
    cells->speed1[(y+1) * (params.task_nx+2) - 1] = recvBuffer[1 + (y*NSPEEDS)];
    cells->speed2[(y+1) * (params.task_nx+2) - 1] = recvBuffer[2 + (y*NSPEEDS)];
    cells->speed3[(y+1) * (params.task_nx+2) - 1] = recvBuffer[3 + (y*NSPEEDS)];
    cells->speed4[(y+1) * (params.task_nx+2) - 1] = recvBuffer[4 + (y*NSPEEDS)];
    cells->speed5[(y+1) * (params.task_nx+2) - 1] = recvBuffer[5 + (y*NSPEEDS)];
    cells->speed6[(y+1) * (params.task_nx+2) - 1] = recvBuffer[6 + (y*NSPEEDS)];
    cells->speed7[(y+1) * (params.task_nx+2) - 1] = recvBuffer[7 + (y*NSPEEDS)];
    cells->speed8[(y+1) * (params.task_nx+2) - 1] = recvBuffer[8 + (y*NSPEEDS)];
  }

  /* Send to right and receive from left */
  for (int y = 0; y < params.task_ny; y++){
    // save the first clo
    sendBuffer[0 + (y*NSPEEDS)] = cells->speed0[y * (params.task_nx+2) + params.task_nx];
    sendBuffer[1 + (y*NSPEEDS)] = cells->speed1[y * (params.task_nx+2) + params.task_nx];
    sendBuffer[2 + (y*NSPEEDS)] = cells->speed2[y * (params.task_nx+2) + params.task_nx];
    sendBuffer[3 + (y*NSPEEDS)] = cells->speed3[y * (params.task_nx+2) + params.task_nx];
    sendBuffer[4 + (y*NSPEEDS)] = cells->speed4[y * (params.task_nx+2) + params.task_nx];
    sendBuffer[5 + (y*NSPEEDS)] = cells->speed5[y * (params.task_nx+2) + params.task_nx];
    sendBuffer[6 + (y*NSPEEDS)] = cells->speed6[y * (params.task_nx+2) + params.task_nx];
    sendBuffer[7 + (y*NSPEEDS)] = cells->speed7[y * (params.task_nx+2) + params.task_nx];
    sendBuffer[8 + (y*NSPEEDS)] = cells->speed8[y * (params.task_nx+2) + params.task_nx];
  }
  MPI_Sendrecv(sendBuffer, (params.ny*NSPEEDS), MPI_FLOAT, params.right, 0, recvBuffer, (params.ny*NSPEEDS), MPI_FLOAT, params.left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (int y = 0; y < params.task_ny; y++){
    // save the last clo
    cells->speed0[y * (params.task_nx+2)] = recvBuffer[0 + (y*NSPEEDS)];
    cells->speed1[y * (params.task_nx+2)] = recvBuffer[1 + (y*NSPEEDS)];
    cells->speed2[y * (params.task_nx+2)] = recvBuffer[2 + (y*NSPEEDS)];
    cells->speed3[y * (params.task_nx+2)] = recvBuffer[3 + (y*NSPEEDS)];
    cells->speed4[y * (params.task_nx+2)] = recvBuffer[4 + (y*NSPEEDS)];
    cells->speed5[y * (params.task_nx+2)] = recvBuffer[5 + (y*NSPEEDS)];
    cells->speed6[y * (params.task_nx+2)] = recvBuffer[6 + (y*NSPEEDS)];
    cells->speed7[y * (params.task_nx+2)] = recvBuffer[7 + (y*NSPEEDS)];
    cells->speed8[y * (params.task_nx+2)] = recvBuffer[8 + (y*NSPEEDS)];
  }
  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */

  /* compute local density total */
  float c_sq = 1.f / 3.f; /* square of speed of sound */
  float w0 = 4.f / 9.f;  /* weighting factor */
  float w1 = 1.f / 9.f;  /* weighting factor */
  float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over _all_ cells */
  for (int jj = 0; jj < params.task_ny; jj++)
  {
    #pragma omp simd
    for (int ii = 1; ii < params.task_nx + 1; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = (jj + 1) % params.task_ny;
      const int x_e = (ii + 1);
      const int y_s = (jj == 0) ? (jj + params.task_ny - 1) : (jj - 1);
      const int x_w = (ii - 1);
      
      const float s0 = cells->speed0[ii + jj*(params.task_nx+2)]; /* central cell, no movement */
      const float s1 = cells->speed1[x_w + jj*(params.task_nx+2)]; /* east */
      const float s2 = cells->speed2[ii + y_s*(params.task_nx+2)]; /* north */
      const float s3 = cells->speed3[x_e + jj*(params.task_nx+2)]; /* west */
      const float s4 = cells->speed4[ii + y_n*(params.task_nx+2)]; /* south */
      const float s5 = cells->speed5[x_w + y_s*(params.task_nx+2)]; /* north-east */
      const float s6 = cells->speed6[x_e + y_s*(params.task_nx+2)]; /* north-west */
      const float s7 = cells->speed7[x_e + y_n*(params.task_nx+2)]; /* south-west */
      const float s8 = cells->speed8[x_w + y_n*(params.task_nx+2)]; /* south-east */

      /* don't consider occupied cells */
      if (obstacles[ii + (jj*params.nx + params.startX) - 1]){
        tmp_cells->speed0[ii + jj*(params.task_nx+2)] = s0;
        tmp_cells->speed1[ii + jj*(params.task_nx+2)] = s3;
        tmp_cells->speed2[ii + jj*(params.task_nx+2)] = s4;
        tmp_cells->speed3[ii + jj*(params.task_nx+2)] = s1;
        tmp_cells->speed4[ii + jj*(params.task_nx+2)] = s2;
        tmp_cells->speed5[ii + jj*(params.task_nx+2)] = s7;
        tmp_cells->speed6[ii + jj*(params.task_nx+2)] = s8;
        tmp_cells->speed7[ii + jj*(params.task_nx+2)] = s5;
        tmp_cells->speed8[ii + jj*(params.task_nx+2)] = s6;
      }
      else
      {
        const float local_density = s0 + s1 + s2 + s3 + s4 + s5 + 
                                    s6 + s7 + s8;
        /* compute x velocity component */
        const float u_x = (s1 + s5 + s8 - (s3 + s6 + s7)) / local_density;
        /* compute y velocity component */
        const float u_y = (s2 + s5 + s6 - (s4 + s7 + s8)) / local_density;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        const float u1 =   u_x;        /* east */
        const float u2 =         u_y;  /* north */
        const float u3 = - u_x;        /* west */
        const float u4 =       - u_y;  /* south */
        const float u5 =   u_x + u_y;  /* north-east */
        const float u6 = - u_x + u_y;  /* north-west */
        const float u7 = - u_x - u_y;  /* south-west */
        const float u8 =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        /* zero velocity density: weight w0 */
        const float d0 = w0 * local_density
                  * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        const float d1 = w1 * local_density * (1.f + u1 / c_sq
                                        + (u1 * u1) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
        const float d2 = w1 * local_density * (1.f + u2 / c_sq
                                        + (u2 * u2) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
        const float d3 = w1 * local_density * (1.f + u3 / c_sq
                                        + (u3 * u3) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
        const float d4 = w1 * local_density * (1.f + u4 / c_sq
                                        + (u4 * u4) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        const float d5 = w2 * local_density * (1.f + u5 / c_sq
                                        + (u5 * u5) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
        const float d6 = w2 * local_density * (1.f + u6 / c_sq
                                        + (u6 * u6) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
        const float d7 = w2 * local_density * (1.f + u7 / c_sq
                                        + (u7 * u7) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
        const float d8 = w2 * local_density * (1.f + u8 / c_sq
                                        + (u8 * u8) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));

        // /* relaxation step */
        tmp_cells->speed0[ii + jj*(params.task_nx+2)] = s0 + params.omega * (d0 - s0);
        tmp_cells->speed1[ii + jj*(params.task_nx+2)] = s1 + params.omega * (d1 - s1);
        tmp_cells->speed2[ii + jj*(params.task_nx+2)] = s2 + params.omega * (d2 - s2);
        tmp_cells->speed3[ii + jj*(params.task_nx+2)] = s3 + params.omega * (d3 - s3);
        tmp_cells->speed4[ii + jj*(params.task_nx+2)] = s4 + params.omega * (d4 - s4);
        tmp_cells->speed5[ii + jj*(params.task_nx+2)] = s5 + params.omega * (d5 - s5);
        tmp_cells->speed6[ii + jj*(params.task_nx+2)] = s6 + params.omega * (d6 - s6);
        tmp_cells->speed7[ii + jj*(params.task_nx+2)] = s7 + params.omega * (d7 - s7);
        tmp_cells->speed8[ii + jj*(params.task_nx+2)] = s8 + params.omega * (d8 - s8);
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf(u_sq);
      }
    }
  }
  return tot_u;
}

float av_velocity( t_param params, grid* restrict cells, char* restrict obstacles)
{
  __assume((params.nx)%16==0);
  __assume((params.ny)%16==0);

  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;
        // #pragma unroll(9)
        // for (int kk = 0; kk < NSPEEDS; kk++)
        // {
        //   local_density += cells->speeds[kk][ii + jj*params.nx];
        // }
        local_density = cells->speed0[ii + jj*params.nx] 
                      + cells->speed1[ii + jj*params.nx] 
                      + cells->speed2[ii + jj*params.nx] 
                      + cells->speed3[ii + jj*params.nx] 
                      + cells->speed4[ii + jj*params.nx] 
                      + cells->speed5[ii + jj*params.nx] 
                      + cells->speed6[ii + jj*params.nx] 
                      + cells->speed7[ii + jj*params.nx] 
                      + cells->speed8[ii + jj*params.nx];

        /* x-component of velocity */
        const float u_x = (cells->speed1[ii + jj*params.nx]
                      + cells->speed5[ii + jj*params.nx]
                      + cells->speed8[ii + jj*params.nx]
                      - (cells->speed3[ii + jj*params.nx]
                         + cells->speed6[ii + jj*params.nx]
                         + cells->speed7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        const float u_y = (cells->speed2[ii + jj*params.nx]
                      + cells->speed5[ii + jj*params.nx]
                      + cells->speed6[ii + jj*params.nx]
                      - (cells->speed4[ii + jj*params.nx]
                         + cells->speed7[ii + jj*params.nx]
                         + cells->speed8[ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
      }
    }
  }
  return tot_u / (float)params.tot_cells;
}

int initialise(const char* paramfile,  const char* obstaclefile,
               t_param* params, grid** cells_ptr, grid** tmp_cells_ptr, grid** result_cells, 
               char** obstacles_ptr, float** av_vels_ptr,
               float** sendBuffer, float** recvBuffer,
               float** sendResultBuffer, float** recvResultBuffer)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /* Get neighbour index */
  if(params->rank != 0){
    params->left = params->rank - 1;
  } else {
    params->left = params->np - 1;
  }
  if(params->rank != params->np - 1){
    params->right = params->rank + 1;
  } else {
    params->right = 0;
  }

  /* Calculate the number of columns for each task */
  params->task_nx = params->nx/params->np;
  int remainder = params->nx % params->np;
  if(remainder != 0){
    if (params->rank < remainder){
      params->task_nx = params->task_nx + 1;
    }
  }
  params->task_ny = params->ny;

  /* Calculate the start and end index of the columns */
  // if the remainder is zero, directly multiply with rank
  // if the task size is total size divide size plus, directly multiply with rank
  if(remainder == 0 || params->task_nx == params->nx/params->np + 1) {
    params->startX = params->rank * params->task_nx;
  } else {
  // this is only for the last task, if the remainder is not zero
  // the start point need add remainder
    params->startX = params->rank * params->task_nx + remainder;
  }
  params->endX = params->startX + params->task_nx - 1;

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* Allocate memory for halo region */
  *recvBuffer = (float*)_mm_malloc(sizeof(float) * (params->task_ny * NSPEEDS), 64);
  *sendBuffer = (float*)_mm_malloc(sizeof(float) * (params->task_ny * NSPEEDS), 64);
  /* main grid */
  *cells_ptr = (grid*)_mm_malloc(sizeof(float*)*NSPEEDS, 64);
  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (grid*)_mm_malloc(sizeof(float*)*NSPEEDS, 64);
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* Allocate the memory based on each task size and two halo region */
  (*cells_ptr)->speed0 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*cells_ptr)->speed1 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*cells_ptr)->speed2 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*cells_ptr)->speed3 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*cells_ptr)->speed4 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*cells_ptr)->speed5 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*cells_ptr)->speed6 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*cells_ptr)->speed7 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*cells_ptr)->speed8 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*tmp_cells_ptr)->speed0 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*tmp_cells_ptr)->speed1 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*tmp_cells_ptr)->speed2 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*tmp_cells_ptr)->speed3 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*tmp_cells_ptr)->speed4 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*tmp_cells_ptr)->speed5 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*tmp_cells_ptr)->speed6 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*tmp_cells_ptr)->speed7 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  (*tmp_cells_ptr)->speed8 = (float*)_mm_malloc(sizeof(float) * (params->task_ny * (params->task_nx+2)), 64);
  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(char) * (params->ny * params->nx), 64);
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  { 
    #pragma omp simd
    for (int ii = 1; ii < params->task_nx+1; ii++)
    {
      /* get the sub grid for each task based on their rank 
         and leave the blank for halo region */
      /* centre */
      (*cells_ptr)->speed0[ii + jj*(params->task_nx+2)] = w0;
      /* axis directions */
      (*cells_ptr)->speed1[ii + jj*(params->task_nx+2)] = w1;
      (*cells_ptr)->speed2[ii + jj*(params->task_nx+2)] = w1;
      (*cells_ptr)->speed3[ii + jj*(params->task_nx+2)] = w1;
      (*cells_ptr)->speed4[ii + jj*(params->task_nx+2)] = w1;
      /* diagonals */
      (*cells_ptr)->speed5[ii + jj*(params->task_nx+2)] = w2;
      (*cells_ptr)->speed6[ii + jj*(params->task_nx+2)] = w2;
      (*cells_ptr)->speed7[ii + jj*(params->task_nx+2)] = w2;
      (*cells_ptr)->speed8[ii + jj*(params->task_nx+2)] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

   /* Allocate memory for result send and receive */
  if (params->rank != 0){
    *sendResultBuffer = (float*)_mm_malloc(sizeof(float) * (params->task_nx * params->task_ny * NSPEEDS), 64);
  } else {
    *recvResultBuffer = (float*)_mm_malloc(sizeof(float) * (params->task_nx * params->task_ny * NSPEEDS), 64);
    *result_cells = (grid*)_mm_malloc(sizeof(float*)*NSPEEDS, 64);
    if (*result_cells == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
    (*result_cells)->speed0 = (float*)_mm_malloc(sizeof(float) * (params->nx * params->ny), 64);
    (*result_cells)->speed1 = (float*)_mm_malloc(sizeof(float) * (params->nx * params->ny), 64);
    (*result_cells)->speed2 = (float*)_mm_malloc(sizeof(float) * (params->nx * params->ny), 64);
    (*result_cells)->speed3 = (float*)_mm_malloc(sizeof(float) * (params->nx * params->ny), 64);
    (*result_cells)->speed4 = (float*)_mm_malloc(sizeof(float) * (params->nx * params->ny), 64);
    (*result_cells)->speed5 = (float*)_mm_malloc(sizeof(float) * (params->nx * params->ny), 64);
    (*result_cells)->speed6 = (float*)_mm_malloc(sizeof(float) * (params->nx * params->ny), 64);
    (*result_cells)->speed7 = (float*)_mm_malloc(sizeof(float) * (params->nx * params->ny), 64);
    (*result_cells)->speed8 = (float*)_mm_malloc(sizeof(float) * (params->nx * params->ny), 64);
  }

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters, 64);
  /* Calculate the total number of cells */
  params->tot_cells = 0;
  for (int jj = 0; jj < params->ny; jj++){ 
    for (int ii = 0; ii < params->nx; ii++){
      /* calculate params->tot_cells after initialisation */
      if(!(*obstacles_ptr)[jj*params->nx + ii]){
        params->tot_cells += 1;
      }
    }
  }
  return EXIT_SUCCESS;
}

int finalise(const char* paramfile,  const char* obstaclefile,
             t_param* params, grid** cells_ptr, grid** tmp_cells_ptr, grid** result_cells, 
             char** obstacles_ptr, float** av_vels_ptr,
             float** sendBuffer, float** recvBuffer,
             float** sendResultBuffer, float** recvResultBuffer)
{
  // /*
  // ** free up allocated memory
  // */

  // _mm_free((*cells_ptr)->speed0);
  // // (*cells_ptr)->speed0 = NULL;
  // _mm_free((*cells_ptr)->speed1);
  // // (*cells_ptr)->speed1 = NULL;
  // _mm_free((*cells_ptr)->speed2);
  // // (*cells_ptr)->speed2 = NULL;
  // _mm_free((*cells_ptr)->speed3);
  // // (*cells_ptr)->speed3 = NULL;
  // _mm_free((*cells_ptr)->speed4);
  // // (*cells_ptr)->speed4 = NULL;
  // _mm_free((*cells_ptr)->speed5);
  // // (*cells_ptr)->speed5 = NULL;
  // _mm_free((*cells_ptr)->speed6);
  // // (*cells_ptr)->speed6 = NULL;
  // _mm_free((*cells_ptr)->speed7);
  // // (*cells_ptr)->speed7 = NULL;
  // _mm_free((*cells_ptr)->speed8);
  // // (*cells_ptr)->speed8 = NULL;

  // _mm_free((*tmp_cells_ptr)->speed0);
  // // (*tmp_cells_ptr)->speed0 = NULL;
  // _mm_free((*tmp_cells_ptr)->speed1);
  // // (*tmp_cells_ptr)->speed1 = NULL;
  // _mm_free((*tmp_cells_ptr)->speed2);
  // // (*tmp_cells_ptr)->speed2 = NULL;
  // _mm_free((*tmp_cells_ptr)->speed3);
  // // (*tmp_cells_ptr)->speed3 = NULL;
  // _mm_free((*tmp_cells_ptr)->speed4);
  // // (*tmp_cells_ptr)->speed4 = NULL;
  // _mm_free((*tmp_cells_ptr)->speed5);
  // // (*tmp_cells_ptr)->speed5 = NULL;
  // _mm_free((*tmp_cells_ptr)->speed6);
  // // (*tmp_cells_ptr)->speed6 = NULL;
  // _mm_free((*tmp_cells_ptr)->speed7);
  // // (*tmp_cells_ptr)->speed7 = NULL;
  // _mm_free((*tmp_cells_ptr)->speed8);
  // // (*tmp_cells_ptr)->speed8 = NULL;

  // _mm_free(*recvBuffer);
  // // *recvBuffer = NULL;
  // _mm_free(*sendBuffer);
  // // *sendBuffer = NULL;

  // if (params->rank == 0){
  //   _mm_free((*result_cells)->speed0);
  //   // (*result_cells)->speed0 = NULL;
  //   // printf("Free 11, from %d. \n", params->rank);
  //   _mm_free((*result_cells)->speed1);
  //   // (*result_cells)->speed1 = NULL;
  //   // printf("Free 12, from %d. \n", params->rank);
  //   _mm_free((*result_cells)->speed2);
  //   // (*result_cells)->speed2 = NULL;
  //   // printf("Free 13, from %d. \n", params->rank);
  //   _mm_free((*result_cells)->speed3);
  //   // (*result_cells)->speed3 = NULL;
  //   // printf("Free 14, from %d. \n", params->rank);
  //   _mm_free((*result_cells)->speed4);
  //   // (*result_cells)->speed4 = NULL;
  //   // printf("Free 15, from %d. \n", params->rank);
  //   _mm_free((*result_cells)->speed5);
  //   // (*result_cells)->speed5 = NULL;
  //   // printf("Free 16, from %d. \n", params->rank);
  //   _mm_free((*result_cells)->speed6);
  //   // (*result_cells)->speed6 = NULL;
  //   // printf("Free 17, from %d. \n", params->rank);
  //   _mm_free((*result_cells)->speed7);
  //   // (*result_cells)->speed7 = NULL;
  //   // printf("Free 18, from %d. \n", params->rank);
  //   _mm_free((*result_cells)->speed8);
  //   // (*result_cells)->speed8 = NULL;
  //   // printf("Free 19, from %d. \n", params->rank);
  //   // _mm_free(result_cells);
  //   // result_cells = NULL;
  //   // printf("Free 20, from %d. \n", params->rank);
  //   _mm_free(*recvResultBuffer);
  //   // *recvResultBuffer = NULL;
  //   // printf("Free 21, from %d. \n", params->rank);
  // } else {
  //   _mm_free(*sendResultBuffer);
  //   // *sendResultBuffer = NULL;
  // }

  // _mm_free(*cells_ptr);
  // *cells_ptr = NULL;
  // // printf("Free 1, from %d. \n", params->rank);

  // _mm_free(*tmp_cells_ptr);
  // *tmp_cells_ptr = NULL;
  // // printf("Free 2, from %d. \n", params->rank);

  // _mm_free(*obstacles_ptr);
  // *obstacles_ptr = NULL;
  // // printf("Free 3, from %d. \n", params->rank);

  // _mm_free(*av_vels_ptr);
  // *av_vels_ptr = NULL;
  // // printf("Free 4, from %d. \n", params->rank);

  /*
  ** free up allocated memory
  */
  _mm_free((*cells_ptr)->speed0);
  _mm_free((*cells_ptr)->speed1);
  _mm_free((*cells_ptr)->speed2);
  _mm_free((*cells_ptr)->speed3);
  _mm_free((*cells_ptr)->speed4);
  _mm_free((*cells_ptr)->speed5);
  _mm_free((*cells_ptr)->speed6);
  _mm_free((*cells_ptr)->speed7);
  _mm_free((*cells_ptr)->speed8);

  _mm_free((*tmp_cells_ptr)->speed0);
  _mm_free((*tmp_cells_ptr)->speed1);
  _mm_free((*tmp_cells_ptr)->speed2);
  _mm_free((*tmp_cells_ptr)->speed3);
  _mm_free((*tmp_cells_ptr)->speed4);
  _mm_free((*tmp_cells_ptr)->speed5);
  _mm_free((*tmp_cells_ptr)->speed6);
  _mm_free((*tmp_cells_ptr)->speed7);
  _mm_free((*tmp_cells_ptr)->speed8);

  if (params->rank == 0){
    _mm_free((*result_cells)->speed0);
    _mm_free((*result_cells)->speed1);
    _mm_free((*result_cells)->speed2);
    _mm_free((*result_cells)->speed3);
    _mm_free((*result_cells)->speed4);
    _mm_free((*result_cells)->speed5);
    _mm_free((*result_cells)->speed6);
    _mm_free((*result_cells)->speed7);
    _mm_free((*result_cells)->speed8);
    //printf("finish free result_cells on %d \n", params->rank);
  }

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  _mm_free(*sendBuffer);
  *sendBuffer = NULL;
  _mm_free(*recvBuffer);
  *recvBuffer = NULL;

  _mm_free(*sendResultBuffer);
  *sendResultBuffer = NULL;
  _mm_free(*recvResultBuffer);
  *recvResultBuffer = NULL;

  //printf("free finished on %d \n", params->rank);
  return EXIT_SUCCESS;
}


float calc_reynolds( t_param params, grid* cells, char* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density( t_param params, grid* cells)
{
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  __assume((params.nx)%16==0);
  __assume((params.ny)%16==0);
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      // for (int kk = 0; kk < NSPEEDS; kk++)
      // {
      //   total += cells->speeds[kk][ii + jj*params.nx];
      // }
      total += cells->speed0[ii + jj*params.nx];
      total += cells->speed1[ii + jj*params.nx];
      total += cells->speed2[ii + jj*params.nx];
      total += cells->speed3[ii + jj*params.nx];
      total += cells->speed4[ii + jj*params.nx];
      total += cells->speed5[ii + jj*params.nx];
      total += cells->speed6[ii + jj*params.nx];
      total += cells->speed7[ii + jj*params.nx];
      total += cells->speed8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(t_param params, grid* cells, char* obstacles, float* av_vels)
{
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);
  __assume((params.nx)%16==0);
  __assume((params.ny)%16==0);

  FILE* fp;                    /* file pointer */
  float c_sq = 1.f / 3.f;      /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        // for (int kk = 0; kk < NSPEEDS; kk++)
        // {
        //   local_density += cells->speeds[kk][ii + jj*params.nx];
        // }
        local_density = cells->speed0[ii + jj*params.nx] + cells->speed1[ii + jj*params.nx] + cells->speed2[ii + jj*params.nx] + cells->speed3[ii + jj*params.nx] + cells->speed4[ii + jj*params.nx] + cells->speed5[ii + jj*params.nx] + cells->speed6[ii + jj*params.nx] + cells->speed7[ii + jj*params.nx] + cells->speed8[ii + jj*params.nx];

        /* compute x velocity component */
        u_x = (cells->speed1[ii + jj*params.nx]
               + cells->speed5[ii + jj*params.nx]
               + cells->speed8[ii + jj*params.nx]
               - (cells->speed3[ii + jj*params.nx]
                  + cells->speed6[ii + jj*params.nx]
                  + cells->speed7[ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speed2[ii + jj*params.nx]
               + cells->speed5[ii + jj*params.nx]
               + cells->speed6[ii + jj*params.nx]
               - (cells->speed4[ii + jj*params.nx]
                  + cells->speed7[ii + jj*params.nx]
                  + cells->speed8[ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die( char* message,  int line,  char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage( char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}