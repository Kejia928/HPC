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
int initialise( char* paramfile,  char* obstaclefile,
               t_param* params, grid** cells_ptr, grid** tmp_cells_ptr,
               char** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
inline float timestep( t_param params, grid** cells, grid** tmp_cells, char* obstacles);
inline int accelerate_flow( t_param params, grid* cells, char* obstacles);
/* compute the next state for the cell */
inline float calculateNextState( t_param params, grid* cells, grid* tmp_cells, char* obstacles);
int write_values( t_param params, grid* cells, char* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise( t_param* params, grid** cells_ptr, grid** tmp_cells_ptr,
             char** obstacles_ptr, float** av_vels_ptr);

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
  char*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

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
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    av_vels[tt] = timestep(params, &cells, &tmp_cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

inline float timestep( t_param params, grid** cells, grid** tmp_cells, char* restrict obstacles)
{
  accelerate_flow(params, *cells, obstacles);
  float result = calculateNextState(params, *cells, *tmp_cells, obstacles);
  grid *tmp = *tmp_cells;
  *tmp_cells = *cells;
  *cells = tmp;
  return result;
}

inline int accelerate_flow( t_param params, grid* restrict cells, char* restrict obstacles)
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
  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;
  //#pragma omp simd
  //#pragma omp parallel for
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speed3[ii + jj*params.nx] - w1) > 0.f
        && (cells->speed6[ii + jj*params.nx] - w2) > 0.f
        && (cells->speed7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed1[ii + jj*params.nx] += w1;
      cells->speed5[ii + jj*params.nx] += w2;
      cells->speed8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speed3[ii + jj*params.nx] -= w1;
      cells->speed6[ii + jj*params.nx] -= w2;
      cells->speed7[ii + jj*params.nx] -= w2;
    }
  }
  return EXIT_SUCCESS;
}

inline float calculateNextState( t_param params, grid* restrict cells, grid* restrict tmp_cells, char* restrict obstacles)
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

  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */

  /* compute local density total */
  float c_sq = 1.f / 3.f; /* square of speed of sound */
  float w0 = 4.f / 9.f;  /* weighting factor */
  float w1 = 1.f / 9.f;  /* weighting factor */
  float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over _all_ cells */
  #pragma omp parallel for schedule(static) reduction(+:tot_u) reduction(+:tot_cells)
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd aligned(cells:64) aligned(tmp_cells:64) aligned(obstacles:64) reduction(+:tot_cells) reduction(+:tot_u)
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = (jj + 1) % params.ny;
      const int x_e = (ii + 1) % params.nx;
      const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      
      const float s0 = cells->speed0[ii + jj*params.nx]; /* central cell, no movement */
      const float s1 = cells->speed1[x_w + jj*params.nx]; /* east */
      const float s2 = cells->speed2[ii + y_s*params.nx]; /* north */
      const float s3 = cells->speed3[x_e + jj*params.nx]; /* west */
      const float s4 = cells->speed4[ii + y_n*params.nx]; /* south */
      const float s5 = cells->speed5[x_w + y_s*params.nx]; /* north-east */
      const float s6 = cells->speed6[x_e + y_s*params.nx]; /* north-west */
      const float s7 = cells->speed7[x_e + y_n*params.nx]; /* south-west */
      const float s8 = cells->speed8[x_w + y_n*params.nx]; /* south-east */

      /* don't consider occupied cells */
      if (obstacles[ii + jj*params.nx]){
        tmp_cells->speed0[ii + jj*params.nx] = s0;
        tmp_cells->speed1[ii + jj*params.nx] = s3;
        tmp_cells->speed2[ii + jj*params.nx] = s4;
        tmp_cells->speed3[ii + jj*params.nx] = s1;
        tmp_cells->speed4[ii + jj*params.nx] = s2;
        tmp_cells->speed5[ii + jj*params.nx] = s7;
        tmp_cells->speed6[ii + jj*params.nx] = s8;
        tmp_cells->speed7[ii + jj*params.nx] = s5;
        tmp_cells->speed8[ii + jj*params.nx] = s6;
      }
      else
      {
        //float new_cell[NSPEEDS];
        /* propagate densities from neighbouring cells, following
        ** appropriate directions of travel and writing into
        ** scratch space grid */
        // new_cell[0] = cells->speeds[0][ii + jj*params.nx]; /* central cell, no movement */
        // new_cell[1] = cells->speeds[1][x_w + jj*params.nx]; /* east */
        // new_cell[2] = cells->speeds[2][ii + y_s*params.nx]; /* north */
        // new_cell[3] = cells->speeds[3][x_e + jj*params.nx]; /* west */
        // new_cell[4] = cells->speeds[4][ii + y_n*params.nx]; /* south */
        // new_cell[5] = cells->speeds[5][x_w + y_s*params.nx]; /* north-east */
        // new_cell[6] = cells->speeds[6][x_e + y_s*params.nx]; /* north-west */
        // new_cell[7] = cells->speeds[7][x_e + y_n*params.nx]; /* south-west */
        // new_cell[8] = cells->speeds[8][x_w + y_n*params.nx]; /* south-east */

        // for (int kk = 0; kk < NSPEEDS; kk++)
        // {
        //   local_density += new_cell[kk];
        // }
        const float local_density = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;

        // /* compute x velocity component */
        //  float u_x = (new_cell[1]
        //               + new_cell[5]
        //               + new_cell[8]
        //               - (new_cell[3]
        //                  + new_cell[6]
        //                  + new_cell[7]))
        //              / local_density;
        // /* compute y velocity component */
        //  float u_y = (new_cell[2]
        //               + new_cell[5]
        //               + new_cell[6]
        //               - (new_cell[4]
        //                  + new_cell[7]
        //                  + new_cell[8]))
        //              / local_density;

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
        // for (int kk = 0; kk < NSPEEDS; kk++)
        // {
        //   tmp_cells->speeds[kk][ii + jj*params.nx] = new_cell[kk] + params.omega * (d_equ[kk] - new_cell[kk]);
        // }
        tmp_cells->speed0[ii + jj*params.nx] = s0 + params.omega * (d0 - s0);
        tmp_cells->speed1[ii + jj*params.nx] = s1 + params.omega * (d1 - s1);
        tmp_cells->speed2[ii + jj*params.nx] = s2 + params.omega * (d2 - s2);
        tmp_cells->speed3[ii + jj*params.nx] = s3 + params.omega * (d3 - s3);
        tmp_cells->speed4[ii + jj*params.nx] = s4 + params.omega * (d4 - s4);
        tmp_cells->speed5[ii + jj*params.nx] = s5 + params.omega * (d5 - s5);
        tmp_cells->speed6[ii + jj*params.nx] = s6 + params.omega * (d6 - s6);
        tmp_cells->speed7[ii + jj*params.nx] = s7 + params.omega * (d7 - s7);
        tmp_cells->speed8[ii + jj*params.nx] = s8 + params.omega * (d8 - s8);
        //local_density = tmp_cells->speed0[ii + jj*params.nx] + tmp_cells->speed1[ii + jj*params.nx] + tmp_cells->speed2[ii + jj*params.nx] + tmp_cells->speed3[ii + jj*params.nx] + tmp_cells->speed4[ii + jj*params.nx] + tmp_cells->speed5[ii + jj*params.nx] + tmp_cells->speed6[ii + jj*params.nx] + tmp_cells->speed7[ii + jj*params.nx] + tmp_cells->speed8[ii + jj*params.nx];
        /* x-component of velocity */
        /*
        u_x = (tmp_cells->speed1[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed8[ii + jj*params.nx]
                      - (tmp_cells->speed3[ii + jj*params.nx]
                         + tmp_cells->speed6[ii + jj*params.nx]
                         + tmp_cells->speed7[ii + jj*params.nx]))
                     / local_density; */
        /* compute y velocity component */
        /*
        u_y = (tmp_cells->speed2[ii + jj*params.nx]
                      + tmp_cells->speed5[ii + jj*params.nx]
                      + tmp_cells->speed6[ii + jj*params.nx]
                      - (tmp_cells->speed4[ii + jj*params.nx]
                         + tmp_cells->speed7[ii + jj*params.nx]
                         + tmp_cells->speed8[ii + jj*params.nx]))
                     / local_density;
                     */
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf(u_sq);
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }
  return tot_u / (float)tot_cells;
}

inline float av_velocity( t_param params, grid* restrict cells, char* restrict obstacles)
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

  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  #pragma omp parallel for schedule(static) reduction(+:tot_u) reduction(+:tot_cells)
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd aligned(cells:64) aligned(obstacles:64) reduction(+:tot_cells) reduction(+:tot_u)
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
        local_density = cells->speed0[ii + jj*params.nx] + cells->speed1[ii + jj*params.nx] + cells->speed2[ii + jj*params.nx] + cells->speed3[ii + jj*params.nx] + cells->speed4[ii + jj*params.nx] + cells->speed5[ii + jj*params.nx] + cells->speed6[ii + jj*params.nx] + cells->speed7[ii + jj*params.nx] + cells->speed8[ii + jj*params.nx];

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
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }
  return tot_u / (float)tot_cells;
}

int initialise( char* paramfile,  char* obstaclefile,
               t_param* params, grid** cells_ptr, grid** tmp_cells_ptr,
               char** obstacles_ptr, float** av_vels_ptr)
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

  /* main grid */
  *cells_ptr = (grid*)_mm_malloc(sizeof(float*)*NSPEEDS, 64);
  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (grid*)_mm_malloc(sizeof(float*)*NSPEEDS, 64);
  // #pragma unroll(9)
  // for(int i = 0; i < NSPEEDS; i++){
  //   (*cells_ptr)->speeds = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  //   (*tmp_cells_ptr)->speeds[i] = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  // }
  (*cells_ptr)->speed0 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed1 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed2 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed3 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed4 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed5 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed6 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed7 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed8 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed0 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed1 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed2 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed3 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed4 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed5 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed6 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed7 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed8 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(char) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  { 
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)->speed0[ii + jj*params->nx] = w0;
      /* axis directions */
      (*cells_ptr)->speed1[ii + jj*params->nx] = w1;
      (*cells_ptr)->speed2[ii + jj*params->nx] = w1;
      (*cells_ptr)->speed3[ii + jj*params->nx] = w1;
      (*cells_ptr)->speed4[ii + jj*params->nx] = w1;
      /* diagonals */
      (*cells_ptr)->speed5[ii + jj*params->nx] = w2;
      (*cells_ptr)->speed6[ii + jj*params->nx] = w2;
      (*cells_ptr)->speed7[ii + jj*params->nx] = w2;
      (*cells_ptr)->speed8[ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
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

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise( t_param* params, grid** cells_ptr, grid** tmp_cells_ptr,
          char** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  // #pragma unroll(9)
  // for(int i = 0; i < NSPEEDS; i++){
  //   _mm_free((*cells_ptr)->speeds[i]);
  //   _mm_free((*tmp_cells_ptr)->speeds[i]);
  // }

  _mm_free((*cells_ptr)->speed0);
  (*cells_ptr)->speed0 = NULL;
  _mm_free((*cells_ptr)->speed1);
  (*cells_ptr)->speed1 = NULL;
  _mm_free((*cells_ptr)->speed2);
  (*cells_ptr)->speed2 = NULL;
  _mm_free((*cells_ptr)->speed3);
  (*cells_ptr)->speed3 = NULL;
  _mm_free((*cells_ptr)->speed4);
  (*cells_ptr)->speed4 = NULL;
  _mm_free((*cells_ptr)->speed5);
  (*cells_ptr)->speed5 = NULL;
  _mm_free((*cells_ptr)->speed6);
  (*cells_ptr)->speed6 = NULL;
  _mm_free((*cells_ptr)->speed7);
  (*cells_ptr)->speed7 = NULL;
  _mm_free((*cells_ptr)->speed8);
  (*cells_ptr)->speed8 = NULL;

  _mm_free((*tmp_cells_ptr)->speed0);
  (*tmp_cells_ptr)->speed0 = NULL;
  _mm_free((*tmp_cells_ptr)->speed1);
  (*tmp_cells_ptr)->speed1 = NULL;
  _mm_free((*tmp_cells_ptr)->speed2);
  (*tmp_cells_ptr)->speed2 = NULL;
  _mm_free((*tmp_cells_ptr)->speed3);
  (*tmp_cells_ptr)->speed3 = NULL;
  _mm_free((*tmp_cells_ptr)->speed4);
  (*tmp_cells_ptr)->speed4 = NULL;
  _mm_free((*tmp_cells_ptr)->speed5);
  (*tmp_cells_ptr)->speed5 = NULL;
  _mm_free((*tmp_cells_ptr)->speed6);
  (*tmp_cells_ptr)->speed6 = NULL;
  _mm_free((*tmp_cells_ptr)->speed7);
  (*tmp_cells_ptr)->speed7 = NULL;
  _mm_free((*tmp_cells_ptr)->speed8);
  (*tmp_cells_ptr)->speed8 = NULL;

  _mm_free(*cells_ptr);
  *cells_ptr = NULL;

  _mm_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

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
