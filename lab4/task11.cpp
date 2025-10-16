#include <mpi.h>
#include <cstdio>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double start_time = MPI_Wtime(); // старт замера

    printf("I am %d process from %d processes!\n", world_rank, world_size);

    // Имитация вычислений
    for (volatile int i = 0; i < 100000000; i++)
    {
    }

    double end_time = MPI_Wtime(); // конец замера

    if (world_rank == 0)
    {
        printf("Execution time: %.6f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}

// mpirun -np 15 --map-by :OVERSUBSCRIBE ./task11