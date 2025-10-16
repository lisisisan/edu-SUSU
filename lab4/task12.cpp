#include <mpi.h>
#include <cstdio>

int main(int argc, char **argv)
{
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int world_rank; // номер процесса
    int world_size; // общее число процессов

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Процесс с номером 0 выводит количество процессов
    if (world_rank == 0)
    {
        printf("%d processes.\n", world_size);
    }

    // Четные процессы: FIRST, нечетные: SECOND
    if (world_rank % 2 == 0)
    {
        printf("I am %d: FIRST!\n", world_rank);
    }
    else
    {
        printf("I am %d: SECOND!\n", world_rank);
    }

    MPI_Finalize();
    return 0;
}
