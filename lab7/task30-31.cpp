#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int n; // количество нитей

    // --- Инициализация MPI ---
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // общее количество процессов

    // --- Ввод числа потоков (только процесс 0) ---
    if (rank == 0)
    {
        fflush(stdout);
        scanf("%d", &n);
    }

    // --- Рассылаем n всем процессам ---
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Устанавливаем число потоков OpenMP ---
    omp_set_num_threads(n);

    // --- Засекаем время выполнения ---
    double start_time = MPI_Wtime();

    // --- Выполним какую-то «работу», чтобы было что измерять ---
    double sum = 0.0;
#pragma omp parallel reduction(+ : sum)
    {
        int thread_num = omp_get_thread_num();
        int process_num = rank;

        // имитация вычислений (нагрузка)
        for (int i = 0; i < 100000000; i++)
        {
            sum += (thread_num + 1) * 0.0000001;
        }

#pragma omp critical
        {
            printf("I am %d thread from %d process. Partial sum = %.2f\n",
                   thread_num, process_num, sum);
            fflush(stdout);
        }
    }

    // --- Замеряем общее время ---
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // --- Выводим время выполнения ---
    printf("Process %d finished in %f seconds\n", rank, elapsed);
    fflush(stdout);

    // --- Синхронизация и замер максимального времени по всем процессам ---
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("=== Total execution time (max among all processes): %f seconds ===\n", max_time);
    }

    MPI_Finalize();
    return 0;
}

// mpicxx -Xpreprocessor -fopenmp \
//   -I/opt/homebrew/opt/libomp/include \
//   -L/opt/homebrew/opt/libomp/lib \
//   -lomp task30-31.cpp -o task30-31

// mpirun -np 2 ./task30-31
