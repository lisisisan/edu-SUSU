#include <mpi.h>
#include <cstdio>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Количество процессов

    long long N = 0; // Количество шагов
    if (rank == 0)
    {
        scanf("%lld", &N); // Ввод точности
    }

    // Рассылка N всем процессам
    MPI_Bcast(&N, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    double h = 1.0 / static_cast<double>(N);
    double local_sum = 0.0;

    // Каждый процесс вычисляет свою часть интеграла
    for (long long i = rank; i < N; i += size)
    {
        double x = (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    // Запуск таймера
    double start_time = MPI_Wtime();

    local_sum *= h;

    double pi = 0.0;

    // Суммирование частичных результатов на процессе 0
    MPI_Reduce(&local_sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double end_time = MPI_Wtime();
        double elapsed = end_time - start_time;

        printf("%.8f\n", pi);
        printf("Elapsed time: %.6f seconds\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}
