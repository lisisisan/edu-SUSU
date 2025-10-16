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

        // Рассылка N процессам 1..size-1
        for (int p = 1; p < size; ++p)
        {
            MPI_Send(&N, 1, MPI_LONG_LONG_INT, p, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        // Получение N от процесса 0
        MPI_Recv(&N, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Запуск таймера
    double start_time = MPI_Wtime();

    double h = 1.0 / static_cast<double>(N);
    double local_sum = 0.0;

    // Каждый процесс считает свою часть интеграла
    for (long long i = rank; i < N; i += size)
    {
        double x = (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    local_sum *= h;

    double pi = 0.0;

    if (rank == 0)
    {
        pi = local_sum;

        // Получение результатов от всех остальных процессов
        for (int p = 1; p < size; ++p)
        {
            double tmp;
            MPI_Recv(&tmp, 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pi += tmp;
        }

        double end_time = MPI_Wtime();
        double elapsed = end_time - start_time;

        printf("%.8f\n", pi);
        printf("Elapsed time: %.6f seconds\n", elapsed);
    }
    else
    {
        // Отправка своей частичной суммы процессу 0
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
