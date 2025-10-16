#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size;
    long long N = 0; // количество шагов интегрирования

    // --- Инициализация MPI ---
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // общее количество процессов

    // --- Ввод данных только процессом 0 ---
    if (rank == 0)
    {
        printf("Enter N: ");
        fflush(stdout);
        scanf("%lld", &N);
    }

    // --- Рассылка N всем процессам ---
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // --- Засекаем время начала ---
    double start_time = MPI_Wtime();

    double h = 1.0 / (double)N;
    double local_sum = 0.0;

    // --- Параллельная область OpenMP ---
#pragma omp parallel
    {
        double private_sum = 0.0;

        // Каждый поток считает часть интеграла
#pragma omp for nowait
        for (long long i = rank; i < N; i += size)
        {
            double x = (i + 0.5) * h;
            private_sum += 4.0 / (1.0 + x * x);
        }

        // Атомарно добавляем частичную сумму потока
#pragma omp atomic
        local_sum += private_sum;
    }

    local_sum *= h;

    double pi = 0.0;

    // --- Суммирование частичных результатов со всех процессов ---
    MPI_Reduce(&local_sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // --- Засекаем время окончания ---
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // --- Определим максимальное время выполнения между процессами ---
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // --- Вывод результата (только процесс 0) ---
    if (rank == 0)
    {
        printf("pi = %.8f\n", pi);
        printf("Execution time (max among all processes): %.6f seconds\n", max_time);
    }

    MPI_Finalize();
    return 0;
}

// mpicc -Xpreprocessor -fopenmp \
//   -I/opt/homebrew/opt/libomp/include \
//   -L/opt/homebrew/opt/libomp/lib \
//   -lomp task32.c -o task32

// mpirun -np 2 ./task32

// --- Суммирование частичных результатов со всех процессов ---
/*
    MPI_Reduce(
        void *sendbuf,    // адрес локальных данных для отправки
        void *recvbuf,    // адрес буфера приёма (только у root)
        int count,        // количество элементов
        MPI_Datatype type,// тип данных
        MPI_Op op,        // операция (например, MPI_SUM, MPI_MAX)
        int root,         // процесс, где собираются данные
        MPI_Comm comm     // коммуникатор
    )
    → На процессе root результат операции (например, сумма всех локальных значений).
*/

// --- Рассылка N всем процессам ---
/*
    MPI_Bcast(
        void *buffer,     // адрес данных для рассылки
        int count,        // количество элементов для рассылки
        MPI_Datatype type,// тип данных (например, MPI_INT, MPI_DOUBLE)
        int root,         // номер процесса-отправителя (обычно 0)
        MPI_Comm comm     // коммуникатор (обычно MPI_COMM_WORLD)
    )
    → Все процессы после вызова получают одинаковое значение из root.
*/