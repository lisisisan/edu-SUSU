#include <mpi.h>
#include <cstdio>
#include <vector>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // количество процессов

    int n = 0;
    std::vector<double> A, B;

    if (rank == 0)
    {
        // Читаем размер матриц и сами матрицы
        scanf("%d", &n);
        A.resize(n * n);
        B.resize(n * n);

        for (int i = 0; i < n * n; i++)
            scanf("%lf", &A[i]);
        for (int i = 0; i < n * n; i++)
            scanf("%lf", &B[i]);
    }

    // Рассылаем размер матрицы всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Разбиваем строки матрицы A между процессами
    std::vector<int> rows_per_proc(size);
    std::vector<int> displs(size);

    int offset = 0;
    for (int i = 0; i < size; i++)
    {
        rows_per_proc[i] = n / size + (i < n % size ? 1 : 0);
        displs[i] = offset;
        offset += rows_per_proc[i];
    }

    // Буферы для локальных данных
    int local_rows = rows_per_proc[rank];
    std::vector<double> local_A(local_rows * n);
    std::vector<double> local_C(local_rows * n, 0.0);

    // Рассылка матрицы A (точка-точка)
    if (rank == 0)
    {
        // Отправляем строки другим процессам
        for (int i = 1; i < size; i++)
        {
            MPI_Send(A.data() + displs[i] * n, rows_per_proc[i] * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        // Сохраняем строки для процесса 0
        for (int i = 0; i < local_rows * n; i++)
            local_A[i] = A[i];
    }
    else
    {
        MPI_Recv(local_A.data(), local_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Рассылка матрицы B всем процессам (B одинакова для всех)
    if (B.empty())
        B.resize(n * n);
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Засекаем время
    double start = MPI_Wtime();

    // Вычисляем локальную часть C
    for (int i = 0; i < local_rows; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }

    // Сбор результатов точка-точка
    if (rank == 0)
    {
        std::vector<double> C(n * n);
        // Копируем свои данные
        for (int i = 0; i < local_rows * n; i++)
            C[i] = local_C[i];

        // Принимаем от других
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(C.data() + displs[i] * n, rows_per_proc[i] * n, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double end = MPI_Wtime();
        printf("Матрица C:\n");
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                printf("%.0lf ", C[i * n + j]);
            printf("\n");
        }
        printf("Время выполнения: %.6f секунд\n", end - start);
    }
    else
    {
        // Отправляем свой кусок обратно
        MPI_Send(local_C.data(), local_rows * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
