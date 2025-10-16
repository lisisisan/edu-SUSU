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

    // Процесс 0 считывает размер и данные матриц
    std::vector<double> A, B;
    if (rank == 0)
    {
        scanf("%d", &n);
        A.resize(n * n);
        B.resize(n * n);
        for (int i = 0; i < n * n; i++)
            scanf("%lf", &A[i]);
        for (int i = 0; i < n * n; i++)
            scanf("%lf", &B[i]);
    }

    // Рассылаем размер n всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Каждому процессу выделяем буфер для своей части строк матрицы A
    int rows_per_proc = n / size + (rank < n % size ? 1 : 0); // равномерное распределение
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    int offset = 0;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = (n / size + (i < n % size ? 1 : 0)) * n; // количество элементов для процесса i
        displs[i] = offset;
        offset += sendcounts[i];
    }

    std::vector<double> local_A(sendcounts[rank]);
    std::vector<double> local_C(sendcounts[rank], 0.0);

    // Рассылаем строки матрицы A
    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Разослать всю матрицу B всем процессам
    if (B.empty())
        B.resize(n * n);
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычисление произведения строк матрицы C
    int local_rows = sendcounts[rank] / n;
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

    // Сбор всех частей матрицы C на процессе 0
    std::vector<double> C;
    if (rank == 0)
        C.resize(n * n);
    MPI_Gatherv(local_C.data(), sendcounts[rank], MPI_DOUBLE,
                C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вывод результата
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                printf("%.0lf ", C[i * n + j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
