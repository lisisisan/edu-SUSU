#include <mpi.h>
#include <cstdio>
#include <cstring>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Количество процессов

    int n = 0;
    char buf[100] = {0}; // Буфер для строки

    // Ввод данных только процессом 0
    if (rank == 0)
    {
        scanf("%d", &n);
        scanf("%s", buf);

        // Рассылка n и строки buf процессам 1..size-1
        for (int p = 1; p < size; ++p)
        {
            MPI_Send(&n, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(buf, n, MPI_CHAR, p, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        // Остальные процессы получают n и строку от процесса 0
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(buf, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int counts[26] = {0}; // Счётчики для каждой буквы 'a'-'z'

    // Каждый процесс обрабатывает "свои" буквы
    for (int i = rank; i < 26; i += size)
    {
        char letter = 'a' + i;
        for (int j = 0; j < n; ++j)
        {
            if (buf[j] == letter)
            {
                counts[i]++;
            }
        }
    }

    // Сбор результатов с использованием MPI_Reduce
    int total_counts[26] = {0};
    MPI_Reduce(counts, total_counts, 26, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Вывод на процессе 0
    if (rank == 0)
    {
        for (int i = 0; i < 26; ++i)
        {
            if (total_counts[i] > 0)
            {
                printf("%c = %d\n", 'a' + i, total_counts[i]);
            }
        }
    }

    MPI_Finalize();
    return 0;
}

// Сравнение эффективности:

// MPI_Bcast — оптимизированная коллективная операция,
// которая под капотом делает рассылку быстрее
// (например, через древовидное распространение данных).
// Это намного быстрее, чем вручную слать всем по очереди.

// Вариант с MPI_Send/MPI_Recv простой, но менее эффективный:
// процесс 0 перегружается, так как он один отправляет всем.