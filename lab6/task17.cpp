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
    }

    // Широковещательная рассылка числа n
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Широковещательная рассылка самой строки
    MPI_Bcast(buf, n, MPI_CHAR, 0, MPI_COMM_WORLD);

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
