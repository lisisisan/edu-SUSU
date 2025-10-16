#include <mpi.h>
#include <cstdio>

int main(int argc, char **argv)
{
    // 2. Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank; // 3. Номер процесса
    int size; // 4. Общее количество процессов

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buf; // 5. Сообщение

    // 6. Разделение кода для master и slave
    if (rank == 0)
    {
        // Master-процесс получает сообщения от всех slave
        for (int src = 1; src < size; src++)
        {
            MPI_Recv(&buf, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // 8. Вывод полученного сообщения
            printf("receive message '%d' from %d\n", buf, src);
        }
    }
    else
    {
        // Slave-процессы отправляют свой номер мастеру
        buf = rank; // 7. Присвоить номер процесса
        MPI_Send(&buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
