#include <mpi.h>
#include <cstdio>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Общее количество процессов

    int buf = rank; // 5. Сообщение, равное номеру процесса
    int recv_buf;   // Буфер для принятого сообщения

    int next = (rank + 1) % size;        // Следующий процесс
    int prev = (rank - 1 + size) % size; // Предыдущий процесс

    MPI_Request requests[2]; // Два запроса: send и recv
    MPI_Status statuses[2];

    // 6.1. Неблокирующая отправка следующему процессу
    MPI_Isend(&buf, 1, MPI_INT, next, 0, MPI_COMM_WORLD, &requests[0]);

    // 6.2. Неблокирующий прием от предыдущего процесса
    MPI_Irecv(&recv_buf, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &requests[1]);

    // 6.3. Ждем завершения обоих обменов
    MPI_Waitall(2, requests, statuses);

    // 7. Вывод принятого сообщения
    printf("[%d]: receive message '%d'\n", rank, recv_buf);

    MPI_Finalize();
    return 0;
}
