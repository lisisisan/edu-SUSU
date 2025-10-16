#include <mpi.h>
#include <cstdio>
#include <vector>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Количество процессов

    int buf = rank; // Сообщение для отправки

    std::vector<int> recv_buf(size);        // Буферы для приема
    std::vector<MPI_Request> send_requests; // Запросы для send
    std::vector<MPI_Request> recv_requests; // Запросы для recv

    // --- Неблокирующая отправка всем остальным ---
    for (int dest = 0; dest < size; dest++)
    {
        if (dest != rank)
        {
            MPI_Request req;
            MPI_Isend(&buf, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, &req);
            send_requests.push_back(req);
        }
    }

    // Ждем завершения всех send
    if (!send_requests.empty())
    {
        MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
    }

    // --- Неблокирующий прием сообщений от всех остальных ---
    for (int src = 0; src < size; src++)
    {
        if (src != rank)
        {
            MPI_Request req;
            MPI_Irecv(&recv_buf[src], 1, MPI_INT, src, 0, MPI_COMM_WORLD, &req);
            recv_requests.push_back(req);
        }
    }

    // Ждем завершения всех recv
    if (!recv_requests.empty())
    {
        MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
    }

    // --- Вывод сообщений ---
    // только для этого процесса, в порядке отправителей
    for (int src = 0; src < size; src++)
    {
        if (src != rank)
        {
            printf("[%d]: receive message '%d' from %d\n", rank, recv_buf[src], src);
        }
    }

    MPI_Finalize();
    return 0;
}
