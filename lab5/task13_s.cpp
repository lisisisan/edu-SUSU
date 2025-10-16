#include <mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Количество процессов

    int buf;

    if (rank == 0)
    {
        // Первый процесс инициализирует эстафету
        buf = 0;
        // Отправляем "палочку" следующему
        MPI_Send(&buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        // Ждём, когда последний вернёт
        MPI_Recv(&buf, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Выводим сообщение для процесса 0
        std::cout << "Process " << rank << " has buf = " << buf << std::endl;
        // printf("[%d]: receive message '%d'\n", rank, buf);
    }
    else
    {
        // Получаем "палочку" от предыдущего
        MPI_Recv(&buf, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        buf++; // увеличиваем значение

        // Вывод строго в порядке
        std::cout << "Process " << rank << " has buf = " << buf << std::endl;
        // printf("[%d]: receive message '%d'\n", rank, buf - 1);

        // Передаём дальше (или возвращаем 0-му, если это последний)
        int next = (rank == size - 1) ? 0 : rank + 1;
        MPI_Send(&buf, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
