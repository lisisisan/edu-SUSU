#include <mpi.h>
#include <cstdio>

int main(int argc, char **argv)
{
    // 2. Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank; // 3. Номер процесса
    int size; // 4. Количество процессов

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buf; // 5. Сообщение ("эстафетная палочка")

    // 6. Разделение кода для процесса 0 и остальных
    if (rank == 0)
    {
        // Процесс 0 начинает передачу
        buf = 0;

        // Отправляем "эстафетную палочку" процессу 1
        MPI_Send(&buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Ждём сообщение от последнего процесса (size-1)
        MPI_Recv(&buf, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 7. Вывод результата
        printf("[%d]: receive message '%d'\n", rank, buf);
    }
    else
    {
        // Остальные процессы
        // Ждём сообщение от предыдущего
        MPI_Recv(&buf, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Увеличиваем сообщение на 1
        buf++;

        // Отправляем следующему процессу (по кругу)
        MPI_Send(&buf, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

        // Выводим сообщение
        printf("[%d]: receive message '%d'\n", rank, buf - 1);
    }

    // Завершение MPI
    MPI_Finalize();
    return 0;
}
