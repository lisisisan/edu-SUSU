#include <iostream>
#include <omp.h>
#include <unistd.h>
using namespace std;

int main()
{
    int rank = 0;
    omp_set_num_threads(15);

    double start_time = omp_get_wtime(); // старт времени

#pragma omp parallel shared(rank)
    {
        rank = omp_get_thread_num();
        usleep(8000); // имитация работы
        printf("I am %d thread.\n", rank);
    }

    double end_time = omp_get_wtime(); // конец времени
    printf("Final rank: %d\n", rank);
    printf("Execution time: %.6f seconds\n", end_time - start_time);

    return 0;
}
