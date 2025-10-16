#include <stdio.h>
#include <omp.h>

int main()
{
    int k = 5, N = 10;
    int sum = 0;

#pragma omp parallel num_threads(k)
    {
#pragma omp for schedule(dynamic, 2) reduction(+ : sum)
        for (int i = 1; i <= N; i++)
        {
            int tid = omp_get_thread_num();
            printf("[%d]: calculation of the iteration number %d\n", tid, i);
            sum += i;
        }
    }

    printf("Sum = %d\n", sum);
    return 0;
}
