#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
    int N;
    cout << "Enter N: ";
    cin >> N;

    int sum = 0;

#pragma omp parallel num_threads(2) reduction(+ : sum)
    {
        int id = omp_get_thread_num();
        int local_sum = 0;

        if (id == 0)
        {
            for (int i = 1; i <= N / 2; i++)
            {
                local_sum += i;
            }
        }
        else
        {
            for (int i = N / 2 + 1; i <= N; i++)
            {
                local_sum += i;
            }
        }

        printf("[%d]: Sum = %d\n", id, local_sum);
        sum += local_sum; // благодаря reduction здесь нет гонок
    }

    cout << "Sum = " << sum << endl;
    return 0;
}
