#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
    int k, N;
    cout << "Enter number of threads (k): ";
    cin >> k;
    cout << "Enter N: ";
    cin >> N;

    int sum = 0;

#pragma omp parallel num_threads(k) reduction(+ : sum)
    {
        int id = omp_get_thread_num();
        int local_sum = 0;

        int chunk = N / k;                              // размер куска
        int start = id * chunk + 1;                     // начало диапазона
        int end = (id == k - 1) ? N : (id + 1) * chunk; // чтобы последний поток досчитал до конца

        for (int i = start; i <= end; i++)
        {
            local_sum += i;
        }

        printf("[%d]: Sum = %d\n", id, local_sum);
        sum += local_sum;
    }

    cout << "Sum = " << sum << endl;
    return 0;
}
