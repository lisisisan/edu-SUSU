#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
    int k;
    cout << "Enter number of threads: ";
    cin >> k;

#pragma omp parallel num_threads(k)
    {
        int id = omp_get_thread_num();     // номер текущего потока
        int total = omp_get_num_threads(); // всего потоков
        printf("I am %d thread from %d threads!\n", id, total);
    }
    return 0;
}
