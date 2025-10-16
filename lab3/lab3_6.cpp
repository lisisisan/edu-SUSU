#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

int main()
{
    int k, N;
    cin >> k >> N;

    int sum = 0;
    vector<int> partial(k, 0); // массив для частичных сумм

#pragma omp parallel num_threads(k) reduction(+ : sum)
    {
        int id = omp_get_thread_num();

// каждый поток считает только свои числа
#pragma omp for
        for (int i = 1; i <= N; i++)
        {
            partial[id] += i;
            sum += i;
        }
    }

    // выводим частичные суммы
    for (int i = 0; i < k; i++)
    {
        printf("[%d]: Sum = %d\n", i, partial[i]);
    }

    cout << "Sum = " << sum << endl;
    return 0;
}
