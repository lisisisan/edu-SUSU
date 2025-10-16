#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
    int k;
    cin >> k; // количество нитей

#pragma omp parallel num_threads(k)
    {
        int tid = omp_get_thread_num();

// блок секций
#pragma omp sections
        {
#pragma omp section
            {
                cout << "[" << tid << "]: came in section 1\n"
                     << endl;
            }
#pragma omp section
            {
                cout << "[" << tid << "]: came in section 2\n"
                     << endl;
            }
#pragma omp section
            {
                cout << "[" << tid << "]: came in section 3\n"
                     << endl;
            }
        }

        // вывод вне секций
        cout << "[" << tid << "]: parallel region" << endl;
    }

    return 0;
}
