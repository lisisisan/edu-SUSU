#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
// Устанавливаем количество потоков = 4
#pragma omp parallel
    {
        printf("Hello World!\n");
    }
    return 0;
}

/*
/opt/homebrew/bin/g++-15 -fopenmp \
  -I/opt/homebrew/opt/libomp/include \
  -L/opt/homebrew/opt/libomp/lib \
  lab1_2.cpp -o  lab1_2
*/
