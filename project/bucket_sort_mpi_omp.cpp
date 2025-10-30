// ===========================================
// Алгоритм: Гибридная реализация сортировки по корзинам (Bucket Sort)
// с использованием MPI (распределение данных между процессами)
// и OpenMP (параллельная сортировка внутри каждого процесса)
//
// ===========================================
//
// export OMPI_CXX=/opt/homebrew/bin/g++-15
// mpicxx -fopenmp -O3 \
//   -I/opt/homebrew/opt/libomp/include \
//   -L/opt/homebrew/opt/libomp/lib \
//   -lomp \
//   bucket_sort_mpi_omp.cpp -o bucket_sort_mpi_omp
//
// Запуск
// mpirun -np 4 ./bucket_sort_mpi_omp -n 10000000 -b 4 -s 12345
//
// Параметры
// -n <N> : количество элементов для сортировки (по умолчанию 1 000 000)
// -b <B> : количество бакетов (корзин) на процесс (по умолчанию 4)
//          общее количество бакетов = B * числу процессов
// -s <seed> : зерно для генератора случайных чисел (по умолчанию 42)
// -v : режим подробного вывода (debug / verbose)
//
// Примечания
// - Сортируются вещественные числа в диапазоне [0, 1).
// - Каждый MPI-процесс получает свою долю бакетов.
// - Внутри каждого процесса используется OpenMP для ускорения сортировки.
// ===========================================

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    // ---------- ИНИЦИАЛИЗАЦИЯ MPI ----------
    MPI_Init(&argc, &argv); // Запускает MPI окружение
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Получаем номер текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // Получаем общее число процессов

    // ---------- ПАРАМЕТРЫ ПО УМОЛЧАНИЮ ----------
    long long N = 1000000;    // Общее количество элементов для сортировки
    int buckets_per_proc = 4; // Количество бакетов (корзин) на процесс
    unsigned int seed = 42;   // Зерно для генерации случайных чисел
    bool verbose = false;     // Флаг для подробного вывода

    // ---------- ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ ----------
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "-n" && i + 1 < argc)
            N = atoll(argv[++i]); // Общее количество элементов
        else if (arg == "-b" && i + 1 < argc)
            buckets_per_proc = atoi(argv[++i]); // Количество бакетов на процесс
        else if (arg == "-s" && i + 1 < argc)
            seed = (unsigned int)atoi(argv[++i]); // Зерно для генератора
        else if (arg == "-v")
            verbose = true; // Подробный режим
        else
        {
            if (rank == 0)
                printf("Неизвестный аргумент: %s\n", arg.c_str());
            MPI_Finalize();
            return 1;
        }
    }

    // ---------- ЗАМЕР ВРЕМЕНИ ----------
    double t_start_all = MPI_Wtime(); // Начало общего времени

    // ---------- ГЕНЕРАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ (только процесс 0) ----------
    vector<double> data;
    if (rank == 0)
    {
        data.resize(N);
        mt19937_64 rng(seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        for (long long i = 0; i < N; ++i)
            data[i] = dist(rng);

        if (verbose)
            printf("[rank 0] Сгенерировано %lld случайных чисел\n", N);
    }

    MPI_Barrier(MPI_COMM_WORLD);      // Синхронизация всех процессов
    double t_after_gen = MPI_Wtime(); // Отметка после генерации

    // ---------- ВЫЧИСЛЕНИЕ ГРАНИЦ БАКЕТОВ ----------
    int total_buckets = buckets_per_proc * nprocs; // Всего корзин
    double bucket_width = 1.0 / total_buckets;     // Ширина диапазона одной корзины

    // ---------- РАСПРЕДЕЛЕНИЕ ДАННЫХ ПО ПРОЦЕССАМ ----------
    vector<vector<double>> send_bufs(nprocs); // Буферы для отправки каждому процессу

    if (rank == 0)
    {
        for (auto &x : data)
        {
            // Определяем номер корзины по значению
            int bucket_idx = (int)floor(x / bucket_width);
            if (bucket_idx >= total_buckets)
                bucket_idx = total_buckets - 1;

            // Определяем, какому процессу принадлежит эта корзина
            int target_proc = bucket_idx / buckets_per_proc;
            if (target_proc >= nprocs)
                target_proc = nprocs - 1;

            send_bufs[target_proc].push_back(x);
        }

        data.clear();         // Освобождаем память
        data.shrink_to_fit(); // Уменьшаем ёмкость контейнера до его размера
    }

    // ---------- ПЕРЕДАЧА КОЛИЧЕСТВА ЭЛЕМЕНТОВ ----------
    vector<int> recv_counts(nprocs, 0);
    if (rank == 0)
    {
        for (int p = 1; p < nprocs; ++p)
        {
            int count = (int)send_bufs[p].size();
            MPI_Send(&count, 1, MPI_INT, p, 100, MPI_COMM_WORLD);
        }
        recv_counts[0] = (int)send_bufs[0].size();
    }
    else
    {
        MPI_Recv(&recv_counts[rank], 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // ---------- ПЕРЕДАЧА ДАННЫХ ----------
    vector<double> recv_local;
    if (rank == 0)
    {
        recv_local = send_bufs[0];
        for (int p = 1; p < nprocs; ++p)
        {
            if (!send_bufs[p].empty())
                MPI_Send(send_bufs[p].data(), send_bufs[p].size(), MPI_DOUBLE, p, 101, MPI_COMM_WORLD);
        }
    }
    else
    {
        int cnt = recv_counts[rank];
        recv_local.resize(cnt);
        if (cnt > 0)
            MPI_Recv(recv_local.data(), cnt, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_after_distribution = MPI_Wtime(); // Время после рассылки

    // ---------- ЛОКАЛЬНАЯ СОРТИРОВКА С ИСПОЛЬЗОВАНИЕМ OpenMP ----------
    double local_start_sort = MPI_Wtime();

    int omp_threads = 1;
#pragma omp parallel
    {
#pragma omp master
        omp_threads = omp_get_num_threads(); // Определяем количество потоков
    }

    size_t L = recv_local.size();
    vector<double> local_sorted;

    if (L > 0)
    {
        int nthreads = omp_threads;
        vector<size_t> starts(nthreads), ends(nthreads);

        // Делим массив на участки между потоками
        for (int i = 0; i < nthreads; ++i)
        {
            starts[i] = L * i / nthreads;
            ends[i] = (i + 1 == nthreads) ? L : L * (i + 1) / nthreads;
        }

        // Параллельная сортировка участков
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            sort(recv_local.begin() + starts[tid], recv_local.begin() + ends[tid]);
        }

        // Последовательное слияние отсортированных участков
        local_sorted.assign(recv_local.begin() + starts[0], recv_local.begin() + ends[0]);
        for (int i = 1; i < nthreads; ++i)
        {
            vector<double> tmp;
            tmp.reserve(local_sorted.size() + (ends[i] - starts[i]));
            merge(local_sorted.begin(), local_sorted.end(),
                  recv_local.begin() + starts[i], recv_local.begin() + ends[i],
                  back_inserter(tmp));
            local_sorted.swap(tmp);
        }
    }

    double local_sort_time = MPI_Wtime() - local_start_sort;

    // ---------- СБОР РЕЗУЛЬТАТОВ НА ROOT ----------
    int local_size = (int)local_sorted.size();
    vector<int> all_sizes(nprocs);
    MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(nprocs);
    vector<double> final_sorted;

    if (rank == 0)
    {
        displs[0] = 0;
        for (int i = 1; i < nprocs; ++i)
            displs[i] = displs[i - 1] + all_sizes[i - 1];

        int total = displs[nprocs - 1] + all_sizes[nprocs - 1];
        final_sorted.resize(total);
    }

    MPI_Gatherv(local_sorted.data(), local_size, MPI_DOUBLE,
                final_sorted.data(), all_sizes.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end_all = MPI_Wtime(); // Время окончания сортировки

    // ---------- ВЫВОД РЕЗУЛЬТАТОВ ----------
    if (rank == 0)
    {
        printf("=== РЕЗУЛЬТАТЫ Bucket Sort (MPI + OpenMP) ===\n");
        printf("Элементов: %lld\n", N);
        printf("Процессов (MPI): %d, бакетов на процесс: %d, всего бакетов: %d\n",
               nprocs, buckets_per_proc, total_buckets);
        printf("Потоков OpenMP на процесс: %d\n", omp_threads);
        printf("Время генерации: %.6f с\n", (t_after_gen - t_start_all));
        printf("Время распределения: %.6f с\n", (t_after_distribution - t_after_gen));
        printf("Время сортировки (локально): %.6f с\n", local_sort_time);
        printf("Полное время выполнения: %.6f с\n", (t_end_all - t_start_all));

        // Проверка корректности сортировки
        bool ok = true;
        for (size_t i = 1; i < final_sorted.size(); ++i)
        {
            if (final_sorted[i - 1] > final_sorted[i])
            {
                ok = false;
                break;
            }
        }

        printf("Проверка сортировки: %s\n", ok ? "OK" : "Ошибка");

        printf("Размеры данных по процессам: ");
        for (int p = 0; p < nprocs; ++p)
        {
            printf("%d%s", all_sizes[p], (p + 1 < nprocs) ? ", " : "\n");
        }
        printf("==============================================\n");
    }

    MPI_Finalize(); // Завершаем работу MPI
    return 0;
}
