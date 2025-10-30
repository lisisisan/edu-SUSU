// ===========================================
// Алгоритм: Последовательная (линейная) реализация сортировки по корзинам (Bucket Sort)
//
// ===========================================
//
// Компиляция
// g++ -O3 bucket_sort_seq.cpp -o bucket_sort_seq
//
// Запуск
// ./bucket_sort_seq -n 10000000 -b 16 -s 42
//
// -------------------------------------------
// Параметры:
// -n <N> : количество элементов для сортировки (по умолчанию 1 000 000)
// -b <B> : количество бакетов (корзин) (по умолчанию 16)
// -s <seed> : зерно генератора случайных чисел (по умолчанию 42)
// -v : режим подробного вывода (verbose / debug)
//
// Алгоритм:
// 1. Генерация случайных чисел в диапазоне [0, 1).
// 2. Распределение элементов по корзинам (buckets).
// 3. Сортировка каждой корзины отдельно (std::sort).
// 4. Слияние всех корзин в единый отсортированный массив.
// 5. Проверка корректности сортировки.
// ===========================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

int main(int argc, char **argv)
{
    // ---------- ПАРАМЕТРЫ ПО УМОЛЧАНИЮ ----------
    long long N = 1000000;  // количество элементов для сортировки
    int total_buckets = 16; // количество корзин
    unsigned int seed = 42; // зерно генератора случайных чисел
    bool verbose = false;   // режим подробного вывода

    // ---------- ЧТЕНИЕ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ ----------
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            N = atoll(argv[++i]);
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc)
            total_buckets = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
            seed = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)
            verbose = true;
        else
        {
            printf("Неизвестный аргумент: %s\n", argv[i]);
            return 1;
        }
    }

    // ---------- ИНФОРМАЦИЯ О ПАРАМЕТРАХ ----------
    printf("=== Последовательная сортировка по корзинам (Bucket Sort) ===\n");
    printf("Количество элементов (N): %lld\n", N);
    printf("Количество корзин (buckets): %d\n", total_buckets);
    printf("Зерно (seed): %u\n", seed);
    printf("--------------------------------------------------------------\n");

    // ---------- ЭТАП 1: ГЕНЕРАЦИЯ СЛУЧАЙНЫХ ДАННЫХ ----------
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<double> data(N);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (long long i = 0; i < N; ++i)
        data[i] = dist(rng);

    auto t_gen = std::chrono::high_resolution_clock::now();
    if (verbose)
        printf("[DEBUG] Сгенерировано %lld случайных чисел.\n", N);

    // ---------- ЭТАП 2: РАСПРЕДЕЛЕНИЕ ЭЛЕМЕНТОВ ПО КОРЗИНАМ ----------
    double bucket_width = 1.0 / total_buckets;
    std::vector<std::vector<double>> buckets(total_buckets);

    for (auto &x : data)
    {
        int idx = (int)std::floor(x / bucket_width);
        if (idx >= total_buckets)
            idx = total_buckets - 1; // защита
        buckets[idx].push_back(x);
    }

    auto t_dist = std::chrono::high_resolution_clock::now();

    if (verbose)
    {
        printf("[DEBUG] Распределение по корзинам завершено.\n");
        for (int i = 0; i < total_buckets; ++i)
            printf("  Корзина %d: %zu элементов.\n", i, buckets[i].size());
    }

    // ---------- ЭТАП 3: СОРТИРОВКА КАЖДОЙ КОРЗИНЫ ----------
    for (int i = 0; i < total_buckets; ++i)
        std::sort(buckets[i].begin(), buckets[i].end());

    auto t_sort = std::chrono::high_resolution_clock::now();
    if (verbose)
        printf("[DEBUG] Каждая корзина отсортирована.\n");

    // ---------- ЭТАП 4: СЛИЯНИЕ ВСЕХ КОРЗИН В ОДИН МАССИВ ----------
    std::vector<double> sorted;
    sorted.reserve(N);
    for (int i = 0; i < total_buckets; ++i)
        sorted.insert(sorted.end(), buckets[i].begin(), buckets[i].end());

    auto t_merge = std::chrono::high_resolution_clock::now();

    // ---------- ЭТАП 5: ПРОВЕРКА КОРРЕКТНОСТИ ----------
    bool ok = true;
    for (size_t i = 1; i < sorted.size(); ++i)
        if (sorted[i - 1] > sorted[i] + 1e-15)
        {
            ok = false;
            break;
        }

    // ---------- ЭТАП 6: ВЫЧИСЛЕНИЕ ВРЕМЕНИ ----------
    double time_gen = std::chrono::duration<double>(t_gen - t_start).count();
    double time_dist = std::chrono::duration<double>(t_dist - t_gen).count();
    double time_sort = std::chrono::duration<double>(t_sort - t_dist).count();
    double time_merge = std::chrono::duration<double>(t_merge - t_sort).count();
    double time_total = std::chrono::duration<double>(t_merge - t_start).count();

    // ---------- ЭТАП 7: ВЫВОД РЕЗУЛЬТАТОВ ----------
    printf("Время генерации: %.6f с\n", time_gen);
    printf("Время распределения: %.6f с\n", time_dist);
    printf("Время сортировки корзин: %.6f с\n", time_sort);
    printf("Время слияния: %.6f с\n", time_merge);
    printf("Полное время: %.6f с\n", time_total);
    printf("Проверка сортировки: %s\n", ok ? "OK" : "Ошибка");
    printf("==============================================================\n");

    return 0;
}
