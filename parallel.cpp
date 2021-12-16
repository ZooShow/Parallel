#include <omp.h>
#include <thread>
#include <vector>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <barrier>
#include <iostream>
#include <type_traits>

#define STEPS 100000000
#define CACHE_LINE_SIZE 64u

/* Compile time assert macro */
#define ASSERT_CONCAT_(a,b) a##b
#define ASSERT_CONCAT(a,b) ASSERT_CONCAT_(a,b)
#define ct_assert(e) enum {ASSERT_CONCAT(asssert_line_, __LINE__) = 1/(!!(e))}

static unsigned num_treads = std::thread::hardware_concurrency();
unsigned get_num_threads()
{
    return num_treads;
}

void set_num_threads(unsigned t)
{
    num_treads = t;
    omp_set_num_threads(t);
}

struct partial_sum
{
    alignas(64) double Value;
};
ct_assert(sizeof(partial_sum) == CACHE_LINE_SIZE);

typedef double (*function)(double);
/*
double IntegrateCppMutex(double a, double b, function F)
{
    using namespace std;
    vector<thread> Threads;
    mutex Mutex;
    unsigned int T = thread::hardware_concurrency();
    double dx = (b-a)/STEPS;
    double Result = 0;
    for(unsigned int t = 0; t < T; t++)
    {
        Threads.emplace_back([=, &Result, &Mutex](){
            double Accum = 0;
            for(unsigned int i = t; i < STEPS; i+=T)
                Accum += Function(dx*i + a);
        }
    }
    {
        unsigned int t = (unsigned int)omp_get_thread_num();
        unsigned int T = (unsigned int)omp_get_num_threads();
        for(unsigned int i = t; i < STEPS; i+=T)
#pragma omp critical
        Result += Accum;
    }
    Result *= dx;
    return Result;
}
*/

typedef double (*unary_function)(double);

double Identity(double x)
{
    return x;
}

double Linear(double x)
{
    return 5 * x;
}

double Quadratic(double x)
{
    return x * x;
}


double IntegratePS(function F, double a, double b)
{
    double Result = 0;
    double dx = (b - a) / STEPS;
    unsigned int T = std::thread::hardware_concurrency();
    auto Vec = std::vector(T, partial_sum{ 0.0 });
    std::vector<std::thread> Threads;

    auto ThreadProcedure = [dx, T, F, a, &Vec](auto t)
    {
        for (auto i = t; i < STEPS; i += T)
            Vec[t].Value += F(dx * i + a);
    };

    for (unsigned t = 1; t < T; t++)
        Threads.emplace_back(ThreadProcedure, t);

    ThreadProcedure(0);
    for (auto& Thread : Threads)
        Thread.join();

    for (auto Elem : Vec)
        Result += Elem.Value;

    Result *= dx;
    return Result;
}

typedef struct experiment_result_
{
    double Result;
    double TimeMS;
} experiment_result;


typedef double (*IntegrateFunction) (function, double, double);
experiment_result RunExperiment(IntegrateFunction I)
{
    double t0 = omp_get_wtime();
    double res = I(Quadratic, -1, 1);
    double t1 = omp_get_wtime();

    experiment_result Result;
    Result.Result = res;
    Result.TimeMS = t1 - t0;

    return Result;
}

void ShowExperimentResult(IntegrateFunction I)
{
    printf("%10s, %10s %10sms\n", "Threads", "Result", "TimeMS");
    for (unsigned T = 1; T <= omp_get_num_procs(); T++)
    {
        experiment_result Experiment;
        omp_set_num_threads(T);
        Experiment = RunExperiment(I);
        printf("%10d, %10g %10gms\n", T, Experiment.Result, Experiment.TimeMS);
    }
    printf("\n");
}

double IntegrateAlign(function Function, double a, double b)
{
    unsigned int T;
    double Result = 0;
    double dx = (b - a) / STEPS;
    partial_sum* Accum = 0;

#pragma omp parallel shared(Accum, T)
    {
        unsigned int t = (unsigned int)omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned int)omp_get_num_threads();
            Accum = (partial_sum*)_aligned_malloc(T * sizeof(*Accum), CACHE_LINE_SIZE);
            memset(Accum, 0, T * sizeof(*Accum));
        }

        for (unsigned int i = t; i < STEPS; i += T)
            Accum[t].Value += Function(dx * i + a);
    }

    for (unsigned int i = 0; i < T; i++)
        Result += Accum[i].Value;

    Result *= dx;
    _aligned_free(Accum);
    return Result;
}


double IntegrateParallel(function Function, double a, double b)
{
    double Result = 0;
    double dx = (b - a) / STEPS;

#pragma omp parallel
    {
        double Accum = 0;
        unsigned int t = (unsigned int)omp_get_thread_num();
        unsigned int T = (unsigned int)omp_get_num_threads();
        for (unsigned int i = t; i < STEPS; i += T)
            Accum += Function(dx * i + a);
#pragma omp critical
        Result += Accum;
    }

    Result *= dx;
    return Result;
}

double IntegrateFalseSharing(function Function, double a, double b)
{
    unsigned int T;
    double Result = 0;
    double dx = (b - a) / STEPS;
    double* Accum = 0;

#pragma omp parallel shared(Accum, T)
    {
        unsigned int t = (unsigned int)omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned int)omp_get_num_threads();
            Accum = (double*)calloc(T, sizeof(double));
        }

        for (unsigned int i = t; i < STEPS; i += T)
            Accum[t] += Function(dx * i + a);
    }

    for (unsigned int i = 0; i < T; i++)
        Result += Accum[i];

    Result *= dx;
    free(Accum);
    return Result;
}

double IntegrateReduction(function Function, double a, double b)
{
    double Result = 0;
    double dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+:Result)
    for (int i = 0; i < STEPS; i++)
        Result += Function(dx * i + a);

    Result *= dx;
    return Result;
}


auto ceil_div(auto x, auto y)
{
    return (x + y - 1) / y;
}

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, std::size_t n, BinaryFn f, ElementType zero)
{
    unsigned T = get_num_threads();

    struct reduction_partial_result_t
    {
        alignas(std::hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
        std::vector<reduction_partial_result_t>(T, reduction_partial_result_t{ zero });
    constexpr std::size_t k = 2;
    std::barrier<> bar{ T };

    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        std::size_t Mt = K / T, it1 = K % T;
        if (t < it1)
            it1 = ++Mt * t;
        else
            it1 += Mt * t;
        it1 *= k;
        std::size_t mt = Mt * k;
        auto it2 = it1 + mt;
        ElementType accum = zero;
        for (std::size_t i = it1; i < it2; ++i)
            accum = f(accum, V[i]);
        reduction_partial_results[t].value = accum;

        std::size_t s = 1;
        while (s < T)
        {
            bar.arrive_and_wait();
            if ((t % (s * k)) == 0 && s + t < T)
            {
                reduction_partial_results[t].value = f(reduction_partial_results[t].value, reduction_partial_results[t + s].value);
                s *= k;
            }
        }
    };

    std::vector<std::thread> threads;
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();

    return reduction_partial_results[0].value;
}

template <class ElementType, class UnaryFn, class BinaryFn>
requires (
    std::is_invocable_r_v<ElementType, UnaryFn, ElementType>&&
    std::is_invocable_r_v<ElementType, BinaryFn, ElementType, ElementType>
    )
    ElementType reduce_range(ElementType a, ElementType b, std::size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero)
{
    unsigned T = get_num_threads();

    struct reduction_partial_result_t
    {
        alignas(std::hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
        std::vector<reduction_partial_result_t>(std::thread::hardware_concurrency(), reduction_partial_result_t{ zero });
    constexpr std::size_t k = 2;
    std::barrier<> bar{ (std::ptrdiff_t)T };

    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        double dx = (b - a) / n;
        std::size_t Mt = K / T, it1 = K % T;
        if (t < it1)
            it1 = ++Mt * t;
        else
            it1 += Mt * t;
        it1 *= k;
        std::size_t mt = Mt * k;
        auto it2 = it1 + mt;
        ElementType accum = zero;
        for (std::size_t i = it1; i < it2; ++i)
            accum = reduce_2(accum, get(a + i * dx));
        reduction_partial_results[t].value = accum;
        for (std::size_t s = 1u, s_next = 2u; s < T; s = s_next, s_next += s_next) //assume: k = 2
        {
            bar.arrive_and_wait();
            if ((t % s_next) == 0 && s + t < T)
                reduction_partial_results[t].value = reduce_2(reduction_partial_results[t].value, reduction_partial_results[t + s].value);
        }
    };

    std::vector<std::thread> threads;
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();

    return reduction_partial_results[0].value;
}


int main()
{
    /*printf("IntegratePS\n");
    ShowExperimentResult(IntegratePS);

    printf("IntegrateReduction\n");
    ShowExperimentResult(IntegrateReduction);

    printf("IntegrateAlign\n");
    ShowExperimentResult(IntegrateAlign);

    printf("IntegrateFalseSharing\n");
    ShowExperimentResult(IntegrateFalseSharing);*/

    set_num_threads(12);
    unsigned V[49];
    for (unsigned i = 0; i < std::size(V); ++i)
        V[i] = i + 1;
    std::cout << "Average: " << reduce_vector(V, std::size(V), [](auto x, auto y) {return x + y; }, 0u) / std::size(V) << '\n';
    return 0;
}