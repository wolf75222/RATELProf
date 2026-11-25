#include <omp.h>

int main(void)
{
    int sum = 0;
    omp_get_initial_device();
    omp_get_num_devices();

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 10; ++i)
        sum += i;

    int offloaded = 0;
    #pragma omp target map(tofrom:sum, offloaded)
    {
        sum += 5;
        offloaded = !omp_is_initial_device();
    }
    return sum != 50 || !offloaded;
}
