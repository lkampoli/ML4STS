#include "rootC.h"
#include <stdio.h>

int main()
{
    double A = 0.4, B = 0.3, t = 0.1, x = 0.0;

    root_(&A,&B,&t,&x);

    printf("A = %f, B = %f, t = %f\n", A, B, t);

    printf("x = %.15f\n", x);

    return 0;
}
