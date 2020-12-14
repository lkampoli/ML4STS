#include "stdio.h"

int change_integer(int *n);

int main(void)
{
    int a;
    float b;
    char c[30];

    printf("Please input an integer: ");
    scanf("%d", &a);

    printf("You new integer is: %d\n", change_integer(&a));

    return 0;
}
