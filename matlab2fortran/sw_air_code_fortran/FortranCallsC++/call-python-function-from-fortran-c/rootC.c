#include "rootC.h"
#include <Python.h>
#include <stdlib.h>

void Initialize ()
{
    Py_Initialize();
}

void Finalize ()
{
    Py_Finalize();
}

void root_(double* A, double* B, double* t, double* x)
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue, *sys, *path;
    static int i;

    if (i == 0)
    {
        ++i;
        Initialize();
        atexit(Finalize);
    }

    sys  = PyImport_ImportModule("sys");
    path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyString_FromString("."));

    pName = PyString_FromString("rootPY");
    pModule = PyImport_Import(pName);

    if (!pModule)
    {
        PyErr_Print();
        printf("ERROR in pModule\n");
        exit(1);
    }

    pFunc = PyObject_GetAttrString(pModule, "root");
    pArgs = PyTuple_New(3);
    PyTuple_SetItem(pArgs, 0, PyFloat_FromDouble((*A)));
    PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble((*B)));
    PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble((*t)));

    pValue = PyObject_CallObject(pFunc, pArgs);
    *x     = PyFloat_AsDouble(pValue);
}

