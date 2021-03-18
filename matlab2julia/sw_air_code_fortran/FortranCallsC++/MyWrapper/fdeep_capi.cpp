#include "fdeep.h"
#include "fdeep.hpp" // original from frugally deep library

#include <iostream>
using namespace std;

FDEEP* create_fdeep(int a, int b){
    cout << "C API, create_fdeep" << endl;
    return new Fdeep(a, b);
}

void delete_fdeep(FDEEP* fdeep){
    cout << "C API, delete_fdeep" << endl;
    delete fdeep;
}

int fdeep_load(const FDEEP* fdeep, int c){
    return fdeep->load(c);
}

double fdeep_predict(const FDEEP* foo, double d){
    return fdeep->predict(d);
}

void fdeep_speaker(const char* s) {
    fdeep_speaker(string(s));
}
