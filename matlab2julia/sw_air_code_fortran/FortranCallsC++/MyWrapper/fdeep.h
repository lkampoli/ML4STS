#ifdef __cplusplus // Are we compiling this with a C++ compiler ?
extern "C" {
    class Fdeep;
    typedef Fdeep FDEEP;
#else
    // From the C side, we use an opaque pointer.
    typedef struct FDEEP FDEEP;
#endif

// Constructor
FDEEP* create_fdeep(int a, int b);

// Destructor
void delete_fdeep(FDEEP* fdeep);

// The const qualificators maps from the member function to pointers to the class instances.
int fdeep_load(const FDEEP* fdeep, int c);
double fdeep_predict(const FDEEP* fdeep, double d);

void fdeep_speaker(const char* s);

#ifdef __cplusplus
}
#endif
