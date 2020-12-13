import cffi
ffibuilder = cffi.FFI()

header = """
extern void add_five (double *);
extern void minus_one (double *);
extern void k_dr_n2 (double *);
"""

module = """
from my_plugin import ffi
import my_module

@ffi.def_extern()
def add_five(a_ptr):
    a = my_module.asarray(ffi, a_ptr, shape=(10,))
    a[:] += 5

@ffi.def_extern()
def minus_one(a_ptr):
    a = my_module.asarray(ffi, a_ptr, shape=(10,))
    a[:] -= 1

@ffi.def_extern()
def k_dr_n2(a_ptr):
    a = my_module.k_dr_n2(a_ptr)
"""

with open("plugin.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_plugin", r''' #include "plugin.h" ''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="libplugin.so", verbose=True)
