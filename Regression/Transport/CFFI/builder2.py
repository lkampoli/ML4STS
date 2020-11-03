import cffi
ffibuilder = cffi.FFI()

header = """
extern void add_one (double *);
"""

module = """
import numpy as np
from my_plugin import ffi
import my_module

@ffi.def_extern()
def add_one(a_ptr):
    a = my_module.asarray(ffi, a_ptr, shape=(10,))
    a[:] += 1
"""

with open("plugin.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_plugin", r'''
    #include "plugin.h" ''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="libplugin.so", verbose=True)
