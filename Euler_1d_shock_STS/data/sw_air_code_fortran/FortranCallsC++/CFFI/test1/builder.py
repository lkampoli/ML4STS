import cffi
ffibuilder = cffi.FFI()

header = """
extern void hello_world(void);
"""

module = """
from my_plugin import ffi
import numpy as np

@ffi.def_extern()
def hello_world():
    print("Hello World!")
"""

with open("plugin.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_plugin", r'''
    #include "plugin.h"
''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="libplugin.so", verbose=True)
