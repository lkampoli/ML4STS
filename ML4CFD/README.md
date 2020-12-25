# TensorFlow in other languages

## Background

This document is intended as a guide for those interested in the creation or
development of TensorFlow functionality in other programming languages. It
describes the features of TensorFlow and recommended steps for making the same
available in other programming languages.

Python was the first client language supported by TensorFlow and currently
supports the most features. More and more of that functionality is being moved
into the core of TensorFlow (implemented in C++) and exposed via a [C API].
Client languages should use the language's [foreign function interface
(FFI)](https://en.wikipedia.org/wiki/Foreign_function_interface) to call into
this [C API] to provide TensorFlow functionality.

## Overview

Providing TensorFlow functionality in a programming language can be broken down
into broad categories:

-   *Run a predefined graph*: Given a `GraphDef` (or
    `MetaGraphDef`) protocol message, be able to create a session, run queries,
    and get tensor results. This is sufficient for a mobile app or server that
    wants to run inference on a pre-trained model.
-   *Graph construction*: At least one function per defined
    TensorFlow op that adds an operation to the graph. Ideally these functions
    would be automatically generated so they stay in sync as the op definitions
    are modified.
-   *Gradients (AKA automatic differentiation)*: Given a graph and a list of
    input and output operations, add operations to the graph that compute the
    partial derivatives (gradients) of the inputs with respect to the outputs.
    Allows for customization of the gradient function for a particular operation
    in the graph.
-   *Functions*: Define a subgraph that may be called in multiple places in the
    main `GraphDef`. Defines a `FunctionDef` in the `FunctionDefLibrary`
    included in a `GraphDef`.
-   *Control Flow*: Construct "If" and "While" with user-specified subgraphs.
    Ideally these work with gradients (see above).
-   *Neural Network library*: A number of components that together support the
    creation of neural network models and training them (possibly in a
    distributed setting). While it would be convenient to have this available in
    other languages, there are currently no plans to support this in languages
    other than Python. These libraries are typically wrappers over the features
    described above.

At a minimum, a language binding should support running a predefined graph, but
most should also support graph construction. The TensorFlow Python API provides
all these features.

## Current Status

New language support should be built on top of the [C API]. However, as you can
see in the table below, not all functionality is available in C yet. Providing
more functionality in the [C API] is an ongoing project.

Feature                                        | Python                                                      | C
:--------------------------------------------- | :---------------------------------------------------------- | :--
Run a predefined Graph                         | `tf.import_graph_def`, `tf.Session`                         | `TF_GraphImportGraphDef`, `TF_NewSession`
Graph construction with generated op functions | Yes                                                         | Yes (The C API supports client languages that do this)
Gradients                                      | `tf.gradients`                                              |
Functions                                      | `tf.python.framework.function.Defun`                        |
Control Flow                                   | `tf.cond`, `tf.while_loop`                                  |
Neural Network library                         | `tf.train`, `tf.nn`, `tf.contrib.layers`, `tf.contrib.slim` |

## Recommended Approach

### Run a predefined graph

A language binding is expected to define the following classes:

-   `Graph`: A graph representing a TensorFlow computation. Consists of
    operations (represented in the client language by `Operation`s) and
    corresponds to a `TF_Graph` in the C API. Mainly used as an argument when
    creating new `Operation` objects and when starting a `Session`. Also
    supports iterating through the operations in the graph
    (`TF_GraphNextOperation`), looking up operations by name
    (`TF_GraphOperationByName`), and converting to and from a `GraphDef`
    protocol message (`TF_GraphToGraphDef` and `TF_GraphImportGraphDef` in the C
    API).
-   `Operation`: Represents a computation node in the graph. Corresponds to a
    `TF_Operation` in the C API.
-   `Output`: Represents one of the outputs of an operation in the graph. Has a
    `DataType` (and eventually a shape). May be passed as an input argument to a
    function for adding operations to a graph, or to a `Session`'s `Run()`
    method to fetch that output as a tensor. Corresponds to a `TF_Output` in the
    C API.
-   `Session`: Represents a client to a particular instance of the TensorFlow
    runtime. Its main job is to be constructed with a `Graph` and some options
    and then field calls to `Run()` the graph. Corresponds to a `TF_Session` in
    the C API.
-   `Tensor`: Represents an N-dimensional (rectangular) array with elements all
    the same `DataType`. Gets data into and out of a `Session`'s `Run()` call.
    Corresponds to a `TF_Tensor` in the C API.
-   `DataType`: An enumerant with all the possible tensor types supported by
    TensorFlow. Corresponds to `TF_DataType` in the C API and often referred to
    as `dtype` in the Python API.

### Graph construction

TensorFlow has many ops, and the list is not static, so we recommend generating
the functions for adding ops to a graph instead of writing them by individually
by hand (though writing a few by hand is a good way to figure out what the
generator should generate). The information needed to generate a function is
contained in an `OpDef` protocol message.

There are a few ways to get a list of the `OpDef`s for the registered ops:

-   `TF_GetAllOpList` in the C API retrieves all registered `OpDef` protocol
    messages. This can be used to write the generator in the client language.
    This requires that the client language have protocol buffer support in order
    to interpret the `OpDef` messages.
-   The C++ function `OpRegistry::Global()->GetRegisteredOps()` returns the same
    list of all registered `OpDef`s (defined in
    [`tensorflow/core/framework/op.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op.h)). This can be used to write the generator
    in C++ (particularly useful for languages that do not have protocol buffer
    support).
-   The ASCII-serialized version of that list is periodically checked in to
    [`tensorflow/core/ops/ops.pbtxt`](https://www.tensorflow.org/code/tensorflow/core/ops/ops.pbtxt) by an automated process.

The `OpDef` specifies the following:

-   Name of the op in CamelCase. For generated functions follow the conventions
    of the language. For example, if the language uses snake_case, use that
    instead of CamelCase for the op's function name.
-   A list of inputs and outputs. The types for these may be polymorphic by
    referencing attributes, as described in the inputs and outputs section of
    [Adding an     op](./op.md).
-   A list of attributes, along with their default values (if any). Note that
    some of these will be inferred (if they are determined by an input), some
    will be optional (if they have a default), and some will be required (no
    default).
-   Documentation for the op in general and the inputs, outputs, and
    non-inferred attributes.
-   Some other fields that are used by the runtime and can be ignored by the
    code generators.

An `OpDef` can be converted into the text of a function that adds that op to the
graph using the `TF_OperationDescription` C API (wrapped in the language's FFI):

-   Start with `TF_NewOperation()` to create the `TF_OperationDescription*`.
-   Call `TF_AddInput()` or `TF_AddInputList()` once per input (depending on
    whether the input has a list type).
-   Call `TF_SetAttr*()` functions to set non-inferred attributes. May skip
    attributes with defaults if you don't want to override the default value.
-   Set optional fields if necessary:
    -   `TF_SetDevice()`: force the operation onto a specific device.
    -   `TF_AddControlInput()`: add requirements that another operation finish
        before this operation starts running
    -   `TF_SetAttrString("_kernel")` to set the kernel label (rarely used)
    -   `TF_ColocateWith()` to colocate one op with another
-   Call `TF_FinishOperation()` when done. This adds the operation to the graph,
    after which it can't be modified.

The existing examples run the code generator as part of the build process (using
a Bazel genrule). Alternatively, the code generator can be run by an automated
cron process, possibly checking in the result. This creates a risk of divergence
between the generated code and the `OpDef`s checked into the repository, but is
useful for languages where code is expected to be generated ahead of time like
`go get` for Go and `cargo ops` for Rust. At the other end of the spectrum, for
some languages the code could be generated dynamically from
[`tensorflow/core/ops/ops.pbtxt`](https://www.tensorflow.org/code/tensorflow/core/ops/ops.pbtxt).

#### Handling Constants

Calling code will be much more concise if users can provide constants to input
arguments. The generated code should convert those constants to operations that
are added to the graph and used as input to the op being instantiated.

#### Optional parameters

If the language allows for optional parameters to a function (like keyword
arguments with defaults in Python), use them for optional attributes, operation
names, devices, control inputs etc. In some languages, these optional parameters
can be set using dynamic scopes (like "with" blocks in Python). Without these
features, the library may resort to the "builder pattern", as is done in the C++
version of the TensorFlow API.

#### Name scopes

It is a good idea to have support for naming graph operations using some sort of
scoping hierarchy, especially considering the fact that TensorBoard relies on it
to display large graphs in a reasonable way. The existing Python and C++ APIs
take different approaches: In Python, the "directory" part of the name
(everything up to the last "/") comes from `with` blocks. In effect, there is a
thread-local stack with the scopes defining the name hierarchy. The last
component of the name is either supplied explicitly by the user (using the
optional `name` keyword argument) or defaults to the name of the type of the op
being added. In C++ the "directory" part of the name is stored in an explicit
`Scope` object. The `NewSubScope()` method appends to that part of the name and
returns a new `Scope`. The last component of the name is set using the
`WithOpName()` method, and like Python defaults to the name of the type of op
being added. `Scope` objects are explicitly passed around to specify the name of
the context.

#### Wrappers

It may make sense to keep the generated functions private for some ops so that
wrapper functions that do a little bit of additional work can be used instead.
This also gives an escape hatch for supporting features outside the scope of
generated code.

One use of a wrapper is for supporting `SparseTensor` input and output. A
`SparseTensor` is a tuple of 3 dense tensors: indices, values, and shape. values
is a vector size [n], shape is a vector size [rank], and indices is a matrix
size [n, rank]. There are some sparse ops that use this triple to represent a
single sparse tensor.

Another reason to use wrappers is for ops that hold state. There are a few such
ops (e.g. a variable) that have several companion ops for operating on that
state. The Python API has classes for these ops where the constructor creates
the op, and methods on that class add operations to the graph that operate on
the state.

#### Other Considerations

-   It is good to have a list of keywords used to rename op functions and
    arguments that collide with language keywords (or other symbols that will
    cause trouble, like the names of library functions or variables referenced
    in the generated code).
-   The function for adding a `Const` operation to a graph typically is a
    wrapper since the generated function will typically have redundant
    `DataType` inputs.

### Gradients, functions and control flow

At this time, support for gradients, functions and control flow operations ("if"
and "while") is not available in languages other than Python. This will be
updated when the [C API] provides necessary support.

[C API]: https://www.tensorflow.org/code/tensorflow/c/c_api.h




The standard Fortran interoperability with C consists of the `BIND(C)` attribute, which enables C calling conventions and changes symbol names and the ISO_C_BINDING module , which provides access to named constants that represent kind type parameters of data representations compatible with C types, the derived type C_PTR corresponding to any C data pointer type, the derived type C_FUNPTR corresponding to any C function pointer type, and four procedures.

    Introduction
    Fortran 2003 provides a standardized mechanism for interoperating with C. This support is widely available in Fortran 95 compilers that partially implement Fortran 2003. This support covers:

        interoperability of procedures - a means of referencing procedures that are defined in the C programming language, or that can be represented by C language prototypes, and a means of specifying that a procedure defined in Fortran can be called from C;

        interoperability of types - a means of declaring types and enumerations in Fortran that correspond to C types;

        interoperability of global data objects - a means of declaring global variables that are associated with C variables with external linkage;

        an intrinsic module (ISO_C_BINDING) that provides access to named constants and procedures relevant to C interoperability.

Clearly, any interoperable entity must be such that equivalent declarations of it may be made in the two languages. This is enforced within the Fortran program by requiring all such entities to be interoperable. We will explain in turn what this requires for types, variables, and procedures. We finish with two examples.

    Interoperability of intrinsic types
    There is an intrinsic module called ISO_C_BINDING that contains named constants holding kind type parameter values for intrinsic types. Their names are shown in Table 1, together with the corresponding C types. The processor is not required to support all of them. Lack of support is indicated with a negative value.

    Table 1. Interoperability between Fortran and C types

    Type        Named constant        C type or types
    INTEGER     C_INT                 int, signed int
                C_SHORT               short int, signed short int
                C_LONG                long int, signed long int
                C_LONG_LONG           long long int, signed long long int
                C_SIGNED_CHAR         signed char, unsigned char
                C_SIZE_T              size_t
                C_INT_LEAST8_T        int_least8_t
                C_INT_LEAST16_T       int_least16_t
                C_INT_LEAST32_T       int_least32_t
                C_INT_LEAST64_T       int_least64_t
                C_INT_FAST8_T         int_fast8_t
                C_INT_FAST16_T        int_fast16_t
                C_INT_FAST32_T        int_fast32_t
                C_INT_FAST64_T        int_fast64_t
                C_INTMAX_T            c intmax_t
    REAL        C_FLOAT               float, float _Imaginary
                C_DOUBLE              double, double _Imaginary
    COMPLEX     C_LONG_DOUBLE         long double, long double _Imaginary
                C_COMPLEX             _Complex
                C_DOUBLE_COMPLEX      double _Complex
                C_LONG_DOUBLE_COMPLEX long double _Complex
    LOGICAL     C_BOOL                _Bool
    CHARACTER   C_CHAR                char

    For character, interoperability also requires that the length type parameter be omitted or be specified by an initialization expression whose value is one. The following named constants (with the obvious meanings) are provided: C_NULL_CHAR, C_ALERT, C_BACKSPACE, C_FORM_FEED, C_NEW_LINE, C_CARRIAGE_RETURN, C_HORIZONTAL_TAB, C_VERTICAL_TAB.

    Interoperability with C pointers
    For interoperating with C pointers (which are just addresses), the module contains a derived type C_PTR that is interoperable with any C pointer type and a named constant C_NULL_PTR with the value NULL of C.
    The module also contains the following procedures:
        C_LOC (X) is an inquiry function that returns the C address of X.
        X is permitted to be a procedure that is interoperable (see para. 5) or a variable that has the TARGET attribute and is either interoperable or is an allocated allocatable variable that has interoperable type and type parameters.
        C_ASSOCIATED (C_PTR1[, C_PTR2]) is an inquiry function that returns a default logical scalar. It has the value false if C_PTR1 is a C null pointer or if C_PTR2 is present with a different value; otherwise, it has the value true.
        C_F_POINTER (CPTR, FPTR [, SHAPE]) is a subroutine with arguments CPTR is a scalar of type C_PTR with intent IN. Its value is the C address of an entity that is is interoperable with variables of the type and type parameters of FPTR. It shall not be the C address of a Fortran variable that does not have the TARGET attribute. FPTR is a pointer that becomes pointer associated with the target of CPTR. If it is an array, its shape is specified by SHAPE.
        SHAPE (optional) is a rank-one array of type integer with intent IN. If present, its size is equal to the rank of FPTR. If FPTR is an array, it must be present.

    This is the mechanism for passing dynamic arrays between the languages. A Fortran pointer or assumed-shape array cannot be passed to C since its elements need not be contiguous in memory. However, an allocated allocatable array may be passed to C and an array allocated in C may be associated with a Fortran pointer.

    Interoperability of derived types
    For a derived type to be interoperable, it must be given the BIND attribute explicitly:

    TYPE, BIND(C) :: MYTYPE
        :
    END TYPE MYTYPE

    Each component must have interoperable type and type parameters, must not be a pointer, and must not be allocatable. This allows Fortran and C types to correspond, for example:

    typedef struct {
        int m, n;
        float r;
    } myctype

    is interoperable with

    USE ISO_C_BINDING
    TYPE, BIND(C) :: MYFTYPE
        INTEGER(C_INT) :: I, J
        REAL(C_FLOAT) :: S
    END TYPE MYFTYPE

    The name of the type and the names of the components are not significant for interoperability.

    No Fortran type is interoperable with a C union type, struct type that contains a bit field, or struct type that contains a flexible array member.

    Interoperability of variables
    A scalar Fortran variable is interoperable if it is of interoperable type and type parameters, and is neither a pointer nor allocatable.

    An array Fortran variable is interoperable if it is of interoperable type and type parameters, and is of explicit shape or assumed size. It interoperates with a C array of the same type types parameters and shape, but with reversal of subscripts. For example, a Fortran array declared as:

    INTEGER :: A(18, 3:7, *)

    is interoperable with a C array declared as

    int b[][5][18]

    Interoperability of procedures A Fortran procedure is interoperable if it has an explicit interface and is declared with the BIND attribute:

    FUNCTION FUNC(I, J, K, L, M) BIND(C)

    All the dummy arguments must be interoperable. For a function, the result must be scalar and interoperable. The procedure has a 'binding label', which has global scope and is the name by which it is known to the C processor. By default, it is the lower-case version of the Fortran name. For example, the above function has the binding label func. Another binding label may be specied:

    FUNCTION FUNC(I, J, K, L, M) BIND(C, NAME='C_Func')

    Such a procedure corresponds to a C function prototype with the same binding label. For a function, the result must be interoperable. For a subroutine, the prototype must have a void result.

    Interoperability of global data An interoperable module variable or a common block with interoperable members may be given the BIND attribute:

    USE ISO_C_BINDING
        INTEGER(C_INT), BIND(C) :: C_EXTERN
        INTEGER(C_LONG) :: C2
        BIND(C, NAME='myVariable') :: C2
        COMMON /COM/ R, S
        REAL(C_FLOAT) :: R, S
        BIND(C) :: /COM/

    It has a binding label defined by the same rules as for procedures and interoperate with a C variable of a corresponding struct type.

    Example of Fortran calling C C Function Prototype:

    int C_Library_Function(void* sendbuf, int sendcount, int *recvcounts)

    Fortran Module:

    MODULE FTN_C
        INTERFACE
            INTEGER (C_INT) FUNCTION C_LIBRARY_FUNCTION  &
                (SENDBUF, SENDCOUNT, RECVCOUNTS)         &
                BIND(C, NAME='C_Library_Function')
                USE ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), VALUE :: SENDBUF
                INTEGER (C_INT), VALUE :: SENDCOUNT
                TYPE (C_PTR), VALUE :: RECVCOUNTS
            END FUNCTION C_LIBRARY_FUNCTION
        END INTERFACE
    END MODULE FTN_C

    Fortran Calling Sequence:

    USE ISO_C_BINDING, ONLY: C_INT, C_FLOAT, C_LOC
    USE FTN_C
    ...
    REAL (C_FLOAT), TARGET :: SEND(100)
    INTEGER (C_INT)        :: SENDCOUNT
    INTEGER (C_INT), ALLOCATABLE, TARGET :: RECVCOUNTS(:)
    ...
    ALLOCATE( RECVCOUNTS(100) )
    ...
    CALL C_LIBRARY_FUNCTION(C_LOC(SEND), SENDCOUNT,     &
    C_LOC(RECVCOUNTS))
    ...

    Example of C calling Fortran
    Fortran Code:

    SUBROUTINE SIMULATION(ALPHA, BETA, GAMMA, DELTA, ARRAYS) BIND(C)
        USE ISO_C_BINDING
        IMPLICIT NONE
        INTEGER (C_LONG), VALUE                 :: ALPHA
        REAL (C_DOUBLE), INTENT(INOUT)          :: BETA
        INTEGER (C_LONG), INTENT(OUT)           :: GAMMA
        REAL (C_DOUBLE),DIMENSION(*),INTENT(IN) :: DELTA
        TYPE, BIND(C) :: PASS
            INTEGER (C_INT) :: LENC, LENF
            TYPE (C_PTR)    :: C, F
        END TYPE PASS
        TYPE (PASS), INTENT(INOUT) :: ARRAYS
        REAL (C_FLOAT), ALLOCATABLE, TARGET, SAVE :: ETA(:)
        REAL (C_FLOAT), POINTER :: C_ARRAY(:)
        ...
        ! Associate C_ARRAY with an array allocated in C
        CALL C_F_POINTER (ARRAYS%C, C_ARRAY, (/ARRAYS%LENC/) )
        ...
        ! Allocate an array and make it available in C
        ARRAYS%LENF = 100
        ALLOCATE (ETA(ARRAYS%LENF))
        ARRAYS%F = C_LOC(ETA)
        ...
    END SUBROUTINE SIMULATION

    C Struct Declaration:

    struct pass {int lenc, lenf; float* f, *c}

    C Function Prototype:

    void simulation(long alpha, double *beta, long *gamma, double delta[],
                    struct pass *arrays)

    C Calling Sequence:

    simulation(alpha, &beta, &gamma, delta, &arrays);


