Module ftf
  use, intrinsic :: iso_c_binding

  enum, bind(C)
    enumerator :: TF_FLOAT      = 1, &
                  TF_DOUBLE     = 2, &
                  TF_INT32      = 3, & ! Int32 tensors are always in 'host' memory.
                  TF_UINT8      = 4, &
                  TF_INT16      = 5, &
                  TF_INT8       = 6, &
                  TF_STRING     = 7, &
                  TF_COMPLEX64  = 8, & ! Single-precision complex
                  TF_COMPLEX    = 8, & ! Old identifier kept for API backwards compatibility
                  TF_INT64      = 9, &
                  TF_BOOL       = 10,&
                  TF_QINT8      = 11,& ! Quantized int8
                  TF_QUINT8     = 12,& ! Quantized uint8
                  TF_QINT32     = 13,& ! Quantized int32
                  TF_BFLOAT16   = 14,& ! Float32 truncated to 16 bits.  Only for cast ops.
                  TF_QINT16     = 15,& ! Quantized int16
                  TF_QUINT16    = 16,& ! Quantized uint16
                  TF_UINT16     = 17,&
                  TF_COMPLEX128 = 18,& ! Double-precision complex
                  TF_HALF       = 19,&
                  TF_RESOURCE   = 20,&
                  TF_VARIANT    = 21,&
                  TF_UINT32     = 22,&
                  TF_UINT64     = 23
  end enum

  enum, bind(C)
    enumerator :: TF_OK                  = 0, &
                  TF_CANCELLED           = 1, &
                  TF_UNKNOWN             = 2, &
                  TF_INVALID_ARGUMENT    = 3, &
                  TF_DEADLINE_EXCEEDED   = 4, &
                  TF_NOT_FOUND           = 5, &
                  TF_ALREADY_EXISTS      = 6, &
                  TF_PERMISSION_DENIED   = 7, &
                  TF_UNAUTHENTICATED     = 16,&
                  TF_RESOURCE_EXHAUSTED  = 8, &
                  TF_FAILED_PRECONDITION = 9, &
                  TF_ABORTED             = 10,&
                  TF_OUT_OF_RANGE        = 11,&
                  TF_UNIMPLEMENTED       = 12,&
                  TF_INTERNAL            = 13,&
                  TF_UNAVAILABLE         = 14,&
                  TF_DATA_LOSS           = 15
  end enum

! TF_AttrType describes the type of the value of an attribute on an operation.
  enum, bind(C)
    enumerator :: TF_ATTR_STRING      = 0, &
                  TF_ATTR_INT         = 1, &
                  TF_ATTR_FLOAT       = 2, &
                  TF_ATTR_BOOL        = 3, &
                  TF_ATTR_TYPE        = 4, &
                  TF_ATTR_SHAPE       = 5, &
                  TF_ATTR_TENSOR      = 6, &
                  TF_ATTR_PLACEHOLDER = 7, &
                  TF_ATTR_FUNC        = 8
  end enum

  type, bind(C) :: TF_Buffer
    type(c_funptr) :: data
    integer(c_size_t) :: length
    type(c_funptr) :: data_deallocator
  end type TF_Buffer

  type, bind(C) :: TF_Graph
  end type TF_Graph

  interface

!! Return a new status object.
!! TF_CAPI_EXPORT extern TF_Status* TF_NewStatus(void);
!  function TF_NewStatus() bind(C) result(tfStatus)
!    implicit none
!    type(TF_Status), value :: tfStatus
!  end function TF_NewStatus
!
!! Delete a previously created status object.
!! TF_CAPI_EXPORT extern void TF_DeleteStatus(TF_Status*);
!  subroutine TF_DeleteStatus(tfStatus) bind(C)
!    implicit none
!    type(TF_Status), value :: tfStatus
!  end subroutine TF_DeleteStatus
!
!! Return a new graph object.
!! TF_CAPI_EXPORT extern TF_Graph* TF_NewGraph(void);
  function TF_NewGraph() bind(C, name="TF_NewGraph") result(tfNewGraph)
    import TF_Graph
    implicit none
    type(TF_Graph) :: tfNewGraph
  end function TF_NewGraph
!
!! Return a new options object.
!! TF_CAPI_EXPORT extern TF_SessionOptions* TF_NewSessionOptions(void);
!  function TF_NewSessionOptions() result(tfSessionOptions)
!    implicit none
!    type(TF_NewSessionOptions), value :: tfSessionOptions
!  end function TF_NewSessionOptions
!
!! Return a new execution session with the associated graph, or NULL on
!! error. Does not take ownership of any input parameters.
!!
!! *`graph` must be a valid graph (not deleted or nullptr). `graph` will be be
!! kept alive for the lifetime of the returned TF_Session. New nodes can still
!! be added to `graph` after this call.
!! TF_CAPI_EXPORT extern TF_Session* TF_NewSession(TF_Graph* graph,
!!                                                 const TF_SessionOptions* opts,
!!                                                 TF_Status* status);
!  function TF_NewSession(graph, opts, tfStatus) result(tfSession)
!    implicit none
!    type(TF_Graph), value :: graph
!    type(TF_SessionOptions), value :: opts
!    type(TF_Status), value :: tfStatus
!  end function TF_NewSession
!
!! Return the code record in *s.
!! TF_CAPI_EXPORT extern TF_Code TF_GetCode(const TF_Status* s);
!  function TF_GetCode(s) result(tfCode)
!    implicit none
!    type(TF_Status), value :: s
!    type(TF_Code), value :: tfCode
!  end function TF_GetCode
!
!! TF_Version returns a string describing version information of the
!! TensorFlow library. TensorFlow using semantic versioning.
!! TF_CAPI_EXPORT extern const char* TF_Version(void);
!  function TF_Version_() bind(C, name=TF_Version) result(version)
!    import c_ptr, c_char
!    implicit none
!    character(c_char) :: version
!  end function

!  subroutine TF_NewBufferFromString(proto, proto_len) bind(C)
!    import c_ptr, c_int, c_funptr
!    implicit none
!    type(c_funptr), value :: proto
!    integer(c_size_t), value :: proto_len
!  end subroutine TF_NewBufferFromString
!
!  subroutine TF_NewBuffer()
!    implicit none
!
!  end subroutine TF_NewBuffer
!
!  subroutine TF_DeleteBuffer()
!    implicit none
!
!  end subroutine TF_NewBuffer
!
!  subroutine TF_GetBuffer()
!    implicit none
!
!  end subroutine TF_GetBuffer
!
!
! function TF_LoadSessionFromSavedModel(session_options, run_options, export_dir, tags, tags_len, graph, meta_graph_def, status)
!   result(TF_Session)
!   import c_char, c_int
!   implicit none
!   type(TF_SessionOptions), value :: session_options
!   type(TF_Buffer), value :: run_options
!   character(c_char), dimension(*) :: export_dir
!   character(c_char), dimension(*) :: tags
!   integer(c_int) :: tags_len
!   type(TF_Graph), value :: graph
!   type(TF_Buffer), value :: meta_graph_def
!   type(TF_Status), value :: status
! end function TF_LoadSessionFromSavedModel
!
! subroutine TF_SessionRun(session, &
!                          run_options, &
!                          inputs, &
!                          input_values, &
!                          ninputs, &
!                          outputs, &
!                          output_values, &
!                          noutputs, &
!                          target_opers, &
!                          ntargets, &
!                          run_metadata, &
!                          TF_Status), bind(C)
!  implicit none
!
!  type(TF_Session), value :: session
!  type(TF_Buffer), value :: run_options
!  type(TF_Output), value :: inputs
!  type(TF_Tensor), value :: input_values
!  integer(c_int) :: ninputs
!  type(TF_Output), value :: outputs
!  type(TF_Tensor), value :: output_values
!  integer(c_int) :: noutputs
!  type(TF_Operation), value :: target_opers
!  integer(c_int) :: ntargets
!  type(TF_Buffer), value :: run_metadata
!  type(TF_Status), value :: TF_Status
!
!
!  function TF_GraphOperationByName(graph, oper_name) result(TF_Operation)
!    implicit none
!    type(TF_Graph), value :: graph
!    character(c_char), dimension(*) :: oper_name
!    type(TF_Operation), value :: tfOperation
!  end function

  end interface

End Module ftf
