program test_TF_Version
  use ftf
  use c_interface_module
  implicit none

  type(TF_Buffer)                :: tfBuffer
  type(TF_Graph)                 :: tfGraph
  type(TF_ImportGraphDefOptions) :: tfImportGraphDefOptions
  type(TF_SessionOptions)        :: tfNewSessionOptions

block
  type(c_ptr)        :: C_version
  character(len=5)   :: F_version
  C_version = TF_Version()
! http://fortranwiki.org/fortran/show/c_interface_module
  call C_F_string_ptr(C_version, F_version)
  write(*,*) " TF_Version --> ", F_version
end block

  tfGraph = TF_NewGraph()
  write(*,*) " TF_NewGraph --> OK "

  !call TF_DeleteGraph(tfGraph)

block
!  type(TF_Status)   :: tfStatus
  !type(c_ptr)      :: C_tfStatus
  !character(len=5) :: F_tfStatus
  !tfStatus = TF_NewStatus();
!  tfStatus = TF_NewStatus();
!  write(*,*) " TF_NewStatus --> OK "
  !call C_F_string_ptr(C_tfStatus, F_tfStatus)
  !write(*,*) F_tfStatus
end block

block
 !type(TF_Message) :: tfMessage
  type(c_ptr)      :: C_message
  character(len=5) :: F_message
  type(c_ptr)   :: tfStatus
  tfStatus = TF_NewStatus();
  C_message = TF_Message(tfStatus)
  call C_F_string_ptr(C_message, F_message)
  write(*,*) " TF_Message --> ", F_message
end block

  !if (TF_GetCode(tfStatus) == "TF_OK") then
  !  write(*,*) "TF_LoadSessionFromSavedModel OK"
  !end if

  !call TF_GraphImportGraphDef(tfGraph, tfBuffer, tfImportGraphDefOptions, tfStatus)

  tfImportGraphDefOptions = TF_NewImportGraphDefOptions()
  write(*,*) " TF_NewImportGraphDefOptions --> OK "

  tfNewSessionOptions = TF_NewSessionOptions()
  write(*,*) " tfNewSessionOptions --> OK "

! Free memory
!  call TF_DeleteGraph(tfGraph)
!  call TF_DeleteSession(tfSession, tfStatus)
!  call TF_DeleteSessionOptions(tfSessionOpts)
!  call TF_DeleteStatus(tfStatus)
!  call TF_DeleteImportGraphDefOptions(tfImportGraphDefOptions)

end program
