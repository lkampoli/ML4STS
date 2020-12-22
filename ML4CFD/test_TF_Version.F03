program test_TF_Version
  use ftf
  implicit none

  type(TF_Buffer)                :: tfBuffer
  type(TF_Graph)                 :: tfGraph
  type(TF_Status)                :: tfStatus
  type(TF_ImportGraphDefOptions) :: tfImportGraphDefOptions

  character(c_char) :: version

  version = TF_Version()
  write(*,*) " TF_Version --> ", version

  tfGraph = TF_NewGraph()
  write(*,*) " TF_NewGraph --> OK "

  !call TF_DeleteGraph(tfGraph)

  tfStatus = TF_NewStatus();
  write(*,*) " TF_NewStatus --> OK "

  tfImportGraphDefOptions = TF_NewImportGraphDefOptions()
  write(*,*) " TF_NewImportGraphDefOptions --> OK "

  !call TF_DeleteImportGraphDefOptions(tfImportGraphDefOptions)
  !write(*,*) " TF_DeleteImportGraphDefOptions --> OK "

  !call TF_GraphImportGraphDef(tfGraph, tfBuffer, tfImportGraphDefOptions, tfStatus)

end program
