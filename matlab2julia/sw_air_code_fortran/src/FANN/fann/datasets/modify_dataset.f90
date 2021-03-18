program iodata
implicit none

   real, dimension(2033,471)   :: dataN2
   real, dimension(2033,361)   :: dataO2
   real, dimension(2033,391)   :: dataNO
   real, dimension(1768,126+1) :: dataXY12
   real, dimension(1326,126+1) :: dataXY12train
   real, dimension(442,126+1)  :: dataXY12test
   integer :: i,j

!   open (1, file = 'train.txt', status = 'old')
!   do j=1,1326
!     read(1,*) dataXY12train(j,:)
!   end do
!   close(1)
!   open(2, file = 'dataXY12train_mod.dat', status='new')
!   do j=1,1326
!     write(2,"(1E15.7)") dataXY12train(j,1)
!     write(2,"(126E15.7)") (dataXY12train(j,i), i=2,126+1)
!   enddo
!   close(2)
!   open (1, file = 'test.txt', status = 'old')
!   do j=1,442
!     read(1,*) dataXY12test(j,:)
!   end do
!   close(1)
!   open(2, file = 'dataXY12test_mod.dat', status='new')
!   do j=1,442
!     write(2,"(1E15.7)") dataXY12test(j,1)
!     write(2,"(126E15.7)") (dataXY12test(j,i), i=2,126+1)
!   enddo
!   close(2)

   ! opening the file for reading
   open (1, file = 'solution_XY_12.dat', status = 'old')
   do j=1,1768
     read(1,*) dataXY12(j,:)
   end do
   close(1)
   !open(2, file = 'dataXY12mod.dat', status='new')
   open(2, file = 'dataXY12modT.dat', status='new')
   do j=1,1768
     write(2,"(1E15.7)") dataXY12(j,1)
     write(2,"(1E15.7)") (dataXY12(j,i), i=126+1,126+1)
     !write(2,"(126E15.7)") (dataXY12(j,i), i=2,126+1)
   enddo
   close(2)

!   ! opening the file for reading
!   open (1, file = 'dataset_STS_kd_kr_N2.txt', status = 'old')
!   !read(1,*) dataN2
!   !read(1,*) ((dataN2(i,j), j=1,470), i=1,2033)
!   do j=1,2033
!     read(1,*) dataN2(j,:)
!   end do
!   close(1)
!   !dataN2c1 = dataN2(:,1)
!   !dataN2c2 = dataN2(:,2:)
!   open(2, file = 'dataN2mod.dat', status='new')
!   !do j=1,2033
!   ! do i=1,470
!   !  write(2,"(1E15.7)")   !dataN2(j,1)
!   !  write(2,"(470E15.7)") !dataN2(j,i+1)
!   ! end do
!   !enddo
!   do j=1,2033
!     write(2,"(1E15.7)") dataN2(j,1) !dataN2c1(j)
!     write(2,"(470E15.7)") (dataN2(j,i), i=2,471) !(dataN2c2(j,i), i=1,470)
!   enddo
!   close(2)
!
!   ! opening the file for reading
!   open (1, file = 'dataset_STS_kd_kr_O2.txt', status = 'old')
!   do j=1,2033
!     read(1,*) dataO2(j,:)
!   end do
!   close(1)
!   open(2, file = 'dataO2mod.dat', status='new')
!   do j=1,2033
!     write(2,"(1E15.7)") dataO2(j,1)
!     write(2,"(360E15.7)") (dataO2(j,i), i=2,361)
!   enddo
!   close(2)
!
!   ! opening the file for reading
!   open (1, file = 'dataset_STS_kd_kr_NO.txt', status = 'old')
!   do j=1,2033
!     read(1,*) dataNO(j,:)
!   end do
!   close(1)
!   open(2, file = 'dataNOmod.dat', status='new')
!   do j=1,2033
!     write(2,"(1E15.7)") dataNO(j,1)
!     write(2,"(390E15.7)") (dataNO(j,i), i=2,391)
!   enddo
!   close(2)

end program
