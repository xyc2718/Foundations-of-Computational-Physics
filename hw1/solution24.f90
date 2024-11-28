module calculator24_module
   ! Define the calculator24 type with a real value and an expression
   ! The expression retains the calculation history
   type calculator24
      real :: value
      character(len=:), allocatable :: expression
      contains
            procedure :: print => print_calculator ! Define the print method to display the type's expression
   end type calculator24

   contains

    subroutine print_calculator(this)  
        class(calculator24), intent(in) :: this
        character(len=5) :: str
        write(str, '(F5.2)') this%value
        print *, this%expression // " = " // str
    end subroutine print_calculator
end module


program main
   use calculator24_module
   integer, parameter :: num_elements = 4 ! Number of input elements
   real, parameter :: cal_value = 24.0    ! Target value for calculation
   integer :: input_array(num_elements)
   type(calculator24) :: type_array(1, num_elements)
   type(calculator24), allocatable :: reduced_pair(:,:)
   type(calculator24) :: answer
   
   print *, 'Enter the elements:'   ! Read input
   
   do i = 1, num_elements
        read *, input_array(i)
   end do

   print *, 'Calculating...'

   type_array = init_array(input_array) ! Initialize array
   reduced_pair = reduce_pair(type_array) ! Iterative calculation
   answer = match_value(reduced_pair(:,1), cal_value) ! Match result

   print *, "The answer for", input_array, "is:" ! Output result
   if (answer%expression == "none") then
      print *, "No solution"
   else
      call answer%print()
   end if
   call system('pause')
contains

   ! Function to perform binary calculations. Symbols 1-6 correspond to addition, subtraction, multiplication, division,
   ! left subtraction, and left division respectively. It computes both value and expression, returning a new calculator24 type.
      function calculate(x1, x2, symbol) result(re) 
      implicit none
      type(calculator24), intent(in) :: x1, x2
      integer, intent(in) :: symbol
      integer :: pos1, pos2
      type(calculator24) :: re

      if (symbol == 1) then
         re%value = x1%value + x2%value
         allocate(re%expression, source="(" // x1%expression // '+' // x2%expression // ")")
         return
      else if (symbol == 2) then
         re%value = x1%value - x2%value
         allocate(re%expression, source="(" // x1%expression // '-' // x2%expression // ")")
         return
      else if (symbol == 3) then
         re%value = x1%value * x2%value
         allocate(re%expression, source=x1%expression // '*' // x2%expression)
         return
      else if (symbol == 4) then
         re%value = x1%value / x2%value
         allocate(re%expression, source=x1%expression // '/' // x2%expression)
         return
      else if (symbol == 5) then
         re%value = x2%value - x1%value
         allocate(re%expression, source="(" // x2%expression // '-' // x1%expression // ")")
         return
      else if (symbol == 6) then
         re%value = x2%value / x1%value
         pos1=index(x1%expression,"/")
         pos2=index(x1%expression,"*")
         if (pos1>0 .or. pos2>0) then
            allocate(re%expression, source=x2%expression // '/' // "("//x1%expression//")")
         else 
            allocate(re%expression, source=x2%expression // '/' // x1%expression)
         end if
         return
      end if
   end function

   ! Initialize an array by converting an integer array to a 1*n calculator24 type 2D array.
   ! The 2D array format is used to unify with other functions' inputs.
   function init_array(array) result(type_array)
      implicit none
      integer, intent(in) :: array(:)
      integer :: num_elements
      type(calculator24), allocatable :: type_array(:,:)
      integer :: i
      character(len=2) :: str
      num_elements = size(array)
      allocate(type_array(1, num_elements))
      do i = 1, num_elements
         type_array(1, i)%value = real(array(i))
         write(str, '(I2)') array(i)
         allocate(type_array(1, i)%expression, source=str)
      end do
   end function

   ! Given an n*m calculator24 type 2D array, return the Cm2 results for each row.
   ! The unpaired array is padded later.
   function generate_pairs(array) result(pairs)
      implicit none
      type(calculator24), intent(in) :: array(:,:)  
      type(calculator24), allocatable :: pairs(:,:) ! Output binary tuple 2D array
      integer :: i, j, k, m, p1, p2, num_elements, num_pairs, rows, r
      num_elements = size(array, 2) 
      rows = size(array, 1)
      num_pairs = rows * (num_elements * (num_elements - 1) / 2) ! Calculate the number of binary combinations
      allocate(pairs(num_pairs, num_elements)) 
      k = 1
      do r = 1, rows
         do i = 1, num_elements - 1
            do j = i + 1, num_elements
               p1 = 3
               p2 = 1
               do m = 1, num_elements
                  if (m == i .or. m == j) then ! Fill the first two columns with permuted elements
                     pairs(k, p2) = array(r, m)
                     p2 = p2 + 1
                  else                      ! Pad other elements
                     pairs(k, p1) = array(r, m)
                     p1 = p1 + 1
                  end if
               end do
               k = k + 1
            end do
         end do
      end do
   end function generate_pairs

   ! Reduce the first two columns of pairs using 6 types of calculations, returning a new rows*6, cols-1 2D array.
   function reduce_pair_step(array) result(reduced_pair) 
      type(calculator24), intent(in) :: array(:,:)
      type(calculator24), allocatable :: reduced_pair(:,:)
      type(calculator24) :: re
      integer :: cols, rows, i, k, symbol
      rows = size(array, 1)
      cols = size(array, 2)
      if (cols < 2) then ! If there is only one column, return as is
         allocate(reduced_pair(rows, cols))
         reduced_pair = array
         return
      else
         allocate(reduced_pair(rows * 6, cols - 1))
         k = 1
         do i = 1, rows
            do symbol = 1, 6 ! Traverse 6 types of calculations for each row
               re = calculate(array(i, 1), array(i, 2), symbol)
               reduced_pair(k, 1) = re
               reduced_pair(k, 2:cols - 1) = array(i, 3:cols)
               k = k + 1
            end do
         end do
      end if
   end function

   ! Iteratively pair and reduce the initial input until only one column remains.
   ! Return an n*1 2D array.
   function reduce_pair(array) result(reduced_pair)
      type(calculator24), intent(in) :: array(:,:)
      type(calculator24), allocatable :: reduced_pair(:,:)
      integer :: i, rows, cols
      integer :: max_iter = 100 ! Limit maximum number of iterations
      reduced_pair = reduce_pair_step(generate_pairs(array))
      do i = 1, max_iter
         reduced_pair = reduce_pair_step(generate_pairs(reduced_pair))
         if (size(reduced_pair, 2) == 1) then ! If only one column remains, return
            return
         end if
      end do
   end function

   ! Search the result with value equal to the given value and return the final calculator24 type.
   function match_value(array, value) result(re)
      type(calculator24), intent(in) :: array(:)
      type(calculator24) :: re
      integer :: i, n
      real :: value
      real :: eps = 0.001 ! Tolerance
      n = size(array, 1)
      do i = 1, n
         if (abs(array(i)%value - value) < eps) then
            re = array(i)
            return
         end if
      end do
      re%expression = "none" ! If not found, set expression to none
   end function

end program main
