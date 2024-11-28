
module calculator24_module !定义calculator24,value为real值，expression为表达式，通过expression保留计算历史，flag将初始化为超越数，用于表达式去重
   type calculator24
      real::value
      real::flag
      character(len=:), allocatable :: expression
      contains
            procedure :: print => print_calculator !定义print方法,用于显示type的表达式
   end type calculator24
   contains
    subroutine print_calculator(this)  
        class(calculator24), intent(in) :: this
        character(len=5) :: str
        write(str, '(F5.2)') this%value
        print *, this%expression//"="//str
    end subroutine print_calculator
end module


program main
   
   use  calculator24_module
   integer,parameter::num_elements=4 !定义输入的元素个数
   real,parameter::cal_value=24.0    !定义计算目标值
   integer input_array(num_elements)
   type(calculator24) type_array(1,num_elements)
   type(calculator24),allocatable :: reduced_pair(:,:)
   type(calculator24) ,allocatable ::answer(:),reduced_answer(:)  
   integer::s
   print*, "This program is used to solve 24 points game"
   print *, "plese Enter" , num_elements ,"elements:"   !读取输入
   do i = 1, num_elements
        read *, input_array(i)
   end do
   print *, 'calculating...'

   type_array=init_array(input_array)!初始化数组
   reduced_pair=reduce_pair(type_array)!迭代计算
   answer=match_value(reduced_pair(:,1),cal_value)!匹配结果
   reduced_answer=reduce_answer(answer)!去重

   print *,"the anwser of",input_array,"is:" !输出结果

   s=size(reduced_answer)
   print *,s, "solution if found" 
   if (s==1 .and. reduced_answer(1)%expression=="no solution") then
      print *,"no solution"
   else
      do i=1,s
      call reduced_answer(i)%print
   end do
   end if
   call system("pause")


   
 
   

contains
   
!定义二元计算函数，通过symbol:1-6分别对应加减乘除和左减左除,会同时计算value和expression和flag
!返回一个新的calculator24类型
   function calculate(x1,x2,symbol) result(re) 
      implicit none
      type(calculator24),intent(in)::x1,x2
      integer,intent(in)::symbol
      type(calculator24) :: re
      integer::pos1,pos2

      if (symbol==1) then
         re%value=x1%value+x2%value
         allocate(re%expression,source="("//x1%expression//'+'//x2%expression//")")
         re%flag=x1%flag+x2%flag
         return
      else if (symbol==2) then
         re%value=x1%value-x2%value
         allocate(re%expression,source="("//x1%expression//'-'//x2%expression//")")
         re%flag=x1%flag-x2%flag
         return
      else if (symbol==3) then
         re%value=x1%value*x2%value
         allocate(re%expression,source=x1%expression//'*'//x2%expression)
         re%flag=x1%flag*x2%flag
         return
      else if (symbol==4) then
         re%value=x1%value/x2%value
         allocate(re%expression,source=x1%expression//'/'//x2%expression)
         re%flag=x1%flag/x2%flag
         return
      else if (symbol==5) then
         re%value=x2%value-x1%value
         allocate(re%expression,source="("//x2%expression//'-'//x1%expression//")")
         re%flag=x2%flag-x1%flag
         return
      else if (symbol==6) then
         re%value=x2%value/x1%value
         re%flag=x2%flag/x1%flag
         pos1=index(x1%expression,"/")
         pos2=index(x1%expression,"*")
         if (pos1>0 .or. pos2>0) then
            allocate(re%expression, source=x2%expression // '/' // "("//x1%expression//")")
         else 
            allocate(re%expression, source=x2%expression // '/'//  x1%expression)
         end if
         return
      end if
   end function

   !初始化数组，将输入的整数数组转换为1*n的calculator24类型二维数组,二维数组是为了和其它函数的输入统一
   function init_array(array) result(type_array)
      implicit none
      integer,intent(in)::array(:)
      integer num_elements
      type(calculator24),allocatable::type_array(:,:)
      integer::i=1
      character(len=2)::str
      num_elements=size(array)
      allocate(type_array(1,num_elements))
      do i =1,num_elements
         type_array(1,i)%value=real(array(i))
         write(str,'(I2)') array(i)
         allocate(type_array(1,i)%expression,source=str)
         type_array(1,i)%flag=sqrt(real(array(i))*3.1415926535) !根据数值初始化一个无理数用于去重
      end do
   end function
    
   !输入n*m的calculator24类型二维数组，返回每一行的Cm2的结果，未参与排列的数组在之后补齐
    function generate_pairs(array) result(pairs)
        implicit none
        type(calculator24), intent(in) :: array(:,:)  
        type(calculator24), allocatable :: pairs(:,:) ! 输出的二元组二维数组
        integer i, j, k,m,p1,p2,num_elements, num_pairs,rows,r 
        num_elements = size(array,2) 
        rows=size(array,1)
        num_pairs = rows*(num_elements * (num_elements - 1) / 2 ) ! 计算二元组合的数量
        allocate(pairs(num_pairs, num_elements)) 
        k=1
        do r = 1,rows
            do i = 1, num_elements - 1
                  do j = i + 1, num_elements
                        p1=3
                        p2=1
                     do m=1,num_elements
                        if (m==i .or. m==j) then !被排列的元素填入前两列
                           pairs(k, p2) = array(r,m)
                           p2=p2+1
                        else                     !其它元素补齐
                           pairs(k,p1)=array(r,m)
                           p1=p1+1
                        end if
                  end do
                  k = k + 1
               end do
            end do
         end do
    end function generate_pairs

!将pairs的前两行用6种计算归约1次，返回一个新的rows*6,cols-1二维数组
    function reduce_pair_step(array) result(reduced_pair) 
      type(calculator24),intent(in)::array(:,:)
      type(calculator24),allocatable::reduced_pair(:,:)
      type(calculator24)::re
      integer::cols,rows,i,k,symbol
      rows=size(array,1)
      cols=size(array,2)
      if (cols<2) then !如果只有一列，原样返回
         allocate(reduced_pair(rows,cols))
         reduced_pair=array
         return
      else
      allocate(reduced_pair(rows*6,cols-1))
      k=1
      do i=1,rows
            do symbol=1,6 !对每一行遍历6种计算
            re=calculate(array(i,1),array(i,2),symbol)
            reduced_pair(k,1)=re
            reduced_pair(k,2:cols-1)=array(i,3:cols)
            k=k+1
            end do
      end do
      end if
      end function


!对初始输入反复pair和reduce，直到只剩下一列，返回一个n*1的二维数组
      function reduce_pair(array) result(reduced_pair)
         type(calculator24),intent(in)::array(:,:)
         type(calculator24),allocatable::reduced_pair(:,:)
         integer::i,rows,cols
         integer::max_iter=100 !限制最大迭代次数
         reduced_pair=reduce_pair_step(generate_pairs(array))
         do i=1,max_iter
            reduced_pair=reduce_pair_step(generate_pairs(reduced_pair))
            if (size(reduced_pair,2)==1) then !如果只剩下一列，返回
               return
            end if
         end do
      end function
      
   !从最终结果中搜索值为value的结果,返回最终的calculator24类型
      function match_value(array,value) result(re)
         type(calculator24),intent(in)::array(:)
         type(calculator24),allocatable::re(:)
         type(calculator24)::temp(100)
         integer::i,n,k
         real::value
         real::eps=0.001 !容差
         n=size(array,1)
         k=1
         do i =1,n
            if ((abs(array(i)%value-value)<eps).and. k<20) then
               temp(k)=array(i)
               k=k+1
            end if
         end do
         if (k>1) then
            allocate(re(k-1))
            re=temp(1:k-1)
         else
            allocate(re(1))
            re(1)%value=-1
            allocate(re(1)%expression,source="no solution")
         end if
         
         end function

         function reduce_answer(answer) result(reduced_answer)  !通过超越数去重
            type(calculator24),allocatable::answer(:),temp(:) 
            type(calculator24),allocatable::reduced_answer(:)
            integer::i,rows,k,j,flag
            real::eps=0.001
            rows=size(answer)
            allocate(temp(rows)) !temp数组用于零时储存结果
            if (rows==1 .and. answer(1)%expression=="no solution") then !处理无解情况
               allocate(reduced_answer(1))
               reduced_answer=answer
               return
            end if
            temp(1)=answer(1)
            k=1
            do i=1,rows
            flag=1
               do j=1,k  !与temp中已有元素比较
                  if (abs(answer(i)%flag-temp(j)%flag)<eps) then !如果超越数相同，则为同一结构表达式，去除
                     flag=0
                  end if
               end do
               if (flag==1) then
                  k=k+1
                  temp(k)=answer(i)
               end if
            end do
            allocate(reduced_answer(k))
            reduced_answer=temp(1:k)
      
   
         end function
      
   
end program main
