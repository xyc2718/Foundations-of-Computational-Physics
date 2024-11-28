program hw11
integer,parameter::kd=selected_int_kind(12) !define the kind of integer,200^5<10^12
integer(kind=kd) a,b,c,d,e
do e=1,200 !loop through all possible values of a,b,c,d,e
    do d=0,e
        do c=0,d
            do b=1,c
                do a=1,b
                    if (a**5+b**5+c**5+d**5==e**5) then !if the condition is true, print the values of a,b,c,d,e
                        print*, "a=",a,"b=",b,"c=",c,"d=",d,"e=",e
                    end if
                end do
            end do
        end do
    end do
end do
print*, "Done" !print "Done" when the program is finished
call system('pause')

end program hw11