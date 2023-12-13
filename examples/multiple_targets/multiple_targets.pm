mdp

const int x_max = 199;
const int x_target = 99;
const int y_max = 99;
const double p = 0.65;
const double q = 0.20;

module Dive_and_Rise

x : [0..x_max] init 0; //step
y : [0..y_max] init 0; //height
z : [0..1] init 0; //game on or not

[dive]  x < x_max & z = 0 ->
		p : (x'= x+1) & (y'= max(y-1, 0)) +
		q : (x'= x+1) & (y'= y) +
		1-p-q : (x'= x+1) & (y'= min(y+1, y_max)) ;
[rise]  x < x_max & z = 0 ->
		p : (x'= x+1) & (y'= min(y+1, y_max)) +
		q : (x'= x+1) & (y'= y) +
		1-p-q: (x'= x+1) & (y'= max(y-1, 0)) ;
[end]	x = x_max -> (x'= 0) & (y'= 0) & (z'= 1) ;
[none]	z = 1 -> 1 : true ;

endmodule

rewards "target_00"
	(x=x_max)&(y=0)&(z=0): 1 ;
endrewards
rewards "target_04"
	(x=x_max)&(y=4)&(z=0): 1 ;
endrewards
rewards "target_09"
	(x=x_max)&(y=9)&(z=0): 1 ;
endrewards
rewards "target_14"
	(x=x_max)&(y=14)&(z=0): 1 ;
endrewards
rewards "target_19"
	(x=x_max)&(y=19)&(z=0): 1 ;
endrewards
rewards "target_24"
	(x=x_max)&(y=24)&(z=0): 1 ;
endrewards
rewards "target_29"
	(x=x_max)&(y=29)&(z=0): 1 ;
endrewards
rewards "target_34"
	(x=x_max)&(y=34)&(z=0): 1 ;
endrewards
rewards "target_39"
	(x=x_max)&(y=39)&(z=0): 1 ;
endrewards
rewards "target_44"
	(x=x_max)&(y=44)&(z=0): 1 ;
endrewards
rewards "target_49"
	(x=x_max)&(y=49)&(z=0): 1 ;
endrewards
rewards "target_54"
	(x=x_max)&(y=54)&(z=0): 1 ;
endrewards
rewards "target_59"
	(x=x_max)&(y=59)&(z=0): 1 ;
endrewards
rewards "target_64"
	(x=x_max)&(y=64)&(z=0): 1 ;
endrewards
rewards "target_69"
	(x=x_max)&(y=69)&(z=0): 1 ;
endrewards
rewards "target_74"
	(x=x_max)&(y=74)&(z=0): 1 ;
endrewards
rewards "target_79"
	(x=x_max)&(y=79)&(z=0): 1 ;
endrewards
rewards "target_84"
	(x=x_max)&(y=84)&(z=0): 1 ;
endrewards
rewards "target_89"
	(x=x_max)&(y=89)&(z=0): 1 ;
endrewards
rewards "target_94"
	(x=x_max)&(y=94)&(z=0): 1 ;
endrewards
rewards "target_99"
	(x=x_max)&(y=99)&(z=0): 1 ;
endrewards
