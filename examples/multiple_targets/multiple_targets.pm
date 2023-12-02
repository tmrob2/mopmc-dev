mdp

const int x_max = 199;
const int x_target = 99;
const int y_max = 99;
const double p = 0.65;
const double q = 0.20;

module M1

x : [0..x_max] init 0; //step
y : [0..y_max] init 0; //height
z : [0..1] init 0; //game on or not

[down]	x < x_max & z = 0 ->
		p : (x'= x+1) & (y'= max(y-1, 0)) +
		q : (x'= x+1) & (y'= y) +
		1-p-q : (x'= x+1) & (y'= min(y+1, y_max)) ;
[up]	x < x_max & z = 0 ->
		p : (x'= x+1) & (y'= min(y+1, y_max)) +
		q : (x'= x+1) & (y'= y) +
		1-p-q: (x'= x+1) & (y'= max(y-1, 0)) ;
[end]	x = x_max -> (x'= 0) & (y'= 0) & (z'= 1) ;
[loop]	z = 1 -> 1 : true ;

endmodule

rewards "target_00"
	(y=0)&(x=x_max): 1;
endrewards
rewards "target_09"
	(y=9)&(x=x_max): 1;
endrewards
rewards "target_19"
	(y=19)&(x=x_max): 1;
endrewards
rewards "target_29"
	(y=29)&(x=x_max): 1;
endrewards
rewards "target_39"
	(y=39)&(x=x_max): 1;
endrewards
rewards "target_49"
	(y=49)&(x=x_max): 1;
endrewards
rewards "target_59"
	(y=59)&(x=x_max): 1;
endrewards
rewards "target_69"
	(y=69)&(x=x_max): 1;
endrewards
rewards "target_79"
	(y=79)&(x=x_max): 1;
endrewards
rewards "target_89"
	(y=89)&(x=x_max): 1;
endrewards
rewards "target_99"
	(y=99)&(x=x_max): 1;
endrewards
