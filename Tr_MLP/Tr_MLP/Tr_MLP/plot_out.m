clear;
load out.dat
load act_y.dat
plot(out);
hold;
plot(act_y,'r');
err= out - act_y;
dim= size(out);
(sum(err.^2)/dim(1,1))^0.5


