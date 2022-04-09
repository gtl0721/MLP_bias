clear;
load out.dat
load test_y.dat
plot(out);
hold;
plot(test_y,'r');
err= out - test_y;
dim= size(out);
(sum(err.^2)/dim(1,1))^0.5


