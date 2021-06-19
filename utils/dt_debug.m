close all; clear; clc;

data = csvread('dt_debug.csv', 1, 0);

iter = data(:,1);
criteria = data(:,2);
angle_gt = data(:,3:9);
angle_command = data(:,10:16);
angle_control = data(:, 17:23);
angle_pnp = data(:, 24:30);
% jacobian = data(:, 31:end);
% 
% 
% J = reshape(jacobian, n, 14, 7);
n = size(data,1);
angle_diff = zeros(n-1,1);
for i=1:n-1
    angle_diff(i) = norm(angle_gt(i+1)-angle_gt(i));
end


figure();
subplot(2,1,1)
plot(iter);
grid on;
subplot(2,1,2)
plot(criteria);
grid on

figure();
subplot(3,2,1)
plot(angle_gt(:,1));
grid on;
hold on;
plot(angle_command(:,1));
plot(angle_pnp(:,1));
legend('gt', 'cmd', 'pnp')

subplot(3,2,2)
plot(angle_gt(:,2));
grid on;
hold on;
plot(angle_command(:,2));
plot(angle_pnp(:,2));
legend('gt', 'cmd', 'pnp')

subplot(3,2,3)
plot(angle_gt(:,3));
grid on;
hold on;
plot(angle_command(:,3));
plot(angle_pnp(:,3));
legend('gt', 'cmd', 'pnp')

subplot(3,2,4)
plot(angle_gt(:,4));
grid on;
hold on;
plot(angle_command(:,4));
plot(angle_pnp(:,4));
legend('gt', 'cmd', 'pnp')

subplot(3,2,5)
plot(angle_gt(:,5));
grid on;
hold on;
plot(angle_command(:,5));
plot(angle_pnp(:,5));
legend('gt', 'cmd', 'pnp')

subplot(3,2,6)
plot(angle_gt(:,6));
grid on;
hold on;
plot(angle_command(:,6));
plot(angle_pnp(:,6));
legend('gt', 'cmd', 'pnp')