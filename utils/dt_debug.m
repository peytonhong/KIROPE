% close all; clear; clc;

data = csvread('dt_debug.csv', 1, 0);

iter = data(:,1);
angle_error = data(:,2);
angle_gt = data(:,3:8)*180/pi;
angle_command = data(:,9:14)*180/pi;
angle_control = data(:, 15:20)*180/pi;
angle_pnp = data(:, 21:26)*180/pi;
% jacobian = data(:, 27:end);
% 
% 
% J = reshape(jacobian, n, 14, 7);
n = size(data,1);
angle_diff = zeros(n-1,1);
for i=1:n-1
    angle_diff(i) = norm(angle_gt(i+1)-angle_gt(i));
end
angle_command = angle_pnp;

figure(1);
clf;
subplot(2,1,1)
plot(iter);
legend('iter');
grid on;
subplot(2,1,2)
plot(angle_error);
grid on
legend('angle error');

figure(2);
clf;
subplot(3,2,1)
plot(angle_gt(:,1));
grid on;
hold on;
plot(angle_command(:,1));
plot(angle_pnp(:,1));
title('joint 1')
ylabel('angle [deg]')
legend('gt', 'cmd', 'pnp')

subplot(3,2,2)
plot(angle_gt(:,2));
grid on;
hold on;
plot(angle_command(:,2));
plot(angle_pnp(:,2));
title('joint 2')
ylabel('angle [deg]')
legend('gt', 'cmd', 'pnp')

subplot(3,2,3)
plot(angle_gt(:,3));
grid on;
hold on;
plot(angle_command(:,3));
plot(angle_pnp(:,3));
title('joint 3')
ylabel('angle [deg]')
legend('gt', 'cmd', 'pnp')

subplot(3,2,4)
plot(angle_gt(:,4));
grid on;
hold on;
plot(angle_command(:,4));
plot(angle_pnp(:,4));
title('joint 4')
ylabel('angle [deg]')
legend('gt', 'cmd', 'pnp')

subplot(3,2,5)
plot(angle_gt(:,5));
grid on;
hold on;
plot(angle_command(:,5));
plot(angle_pnp(:,5));
title('joint 5')
ylabel('angle [deg]')
legend('gt', 'cmd', 'pnp')

subplot(3,2,6)
plot(angle_gt(:,6));
grid on;
hold on;
plot(angle_command(:,6));
plot(angle_pnp(:,6));
title('joint 6')
ylabel('angle [deg]')
legend('gt', 'cmd', 'pnp')