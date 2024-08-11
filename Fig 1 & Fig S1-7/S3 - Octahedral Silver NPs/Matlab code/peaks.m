close all
clear all
clc

c = 299792458; %speed of light in m/s
lambda_1= 300; %lowest wavelength
lambda_2 = 800; %highest wavelength
freq_1 = c/ lambda_1;
freq_2 = c/ lambda_2;
freq = linspace(freq_2,freq_1,501);

lambda = c./freq;
average = 1; %turn on average peak


datafiles = dir('*.mat');
E_mean_tot = zeros(length(datafiles),length(freq));
for ii=1:length(datafiles)
    name(ii) = string(datafiles(ii).name);
    tmp = load(name(ii));  
    E_mean_tot(ii,:) = tmp.E_mean;
    plot(lambda,tmp.E_mean)
    max_E(ii) = max(E_mean_tot(ii,:));
    hold on 
end

figure
data_peaks = [name', max_E'];
angle = [0,10,15,1,20,22.5,2,30,3,45,4,5,6,7,8,9];
scatter(angle,max_E,'o','lineWidth',3)
set(gca,'fontsize',20);
xlabel('Angle [deg]')
ylabel('$\displaystyle\frac{|E|^2}{|E_0|^2}$','interpreter','latex','rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')





%plot(lambda,E_mean_tot)