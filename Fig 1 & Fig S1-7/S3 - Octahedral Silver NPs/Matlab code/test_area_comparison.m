close all
clear all
clc

c = 299792458; %speed of light in m/s
lambda_1= 300; %lowest wavelength
lambda_2 = 800; %highest wavelength
freq_1 = c/ lambda_1;
freq_2 = c/ lambda_2;
freq = linspace(freq_2,freq_1,lambda_2-lambda_1+1);

lambda = c./freq;
average = 0; %turn on average peak


datafiles = dir('*.mat');

for ii=1:length(datafiles)
    name(ii) = string(datafiles(ii).name);
    tmp = load(name(ii));    
    temp_E = tmp.E_mean;
    temp_E(temp_E==0) = [];
    E_mean_all(ii,:) = temp_E;
    [max_E(ii), id(ii)] = max(temp_E);
end

m = {'-','--',':','-.','-','--'} ;

hold on 
for k=1:length(m)
plot(lambda,E_mean_all(k,:),m{k},'lineWidth',3)
end
hold off 
legend(name)
xlabel('Wavelength [nm]')
ylabel('$\displaystyle\frac{|E|^2}{|E_0|^2}$','interpreter','latex','rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
set(gca,'fontsize',20);
xlim([lambda_1,lambda_2])

conal_angle = [38,39,40,45,50,60];
% calculation length

%%
R = 20*1e-9;
alpha = conal_angle.*pi./180;

L = 2.*R.*(pi-alpha+2.*sqrt(sin(alpha./2).^2+9/4));
figure
hold on
plot(L*10^9,max_E,'lineWidth',3)
xlabel('L [nm]')
ylabel('$\displaystyle\frac{|E|^2}{|E_0|^2}$','interpreter','latex','rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
set(gca,'fontsize',20);

lambda_max = lambda(id);

figure
plot(L*10^9,lambda_max,'lineWidth',3)
xlabel('L [nm]')
ylabel('wavelength [nm]')
set(gca,'fontsize',20);
