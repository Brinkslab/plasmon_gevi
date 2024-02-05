
%put .mat file in this folder.
%this works only for 2d full monitors
close all
clear all

c = 299792458; %speed of light in m/s
lambda_1= 300; %lowest wavelength
lambda_2 = 800; %highest wavelength
freq_1 = c/ lambda_1;
freq_2 = c/ lambda_2;
freq = linspace(freq_2,freq_1,501);

lambda = c./freq;
average = 0; %turn on average peak


datafiles = dir('*.mat');

for ii=1:length(datafiles)
    name(ii) = string(datafiles(ii).name);
    tmp = load(name(ii));    
    plot(lambda,tmp.E_mean,'lineWidth',3)
    hold on 
end
legend(name)
xlabel('Wavelength [nm]')
ylabel('$\displaystyle\frac{|E|^2}{|E_0|^2}$','interpreter','latex','rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
set(gca,'fontsize',20);
xlim([lambda_1,lambda_2])

if average == 1
    E_all = zeros(length(datafiles),length(lambda));
    for ii=1:length(datafiles)
        name(ii) = string(datafiles(ii).name);
        tmp = load(name(ii));   
        E_all(ii,:) = tmp.E_mean;
    end
    E_average = mean(E_all,1);

    figure
    plot(lambda, E_average)
    title("average")
end


