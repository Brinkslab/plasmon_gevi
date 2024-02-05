%put .mat file in this folder.
%this works only for 2d full monitors
close all
clear all

datafiles = dir('*.mat');

for ii=1:length(datafiles)
    name(ii) = string(datafiles(ii).name);
    tmp = load(name(ii));
    plot(tmp.lambda_plot,tmp.E_mean)
    hold on 
end

legend(name)
xlabel('Wavelength [nm]')
ylabel('$\displaystyle\frac{|E|^2}{|E_0|^2}$','interpreter','latex','rotation',0)
set(gca,'fontsize',14);