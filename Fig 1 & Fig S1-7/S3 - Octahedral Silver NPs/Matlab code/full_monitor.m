%put .mat file in this folder.
%this works only for 2d full monitors
close all
clear all
datafiles = dir('*.mat');
name = string(datafiles.name);
load(name);


index =50; 


radius = 16; %nm

%converting to linear frequency
c = 299792458;
lambda_1= 300; %lowest wavelength
lambda_2 = 800; %highest wavelength
freq_1 = c/ lambda_1;
freq_2 = c/ lambda_2;
freq = linspace(freq_2,freq_1,501);

lambda = c./freq;



E_full_monitor = squeeze(E_large);

length_mon = size(E_full_monitor,1);
rad_im = 32;
length_im = size(E_full_monitor,1)*radius/rad_im;

x = [-length_im/2, length_im/2];
y = [-length_im/2, length_im/2];
imagesc(x,y,rot90(E_full_monitor(:,:,index)));
title(['Wavelength = ', num2str(lambda(index)),' nm'])
ylabel('z-location [nm]'),xlabel('x-location [nm]')

colorbar
colormap hot
h = colorbar;
ylabel(h,'$\displaystyle\frac{|E|^2}{|E_0|^2}$','interpreter','latex','rotation',0,'VerticalAlignment','top', 'HorizontalAlignment','right')
set(gca,'fontsize',20);

%%
asdf = E_full_monitor(2:end-1,:,index);
x_new = linspace(x(1),x(end),size(asdf,1));
figure
hold on
line([20 20], [0 20],'lineStyle','--','lineWidth',3,'color','black');
line([-20 -20], [0 20],'lineStyle','--','lineWidth',3,'color','black');
plot(x_new,asdf(:,120),'lineWidth',3)
xlabel('x-position [nm]')
ylabel('$\displaystyle\frac{|E|^2}{|E_0|^2}$','interpreter','latex','rotation',0,'VerticalAlignment','middle', 'HorizontalAlignment','right')
set(gca,'fontsize',20);
xlim([x_new(1),x_new(end)])