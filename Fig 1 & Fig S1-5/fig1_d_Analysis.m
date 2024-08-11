%% 90%
load('exp00690.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
figure(1)
subplot(2,3,1)
for k=1:25
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    [MaxFieldEnhP90(k) index]=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end
AvgEnh=TotEnh/25;
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)
legend ('simulation','mean over 25')

%title('Avg. field enhancement |E|^2/|E^0|^2 of GNSs 4 nm away from spikes on a R=30 nm L_t_i_p=10 nm and 90%spikes')
title('90%')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement |E|^2/|E^0|^2 [a.u.]');
%% 80%
load('exp00680.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
figure (1)
subplot(2,3,2)
for k=1:25
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    [MaxFieldEnhP80(k) index]=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end
AvgEnh=TotEnh/25;
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)

 
legend ('simulation','mean over 25')
title('80%')

%title('Avg. field enhancement |E|^2/|E^0|^2 of GNSs 4 nm away from spikes on a R=30 nm L_t_i_p=10 nm and 80%spikes')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement |E|^2/|E^0|^2 [a.u.]');

%% 70%
load('exp00670.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
figure (1)
subplot(2,3,3)
for k=1:25
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    [MaxFieldEnhP70(k) index]=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end
AvgEnh=TotEnh/25;
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)

 
legend ('simulation','mean over 25')
title('70%')
%title('Avg. field enhancement |E|^2/|E^0|^2 of GNSs 4 nm away from spikes on a R=30 nm L_t_i_p=10 nm and 70%spikes')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement |E|^2/|E^0|^2 [a.u.]');

%% 60%
load('exp00660.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
figure(1)
subplot(2,3,4)
for k=1:25
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    [MaxFieldEnhP60(k) index]=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end
AvgEnh=TotEnh/25;
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)

 
legend ('simulation','mean over 25')
title('60%')
%title('Avg. field enhancement |E|^2/|E^0|^2 of GNSs 4 nm away from spikes on a R=30 nm L_t_i_p=10 nm and 60%spikes')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement |E|^2/|E^0|^2 [a.u.]');

%% 50%
load('exp00650.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
figure(1)
subplot(2,3,5)
for k=1:25
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    [MaxFieldEnhP50(k) index]=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end
AvgEnh=TotEnh/25;
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)

 
legend ('simulation','mean over 25')
title('50%')
%title('Avg. field enhancement |E|^2/|E^0|^2 of GNSs 4 nm away from spikes on a R=30 nm L_t_i_p=10 nm and 50%spikes')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement |E|^2/|E^0|^2 [a.u.]');

%% CROSS SECTIONS
% legendname={}
% currentFolder=pwd;
% fign=2;
% 
% for V=1:25
%     pathtofile=strcat('\cs', '\[4] cs_abs_ R30L10S3B0.9D35T40V',num2str(V),'.txt')
%     filenamething=fullfile(currentFolder,pathtofile)
%     scatt1=importdata(filenamething);
%     Lam=scatt1(:,1);
%     abs=scatt1(:,2);
%     %legendname{V}=strcat(num2str(V*20),'fs');
%     figure (fign)
%     plot(Lam,abs,'b');
%     hold on
% end
% title('Absorption cross sections of GNSs of l_t_i_p=10 nm on a R=30 nm sphere')
% xlabel('\lambda [nm]');
% ylabel('Cross section [m^2]');
% 
% for V=1:25
%     pathtofile=strcat('\cs', '\[4] cs_sca_ R30L10S3B0.9D35T40V',num2str(V),'.txt')
%     filenamething=fullfile(currentFolder,pathtofile)
%     scatt1=importdata(filenamething);
%     Lam=scatt1(:,1);
%     sca=scatt1(:,2);
%     %legendname{V}=strcat(num2str(V*20),'fs');
%     figure (fign)
%     plot(Lam,sca,'r');
%     hold on
% end
% title('Scattering cross sections of GNSs of l_t_i_p=10 nm on a R=30 nm sphere')
% xlabel('\lambda [nm]');
% ylabel('Cross section [m^2]');
% 
% for V=1:25
%     pathtofile1=strcat('\cs', '\[4] cs_abs_ R30L10S3B0.9D35T40V',num2str(V),'.txt')
%     filenamething1=fullfile(currentFolder,pathtofile1)
%     scatt1=importdata(filenamething1);
%     pathtofile2=strcat('\cs', '\[4] cs_sca_ R30L10S3B0.9D35T40V',num2str(V),'.txt')
%     filenamething2=fullfile(currentFolder,pathtofile2)
%     scatt2=importdata(filenamething2);
%     Lam=scatt1(:,1);
%     abs=scatt1(:,2);
%     sca=scatt2(:,2);
%     ext=abs+sca;
%     %legendname{V}=strcat(num2str(V*20),'fs');
%     figure (fign)
%     plot(Lam,ext,'m');
%     hold on
% end
% title('extinction cross sections of GNSs of l_t_i_p=10 nm on a R=30 nm sphere')
% xlabel('\lambda [nm]');
% ylabel('Cross section [m^2]');

%% hopefully right violinplot
cd
disp('this example uses the statistical toolbox')

Y=[]
%Y100=sort(MaxFieldEnhControl)';
Y90=sort(MaxFieldEnhP90)';
Y80=sort(MaxFieldEnhP80)';
Y70=sort(MaxFieldEnhP70)';
Y60=sort(MaxFieldEnhP60)';
Y50=sort(MaxFieldEnhP50)';

figure(3)
set(figure(3),'DefaultTextFontSize', 24);
set(gcf,'Units','centimeters','Position',[0 0 32 16]); % set figure size
violin([Y90,Y80,Y70,Y60,Y50],'facecolor',[0.35, 0.70, 0.90],'edgecolor','k','bw',1,'mc','','medc','');
hold on
boxplot([Y90,Y80,Y70,Y60,Y50],'PlotStyle','compact','Jitter',0,'Labels',{'90%','80%','70%','60%','50%'},'color',[0,0.45,0.70]);
hold on

Yexp006=[Y90,Y80,Y70,Y60,Y50];
SDexp006=std(Yexp006);
Medianexp006=median(Yexp006);
x=1:5;
P = polyfit(x,Medianexp006,1);
x=0.5:5.5;
yfit = P(1)*x+P(2);
plot(x,yfit,'LineStyle','--','LineWidth',3,'color','#e69f00');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ax = gca(); 
set(gca,'FontSize',24);
% Get handle to the hggroup
boxplotGroupHandle = findobj(ax,'Type','hggroup','Tag','boxplot'); 
% Get handle to the text labels that serve as x-ticks
textHandles = findobj(boxplotGroupHandle, 'type', 'text');
% The handles are in reverse order so we'll flip the vector to make indexing easier
textHandles = flipud(textHandles); 
% Rotate and recenter text 
set(textHandles, 'rotation', 0,'HorizontalAlignment','center','VerticalAlignment','bottom')
% % Keep 1 "tick label" out of every 3
% % Instead of deleting handles, replace their string with an empty ''
% rmIndx = ~ismember(1:numel(textHandles), 1:3:numel(textHandles));
% set(textHandles(rmIndx), 'String', '')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ylabel('Avg. field enhancement [a.u.]', 'FontSize', 28)
xlabel('Tip density [a.u.]', 'FontSize', 28)

hold off

ylim([1 41]); % set y-axis limits to include all data
% Set resolution to 300dpi and save as PNG file
print(gcf,'-dpng','-r300','TipDensity_poster.png');

% violin([Y1,Y2],'xlabel',{'0.4','0.8'},'facecolor','b','edgecolor','k','bw',1,'mc','','medc','');
% hold on
% boxplot([Y1,Y2],'PlotStyle','compact','Jitter',0,'Labels',{'0.4','0.8'});
% ylabel('Avg. field enhancement |E|^2/|E^0|^2 at \lambda_{peak}  [a.u.]','FontSize',14)
% hold off
