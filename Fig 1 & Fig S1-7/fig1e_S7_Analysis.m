%% only random rotation
%title('Average field enhancement of GNSs 4 nm away from spikes on a R=30 nm L_t_i_p=10 nm')
load('exp00900.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
subplot(3,2,1);

for k=1:25    
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    [MaxFieldEnhR0(k) MaxFieldEnhR0Lambda(k)]=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end
load('exp009control.mat', 'AvgELambda')
name = strcat('AvgELambda.avarage_field_enhancement',num2str(1));
filename = sprintf('%s',name);
FieldEnhControl=squeeze(eval(filename));
[MaxFieldEnhControl MaxFieldEnhControlLambda]=max(FieldEnhControl);
AvgEnh=TotEnh/25;
MaxAvgEnh=max(AvgEnh);
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)
hold on
plot(Lambda,FieldEnhControl,':','color','#e69f00','LineWidth',2);

title('Randomness = 0')
legend ('simulation','mean over 25','control')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement [a.u]');
ylim([0 22]);


%% 0.2
load('exp00902.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
subplot(3,2,2);

for k=1:25    
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    MaxFieldEnhR2(k)=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end
% name = strcat('AvgELambda009control.avarage_field_enhancement',num2str(1));
% filename = sprintf('%s',name);
% FieldEnhControl=squeeze(eval(filename));
% MaxFieldEnhControl=max(FieldEnhControl);
AvgEnh=TotEnh/25;
MaxAvgEnh=max(AvgEnh);
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)
hold on
plot(Lambda,FieldEnhControl,':','color','#e69f00','LineWidth',2);

title('Randomness = 0.2')
%title('Average field enhancement of GNSs 4 nm away from spikes on a R=30 nm L_t_i_p=10 nm')
legend ('simulation','mean over 25','control')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement [a.u]');
ylim([0 22]);

%% 0.4
load('exp00904.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
subplot(3,2,3);
for k=1:10    
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    MaxFieldEnhR4(k)=max(FieldEnh);
    plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    hold on
    TotEnh=TotEnh+FieldEnh;
end
load('exp00904newones.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
for k=1:15    
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    MaxFieldEnhR4(k+10)=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end
% name = strcat('AvgELambda009control.avarage_field_enhancement',num2str(1));
% filename = sprintf('%s',name);
% FieldEnh=squeeze(eval(filename));
AvgEnh=TotEnh/25;
MaxAvgEnh=max(AvgEnh);
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)
hold on
plot(Lambda,FieldEnhControl,':','color','#e69f00','LineWidth',2);

title('Randomness = 0.4')
legend ('simulation','mean over 25','control')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement [a.u]');
ylim([0 22]);
%% 0.6
load('exp00906.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
subplot(3,2,4);
% hold on
for k=1:25    
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    MaxFieldEnhR6(k)=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end

% name = strcat('AvgELambda009control.avarage_field_enhancement',num2str(1));
% filename = sprintf('%s',name);
% FieldEnh=squeeze(eval(filename));
AvgEnh=TotEnh/25;
MaxAvgEnh=max(AvgEnh);
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)
hold on
plot(Lambda,FieldEnhControl,':','color','#e69f00','LineWidth',2);

title('Randomness = 0.6')
legend ('simulation','mean over 25','control')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement [a.u]');
ylim([0 22]);

%% 0.8
load('exp00908.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
% FieldEnh=[];
TotEnh=0;
subplot(3,2,5);
for k=1:10
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    MaxFieldEnhR8(k)=max(FieldEnh);
    plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    hold on
    TotEnh=TotEnh+FieldEnh;
    
end
load('exp00908newones.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
for k=1:15
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    MaxFieldEnhR8(k+10)=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
    
end

% name = strcat('AvgELambda009control.avarage_field_enhancement',num2str(1));
% filename = sprintf('%s',name);
% FieldEnh=squeeze(eval(filename));
AvgEnh=TotEnh/25;
MaxAvgEnh=max(AvgEnh);
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)
hold on
plot(Lambda,FieldEnhControl,':','color','#e69f00','LineWidth',2);

title('Randomness = 0.8')
legend ('simulation','mean over 25','control')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement [a.u]');
ylim([0 22]);
%% 1
load('exp00910.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
TotEnh=0;
subplot(3,2,6);

for k=1:25    
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    MaxFieldEnhR10(k)=max(FieldEnh);
    if k==1
        plot(Lambda,FieldEnh,'color',' #56b4e9');
    else
        plot(Lambda,FieldEnh,'color',' #56b4e9','HandleVisibility','off');
    end
    hold on
    TotEnh=TotEnh+FieldEnh;
end
% name = strcat('AvgELambda009control.avarage_field_enhancement',num2str(1));
% filename = sprintf('%s',name);
% FieldEnh=squeeze(eval(filename));
AvgEnh=TotEnh/25;
MaxAvgEnh=max(AvgEnh);
plot(Lambda,AvgEnh,'color','#0072b2','LineWidth',2)
hold on
plot(Lambda,FieldEnhControl,':','color','#e69f00','LineWidth',2);

title('Randomness = 1')
legend ('simulation','mean over 25','control')
xlabel('\lambda [nm]');
ylabel('Avg. field enhancement [a.u]');
ylim([0 22]);

hold off

%% violin plot overview
figure(2);
set(figure(2),'DefaultTextFontSize', 24);
set(gcf,'Units','centimeters','Position',[0 0 32 16]); % set figure size
Y90=MaxFieldEnhR0';
Y92=MaxFieldEnhR2';
Y94=MaxFieldEnhR4';
Y96=MaxFieldEnhR6';
Y98=MaxFieldEnhR8';
Y910=MaxFieldEnhR10';
violin([Y90,Y92,Y94,Y96,Y98,Y910],'facecolor',[0.35, 0.70, 0.90],'edgecolor','k','bw',1,'mc','','medc','');
hold on
boxplot([Y90,Y92,Y94,Y96,Y98,Y910],'PlotStyle','compact','Jitter',0,'Labels',{'0','0.2','0.4','0.6','0.8','1'},'color',[0,0.45,0.70]);
hold on

Yexp009=[Y90,Y92,Y94,Y96,Y98,Y910];
SDexp009=std(Yexp009);
Medianexp009=median(Yexp009);
x=1:6;
P = polyfit(x,Medianexp009,1);
x=0.5:6.5;
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
xlabel('Randomness factor [a.u.]', 'FontSize', 28)
hold off

ylim([4.8 24]); % set y-axis limits to include all data
% Set resolution to 300dpi and save as PNG file
print(gcf,'-dpng','-r300','Randomness_poster.png');
%% peak shift
load('exp00900D50.mat', 'AvgELambda')
Lambda=squeeze(AvgELambda.lambda);
for k=1:25    
    name = strcat('AvgELambda.avarage_field_enhancement',num2str(k));
    filename = sprintf('%s',name);
    FieldEnh=squeeze(eval(filename));
    [MaxFieldEnhR0D50(k) MaxFieldEnhR0D50Lambda(k)]=max(FieldEnh);
    TotEnh=TotEnh+FieldEnh;
end
Y90ps=Lambda(MaxFieldEnhR0Lambda);
Y90D50ps=Lambda(MaxFieldEnhR0D50Lambda);

figure(3);
violin([Y90ps,Y90D50ps],'facecolor',[ 0.5843 0.8157 0.9882],'edgecolor','k','bw',1,'mc','','medc','');
boxplot([Y90ps,Y90D50ps],'PlotStyle','compact','Jitter',0,'Labels',{'35 MPR','50 MPR'})
ylabel('\lambda_{peak} [nm]')
yline(Lambda(MaxFieldEnhControlLambda))
hold off

figure(4);
plot([35 50],[Y90ps Y90D50ps]','-o');
xlabel('MPR');
ylabel('\lambda_{peak} [nm]');
hold off
