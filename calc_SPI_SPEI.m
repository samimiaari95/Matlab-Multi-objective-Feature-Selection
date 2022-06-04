clc; clear;

addpath('C:\Users\samim\OneDrive - Politecnico di Milano\tesi\thesis cananzi\original data\Code\Code\function')
%addpath('C:\Users\samim\Downloads\SDAT.zip\SDAT')

filePath = 'agg_precipitation_white_nile.csv';
M = readtable(filePath);
P = M(:,3);
%Y = M(:,'year');
%W = M(:,'week');
year = table2array(M(:,'year'));
week = table2array(M(:,'week'));
V = table2array(P);

SPEI = calc_SPEI(V);
SPI = calc_SPI(V, 'gamma');

all_SPI = [year,week,SPI];
all_SPEI = [year,week,SPEI];
SPI_T = array2table(all_SPI,'VariableNames',{'year','week','SPI'});
SPEI_T = array2table(all_SPEI,'VariableNames',{'year','week','SPEI'});
writetable(SPI_T, 'SPI_white_nile.csv');
writetable(SPEI_T, 'SPEI_white_nile.csv');


%% Open nc
%startLoc = [1 1]; % Start location along each coordinate
%count  = [10 10];
%ssw  = ncread('MainNile_sm_.nc','sm',startLoc,count);
%disp(ssw(:,:))
%writematrix(ssw(:,:,1),'unsat2.csv')
