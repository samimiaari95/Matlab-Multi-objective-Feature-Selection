%Regression

clear
clc
%% Define input file name

filename = 'subbasin.xls';

%% specify include paths

mi_path = what("mi");
borg_path = what('borg_moea\C\plugins\Matlab\');
paretofront_path = what("pareto_front");
inputs_path = what("inputs");
MOFS_path = what("MOFS");
supporting_functions = what("supporting_functions");
results_path = what("results");

addpath(mi_path.path);  % Peng's mutual information
addpath(borg_path.path);   % Borg
addpath(paretofront_path.path);   % paretofront toolbox
addpath(inputs_path.path);  % inputs folder
addpath(MOFS_path.path);    % Multi-Objective Feature Selection folder
addpath(supporting_functions.path); % supporting functions folder
addpath(results_path.path)  % results folder



%% Model settings

nFolds=32; % equal to the number of years
nRuns = 200; % the higher the better for ANN
weeks=1:1:1664; % number of weeks
nELM=5000; % number of ELM function evaluations
nUnits = 100; % The higher the better for ELM but increases running time


%% Nile dataset

Nile = readtable(filename);

%depending on the variables chosen from IVS
p1_Nile  = Nile.year;
p2_Nile  = Nile.week;
p3_Nile  = Nile.Prec;
p4_Nile  = Nile.Prec2w;
p5_Nile  = Nile.Prec4w;
p6_Nile  = Nile.Tmin;
p7_Nile  = Nile.Tmin2w;
p8_Nile  = Nile.Tmax;
p9_Nile  = Nile.Tmax2w;
p10_Nile  = Nile.Tmean;
p11_Nile  = Nile.Tmean2w;
p12_Nile  = Nile.Tmean4w;
p13_Nile  = Nile.Evapotranspiration;
p14_Nile  = Nile.Evapotranspiration3w;
p15_Nile  = Nile.Evapotranspiration6w;
p16_Nile  = Nile.Evapotranspiration16w;
p17_Nile  = Nile.Riverdischarge;
p18_Nile  = Nile.Riverdischarge2w;
p19_Nile  = Nile.Riverdischarge4w;
p20_Nile  = Nile.Riverdischarge16w;
p21_Nile  = Nile.Soilmoisture;
p22_Nile  = Nile.Soilmoisture16w;
p23_Nile  = Nile.Soilmoisture52w;
p24_Nile  = Nile.SPI;
p25_Nile  = Nile.SPI3w;
p26_Nile  = Nile.SPI6w;
p27_Nile  = Nile.SPI16w;
p28_Nile  = Nile.SPI52w;

Y_Nile   = Nile.NDVI;


T = 52;

%% NDVI processing
%Deseasonalizing NDVI Nile
[ mi_Nile , m_Nile ] = moving_average( Y_Nile , T , 5 ) ; 
[ sigma2_Nile , s2_Nile ] = moving_average( ( Y_Nile - m_Nile ).^2 , T , 5 ) ;
sigma_Nile           = sigma2_Nile .^ 0.5                        ;
s_Nile               = s2_Nile .^ 0.5                            ;

% deseasonalized NDVI Nile
x_Nile = ( Y_Nile - m_Nile ) ./ s_Nile ; 

figure()
plot(1:1664,x_Nile) % 884 representing the number of weeks
titles = 'Deseasonalized NDVI';
title(titles);
ylabel(titles)
xlabel( 'Years' )
xlim([0 size(weeks,2)])
xticks(0:52:size(weeks,2))
% modify the labels
xticklabels({'1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'})


 % periodicity 
%tt = mod( t - 1 , T ) + 1   ; % indices tt = 1 , 2 , ... , 365 , 1 , 2 , ...
tt = repmat( [1:T]' , 1664/T, 1 ) ; % indices tt = 1 , 2 , ... , 365 , 1 , 2 , ...

figure ; plot( tt ,Y_Nile , '.' ) ; 
titles = 'Yearly NDVI';
title(titles);
xlabel( 'time t (1 year)' ) ; ylabel( 'NDVI' )

Q = reshape(Y_Nile,T,32); % 32 stands for the number of years

% cyclo-stationary mean
Cm = mean(Q')' ;
Cm = mean(Q,2) ;

% cyclo-stationary variance
Cv = var(Q')'  ;

% graphical analysis
figure ; 
plot( tt , Y_Nile , '.' ) ; hold on;
plot(Cm,'r','LineWidth',2); 
titles = 'Observed-Cyclo mean';
title(titles);
legend('observed','cyclo mean');
xlabel( 'time t (1 year)' ) ; ylabel( 'NDVI' );


figure();
plot(1:1664,repmat(Cm,32),'-k') % 32 represents the number of years
hold on
plot(1:1664,Y_Nile,'-r')
titles = 'Deseasonalized - Observed NDVI';
title(titles);
ylabel('NDVI')
xlabel( 'Years' )
xlim([0 size(weeks,2)])
xticks(0:52:size(weeks,2))
xticklabels({'1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'})

%% Prepare the dataset
% Deseasonalize variables

x_p3_Nile = deseasonalize_var(p3_Nile, 'precipitation');
x_p4_Nile = deseasonalize_var(p4_Nile, 'precipitation 2w');
x_p5_Nile = deseasonalize_var(p5_Nile, 'precipitation 4w');
x_p6_Nile = deseasonalize_var(p6_Nile, 'Tmin');
x_p7_Nile = deseasonalize_var(p7_Nile, 'Tmin 2w');
x_p8_Nile = deseasonalize_var(p8_Nile, 'Tmax');
x_p9_Nile = deseasonalize_var(p9_Nile, 'Tmax 2w');
x_p10_Nile = deseasonalize_var(p10_Nile, 'Tmean');
x_p11_Nile = deseasonalize_var(p11_Nile, 'Tmean 2w');
x_p12_Nile = deseasonalize_var(p12_Nile, 'Tmean 4w');
x_p13_Nile = deseasonalize_var(p13_Nile, 'evapotranspiration');
x_p14_Nile = deseasonalize_var(p14_Nile, 'evapotranspiration 3w');
x_p15_Nile = deseasonalize_var(p15_Nile, 'evapotranspiration 6w');
x_p16_Nile = deseasonalize_var(p16_Nile, 'evapotranspiration 16w');
x_p17_Nile = deseasonalize_var(p17_Nile, 'river discharge');
x_p18_Nile = deseasonalize_var(p18_Nile, 'river discharge 2w');
x_p19_Nile = deseasonalize_var(p19_Nile, 'river discharge 4w');
x_p20_Nile = deseasonalize_var(p20_Nile, 'river discharge 16w');
x_p21_Nile = deseasonalize_var(p21_Nile, 'soil moisture');
x_p22_Nile = deseasonalize_var(p22_Nile, 'soil moisture 16w');
x_p23_Nile = deseasonalize_var(p23_Nile, 'soil moisture 52w');
x_p24_Nile = deseasonalize_var(p24_Nile, 'SPI');
x_p25_Nile = deseasonalize_var(p25_Nile, 'SPI 3w');
x_p26_Nile = deseasonalize_var(p26_Nile, 'SPI 6w');
x_p27_Nile = deseasonalize_var(p27_Nile, 'SPI 16w');
x_p28_Nile = deseasonalize_var(p28_Nile, 'SPI 52w');


x_PHI_Nile = [p2_Nile,x_p27_Nile] ; % predictors to be considered

Y          = x_Nile ;

% number of predictors
featIxes=1:1:2;


%% Linear model

 % k-fold cross validation
    lData  = size(Y,1);
    lFold  = floor(lData/nFolds);
    Yhat_Nile_lin = zeros(size(Y,1),1);
x__Nile = zeros(size(Y,1),1);


for i = 1 : nFolds
        % select training and validation data
        ix1 = (i-1)*lFold+1;
        if i == nFolds
            ix2 = lData;
        else
            ix2 = i*lFold;
        end
        valIxes  = ix1:ix2; % select the validation chunk
        trIxes = setdiff(1:lData,valIxes); % obtain training indexes by set difference
        
        % create datasets
        trX  = x_PHI_Nile(trIxes,featIxes);  
        trY  = Y(trIxes,:);
        valX = x_PHI_Nile(valIxes,featIxes);
        
        % train and test Linear Model
        
        theta_Nile = trX\trY;
        
        x__Nile = valX*theta_Nile; 
        
        Yhat_Nile_lin(valIxes) =  x__Nile .* s_Nile(valIxes) + m_Nile(valIxes)           ;
end

Erre2_Nile_LIN = rsq(Y_Nile,Yhat_Nile_lin);
RMSE_LIN = sqrt(mean((Y_Nile - Yhat_Nile_lin).^2));
fprintf('R2 Linear = %s\n', Erre2_Nile_LIN);
fprintf('RMSE Linear = %s\n', RMSE_LIN);

figure; plot(Y_Nile,Yhat_Nile_lin, '.');
title('Scatterplot Observed-Predicted NDVI LINEAR')
xlabel('Observed NDVI');
ylabel('Predicted NDVI');

figure; plot (weeks,Y_Nile);
hold on; plot (weeks, Yhat_Nile_lin);
title('Predicted NDVI Linear');
legend('Observed NDVI', 'Predicted NDVI Linear');
ylabel('NDVI')
xlabel( 'Years' )
xlim([0 size(weeks,2)])
xticks(0:52:size(weeks,2))
xticklabels({'1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'})
    


%% ANN model

R2_i_Nile = zeros(nRuns,1);

Yhat_Nile_ANN = zeros(size(x_Nile,1),nRuns); 
 
for j = 1:nRuns
    fprintf('Run %d/%d\n', j , nRuns);
    % k-fold cross validation
    lData  = size(x_Nile,1);
    lFold  = floor(lData/nFolds);
    
    for i = 1 : nFolds
        % select trainind and validation data
        ix1 = (i-1)*lFold+1;
        if i == nFolds
            ix2 = lData;
        else
            ix2 = i*lFold;
        end
        valIxes  = ix1:ix2; % select the validation chunk
        trIxes = setdiff(1:lData,valIxes); % obtain training indexes by set difference
        
        % create datasets
        trX_Nile  = x_PHI_Nile(trIxes,featIxes); 
        trY_Nile  = x_Nile(trIxes,:);
        valX_Nile = x_PHI_Nile(valIxes,featIxes);
        
       
        net_i_Nile = newff(trX_Nile',trY_Nile',3) ; % initialization of ANN
        net_i_Nile = train( net_i_Nile, trX_Nile',trY_Nile' ) ;
        Yhat_Nile_ANN(valIxes,j) = sim( net_i_Nile, valX_Nile' ) ;
        % remove deseasonality
        Yhat_Nile_ANN(valIxes,j) =  Yhat_Nile_ANN(valIxes,j) .* s_Nile(valIxes) + m_Nile(valIxes) ;
    end
 
    R2_i_Nile(j)=rsq(Y_Nile,Yhat_Nile_ANN(:,j));
    
    if R2_i_Nile(j) >= max(R2_i_Nile)
        net_opt_Nile = net_i_Nile ;
    end
end

[Erre2_Nile_ANN,pos] = max(R2_i_Nile);
predicted_NDVI_ANN_Nile = Yhat_Nile_ANN(:,pos);

RMSE_ANN = sqrt(mean((Y_Nile - predicted_NDVI_ANN_Nile).^2));
fprintf('R2 ANN = %s\n', Erre2_Nile_ANN);
fprintf('RMSE ANN = %s\n', RMSE_ANN);

figure; plot(Y_Nile,predicted_NDVI_ANN_Nile, '.'); 
title('Scatterplot Observed-Predicted NDVI ANN')
xlabel('Observed NDVI');
ylabel('Predicted NDVI');

figure; plot (weeks,Y_Nile);
hold on; plot (weeks, predicted_NDVI_ANN_Nile);
title('Predicted NDVI ANN');
legend('Observed', 'Predicted ANN');
ylabel('NDVI')
xlabel( 'Years' )
xlim([0 size(weeks,2)])
xticks(0:52:size(weeks,2))
xticklabels({'1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'})


%% Extreme Learning Machine model with leave-one-out crossvalidation

Yhat_Nile_ELM = zeros(size(x_Nile,1),1);
SU_Nile = zeros(1,nELM) + Inf;

Rmax_Nile_ELM = 0 ;

maxSU=-inf;
k=1;

for j = 1 : nELM
    
    % k-fold cross validation
    lData  = size(x_Nile,1);
    lFold  = floor(lData/nFolds);
    
    for i = 1 : nFolds
        % select trainind and validation data
        ix1 = (i-1)*lFold+1;
        if i == nFolds
            ix2 = lData;
        else
            ix2 = i*lFold;
        end
        valIxes  = ix1:ix2; % select the validation chunk
        trIxes = setdiff(1:lData,valIxes); % obtain training indexes by set difference
        
        % create datasets
        trX  = x_PHI_Nile(trIxes,featIxes); 
        trY  = x_Nile(trIxes,:);
        valX = x_PHI_Nile(valIxes,featIxes);
        
        % train and test ELM
        [~,Yhat_Nile_ELM(valIxes), W1, W2, bias] =...
            ELMregression(trX', trY', valX', nUnits);
        % remove deseasonality
        Yhat_Nile_ELM(valIxes) =  Yhat_Nile_ELM(valIxes) .* s_Nile(valIxes) + m_Nile(valIxes) ;
    end
    
    Traj_Nile(:,j)=Yhat_Nile_ELM; 
    SU_Nile(j) = computeSU(Y_Nile,Yhat_Nile_ELM);
    R2_Nile_ELM(j)=rsq(Y_Nile,Yhat_Nile_ELM);

    if R2_Nile_ELM(j) > Rmax_Nile_ELM   %SU(j)>maxSU
        maxSU=SU_Nile;
        k=j;
        Rmax_Nile_ELM=R2_Nile_ELM(j);
        fprintf('R2 ELM = %s\n', Rmax_Nile_ELM);
        W1opt_Nile = W1;
        W2opt_Nile = W2;
        biasopt_Nile = bias;
    end
    
    
end


predicetd_NDVI_Nile_ELM=Traj_Nile(:,k); %traiettoria modello migliore

Corrcoef_Nile = corrcoef(predicetd_NDVI_Nile_ELM,Y_Nile);


% ELM results

R2_ELM = rsq(Y_Nile,predicetd_NDVI_Nile_ELM);
RMSE_ELM = sqrt(mean((Y_Nile - predicetd_NDVI_Nile_ELM).^2));
fprintf('R2 ELM = %s\n', R2_ELM);
fprintf('RMSE ELM = %s\n', RMSE_ELM);

figure; plot(Y_Nile,predicetd_NDVI_Nile_ELM, '.'); 
title('Scatterplot Observed-Predicted NDVI ELM')
xlabel('Observed NDVI');
ylabel('Predicted NDVI');

figure; plot (weeks,Y_Nile, 'k-');
hold on; plot (weeks, predicetd_NDVI_Nile_ELM, 'r-');
title('Predicted NDVI ELM');
legend('Observed', 'Predicted ELM');
ylabel('NDVI')
xlabel( 'Years' )
xlim([0 size(weeks,2)])
xticks(0:52:size(weeks,2))
xticklabels({'1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'})

Erre2_Nile_ELM = rsq(Nile.NDVI,predicetd_NDVI_Nile_ELM);
fprintf('R2 ELM = %s\n', Erre2_Nile_ELM);

%% Nile final results

figure; plot (weeks,Nile.NDVI);
hold on; plot (weeks, predicetd_NDVI_Nile_ELM);
hold on; plot (weeks, Yhat_Nile_lin);
hold on; plot (weeks, predicted_NDVI_ANN_Nile);
title('Predicted NDVI Linear - ANN - ELM');
legend('Observed', 'Predicted ELM','Predicted Linear Model','Predicted ANN');
ylabel('NDVI')
xlabel( 'Years' )
xlim([0 size(weeks,2)])
xticks(0:52:size(weeks,2))
xticklabels({'1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'})

%% Create the results folder
[pathstr, subbasin, ext] = fileparts(filename);
FolderName = append(results_path.path,'\',subbasin);
vlist = size(dir(FolderName)) ;
newfolder = append(FolderName,'\','v',num2str( vlist(1) ));
mkdir(newfolder)

%% Save figures in the results folder
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
  FigHandle = FigList(iFig);
  ax = FigHandle.CurrentAxes; 
  FigName   = get(ax,'title');
  FigName = get(FigName, 'string');
  savefig(FigHandle, fullfile(newfolder, strjoin({FigName, '.fig'},'')));
end
