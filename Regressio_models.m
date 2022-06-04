%Regression

clear
clc
%% Define input file name

filename = 'blue_nile.xls';

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
weeks=1:1:1664; % change number of weeks from 884
nELM=5000;
nUnits = 100; % change from 28 and see the output


%% Nile dataset

Nile = readtable(filename);

%depending on the variables chosen from IVS
p1_Nile  = Nile.week;
p2_Nile  = Nile.Riverdischarge2w;
p3_Nile  = Nile.SPI16w;
%p4_Nile  = Nile.Evapotranspiration;
%p5_Nile  = Nile.Soilmoisture;
%p6_Nile  = Nile.SPI6w;
%p7_Nile  = Nile.SPI16w;
Y_Nile   = Nile.NDVI;


T = 52;

%% NDVI processing
%Deseasonalizing NDVI Nile
[ mi_Nile , m_Nile ] = moving_average( Y_Nile , 52 , 5 ) ; 
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
tt = repmat( [1:52]' , 1664/T, 1 ) ; % indices tt = 1 , 2 , ... , 365 , 1 , 2 , ...

figure ; plot( tt ,Y_Nile , '.' ) ; 
titles = 'Yearly NDVI';
title(titles);
xlabel( 'time t (1 year)' ) ; ylabel( 'NDVI' )

Q = reshape(Y_Nile,T,32); % 34 stands for the number of years

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
plot(1:1664,repmat(Cm,32),'-k') % 34 represents the number of years
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
x_p2_Nile = deseasonalize_var(p2_Nile, 'river discharge 2w');
x_p3_Nile = deseasonalize_var(p3_Nile, 'SPI 16w');
%x_p4_Nile = deseasonalize_var(p4_Nile, 'evapotranspiration');
%x_p5_Nile = deseasonalize_var(p5_Nile, 'soil moisture');
%x_p6_Nile = deseasonalize_var(p6_Nile, 'SPI 6w');
%x_p7_Nile = deseasonalize_var(p7_Nile, 'SPI 16w');

x_PHI_Nile = [p1_Nile, x_p2_Nile,x_p3_Nile] ;

Y            = x_Nile;

% number of candidates
featIxes=1:1:3;


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

Erre2_Nile_LIN = rsq(Nile.NDVI,Yhat_Nile_lin);
fprintf('R2 Linear = %s\n', Erre2_Nile_LIN);
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
fprintf('R2 ANN = %s\n', Erre2_Nile_ANN);

predicted_NDVI_ANN_Nile = Yhat_Nile_ANN(:,pos);
 
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
list = size(dir('results')) ;
[pathstr, subbasin, ext] = fileparts(filename);
FolderName = append(results_path.path,'\',subbasin,'\','v',num2str( list(1) ));
mkdir(FolderName)

%% Save figures in the results folder
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
  FigHandle = FigList(iFig);
  ax = FigHandle.CurrentAxes; 
  FigName   = get(ax,'title');
  FigName = get(FigName, 'string');
  savefig(FigHandle, fullfile(FolderName, strjoin({FigName, '.fig'},'')));
end

