% -------------------------------------------------------------------------
% Leave-one-out cross-validation framework using MKL-SVM, linear kernels
% with permutation test
%
% N - number of subjects
% D - number of features
% K - number of feature sets
% data   - [D x N x K] matrix
% labels - [N x 1] vector; 1 for controls, -1 for patients
%
% M. J. Rosa, Centre for Neuroimaging Sciences, King's College London
% -------------------------------------------------------------------------
clear all
clc

addpath(genpath('/cns_zfs/system/system_ce51_64/PRoNTo/PRoNTo_v1.1_beta'))
addpath(genpath('/cns_zfs/system/system_ce51_64/shogun/shogun-2.0.0'))
addpath(genpath('/cns_zfs/system/system_s9_sparc/spm/spm-8-5236'))


% Load all data [N x D x K]
% -------------------------------------------------------------------------
p  = spm_select(Inf,'any','Select feature sets (one file per set, .csv or .mat)');
nK = size(p,1);
tmpstr = strtrim(p(1,1:end));

switch tmpstr(end-3:end)
    case '.mat'
        for k = 1:nK
            tmp = load(strtrim(p(k,:)));
            data{k} = tmp.data;
        end
    case '.csv'
        for k = 1:nK
            data{k} = csvread(strtrim(p(k,:)),0,0);
        end
    otherwise
        disp('File format not supported!')
        break
end

% Load all labels [N x 1] vector called 'labels'
% -------------------------------------------------------------------------
p  = spm_select(1,'any','Select labels (.csv or .mat)');
tmpstr = strtrim(p(1,1:end));

switch tmpstr(end-3:end)
    case '.mat'
            tmp    = load(strtrim(p(1,:)));
            labels = tmp.labels;
    case '.csv'
            labels = csvread(strtrim(p(1,:)),0,0);
    otherwise
        disp('File format not supported!')
        break
end

% Initialise SVM
% -------------------------------------------------------------------------
nperm     = 1;    % Number of permutations (leave = 1 if you don't want to run a permutation test)                    
pval      = 0;
N         = size(data{1},2);
C_values  = [0.000001 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000]; 
mkl_norm  = 2;

% Bluid linear kernels
% -------------------------------------------------------------------------
for n = 1:nK
    Kt(:,:,n) = data{n}'*data{n};
end

% Permutations loop
% -------------------------------------------------------------------------
for p = 1:nperm,   
    disp(sprintf('Permutation>>>: %d out of %d',p,nperm));
    
    % Permutes labels for permutation test
    % ---------------------------------------------------------------------
    if p~=1  
        rpidx = randperm(length(labels));
    else
        rpidx = 1:length(labels);
    end
    uselabels = labels(rpidx);
    
    % Finds classes
    % ---------------------------------------------------------------------
    c1 = find(uselabels == 1);  % Positive class
    c2 = find(uselabels == -1); % Negative class
    c  = [c1; c2];

    % Main cross-validation loop
    % ---------------------------------------------------------------------
    for stest = 1:N,   
        disp(sprintf('Fold---------------------->>>: %d out of %d',stest,N));
        
        % Find train and test
        train = c(find(c~=c(stest)));       
        test  = c(stest);
        
        % Get part of kernel for train and test
        K_train   = Kt(train,train,:);
        K_test    = Kt(test,train,:);
        K_testcov = Kt(test,test,:);  
        
        % Optimise SVM-C parameter
        c_cv  = nested_cv_mkl(K_train,uselabels(train),C_values,mkl_norm);
        
        % Mean centre kernel
        for k = 1:nK
            [K_train_k,K_test_k,K_testcov_k] = prt_centre_kernel(K_train(:,:,k), K_test(:,:,k), K_testcov(:,:,k));
            K_train(:,:,k)   = K_train_k;
            K_test(:,:,k)    = K_test_k;
            K_testcov(:,:,k) = K_testcov_k;
            
            % Normalize kernel
            Phi = [K_train(:,:,k), K_test(:,:,k)'; K_test(:,:,k), K_testcov(:,:,k)];
            Phi = prt_normalise_kernel(Phi);
            
            tr = 1:size(K_train(:,:,k),1);
            te = (1:size(K_test(:,:,k),1))+max(tr);
            K_train_norm(:,:,k)    = Phi(tr,tr);
            K_test_norm(:,:,k)     = Phi(te,tr)';
            K_testcov_norm(:,:,k)  = Phi(te,te);
        end
        
        % Train model
        y = uselabels(train)';
        svm = train_sgmklSVM( 'K', K_train_norm, 'y', y, 'mkl_norm', mkl_norm, 'C', c_cv);
        
        % Test model
        y = uselabels(test)';
        test_outputs = apply_sgmklSVM( 'K', K_test_norm, 'svm', svm);
        acc_fold(stest)  = 1 - mean(y ~= sign(test_outputs));
    end   
    
    % Calculate p-value (permutation test)
    % ---------------------------------------------------------------------
    if p == 1,
        final_accuracy = sum(acc_fold)/N;
        final_sens     = sum(acc_fold(c1)/length(c1));
        final_spec     = sum(acc_fold(c2)/length(c2));
        
        disp('-----------------------')
        disp('Final results----------')
        disp(sprintf('Total accuracy: %d%%',final_accuracy));
        disp(sprintf('Total sensitivity: %d%%',final_sens));
        disp(sprintf('Total specificity: %d%%',final_spec));
    else
        if final_accuracy >= acc_1, pval = pval + 1; end
    end

end

% Display results of permutation test
% -------------------------------------------------------------------------
pval = pval/nperm;
if p>1, disp(sprintf('Permutation p-val: %d',pval)); end