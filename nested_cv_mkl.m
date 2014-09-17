function c_cv = nested_cv_mkl(Kt_cv,labels_cv, C_values, mkl_norm) 

% -------------------------------------------------------------------------
% FORMAT c_cv = nested_cv_mkl(Kt_cv,labels_cv, C_values, mkl_norm)
% 
% Nested cross-validation to optimise C parameter (MKL-SVM)
%
% Kt_cv     - kernels
% labels_cv - labels
% C_values  - C values
% mkl_norm  - MKL norm
%
% M. J. Rosa, Centre for Neuroimaging Sciences, King's College London
% -------------------------------------------------------------------------
 
n_subjects = size(Kt_cv,1);
idx        = 1:n_subjects;
nK         = size(Kt_cv,3);
 
for i = 1:n_subjects
    
    % Index test/train subjects
    cv_test  = i;
    cv_train = idx(idx~=i);
    
    % Get kernels
    K_train_cv    = Kt_cv(cv_train,cv_train,:);
    K_test_cv     = Kt_cv(cv_test,cv_train,:);
    K_testcov_cv  = Kt_cv(cv_test,cv_test,:);
    
    for k = 1:nK
        % Mean centre kernel
        [K_train_k,K_test_k,K_testcov_k] = prt_centre_kernel(K_train_cv(:,:,k), K_test_cv(:,:,k), K_testcov_cv(:,:,k));
        K_train(:,:,k)   = K_train_k;
        K_test(:,:,k)    = K_test_k;
        K_testcov(:,:,k) = K_testcov_k;
        
        % Normalize kernel
        Phi = [K_train(:,:,k), K_test(:,:,k)'; K_test(:,:,k), K_testcov(:,:,k)];
        Phi = prt_normalise_kernel(Phi);
        tr  = 1:size(K_train(:,:,k),1);
        te  = (1:size(K_test(:,:,k),1))+max(tr);
        K_train_norm(:,:,k)    = Phi(tr,tr);
        K_test_norm(:,:,k)     = Phi(te,tr)';
        K_testcov_norm(:,:,k)  = Phi(te,te);
    end
    
    for j = 1:length(C_values)
        
        C = C_values(j);
        
        % Train MKL-SVM
        y   = labels_cv(cv_train)';
        svm = train_sgmklSVM( 'K', K_train_norm, 'y', y, 'mkl_norm', mkl_norm, 'C', C);
        
        % Test MKL-SVM
        y = labels_cv(cv_test)';
        test_outputs = apply_sgmklSVM( 'K', K_test_norm, 'svm', svm);
        acc_cv(i,j)  = 1 - mean(y ~= sign(test_outputs));
        
    end
    
end

mean_cv_acc = mean(acc_cv,1); 
I           = round(median(find(mean_cv_acc==max(mean_cv_acc))));
c_cv        = C_values(I);