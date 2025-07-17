function [model,runtime,iter_time] = TWRR_ALS(para, rank, rank_l, X,Y, Xv, Yv )
%% set parameters

iterStart = 0;
iterTotal = para.maxiter;
shakyRate = 1.5;
N=para.N;
L=para.L;
M=para.M;
lambda=para.lambda;
NN = size(X,1);

% X = randn([20 7 8 9]); Y = randn([20 4]);

%% initialization
[model,core,wheels] = init_model(para,rank,rank_l);

%% 停止条件准备
minTrainingFuncValue = calcobjfunction(para,model, X,Y);
minValidationFuncValue = calcobjfunction(para,model,Xv,Yv);

disp('Train the model. Running...');
tic;

%% main loop
iter_time(1) = 0;
for iter = iterStart+1:iterTotal
    iniIterTime = tic;
    %% update first L dimensions  
    for i = 1:L 
        V = wheels{L+1};
        for j = L+2:N
            V = tensor_prod(V,wheels{j});
        end
        if L == 1
            %% test 用结构
            cc = classical_mode_unfolding(core{1},1);
            ccc = tensorprod(V,cc,3,2);
            ccc = permute(ccc,[2 3 4 1]);
            szz = rank(1) * rank(2) * rank_l(1);
            ccc = reshape(ccc,numel(ccc)/szz,szz);
            CV = kron(X,ccc);
            ATA = CV' * CV;
            
            y = reshape(Y,numel(Y),1);
            AY = CV' * y;
            
            HTH = ccc' * ccc;
            
            vec_u = (ATA + lambda * (kron(HTH,eye(para.P(1)))))\AY;
            wheels{1} = permute(reshape(vec_u,[para.P(1),rank(1),rank_l(1),rank(2)]),[2,1,3,4]);
        else
            u_left = wheels{1};
            for j = 2:i-1
                u_left = tensor_prod(u_left,wheels{j});
            end
            u_right = wheels{i+1};
            for j = i+2:L
                u_right = tensor_prod(u_right,wheels{j});
            end
            szz = horzcat(NN,prod(para.P(1:i-1)),para.P(i),prod(para.P(i+1:L)));
            po = szz == 1;
            szz(po) = [];
            xx = reshape(X,szz);

            szz = horzcat(prod(rank_l(1:i-1)),rank_l(i),prod(rank_l(i+1:L)),prod(rank_l(L+1:N)));
            po = szz == 1;
            szz(po) = [];
            cc = reshape(core{1},szz);
            %% test 用结构
            if i == 1
                C = tensorprod(xx,u_right,3,2);
                CV = tensorprod(C,V,5,1);
                CV = tensorprod(CV,cc,[4 6],[2 3]);
                CV = permute(CV,[1 4 2 5 6 3]);
                szz = rank(1) * rank(2) * rank_l(1) * para.P(1);
                CV = reshape(CV,numel(CV)/szz,szz);
                ATA = CV' * CV;
                
                y = reshape(Y,numel(Y),1);
                AY = CV' * y;
                
                vv = tensorprod(V,V,2,2);
                uu_right = tensorprod(u_right,u_right,2,2);
                ccc = tensorprod(vv,uu_right,[1 4],[3 6]);
                ccc = tensorprod(ccc,cc,[1 6],[3 2]);
                ccc = tensorprod(ccc,cc,[2 6],[3 2]);
                ccc = permute(ccc,[1 5 3 2 6 4]);
                szz = rank(1) * rank(2) * rank_l(1);
                HTH = reshape(ccc,szz,szz);

                vec_u = (ATA + lambda * (kron(HTH,eye(para.P(i)))))\AY;
                wheels{i} = permute(reshape(vec_u,[para.P(i),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            elseif i == L
                C = tensorprod(xx,u_left,2,2);
                CV = tensorprod(C,V,3,4);
                CV = tensorprod(CV,cc,[3 7],[1 3]);
                CV = permute(CV,[1 5 2 3 6 4]);
                szz = rank(i) * rank(i+1) * rank_l(i) * para.P(i);
                CV = reshape(CV,numel(CV)/szz,szz);
                ATA = CV' * CV;

                y = reshape(Y,numel(Y),1);
                AY = CV' * y;

                vv = tensorprod(V,V,2,2);
                uu_left = tensorprod(u_left,u_left,2,2);
                ccc = tensorprod(uu_left,vv,[1 4],[3 6]);
                ccc = tensorprod(ccc,cc,[1 6],[1 3]);
                ccc = tensorprod(ccc,cc,[2 6],[1 3]);
                ccc = permute(ccc,[1 5 3 2 6 4]);
                szz = rank(i) * rank(i+1) * rank_l(i);
                HTH = reshape(ccc,szz,szz);

                vec_u = (ATA + lambda * (kron(HTH,eye(para.P(i)))))\AY;
                wheels{i} = permute(reshape(vec_u,[para.P(i),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            else
                C = tensorprod(xx,u_left,2,2);
                C = tensorprod(C,u_right,3,2);
                CV = tensorprod(C,V,[3 8],[4 1]);
                CV = tensorprod(CV,cc,[3 6 8],[1 3 4]);
                CV = permute(CV,[1 5 2 3 6 4]);
                szz = rank(i) * rank(i+1) * rank_l(i) * para.P(i);
                CV = reshape(CV,numel(CV)/szz,szz);
                ATA = CV' * CV;

                y = reshape(Y,numel(Y),1);
                AY = CV' * y;

                vv = tensorprod(V,V,2,2);
                uu_left = tensorprod(u_left,u_left,2,2);
                uu_right = tensorprod(u_right,u_right,2,2);
                ccc = tensorprod(uu_right,vv,[3 6],[1 4]);
                ccc = tensorprod(ccc,uu_left,[6 8],[1 4]);
                ccc = tensorprod(ccc,cc,[2 5 7],[3 4 1]);
                ccc = tensorprod(ccc,cc,[3 4 6],[3 4 1]);
                ccc = permute(ccc,[3 5 1 4 6 2]);
                szz = rank(i) * rank(i+1) * rank_l(i);
                HTH = reshape(ccc,szz,szz);
                
                vec_u = (ATA + lambda * (kron(HTH,eye(para.P(i)))))\AY;
                wheels{i} = permute(reshape(vec_u,[para.P(i),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            end
    
        end
    end
    %% update last M dimensions
    for i = L+1:N
        U = wheels{1};
        for j = 2:L
            U = tensor_prod(U,wheels{j});
        end
        D = classical_mode_unfolding(X,1);
        D = tensorprod(D,U,2,2);

        szz = horzcat(prod(rank_l(1:L)),prod(rank_l(L+1:i-1)),rank_l(i),prod(rank_l(i+1:N)));
        po = szz == 1;
        szz(po) = [];
        cc = reshape(core{1},szz);
        if M == 1
            %cc = classical_mode_unfolding(core{1},N);
            D = tensorprod(D,cc,3,1);
            %% test 用结构
            szz = rank(1) * rank(L+1) * rank_l(L+1);
            D = reshape(permute(D,[1 3 4 2]),[NN,szz]);
            BTB = D' * D;
            
            YB = tensorprod(Y,D,1,1);
            
            uu = tensorprod(U,U,2,2);
            FTF = tensorprod(uu,cc,2,1);
            FTF = tensorprod(FTF,cc,4,1);
            FTF = reshape(permute(FTF,[2 5 1 4 6 3]),[szz,szz]);

            unf_v = YB / (BTB + lambda * FTF);
            wheels{i} = permute(reshape(unf_v,[para.Q(i-L),rank(i),rank_l(i),rank(1)]),[2,1,3,4]);
        else
            if i == L+1
                v_right = wheels{i+1};
                for j = i+2:N
                    v_right = tensor_prod(v_right,wheels{j});
                end
                vv_right = tensorprod(v_right,v_right,2,2);
                DD = tensorprod(D,D,1,1);
                BTB = tensorprod(vv_right,DD,[3 6],[1 4]);
                BTB = tensorprod(BTB,cc,[2 5],[3 1]);
                BTB = tensorprod(BTB,cc,[3 5],[3 1]);
                szz = rank(i) * rank(i+1) * rank_l(i);
                BTB = reshape(permute(BTB,[3 5 1 4 6 2]),[szz szz]);

                UU = tensorprod(U,U,2,2);
                FTF = tensorprod(vv_right,UU,[3 6],[1 4]);
                FTF = tensorprod(FTF,cc,[2 5],[3 1]);
                FTF = tensorprod(FTF,cc,[3 5],[3 1]);
                szz = rank(i) * rank(i+1) * rank_l(i);
                FTF = reshape(permute(FTF,[3 5 1 4 6 2]),[szz szz]);

                szz = horzcat(NN,prod(para.Q(1:i-L-1)),para.Q(i-L),prod(para.Q(i-L+1:M)));
                po = szz == 1;
                szz(po) = [];
                
                y = reshape(Y,szz);
                YB = tensorprod(y,v_right,3,2);
                YB = tensorprod(YB,D,[1 5],[1 2]);
                YB = tensorprod(YB,cc,[3 4],[3 1]);
                YB = permute(YB,[1 3 4 2]);
                YB = reshape(YB,para.Q(i-L),numel(YB)/para.Q(i-L));
    
                unf_v = YB / (BTB + lambda * FTF);
                wheels{i} = permute(reshape(unf_v,[para.Q(i-L),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            elseif i == N
                v_left = wheels{L+1};
                for j = L+2:i-1
                    v_left = tensor_prod(v_left,wheels{j});
                end

                vv_left = tensorprod(v_left,v_left,2,2);
                DD = tensorprod(D,D,1,1);
                BTB = tensorprod(vv_left,DD,[1 4],[3 6]);
                BTB = tensorprod(BTB,cc,[1 6],[2 1]);
                BTB = tensorprod(BTB,cc,[2 6],[2 1]);
                szz = rank(i) * rank(1) * rank_l(i);
                BTB = reshape(permute(BTB,[1 5 3 2 6 4]),[szz szz]);

                UU = tensorprod(U,U,2,2);
                FTF = tensorprod(vv_left,UU,[1 4],[3 6]);
                FTF = tensorprod(FTF,cc,[1 6],[2 1]);
                FTF = tensorprod(FTF,cc,[2 6],[2 1]);
                FTF = reshape(permute(FTF,[1 5 3 2 6 4]),[szz szz]);

                szz = horzcat(NN,prod(para.Q(1:i-L-1)),para.Q(i-L),prod(para.Q(i-L+1:M)));
                po = szz == 1;
                szz(po) = [];
                
                y = reshape(Y,szz);
                YB = tensorprod(y,v_left,2,2);
                YB = tensorprod(YB,D,[1 3],[1 4]);
                YB = tensorprod(YB,cc,[2 5],[2 1]);
                YB = permute(YB,[1 2 4 3]);
                YB = reshape(YB,para.Q(i-L),numel(YB)/para.Q(i-L));
    
                unf_v = YB / (BTB + lambda * FTF);
                wheels{i} = permute(reshape(unf_v,[para.Q(i-L),rank(i),rank_l(i),rank(1)]),[2,1,3,4]);

            else
                v_left = wheels{L+1};
                for j = L+2:i-1
                    v_left = tensor_prod(v_left,wheels{j});
                end
                v_right = wheels{i+1};
                for j = i+2:N
                    v_right = tensor_prod(v_right,wheels{j});
                end
                %% test 用结构
                vv_left = tensorprod(v_left,v_left,2,2);
                vv_right = tensorprod(v_right,v_right,2,2);
                DD = tensorprod(D,D,1,1);
                
                BTB = tensorprod(DD,vv_left,[3 6],[1 4]);
                BTB = tensorprod(BTB,vv_right,[1 3],[3 6]);
                BTB = tensorprod(BTB,cc,[1 3 8],[1 2 4]);
                BTB = tensorprod(BTB,cc,[1 3 7],[1 2 4]);
                BTB = permute(BTB,[1 5 3 2 6 4]);
                szz = rank(i) * rank(i+1) * rank_l(i);
                BTB = reshape(BTB,szz,szz);

                UU = tensorprod(U,U,2,2);
                FTF = tensorprod(UU,vv_left,[3 6],[1 4]);
                FTF = tensorprod(FTF,vv_right,[1 3],[3 6]);
                FTF = tensorprod(FTF,cc,[1 3 8],[1 2 4]);
                FTF = tensorprod(FTF,cc,[1 3 7],[1 2 4]);
                FTF = permute(FTF,[1 5 3 2 6 4]);
                szz = rank(i) * rank(i+1) * rank_l(i);
                FTF = reshape(FTF,szz,szz);
                
                szz = horzcat(NN,prod(para.Q(1:i-L-1)),para.Q(i-L),prod(para.Q(i-L+1:M)));
                y = reshape(Y,szz);
                YB = tensorprod(y,v_left,2,2);
                YB = tensorprod(YB,v_right,3,2);
                YB = tensorprod(YB,D,[1 3 8],[1 4 2]);
                YB = tensorprod(YB,cc,[6 2 5],[1 2 4]);
                YB = permute(YB,[1 2 4 3]);
                YB = reshape(YB,para.Q(i-L),numel(YB)/para.Q(i-L));
    
                unf_v = YB / (BTB + lambda * FTF);
                wheels{i} = permute(reshape(unf_v,[para.Q(i-L),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            end 
        end
    end
    
    
    %% updata core factor
    U = wheels{1};
    for j = 2:L
        U = tensor_prod(U,wheels{j});
    end
    V = wheels{L+1};
    for j = L+2:N
        V = tensor_prod(V,wheels{j});
    end
    
    xx = classical_mode_unfolding(X,1);
    C = tensorprod(xx,U,2,2);
    cc = tensorprod(C,C,1,1);
    vv = tensorprod(V,V,2,2);
    ATA = tensorprod(cc,vv,[1 3 4 6],[3 1 6 4]);
    ATA = permute(ATA,[1 3 2 4]);
    ATA = reshape(ATA,prod(rank_l),prod(rank_l));
    
    uu = tensorprod(U,U,2,2);
    HTH = tensorprod(uu,vv,[1 3 4 6],[3 1 6 4]);
    HTH = permute(HTH,[1 3 2 4]);
    HTH = reshape(HTH,prod(rank_l),prod(rank_l));

    y = classical_mode_unfolding(Y,1);
    AY = tensorprod(y,V,2,2);
    AY = tensorprod(C,AY,[1 2 4],[1 4 2]);
    AY = reshape(AY,numel(AY),1);
    
    vec_u = (ATA + lambda * HTH)\AY;
    core{1} = reshape(vec_u,rank_l);
    iter_time(iter+1) = iter_time(iter) + toc(iniIterTime);
    
    %% compute error
    model = tensor_core_tensor(wheels,core,rank_l);
   
   
    trainingFuncValue = calcobjfunction(para,model,X ,Y);
    trainerror(iter)=trainingFuncValue;
    if abs(trainingFuncValue - minTrainingFuncValue)/minTrainingFuncValue<=1e-3
        break;
    end      
    if trainingFuncValue <= shakyRate * minTrainingFuncValue
        minTrainingFuncValue = min(minTrainingFuncValue, trainingFuncValue);
        disp('descening');
    else
        disp('not descening, error');
        break;
    end
       
    validationFuncValue = calcobjfunction(para,model,Xv,Yv);
    
    disp(['    Iter: ', num2str(iter), '. Training: ', num2str(trainingFuncValue), '. Validation: ', num2str(validationFuncValue)]);
    
    minValidationFuncValue = min(minValidationFuncValue, validationFuncValue);
   
    
end
runtime=toc;
disp('Train the model. Finish.');

end