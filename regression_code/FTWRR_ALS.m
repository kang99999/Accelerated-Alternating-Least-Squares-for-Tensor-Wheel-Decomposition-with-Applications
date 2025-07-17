function [model,runtime,iter_time] = FTWRR_ALS(para, rank, rank_l, X,Y, Xv, Yv )
%% set parameters
%% ALS_NE with fast GRAM
iterStart = 0;
iterTotal = para.maxiter;
shakyRate = 1.5;
N=para.N;
L=para.L;
M=para.M;
lambda=para.lambda;
% eta=para.eta;
% gm=para.gm;
NN = size(X,1);

% X = randn([20 7 8 9]); Y = randn([20 4]);

%% initialization
[model,core,wheels] = init_model(para,rank,rank_l);
pp = cell(N,1);
%% 停止条件准备
minTrainingFuncValue = calcobjfunction(para,model, X,Y);
minValidationFuncValue = calcobjfunction(para,model,Xv,Yv);

disp('Train the model. Running...');
tic;
iter_time(1) = 0;
%% main loop
for iter = iterStart+1:iterTotal
    iniIterTime = tic;
    %% update first L dimensions  
    for i = 1:L 
        V = wheels{L+1};
        for j = L+2:N
            V = tensor_prod(V,wheels{j});
        end
        for j = L+1:N
            pp{j} = tensorprod(wheels{j},wheels{j},2,2);
        end

        vv = pp{L+1};
        for j = L+2:N
            vv = y_prod(vv,pp{j});
        end

        if L == 1
            %% test 用结构
            C = X;
            co = classical_mode_unfolding(core{1},1);
            vv = tensorprod(vv,co,2,2);
            vv = tensorprod(vv,co,4,2);
            vv = permute(vv,[2 5 1 4 6 3]);
            szz = rank(1) * rank_l(1) * rank(2);
            HTH = reshape(vv,szz,szz);
            cc = C' * C;
            ATA = kron(cc,HTH);
            
            y = reshape(Y,NN,numel(Y)/NN);
            AY = tensorprod(X,y,1,1);
            AY = tensorprod(AY,V,2,2);
            AY = tensorprod(AY,co,3,2);
            AY = permute(AY,[1 3 4 2]);
            AY = reshape(AY,numel(AY),1);
            
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
            for j = 1:L
                if j ~= i
                    pp{j} = tensorprod(wheels{j},wheels{j},2,2);
                end
            end
    
            up_left = pp{1};
            for j = 2:i-1
                up_left = y_prod(up_left,pp{j});
            end
            up_right = pp{i+1};
            for j = i+2:L
                up_right = y_prod(up_right,pp{j});
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
                UU = tensorprod(up_right,cc,2,2);
                UU = tensorprod(UU,cc,4,2);
                UU = tensorprod(UU,vv,[2 4 6 8],[1 4 2 5]);
                UU = permute(UU,[5 3 1 6 4 2]);
                szz = rank(1) * rank_l(1) * rank(2);
                HTH = reshape(UU,szz,szz);

                C = tensorprod(xx,u_right,3,2);
                CC = tensorprod(C,C,1,1);
                CC = tensorprod(CC,cc,3,2);
                CC = tensorprod(CC,cc,6,2);
                ATA = tensorprod(CC,vv,[3 6 8 10],[1 4 2 5]);
                ATA = permute(ATA,[1 7 5 2 3 8 6 4]);
                szz = szz * para.P(1);
                ATA = reshape(ATA,szz,szz);
                
                y = reshape(Y,NN,numel(Y)/NN);
                AY = tensorprod(y,C,1,1);
                AY = tensorprod(AY,V,[1 5],[2 1]);
                AY = tensorprod(AY,cc,[3 4],[2 3]);
                AY = permute(AY,[1 3 4 2]);
                AY = reshape(AY,numel(AY),1);

                vec_u = (ATA + lambda * (kron(HTH,eye(para.P(i)))))\AY;
                wheels{i} = permute(reshape(vec_u,[para.P(i),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            elseif i == L
                UU = tensorprod(up_left,cc,2,1);
                UU = tensorprod(UU,cc,4,1);
                UU = tensorprod(UU,vv,[1 3 6 8],[3 6 2 5]);
                UU = permute(UU,[1 3 5 2 4 6]);
                szz = rank(L) * rank_l(L) * rank(L+1);
                HTH = reshape(UU,szz,szz);

                C = tensorprod(xx,u_left,2,2);
                CC = tensorprod(C,C,1,1);
                CC = tensorprod(CC,cc,3,1);
                CC = tensorprod(CC,cc,6,1);
                ATA = tensorprod(CC,vv,[2 5 8 10],[3 6 2 5]);
                ATA = permute(ATA,[1 2 5 7 3 4 6 8]);
                szz = szz * para.P(L);
                ATA = reshape(ATA,szz,szz);
                
                y = reshape(Y,NN,numel(Y)/NN);
                AY = tensorprod(y,C,1,1);
                AY = tensorprod(AY,V,[1 3],[2 4]);
                AY = tensorprod(AY,cc,[2 5],[1 3]);
                AY = permute(AY,[1 2 4 3]);
                AY = reshape(AY,numel(AY),1);

                vec_u = (ATA + lambda * (kron(HTH,eye(para.P(i)))))\AY;
                wheels{i} = permute(reshape(vec_u,[para.P(i),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            else
                medi = tensorprod(up_right,vv,[3 6],[1 4]);
                medi = tensorprod(medi,cc,[2 5],[3 4]);
                medi = tensorprod(medi,cc,[3 5],[3 4]);
                UU = tensorprod(up_left,medi,[1 4 2 5],[3 4 5 7]);
                UU = permute(UU,[5 3 1 6 4 2]);
                szz = rank(i) * rank_l(i) * rank(i+1);
                HTH = reshape(UU,szz,szz);

                C = tensorprod(xx,u_left,2,2);
                C = tensorprod(C,u_right,3,2);
                %% 14 阶 我了个去
                CC = tensorprod(C,C,1,1);
                CC = tensorprod(CC,cc,[3 6],[1 3]);
                CC = tensorprod(CC,cc,[8 11],[1 3]);
                ATA = tensorprod(CC,vv,[2 5 7 10 12 14],[3 1 6 4 2 5]);
                ATA = permute(ATA,[1 2 7 3 4 5 8 6]);
                szz = szz * para.P(i);
                ATA = reshape(ATA,szz,szz);
                
                y = reshape(Y,NN,numel(Y)/NN);
                AY = tensorprod(y,C,1,1);
                AY = tensorprod(AY,V,[1 3 8],[2 4 1]);
                AY = tensorprod(AY,cc,[2 5 6],[1 3 4]);
                AY = permute(AY,[1 2 4 3]);
                AY = reshape(AY,numel(AY),1);
                
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

        for j = 1:L
            pp{j} = tensorprod(wheels{j},wheels{j},2,2);
        end
        uu = pp{1};
        for j = 2:L
            uu = y_prod(uu,pp{j});
        end

        szz = horzcat(prod(rank_l(1:L)),prod(rank_l(L+1:i-1)),rank_l(i),prod(rank_l(i+1:N)));
        po = szz == 1;
        szz(po) = [];
        cc = reshape(core{1},szz);
        if M == 1
            D = tensorprod(D,cc,3,1);

            FTF = tensorprod(uu,cc,2,1);
            FTF = tensorprod(FTF,cc,4,1);
            FTF = permute(FTF,[2 5 1 4 6 3]);
            szz = rank(i) * rank_l(i) * rank(1);
            FTF = reshape(FTF,szz,szz);
            dd = tensorprod(D,D,1,1);
            BTB = permute(dd,[2 3 1 5 6 4]);
            BTB = reshape(BTB,szz,szz);
            
            YB = tensorprod(Y,D,1,1);
            YB = reshape(YB,para.Q(i-L),numel(YB)/para.Q(i-L));

            unf_v = YB / (BTB + lambda * FTF);
            wheels{i} = permute(reshape(unf_v,[para.Q(i-L),rank(i),rank_l(i),rank(1)]),[2,1,3,4]);
        else
            if i == L+1
                v_right = wheels{i+1};
                for j = i+2:N
                    v_right = tensor_prod(v_right,wheels{j});
                end
                for j = i+1:N
                    pp{j} = tensorprod(wheels{j},wheels{j},2,2);
                end
                vv = pp{i+1};
                for j = i+2:N
                    vv = y_prod(vv,pp{j});
                end
                
                FTF = tensorprod(uu,vv,[1 4],[3 6]);
                FTF = tensorprod(FTF,cc,[1 6],[1 3]);
                FTF = tensorprod(FTF,cc,[2 6],[1 3]);
                FTF = permute(FTF,[1 5 3 2 6 4]);
                szz = rank(i) * rank_l(i) * rank(i+1);
                FTF = reshape(FTF,szz,szz);

                dd = tensorprod(D,D,1,1);
                BTB = tensorprod(dd,vv,[1 4],[3 6]);
                BTB = tensorprod(BTB,cc,[1 6],[1 3]);
                BTB = tensorprod(BTB,cc,[2 6],[1 3]);
                BTB = permute(BTB,[1 5 3 2 6 4]);
                BTB = reshape(BTB,szz,szz);
                
                szz = horzcat(NN,prod(para.Q(1:i-L-1)),para.Q(i-L),prod(para.Q(i-L+1:M)));
                po = szz == 1;
                szz(po) = [];
                
                y = reshape(Y,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,v_right,[2 3],[2 4]);
                YB = tensorprod(YB,cc,[2 5],[1 3]);
                YB = permute(YB,[1 2 4 3]);
                YB = reshape(YB,para.Q(i-L),numel(YB)/para.Q(i-L));

                unf_v = YB / (BTB + lambda * FTF);
                wheels{i} = permute(reshape(unf_v,[para.Q(i-L),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            elseif i == N
                v_left = wheels{L+1};
                for j = L+2:i-1
                    v_left = tensor_prod(v_left,wheels{j});
                end
                
                for j = L+1:N-1
                    pp{j} = tensorprod(wheels{j},wheels{j},2,2);
                end
                vv = pp{L+1};
                for j = i+2:N-1
                    vv = y_prod(vv,pp{j});
                end
                
                FTF = tensorprod(uu,vv,[3 6],[1 4]);
                FTF = tensorprod(FTF,cc,[2 5],[1 2]);
                FTF = tensorprod(FTF,cc,[3 5],[1 2]);
                FTF = permute(FTF,[3 5 1 4 6 2]);
                szz = rank(i) * rank_l(i) * rank(1);
                FTF = reshape(FTF,szz,szz);

                dd = tensorprod(D,D,1,1);
                BTB = tensorprod(dd,vv,[3 6],[1 4]);
                BTB = tensorprod(BTB,cc,[2 5],[1 2]);
                BTB = tensorprod(BTB,cc,[3 5],[1 2]);
                BTB = permute(BTB,[3 5 1 4 6 2]);
                BTB = reshape(BTB,szz,szz);
                
                szz = horzcat(NN,prod(para.Q(1:i-L-1)),para.Q(i-L),prod(para.Q(i-L+1:M)));
                po = szz == 1;
                szz(po) = [];
                
                y = reshape(Y,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,v_left,[1 5],[2 1]);
                YB = tensorprod(YB,cc,[3 4],[1 2]);
                YB = permute(YB,[1 3 4 2]);
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
                
                for j = L+1:N
                    if j ~= i
                        pp{j} = tensorprod(wheels{j},wheels{j},2,2);
                    end
                end
                vv_left = pp{L+1};
                for j = L+2:i-1
                    vv_left = y_prod(vv_left,pp{j});
                end
                vv_right = pp{i+1};
                for j = i+2:N
                    vv_right = y_prod(vv_right,pp{j});
                end
                
                FTF = tensorprod(uu,vv_left,[3 4],[1 4]);
                FTF = tensorprod(FTF,vv_right,[1 3],[3 6]);
                FTF = tensorprod(FTF,cc,[1 3 8],[1 2 4]);
                FTF = tensorprod(FTF,cc,[1 3 7],[1 2 4]);
                FTF = permute(FTF,[1 5 3 2 6 4]);
                szz = rank(i) * rank_l(i) * rank(i+1);
                FTF = reshape(FTF,szz,szz);

                dd = tensorprod(D,D,1,1);
                BTB = tensorprod(dd,vv_left,[3 4],[1 4]);
                BTB = tensorprod(BTB,vv_right,[1 3],[3 6]);
                BTB = tensorprod(BTB,cc,[1 3 8],[1 2 4]);
                BTB = tensorprod(BTB,cc,[1 3 7],[1 2 4]);
                BTB = permute(BTB,[1 5 3 2 6 4]);
                BTB = reshape(BTB,szz,szz);
                
                szz = horzcat(NN,prod(para.Q(1:i-L-1)),para.Q(i-L),prod(para.Q(i-L+1:M)));
                y = reshape(Y,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,v_left,[1 6],[2 1]);
                YB = tensorprod(YB,v_right,[2 3],[2 4]);
                YB = tensorprod(YB,cc,[2 3 6],[1 2 4]);
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
%     vv = tensorprod(V,V,2,2);
    for j = L+1:N
        pp{j} = tensorprod(wheels{j},wheels{j},2,2);
    end
    vv = pp{L+1};
    for j = L+2:N
        vv = y_prod(vv,pp{j});
    end

    ATA = tensorprod(cc,vv,[1 3 4 6],[3 1 6 4]);
    ATA = permute(ATA,[1 3 2 4]);
    ATA = reshape(ATA,prod(rank_l),prod(rank_l));
    
%     uu = tensorprod(U,U,2,2);
    for j = 1:L
        pp{j} = tensorprod(wheels{j},wheels{j},2,2);
    end
    uu = pp{1};
    for j = 2:L
        uu = y_prod(uu,pp{j});
    end

    HTH = tensorprod(uu,vv,[1 3 4 6],[3 1 6 4]);
    HTH = permute(HTH,[1 3 2 4]);
    HTH = reshape(HTH,prod(rank_l),prod(rank_l));

    y = classical_mode_unfolding(Y,1);
    AY = tensorprod(y,V,2,2);
    AY = tensorprod(C,AY,[1 2 4],[1 4 2]);
    AY = reshape(AY,numel(AY),1);
    
    vec_u = (ATA + lambda * HTH)\AY;
    core{1} = reshape(vec_u,rank_l);
%     core{1}(abs(core{1})<gm)=0;
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