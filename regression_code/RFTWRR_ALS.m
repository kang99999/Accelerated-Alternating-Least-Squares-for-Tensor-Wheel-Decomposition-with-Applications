function [model,runtime,iter_time] = RFTWRR_ALS(para, rank, rank_l, MM,X,Y, Xv, Yv )
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
%计算QR分解
sz=[para.P,para.Q];
Q=cell(L+M,1);
R=cell(L+M,1);
YYY=cell(M+1,1);
XXX=cell(L+1,1);
for n=1:L
    if MM(n)==sz(n)
        Q{n}=eye(sz(n));
        R{n}=wheels{n};
    else
        X1=X;
        for i=1:n-1
            X1=tensorprod(X1,randn(MM(i),sz(i)),2,2);
        end
        for i=n+1:L
            X1=tensorprod(X1,randn(MM(i),sz(i)),3,2);
        end
        X1=permute(X1,[2,1,3:L+1]);
        X1=reshape(X1,sz(n),[]);
        [u,~,~]=svd(X1,'econ');
        Q{n}=u(:,1:MM(n));
        R{n}=permute(tensorprod(wheels{n},Q{n},2,1),[1,4,2,3]);
    end
end
for n=1:L+1
    if n~=L+1
        XXX{n}=X;
        for i=1:n-1
            XXX{n}=tensorprod(XXX{n},Q{i},2,1);
        end
        for i=n+1:L
            XXX{n}=tensorprod(XXX{n},Q{i},3,1);
        end
        XXX{n}=ipermute(XXX{n},[1,n+1,2:n,n+2:L+1]);
    else
        XXX{L+1}=tensorprod(XXX{L},Q{L},L+1,1);
    end
end
for n=1:M
    if MM(L+n)==sz(L+n)
        Q{L+n}=eye(sz(L+n));
        R{L+n}=wheels{L+n};
    else
        Y1=Y;
        for i=1:n-1
            Y1=tensorprod(Y1,randn(MM(L+i),sz(L+i)),2,2);
        end
        for i=n+1:M
            Y1=tensorprod(Y1,randn(MM(L+i),sz(L+i)),3,2);
        end
        Y1=permute(Y1,[2,1,3:M+1]);
        Y1=reshape(Y1,sz(L+n),[]);
        [u,~,~]=svd(Y1,'econ');
        Q{L+n}=u(:,1:MM(L+n));
        R{L+n}=permute(tensorprod(wheels{L+n},Q{L+n},2,1),[1,4,2,3]);
    end
end
for n=1:M+1
    if n~=M+1
        YYY{n}=Y;
        for i=1:n-1
            YYY{n}=tensorprod(YYY{n},Q{L+i},2,1);
        end
        for i=n+1:M
            YYY{n}=tensorprod(YYY{n},Q{L+i},3,1);
        end
        YYY{n}=ipermute(YYY{n},[1,n+1,2:n,n+2:M+1]);
    else
        YYY{M+1}=tensorprod(YYY{M},Q{L+M},M+1,1);
    end
end
% for i=1:L
%     unG=mode_unfolding_1(wheels{i},2);
%     [Q{i},unR]=qr(unG,'econ');
%     szR=size(wheels{i});
%     szR(2)=size(unR,1);
%     R{i}=permute(reshape(unR,szR([2,3,4,1])),[4,1,2,3]);
% end

%% main loop
% YY=Y;
% for i=1:M
%     YY=tensorprod(YY,Q{L+i},2,1);
% end
iter_time(1) = 0;
for iter = iterStart+1:iterTotal
    iniIterTime = tic;
    %% update first L dimensions  
    
    
%     XX=X;
    YY=YYY{M+1};
    for i = 1:L 
        XX=XXX{i};
%         for j=i+1:L
%             XX=tensorprod(XX,Q{j},i+2,1);
%         end
%         for j=1:i-1
%             XX=tensorprod(XX,Q{j},2,1);
%         end
%         XX=ipermute(XX,[1,i+1:L+1,2:i]);
        V = R{L+1};
        for j = L+2:N
            V = tensor_prod(V,R{j});
        end
        for j = L+1:N
            pp{j} = tensorprod(R{j},R{j},2,2);
        end

        vv = pp{L+1};
        for j = L+2:N
            vv = y_prod(vv,pp{j});
        end

        if L == 1
            %% test 用结构
            C = XX;
            co = classical_mode_unfolding(core{1},1);
            vv = tensorprod(vv,co,2,2);
            vv = tensorprod(vv,co,4,2);
            vv = permute(vv,[2 5 1 4 6 3]);
            szz = rank(1) * rank_l(1) * rank(2);
            HTH = reshape(vv,szz,szz);
            cc = C' * C;
            ATA = kron(cc,HTH);
            
            y = reshape(YY,NN,numel(YY)/NN);
            AY = tensorprod(XX,y,1,1);
            AY = tensorprod(AY,V,2,2);
            AY = tensorprod(AY,co,3,2);
            AY = permute(AY,[1 3 4 2]);
            AY = reshape(AY,numel(AY),1);
            
            vec_u = (ATA + lambda * (kron(HTH,eye(para.P(1)))))\AY;
            wheels{1} = permute(reshape(vec_u,[para.P(1),rank(1),rank_l(1),rank(2)]),[2,1,3,4]);
        else
            u_left = R{1};
            for j = 2:i-1
                u_left = tensor_prod(u_left,R{j});
            end
            u_right = R{i+1};
            for j = i+2:L
                u_right = tensor_prod(u_right,R{j});
            end
            for j = 1:L
                if j ~= i
                    pp{j} = tensorprod(R{j},R{j},2,2);
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
            szXX=size(XX,2:L+1);
            szz = horzcat(NN,prod(szXX(1:i-1)),szXX(i),prod(szXX(i+1:L)));
            po = szz == 1;
            szz(po) = [];
            xx = reshape(XX,szz);

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
                szz = sqrt(numel(ATA));
                ATA = reshape(ATA,szz,szz);
                
                y = reshape(YY,NN,numel(YY)/NN);
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
                szz = sqrt(numel(ATA));
                ATA = reshape(ATA,szz,szz);
                
                y = reshape(YY,NN,numel(YY)/NN);
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
                
                y = reshape(YY,NN,numel(YY)/NN);
                AY = tensorprod(y,C,1,1);
                AY = tensorprod(AY,V,[1 3 8],[2 4 1]);
                AY = tensorprod(AY,cc,[2 5 6],[1 3 4]);
                AY = permute(AY,[1 2 4 3]);
                AY = reshape(AY,numel(AY),1);
                
                vec_u = (ATA + lambda * (kron(HTH,eye(para.P(i)))))\AY;
                wheels{i} = permute(reshape(vec_u,[para.P(i),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            end
    
        end
%         unG=mode_unfolding_1(wheels{i},2);
%         [Q{i},unR]=qr(unG,'econ');
%         szR=size(wheels{i});
%         szR(2)=size(unR,1);
%         R{i}=permute(reshape(unR,szR([2,3,4,1])),[4,1,2,3]);
        R{i}=permute(tensorprod(wheels{i},Q{i},2,1),[1,4,2,3]);
        pp{i}=tensorprod(R{i},R{i},2,2);
    end
    %% update last M dimensions
    XX=XXX{L+1};
%     YY=Y;
    for i = L+1:N
        YY=YYY{i-L};
%         for j=i+1:N
%             YY=tensorprod(YY,Q{j},i-L+2,1);
%         end
%         for j=L+1:i-1
%             YY=tensorprod(YY,Q{j},2,1);
%         end
%         YY=ipermute(YY,[1,i-L+1:M+1,2:i-L]);
        U = R{1};
        for j = 2:L
            U = tensor_prod(U,R{j});
        end
        D = classical_mode_unfolding(XX,1);
        D = tensorprod(D,U,2,2);

        for j = 1:L
            pp{j} = tensorprod(R{j},R{j},2,2);
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
            
            YB = tensorprod(YY,D,1,1);
            YB = reshape(YB,para.Q(i-L),numel(YB)/para.Q(i-L));

            unf_v = YB / (BTB + lambda * FTF);
            wheels{i} = permute(reshape(unf_v,[para.Q(i-L),rank(i),rank_l(i),rank(1)]),[2,1,3,4]);
        else
            if i == L+1
                v_right = R{i+1};
                for j = i+2:N
                    v_right = tensor_prod(v_right,R{j});
                end
                for j = i+1:N
                    pp{j} = tensorprod(R{j},R{j},2,2);
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
                szYY=size(YY,2:M+1);
                szz = horzcat(NN,prod(szYY(1:i-L-1)),szYY(i-L),prod(szYY(i-L+1:M)));
                po = szz == 1;
                szz(po) = [];
                
                y = reshape(YY,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,v_right,[2 3],[2 4]);
                YB = tensorprod(YB,cc,[2 5],[1 3]);
                YB = permute(YB,[1 2 4 3]);
                YB = reshape(YB,para.Q(i-L),numel(YB)/para.Q(i-L));

                unf_v = YB / (BTB + lambda * FTF);
                wheels{i} = permute(reshape(unf_v,[para.Q(i-L),rank(i),rank_l(i),rank(i+1)]),[2,1,3,4]);
            elseif i == N
                v_left = R{L+1};
                for j = L+2:i-1
                    v_left = tensor_prod(v_left,R{j});
                end
                
                for j = L+1:N-1
                    pp{j} = tensorprod(R{j},R{j},2,2);
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
                szYY=size(YY,2:M+1);
                szz = horzcat(NN,prod(szYY(1:i-L-1)),szYY(i-L),prod(szYY(i-L+1:M)));
                po = szz == 1;
                szz(po) = [];
                
                y = reshape(YY,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,v_left,[1 5],[2 1]);
                YB = tensorprod(YB,cc,[3 4],[1 2]);
                YB = permute(YB,[1 3 4 2]);
                YB = reshape(YB,para.Q(i-L),numel(YB)/para.Q(i-L));

                unf_v = YB / (BTB + lambda * FTF);
                wheels{i} = permute(reshape(unf_v,[para.Q(i-L),rank(i),rank_l(i),rank(1)]),[2,1,3,4]);

            else
                v_left = R{L+1};
                for j = L+2:i-1
                    v_left = tensor_prod(v_left,R{j});
                end
                v_right = R{i+1};
                for j = i+2:N
                    v_right = tensor_prod(v_right,R{j});
                end
                
                for j = L+1:N
                    if j ~= i
                        pp{j} = tensorprod(R{j},R{j},2,2);
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
                szYY=size(YY,2:M+1);
                szz = horzcat(NN,prod(szYY(1:i-L-1)),szYY(i-L),prod(szYY(i-L+1:M)));
                y = reshape(YY,szz);
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
%         unG=mode_unfolding_1(wheels{i},2);
%         [Q{i},unR]=qr(unG,'econ');
%         szR=size(wheels{i});
%         szR(2)=size(unR,1);
%         R{i}=permute(reshape(unR,szR([2,3,4,1])),[4,1,2,3]);
        R{i}=permute(tensorprod(wheels{i},Q{i},2,1),[1,4,2,3]);
        pp{i}=tensorprod(R{i},R{i},2,2);
    end
    
    
    %% updata core factor
    XX=XXX{L+1};
    YY=YYY{M+1};
    U = R{1};
    for j = 2:L
        U = tensor_prod(U,R{j});
    end
    V = R{L+1};
    for j = L+2:N
        V = tensor_prod(V,R{j});
    end
    
    xx = classical_mode_unfolding(XX,1);
    C = tensorprod(xx,U,2,2);
    cc = tensorprod(C,C,1,1);
%     vv = tensorprod(V,V,2,2);
    for j = L+1:N
        pp{j} = tensorprod(R{j},R{j},2,2);
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
        pp{j} = tensorprod(R{j},R{j},2,2);
    end
    uu = pp{1};
    for j = 2:L
        uu = y_prod(uu,pp{j});
    end

    HTH = tensorprod(uu,vv,[1 3 4 6],[3 1 6 4]);
    HTH = permute(HTH,[1 3 2 4]);
    HTH = reshape(HTH,prod(rank_l),prod(rank_l));
%     YY=tensorprod(YY,Q{N},M+1,1);
    y = classical_mode_unfolding(YY,1);
    AY = tensorprod(y,V,2,2);
    AY = tensorprod(C,AY,[1 2 4],[1 4 2]);
    AY = reshape(AY,numel(AY),1);
    
    vec_u = (ATA + lambda * HTH)\AY;
    core{1} = reshape(vec_u,rank_l);
    iter_time(iter+1) = iter_time(iter) + toc(iniIterTime);
%     core{1}(abs(core{1})<gm)=0;
    
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