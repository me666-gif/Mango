function [Gbest, Gbest_val] = SLPSO_func2( func_num, fhd, dim, pop, MaxFes, VRmin, VRmax, MaxIter, varargin)

    % 参数初始化
    fbias = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, ...
             1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, ...
             2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000];
    group_ps = 3;  % 子群中的粒子数
    Rmin = 0.4;
    Rmax = 0.6;
    flag = 0;
    sigmax = 1.2;
    sigmin = 0.05;
    sig = 1;
    gap1 = 5;
    c1 = 2.5 - (1:MaxIter)*2 / MaxIter;  % 加速系数
    c2 = 0.5 + (1:MaxIter)*2 / MaxIter;

    FEs = 0;
    Xmin = VRmin;
    Xmax = VRmax;


    % 种群初始速度
    lu = [Xmin * ones(1, dim); Xmax * ones(1, dim)];
    mv = 0.1 * (lu(2, :) - lu(1, :));
    Vmin = repmat(-mv, pop, 1);
    Vmax = -Vmin;
    vel = Vmin + 2 .* Vmax .* rand(pop, dim);

    % 使用混沌映射生成初始种群位置
    pos = initializationNew(pop, dim, Xmax, Xmin);
    

    % 计算适应度值
    fitness = (feval(fhd, pos', varargin{:}) - fbias(func_num))';
     % fitness = (feval(fhd, pos', varargin{:}))';
    FEs = FEs + pop;

    Pbest = pos;  % 个体历史最优位置
    Pbest_val = fitness;
    [val, indfitness] = sort(fitness, 'ascend');  % 将个体最优适应度值按照升序排序
    Gbest_val = val(1);  % 初始化全局最优适应度值
    Gbest = pos(indfitness(1), :);  % 初始化全局最优位置
    g=0;
    recorded = 0;  % 初始化 recorded 变量
    suc = 0;  % 初始化 suc 变量

    while g <= MaxIter & FEs <= MaxFes
        g=g+1;

        pm=0.1-(0.1-0.02)*g/MaxIter;
        fitness_norm = (fitness - min(fitness)) / (max(fitness) - min(fitness));
        [val1, index] = sort(fitness_norm, 'ascend');  % 将个体最优适应度值按照升序排序
        K = ceil(pop * Rmax - (pop * Rmax - pop * Rmin) * (g / MaxIter));
        K = max(1, min(K, pop));

        while mod(K, 3) ~= 0
            K = K + 1;
            K = max(1, min(K, pop));
        end

        shuffled_indices = index(1:K);  % 提取前 K 个索引
        shuffled_indices = shuffled_indices(randperm(K));  % 打乱前 K 个索引
        changedIndex = [shuffled_indices;index(K+1:end)];


        Pbest_new = Pbest(changedIndex, :);
        pos_new = pos(changedIndex, :);
        Pbest_valnew = Pbest_val(changedIndex, :);
        vel_new = vel(changedIndex, :);
        scorenew = fitness_norm(changedIndex);

        group_num = K / group_ps;
        group_id = zeros(group_num, group_ps);
        pos_group = zeros(1, pop);
        for h = 1:group_num
            group_id(h, :) = [((h-1)*group_ps+1):h*group_ps];
            pos_group(group_id(h, :)) = h;
            [gbestval(h), gbestid] = min(Pbest_valnew(group_id(h, :)));  % 初始化局部最优适应度值
            gbest(h, :) = Pbest_new(group_id(h, gbestid), :);  % 初始化局部最优位置
        end

        OO = zeros(1, dim);
        for kk = 1:pop
              wx(kk) = (0.756*exp(-g/MaxIter)+0.144)*(1 / (1 + 0.1* exp(2.6 * scorenew(kk))));  % 加权惯性权重

        end

        scorelite = scorenew(1:K);
        score=zeros(1,K);
        for aa=1:K
            score(aa)=1-scorelite(aa);
        end
        for jj = 1:K
            OO = OO + score(jj) / sum(score) * Pbest_new(jj, :);
        end

        FDBbest = Pbest(index(1), :);

        exploration_ve = zeros(pop, dim);
        exploration_pos = zeros(pop, dim);
        exploitation_ve = zeros(pop, dim);
        exploitation_pos = zeros(pop, dim);

        for i = 1:pop
            if i <= K
                % 探索粒子速度更新
                exploration_ve(i, :) = wx(i)* vel_new(i, :) + (c1(g).* rand(1, dim) .* (Pbest_new(i, :) - pos_new(i, :))) + (c2(g).* rand(1, dim) .* (gbest(pos_group(i), :) - pos_new(i, :)));
                exploration_ve(i, :) = ((exploration_ve(i, :) < Vmin(i, :)) .* Vmin(i, :)) + ((exploration_ve(i, :) > Vmax(i, :)) .* Vmax(i, :)) + (((exploration_ve(i, :) < Vmax(i, :)) & (exploration_ve(i, :) > Vmin(i, :))) .* exploration_ve(i, :));
                exploration_pos(i, :) = pos_new(i, :) + exploration_ve(i, :);
                exploration_pos(i, :) = Non_uniform_mutation(exploration_pos(i, :), pm, g, MaxIter, Xmin, Xmax);  % 非均匀变异
            else
                % 开发粒子速度和位置更新
                exploitation_ve(i, :) = wx(i)* vel_new(i, :) + (c1(g).* rand(1, dim) .* (OO - pos_new(i, :))) + (c2(g).* rand(1, dim) .* (FDBbest - pos_new(i, :)));
                exploitation_ve(i, :) = ((exploitation_ve(i, :) < Vmin(i, :)) .* Vmin(i, :)) + ((exploitation_ve(i, :) > Vmax(i, :)) .* Vmax(i, :)) + (((exploitation_ve(i, :) < Vmax(i, :)) & (exploitation_ve(i, :) > Vmin(i, :))) .* exploitation_ve(i, :));
                exploitation_pos(i, :) = pos_new(i, :) + exploitation_ve(i, :);
            end
        end

        exploitation_poslate = exploitation_pos(K+1:pop, :);
        pos = [exploration_pos(1:K, :); exploitation_poslate];
        exploitation_velate = exploitation_ve(K+1:pop, :);
        vel = [exploration_ve(1:K, :); exploitation_velate];


        if rand > 0.5
            pos = (pos > Xmax) .* Xmax + (pos <= Xmax) .* pos;
            pos = (pos < Xmin) .* Xmin + (pos >= Xmin) .* pos;
        else
            pos = ((pos >= Xmin) & (pos <= Xmax)) .* pos + (pos < Xmin) .* (Xmin + 0.2 .* (Xmax - Xmin) .* rand(pop, dim)) + (pos > Xmax) .* (Xmax - 0.2 .* (Xmax - Xmin) .* rand(pop, dim));
        end

        fitness = (feval(fhd, pos', varargin{:}) - fbias(func_num))';
         % fitness = (feval(fhd, pos', varargin{:}) )';
        FEs = FEs + pop;
        improved = (Pbest_valnew > fitness);  % 个体是否改进
        temp = repmat(improved, 1, dim);
        Pbest = temp .* pos + (1 - temp) .* Pbest_new;
        Pbest_val = improved .* fitness + (1 - improved) .* Pbest_valnew;  % 更新个体最优适应度值
        [val, indFitness] = sort(Pbest_val, 'ascend');
        Gbest_valtemp = val(1);
        Gbesttemp = Pbest(indFitness(1), :);

        % 判断Gbest是否得到改进
        if Gbest_valtemp < Gbest_val
            Gbest = Gbesttemp;
            Gbest_val = Gbest_valtemp;
            flag = 0;
        else
            flag = flag + 1;
        end


    if  flag >= gap1

        num_perturb_dims = ceil(dim * (0.1 + 0.1 * (g / MaxIter))); % 扰动维度随迭代次数增加

       for d = 1:num_perturb_dims
            pt = Gbest;
            d1 = unidrnd(dim); 
            randdata = 2 * rand(1, 1) - 1;
            pt(d1) = pt(d1) + sign(randdata) * (Xmax - Xmin) * normrnd(0, sig^2);
            pt(pt > Xmax) = Xmax * rand;
            pt(pt < Xmin) = Xmin * rand;
            cv = (feval(fhd, pt', varargin{:}) - fbias(func_num))';
            FEs = FEs + 1;
            if cv < Gbest_val
                Gbest = pt;
                Gbest_val = cv;
                flag = 0;
                % 若找到更优解，跳出循环

            end

       end
    end

sig=sigmax-(sigmax-sigmin)*(FEs/MaxFes);

        if FEs >= MaxFes
            break;
        end

        if (g == MaxIter) && (FEs < MaxFes)
            g = g - 1;
        end

    end

end

function [newpop] = Non_uniform_mutation(psd, pm, t, T, Xmin, Xmax)
      b = 4;  
    [ps, D] = size(psd);
    VRmin = Xmin;
    VRmax = Xmax;

    newpop = psd;
    for i = 1:ps
        for j = 1:D
            if rand() < pm
                aa = rand(1, D);
                N_mm = diag(aa);
                if round(rand()) == 0
                    newpop(i, j) = psd(i, j) + N_mm(j, j) * (VRmax - psd(i, j)) * (1 - t / T)^b;
                else
                    newpop(i, j) = psd(i, j) - N_mm(j, j) * (psd(i, j) - VRmin) * (1 - t / T)^b;
                end
            end
        end
    end
end
