close all
clear
load bc_source.mat 

% ==================== CONFIGURATION ====================
num_samples = 4000; 

% Hole Control: 'free' or 'fixed'
HOLE_BC_TYPE = 'fixed'; 

% ==================== Initialize Storage ====================
coors_dict = cell(1, num_samples);
final_u = cell(1, num_samples);
final_u_y = cell(1, num_samples);
geo_param_dict = cell(1, num_samples); 
flag_BC_load_dict = cell(1, num_samples); 
flag_BCxy_dict = cell(1, num_samples);    
flag_BCy_dict = cell(1, num_samples);     
flag_hole_dict = cell(1, num_samples);    

input_force_data = cell(1, num_samples);
input_disp_data = cell(1, num_samples);
input_material = zeros(num_samples, 2);

fprintf('Running FEM with Hole BC: [%s]...\n', HOLE_BC_TYPE);

% [OPTIMIZATION] Define Names and Formula ONCE (Static)
% Since we always have 1 Rect + 4 Circles, we don't need to build this in the loop.
% decsg expects names in columns, so we transpose (') the char array.
fixed_names = char('R1', 'C1', 'C2', 'C3', 'C4')'; 
fixed_formula = 'R1-C1-C2-C3-C4';

for i = 1:num_samples
    if mod(i, 50) == 0, fprintf('Sample %d / %d...\n', i, num_samples); end
    
    % --- 1. Load Data & Material ---
    curve_data = f_bc(i, :);
    type_id = f_type(i);
    
    E_val = max(5, min(20, normrnd(12.5, 2.5))); 
    v_val = max(0.2, min(0.35, normrnd(0.275, 0.025)));
    input_material(i, :) = [E_val, v_val];
    
    if type_id == 1 || type_id == 2
        mode = 'force';
        input_force_data{i} = curve_data;
        input_disp_data{i} = zeros(size(curve_data));
    else
        mode = 'disp';
        input_force_data{i} = zeros(size(curve_data));
        input_disp_data{i} = curve_data;
    end
    
    % --- 2. Geometry Generation ---
    model = createpde('structural','static-planestrain');
    
    % Rect Geometry (Variable renamed to avoid conflict)
    rect_geo = [3, 4, 0, 1, 1, 0, 0, 0, 1, 1]';
    
    % Hole Parameters
    base_centers = [0.75, 0.75; 0.25, 0.75; 0.25, 0.25; 0.75, 0.25];
    perturb = 0.075; hole_range = [0.14, 0.075];
    
    circles_geo = []; 
    current_holes = zeros(4,3);
    
    for k = 1:4
        r_rand = sqrt(rand())*perturb; th = rand()*2*pi;
        cx = base_centers(k,1) + r_rand*cos(th);
        cy = base_centers(k,2) + r_rand*sin(th);
        cr = hole_range(1) + diff(hole_range)*rand();
        current_holes(k,:) = [cx, cy, cr];
        C_col = [1; cx; cy; cr];
        circles_geo = [circles_geo, [C_col; zeros(6,1)]];
    end
    
    gm = [rect_geo, circles_geo];
    
    % [CRITICAL FIX]
    % 1. We use the static 'fixed_names' (which is correctly transposed).
    % 2. We use the static 'fixed_formula'.
    % 3. 'rect_geo' is used in gm, so no variable name conflict with 'R1'.
    dl = decsg(gm, fixed_formula, fixed_names); 
    
    geometryFromEdges(model, dl);
    
    % --- 3. Boundary Edge Identification ---
    edges_top = []; 
    edges_bot = [];
    edges_holes = []; 

    tol_geo = 1e-4;
    for e = 1:size(dl, 2)
        xm = (dl(2,e) + dl(3,e))/2;
        ym = (dl(4,e) + dl(5,e))/2;
        if abs(ym - 1.0) < tol_geo
            edges_top = [edges_top, e];
        elseif abs(ym - 0) < tol_geo
            edges_bot = [edges_bot, e];
        elseif xm > tol_geo && xm < (1.0 - tol_geo) && ym > tol_geo && ym < (1.0 - tol_geo)
            edges_holes = [edges_holes, e];
        end
    end
    
    % --- 4. Physics Application ---
    structuralProperties(model, 'YoungsModulus', E_val, 'PoissonsRatio', v_val);
    structuralBC(model, 'Edge', edges_bot, 'Constraint', 'fixed');
    
    % Hole Constraints
    if strcmp(HOLE_BC_TYPE, 'fixed')
        structuralBC(model, 'Edge', edges_holes, 'Constraint', 'fixed');
    end
    
    global global_ubc_curve
    global_ubc_curve = curve_data; 
    
    if strcmp(mode, 'disp')
        structuralBC(model, 'Edge', edges_top, 'YDisplacement', @myload_wrapper, 'Vectorized', 'on');
    else
        structuralBoundaryLoad(model, 'Edge', edges_top, ...
            'SurfaceTraction', @(l,s) [zeros(1,numel(l.x)); myload_wrapper(l,s)], ...
            'Vectorized', 'on');
    end
    
    % --- 5. Solve & Save ---
    generateMesh(model, 'Hmax', 0.02);
    R = solve(model);
    
    nodes = R.Mesh.Nodes;
    final_u{i} = R.Displacement.ux;
    final_u_y{i} = R.Displacement.uy;
    coors_dict{i} = nodes';
    geo_param_dict{i} = current_holes; 
    
    % Flags
    xx = nodes(1,:); yy = nodes(2,:);
    tol = 1e-4;

    % 优先定义上下边界（包含角点）
    %Top: (0,1) -> (1,1)
    is_top = abs(yy - 1.0) < tol;
    % Bottom: (0,0) -> (1,0)
    is_bottom = abs(yy - 0) < tol;

    % 定义原始左右边界
    is_left_raw = abs(xx - 0) < tol;
    is_right_raw = abs(xx - 1.0) < tol;

    % 从左右边界中剔除角点
    % 逻辑：是左边 AND 不是上边 AND 不是下边
    is_left = is_left_raw & (~is_top) & (~is_bottom);
    is_right = is_right_raw & (~is_top) & (~is_bottom);

    is_hole_node = false(size(xx));
    for k = 1:4
        cx = current_holes(k, 1); cy = current_holes(k, 2); cr = current_holes(k, 3);
        dist_to_center = sqrt((xx - cx).^2 + (yy - cy).^2);
        is_hole_node = is_hole_node | (abs(dist_to_center - cr) < tol);
    end
    
    flag_BC_load_dict{i} = double(is_top)';                   
    flag_BCxy_dict{i} = double(is_bottom)';                   
    flag_BCy_dict{i} = double((is_left | is_right))';         
    flag_hole_dict{i} = double(is_hole_node)';                
end

save('plate_holes_mixed_LG_PIDON.mat', 'coors_dict', 'final_u', 'final_u_y', ...
    'input_force_data', 'input_disp_data', 'input_material', 'geo_param_dict', ...
    'flag_BC_load_dict', 'flag_BCxy_dict', 'flag_BCy_dict', 'flag_hole_dict');
fprintf('Done. Dataset saved.\n');