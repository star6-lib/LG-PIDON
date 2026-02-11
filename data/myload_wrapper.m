function val = myload_wrapper(location, state)
    % Reads the current curve from global variable to avoid file I/O speed bottlenecks
    global global_ubc_curve 
    
    % Mapping: Physical [0, 20] -> Data [0, 1] (Index 1 to 101)
    N = 101;
    X_data = linspace(0, 1, N);
    
    % Interpolate the 101-point curve to the FEM query points
    val = interp1(X_data, global_ubc_curve, location.x, 'linear', 0);
end