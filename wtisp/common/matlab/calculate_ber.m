function eval_info = calculate_ber(source, gt_symbol)

    fs = 400;
    sps = 4;
    span = 20; 
    rolloff = .3; 
    f_carrier1 = 275;
    rrcFilter=rcosdesign(rolloff,span,sps,'sqrt'); 

    t = (0:1/fs:((length(source)-1)/fs));
    ccos = cos(2*pi*f_carrier1 * t);
    csin = sin(2*pi*f_carrier1 * t);

    source = real_to_complex(source, ccos, csin);
    source = upfirdn(source, rrcFilter, 1, sps);
    source = source(span+1:end-span); 
    symbol = pskdemod(source, 4); % Demodulate detected signal from equalizer.
    eval_info = step(comm.ErrorRate, gt_symbol.', symbol.');
    eval_info = eval_info(1:2);
    
end

function x = real_to_complex(x, cos1, sin1)

    xi_dnconv = x .* cos1;
    xq_dnconv = x .* sin1;
    x = xi_dnconv + 1j * xq_dnconv;
    
end