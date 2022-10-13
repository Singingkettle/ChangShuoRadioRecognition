%%

M = 16;            % Modulation order
k = log2(M);       % Bits per symbol
numBits = k*7.5e4; % Bits to process
sps = 4;           % Samples per symbol (oversampling factor)

filtlen = 10;      % Filter length in symbols
rolloff = 0.25;    % Filter rolloff factor

rrcFilter = rcosdesign(rolloff,filtlen,sps);

fvtool(rrcFilter,'Analysis','Impulse')

rng default;                     % Use default random number generator
dataIn = randi([0 1],numBits,1); % Generate vector of binary data

dataSymbolsIn = bit2int(dataIn,k);

dataMod = rand(100, 1);


txFiltSignal = upfirdn(dataMod,rrcFilter,sps,1);
txFiltSignal2 = myupfirdn(dataMod,rrcFilter,sps,1);

a = myupfirdn(txFiltSignal2,rrcFilter,1, sps);

function yout = myupfirdn(xin,h,p,q)

    yout = upsample(xin,p);
    yout = filter(h,1,yout);
    yout = downsample(yout,q);
end