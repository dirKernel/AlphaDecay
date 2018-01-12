function para = fit_expGauss(data)
[xData, yData] = prepareCurveData(1:1:numel(data), data);

func_str = 'l/2*exp(1/2*(2*x+l*s*s-2*m))*(1-erf((x+l*s*s-m)/(sqrt(2)*s)))';

ft = fittype( func_str, 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';

% change the iteration number and upper lower bound on the value and
% starting point if needed
opts.MaxIter = 500000;
%opts.Upper = [100 100 1300];
%opts.Lower = [0 0 1100];
%opts.StartPoint = [2 2 1200];
%opts.Weight = weights;

[fitresult, gof] = fit( xData, yData, ft, opts );
para = coeffvalues(fitresult);

end
