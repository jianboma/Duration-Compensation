
function Ey = twin_model_plda_projection(plda, model_iv, L_S)
% Project original i-vectors to a new space with twin model GPLDA.
%
% Inputs:
%   plda            : structure containing twin model GPLDA hyperparameters
%   model_iv        : data matrix for enrollment i-vectors (column observations)
%   L_S         : String variables contain either 'long' or 'short'
%
% Outputs:
%    Ey         : Projected vectors
%
% References:
%   [1] J. Ma, V. Sethu, E. Ambikairajah, and K. A. Lee, "Twin Model G-PLDA
%       for Duration Mismatch Compensation in Text-Independent Speaker 
%       Verification," Interspeech 2016, pp. 1853-1857, 2016.
%   [2] J. Ma, V. Sethu, E. Ambikairajah, and K. A. Lee, "Duration compensation 
%       of i-vectors for short duration speaker verification," Electronics Letters,
%       vol. 53, pp. 405-407, 2017
%
% Jianbo Ma <jianbo.ma@student.unsw.edu.au>
% EE&T, UNSW, Sydney

ndim = size(model_iv,1);
Phi_short     = plda.Phi(1:ndim,:);
Phi_long     = plda.Phi(ndim+1:end,:);
Sigma_short   = plda.Sigma(1:ndim,1:ndim);
Sigma_long   = plda.Sigma(ndim+1:end,ndim+1:end);


W_short       = plda.W_short;
M_short       = plda.M_short;
W_long       = plda.W_long;
M_long       = plda.M_long;

%%%%% post-processing the model i-vectors %%%%%
if strcmp(L_S,'long')
    model_iv = bsxfun(@minus, model_iv, M_long); % centering the data
    model_iv = length_norm(model_iv); % normalizing the length
    model_iv = W_long' * model_iv; % whitening data
    
    [Ey, ~] = expectation_plda(model_iv, Phi_long, Sigma_long);
else
    model_iv = bsxfun(@minus, model_iv, M_short); % centering the data
    model_iv = length_norm(model_iv); % normalizing the length
    model_iv = W_short' * model_iv; % whitening data
    
    [Ey, ~] = expectation_plda(model_iv, Phi_short, Sigma_short);
end


function [Ey, Eyy] = expectation_plda(data, Phi, Sigma)
% computes the posterior mean and covariance of the factors
nphi     = size(Phi, 2);
nsamples = size(data, 2);
nspks    = size(data, 2);

Ey  = zeros(nphi, nsamples);
Eyy = zeros(nphi);

% initialize common terms to save computations

PhiT_invS_Phi = ( Phi'/Sigma ) * Phi;
I = eye(nphi);

nPhiT_invS_Phi = PhiT_invS_Phi;
Cyy =  pinv(I + nPhiT_invS_Phi);
invTerms = Cyy;


data = Sigma\data;

for spk = 1 : nspks
    nsessions = 1;
    Data = data(:, spk);
    PhiT_invS_y = sum(Phi' * Data, 2);
    Cyy = invTerms;
    Ey_spk  = Cyy * PhiT_invS_y;
    Eyy_spk = Cyy + Ey_spk * Ey_spk';
    Eyy     = Eyy + nsessions * Eyy_spk;
    Ey(:, spk) = repmat(Ey_spk, 1, nsessions);
end