function [ err ] = cross_ent_err(true, pred)
   err = sum(-1 * sum(true .* log(pred) + (1 - true) .* log(1 - pred))); 
end