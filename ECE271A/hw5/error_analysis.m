clc;
clear;

error_struct = load('error_storage.mat');

error_storage = error_struct.error_storage;

max_vals = [];
min_vals = [];
for i=1:25
    max_val = max(error_storage(num2str(i)));
    min_val = min(error_storage(num2str(i)));
    max_vals = [max_vals max_val];
    min_vals = [min_vals min_val];
end