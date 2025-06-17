% Customer Lifetime Value (CLV) Prediction and Segmentation in MATLAB

% Load enhanced customer dataset
data = readtable('enhanced_customer_data.csv');

% STEP 1: Define CLV manually (for safety; already included in CSV)
data.CLV = data.PurchasesPerMonth .* data.TenureInMonths .* data.AvgBasketValue;

% STEP 2: Prepare features for clustering (excluding CLV for now)
X = [data.AnnualIncome, data.SpendingScore, data.PurchasesPerMonth, data.TenureInMonths, data.AvgBasketValue];

% STEP 3: K-Means Clustering to segment customers
k = 3; % choose 3 segments
[idx, centroids] = kmeans(X, k);

% STEP 4: Add cluster labels to dataset
data.Cluster = idx;

% STEP 5: Visualize Clustering in 3D
figure;
scatter3(X(:,1), X(:,2), data.CLV, 50, idx, 'filled');
xlabel('Annual Income'); ylabel('Spending Score'); zlabel('CLV');
title('3D Customer Clustering by CLV');
grid on;

% STEP 6: Train Regression Model to Predict CLV
features = [data.Age, data.AnnualIncome, data.SpendingScore, data.PurchasesPerMonth, data.TenureInMonths, data.AvgBasketValue];
target = data.CLV;

model = fitlm(features, target);  % Train linear regression

% STEP 7: Show regression summary
disp(model);

% STEP 8: Predict CLV for future customer (example input)
new_customer = [30, 45000, 78, 6, 12, 1000];  % age, income, score, purchases, tenure, basket
predicted_clv = predict(model, new_customer);

fprintf('\nPredicted CLV for new customer: ₹%.2f\n', predicted_clv);

% STEP 9: Save updated table
writetable(data, 'customer_segments_clv.csv');
