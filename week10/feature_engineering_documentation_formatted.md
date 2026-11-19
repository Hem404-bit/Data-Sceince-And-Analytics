# Feature Engineering Documentation

## 1. Overview

All features are created by the `FeatureEngineer` transformer. The
transformer is stateless, making it safe to use inside the scikit-learn
pipeline. It is applied consistently to both training data and new data.

**Division safety:** All divisions use `Tenure.clip(lower=1)` to avoid
division-by-zero errors.

## 2. Engineered Features

The `FeatureEngineer` transformer creates the following nine features.

  ----------------------------------------------------------------------------------------------------------------------------------
            No. Feature                 Formula / Definition                             Business        Mean (Stayed)          Mean
                                                                                         Meaning                           (Churned)
  ------------- ----------------------- ------------------------------------------------ --------------- ------------- -------------
              1 **CLV** (Customer       `Tenure × MonthlyCharges`                        Expected                4,479           785
                Lifetime Value)                                                          revenue at the                
                                                                                         current rate                  
                                                                                         over the                      
                                                                                         customer's                    
                                                                                         tenure.                       

              2 **AvgMonthlySpend**     `TotalCharges / Tenure`                          Average amount            200         1,150
                                                                                         actually billed               
                                                                                         per month.                    

              3 **PaymentEfficiency**   `TotalCharges / CLV`                             Actual versus            2.94         11.13
                                                                                         expected                      
                                                                                         billing;                      
                                                                                         approximately 1               
                                                                                         means the                     
                                                                                         customer pays                 
                                                                                         as expected.                  

              4 **ChargePerTenure**     `MonthlyCharges / Tenure`                        Bill amount              4.79         35.73
                                                                                         relative to                   
                                                                                         customer                      
                                                                                         tenure; new                   
                                                                                         customers with                
                                                                                         expensive plans               
                                                                                         may be at risk.               

              5 **IsNewCustomer**       `Tenure ≤ 12`                                    Flag indicating          0.07          1.00
                                                                                         whether the                   
                                                                                         customer is in                
                                                                                         their first                   
                                                                                         year.                         

              6 **TenureGroup**         Bins: 0--1 year / 1--2 years / 2--4 years / 4+   Represents the            ---     0--1 year
                                        years                                            customer's                     churn rate =
                                                                                         lifecycle                               63%
                                                                                         stage; one-hot                
                                                                                         encoded.                      

              7 **AutoPay**             `PaymentMethod ∈ {Bank Transfer, Credit Card}`   Indicates                0.68          0.66
                                                                                         automatic                     
                                                                                         versus manual                 
                                                                                         payment.                      

              8 **ContractToTenure**    `contract_months / Tenure` (M2M = 1, 1yr = 12,   Measures                 0.69          2.09
                                        2yr = 24)                                        contract                      
                                                                                         commitment                    
                                                                                         relative to                   
                                                                                         customer                      
                                                                                         tenure.                       

              9 **HighMonthlyCharge**   `MonthlyCharges > 110` (approximately median     Flag indicating          0.51          0.70
                                        115)                                             a premium plan.               
  ----------------------------------------------------------------------------------------------------------------------------------

## 3. Key Findings

### 3.1 Strongest Engineered Features

The following engineered features show the strongest relationships with
churn.

  ------------------------------------------------------------------------
  Feature                     Correlation with Churn Finding
  --------------------- ---------------------------- ---------------------
  `IsNewCustomer`                               0.77 Strongest engineered
                                                     feature.

  `ChargePerTenure`                             0.52 Higher charge
                                                     relative to tenure is
                                                     associated with
                                                     churn.

  `AvgMonthlySpend`                             0.41 Shows a relationship
                                                     with churn.

  `CLV`                                        −0.37 Negatively correlated
                                                     with churn.
  ------------------------------------------------------------------------

Each of these features performs better than every original column except
`Tenure`.

### 3.2 Weak Feature

`AutoPay` is a weak feature in this dataset. Payment method has little
effect on churn and is perfectly correlated with the
`PaymentMethod_Electronic Check` dummy variable, making it redundant.

### 3.3 Skew Handling

Ratio features are heavily right-skewed. Therefore, they are transformed
using `log1p` before being passed to `StandardScaler` in the pipeline.

## 4. Feature Transformation Flow

The feature engineering and preprocessing workflow is:

``` text
Raw columns
    |
    v
FeatureEngineer
    |
    v
16 columns (7 raw + 9 new)
    |
    v
ColumnTransformer
    |
    v
21 numeric model inputs
```

## 5. How to Add a New Feature

To add another feature to the pipeline, follow these steps:

1.  Add the feature calculation inside `FeatureEngineer.transform()`.
2.  Add the new column name to the appropriate list in the Day 7
    notebook cell, such as `RATIO_NUM`, `BIN_FEATURES`, or
    `NOM_FEATURES`.
3.  Re-run the notebook from the beginning.
