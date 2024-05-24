# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire, os, sys

# 获取外部文件的绝对路径
external_path = os.path.abspath('../../llama3')
print(external_path)

# 添加路径到 sys.path
if external_path not in sys.path:
    sys.path.append(external_path)

# 获取外部文件的绝对路径
external_path = os.path.abspath('./llama3')
print(external_path)

# 添加路径到 sys.path
if external_path not in sys.path:
    sys.path.append(external_path)

from llama import Dialog, Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 2048,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
    {
        "role": "system",
        "content": "You are an AI Python assistant. You will be given your past implementation, a feedback of running your code, and a hint to change the implementation appropriately. Write your full implementation.\nUse a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
    },
    {
        "role": "assistant",
        "content": "```python\nimport pandas as pd\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error\nimport numpy as np\n\n# Load the training data\ntrain_df = pd.read_csv('Tasks/house-prices-advanced-regression-techniques/train.csv')\n\n# Define the target variable and the feature variables\ntarget = 'SalePrice'\nfeatures = [col for col in train_df.columns if col != target]\n\n# Convert categorical variables to numerical variables\nfrom sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\nfor col in features:\n    if train_df[col].dtype == 'object':\n        train_df[col] = le.fit_transform(train_df[col])\n\n# Split the data into training and validation sets\nX_train, X_val, y_train, y_val = train_test_split(train_df[features], train_df[target], test_size=0.2, random_state=42)\n\n# Train a random forest regressor model\nrf = RandomForestRegressor(n_estimators=100, random_state=42)\nrf.fit(X_train, y_train)\n\n# Make predictions on the validation set\ny_pred = rf.predict(X_val)\n\n# Calculate the mean squared error\nmse = mean_squared_error(y_val, y_pred)\nprint(f'Mean squared error: {mse:.2f}')\n\n# Load the test data\ntest_df = pd.read_csv('Tasks/house-prices-advanced-regression-techniques/test.csv')\n\n# Convert categorical variables to numerical variables\nfor col in features:\n    if test_df[col].dtype == 'object':\n        test_df[col] = le.transform(test_df[col])\n\n# Make predictions on the test set\ny_pred_test = rf.predict(test_df[features])\n\n# Save the predictions to a submission file\nsubmission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_pred_test})\nsubmission_df.to_csv('submission.csv', index=False)\n```"
    },
    {
        "role": "user",
        "content": "[feedback from previous impl]:\nYour implementation is wrong. Here are some error information:\n\n-------------------------------------------------------------------------------\ngen_code.py 26 <module>\nrf.fit(X_train, y_train)\n\nbase.py 1152 wrapper\nreturn fit_method(estimator, *args, **kwargs)\n\n_forest.py 348 fit\nX, y = self._validate_data(\n\nbase.py 622 _validate_data\nX, y = check_X_y(X, y, **check_params)\n\nvalidation.py 1146 check_X_y\nX = check_array(\n\nvalidation.py 957 check_array\n_assert_all_finite(\n\nvalidation.py 122 _assert_all_finite\n_assert_all_finite_element_wise(\n\nvalidation.py 171 _assert_all_finite_element_wise\nraise ValueError(msg_err)\n\nValueError:\nInput X contains NaN.\nRandomForestRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values"
    },
    {
        "role": "assistant",
        "content": "My implementation is wrong because I did not handle missing values in the data. The error message indicates that the input X contains NaN values, which are not accepted by the RandomForestRegressor. I need to preprocess the data to handle missing values, such as using an imputer transformer in a pipeline or dropping samples with missing values."
    },
    {
        "role": "user",
        "content": "[improved impl]:\n\nThe dataframe 'Tasks/house-prices-advanced-regression-techniques/train.csv' is loaded and in memory. Columns are also named attributes. \nDescription of the dataset in 'train.csv':\n```\n\nId,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice\n1,60,RL,65,8450,Pave,NA,Reg,Lvl,AllPub,Inside,Gtl,CollgCr,Norm,Norm,1Fam,2Story,7,5,2003,2003,Gable,CompShg,VinylSd,VinylSd,BrkFace,196,Gd,TA,PConc,Gd,TA,No,GLQ,706,Unf,0,150,856,GasA,Ex,Y,SBrkr,856,854,0,1710,1,0,2,1,3,1,Gd,8,Typ,0,NA,Attchd,2003,RFn,2,548,TA,TA,Y,0,61,0,0,0,0,NA,NA,NA,0,2,2008,WD,Normal,208500\n2,20,RL,80,9600,Pave,NA,Reg,Lvl,AllPub,FR2,Gtl,Veenker,Feedr,Norm,1Fam,1Story,6,8,1976,1976,Gable,CompShg,MetalSd,MetalSd,None,0,TA,TA,CBlock,Gd,TA,Gd,ALQ,978,Unf,0,284,1262,GasA,Ex,Y,SBrkr,1262,0,0,1262,0,1,2,0,3,1,TA,6,Typ,1,TA,Attchd,1976,RFn,2,460,TA,TA,Y,298,0,0,0,0,0,NA,NA,NA,0,5,2007,WD,Normal,181500\n3,60,RL,68,11250,Pave,NA,IR1,Lvl,AllPub,Inside,Gtl,CollgCr,Norm,Norm,1Fam,2Story,7,5,2001,2002,Gable,CompShg,VinylSd,VinylSd,BrkFace,162,Gd,TA,PConc,Gd,TA,Mn,GLQ,486,Unf,0,434,920,GasA,Ex,Y,SBrkr,920,866,0,1786,1,0,2,1,3,1,Gd,6,Typ,1,TA,Attchd,2001,RFn,2,608,TA,TA,Y,0,42,0,0,0,0,NA,NA,NA,0,9,2008,WD,Normal,223500\n4,70,RL,60,9550,Pave,NA,IR1,Lvl,AllPub,Corner,Gtl,Crawfor,Norm,Norm,1Fam,2Story,7,5,1915,1970,Gable,CompShg,Wd Sdng,Wd Shng,None,0,TA,TA,BrkTil,TA,Gd,No,ALQ,216,Unf,0,540,756,GasA,Gd,Y,SBrkr,961,756,0,1717,1,0,1,0,3,1,Gd,7,Typ,1,Gd,Detchd,1998,Unf,3,642,TA,TA,Y,0,35,272,0,0,0,NA,NA,NA,0,2,2006,WD,Abnorml,140000\n5,60,RL,84,14260,Pave,NA,IR1,Lvl,AllPub,FR2,Gtl,NoRidge,Norm,Norm,1Fam,2Story,8,5,2000,2000,Gable,CompShg,VinylSd,VinylSd,BrkFace,350,Gd,TA,PConc,Gd,TA,Av,GLQ,655,Unf,0,490,1145,GasA,Ex,Y,SBrkr,1145,1053,0,2198,1,0,2,1,4,1,Gd,9,Typ,1,TA,Attchd,2000,RFn,3,836,TA,TA,Y,192,84,0,0,0,0,NA,NA,NA,0,12,2008,WD,Normal,250000\n\n```\nColumns data types are all unknown, and you should carefully think the data type of each column by yourself.\nPlease choose the best way to implement a model to predict the SalePrice. 'Tasks/house-prices-advanced-regression-techniques/train.csv' is the training data. 'Tasks/house-prices-advanced-regression-techniques/test.csv' is the test data. And you should save your prediction in 'submission.csv'.\nPlease try your best to avoiding any possible errors when running your code!\n    "
    },
]
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
