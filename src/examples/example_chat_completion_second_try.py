# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

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
        [{
            "role": "task", 
            "content": """
                You are a Python programming assistant. 
                The dataframe 'train.csv' is loaded and in memory. Columns are also named attributes.
                Please choose the best way to implement a model to predict the sales price for houses. '../house-prices-advanced-regression-techniques/train.csv' is the training data, where 'SalePrice' is the target feature, others are judgment features. '../house-prices-advanced-regression-techniques/test.csv' is the test data.
                Description of the dataset in 'train.csv' (columns dtypes are unknown):
                ```
                Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice
                1,60,RL,65,8450,Pave,NA,Reg,Lvl,AllPub,Inside,Gtl,CollgCr,Norm,Norm,1Fam,2Story,7,5,2003,2003,Gable,CompShg,VinylSd,VinylSd,BrkFace,196,Gd,TA,PConc,Gd,TA,No,GLQ,706,Unf,0,150,856,GasA,Ex,Y,SBrkr,856,854,0,1710,1,0,2,1,3,1,Gd,8,Typ,0,NA,Attchd,2003,RFn,2,548,TA,TA,Y,0,61,0,0,0,0,NA,NA,NA,0,2,2008,WD,Normal,208500
                2,20,RL,80,9600,Pave,NA,Reg,Lvl,AllPub,FR2,Gtl,Veenker,Feedr,Norm,1Fam,1Story,6,8,1976,1976,Gable,CompShg,MetalSd,MetalSd,None,0,TA,TA,CBlock,Gd,TA,Gd,ALQ,978,Unf,0,284,1262,GasA,Ex,Y,SBrkr,1262,0,0,1262,0,1,2,0,3,1,TA,6,Typ,1,TA,Attchd,1976,RFn,2,460,TA,TA,Y,298,0,0,0,0,0,NA,NA,NA,0,5,2007,WD,Normal,181500
                3,60,RL,68,11250,Pave,NA,IR1,Lvl,AllPub,Inside,Gtl,CollgCr,Norm,Norm,1Fam,2Story,7,5,2001,2002,Gable,CompShg,VinylSd,VinylSd,BrkFace,162,Gd,TA,PConc,Gd,TA,Mn,GLQ,486,Unf,0,434,920,GasA,Ex,Y,SBrkr,920,866,0,1786,1,0,2,1,3,1,Gd,6,Typ,1,TA,Attchd,2001,RFn,2,608,TA,TA,Y,0,42,0,0,0,0,NA,NA,NA,0,9,2008,WD,Normal,223500
                4,70,RL,60,9550,Pave,NA,IR1,Lvl,AllPub,Corner,Gtl,Crawfor,Norm,Norm,1Fam,2Story,7,5,1915,1970,Gable,CompShg,Wd Sdng,Wd Shng,None,0,TA,TA,BrkTil,TA,Gd,No,ALQ,216,Unf,0,540,756,GasA,Gd,Y,SBrkr,961,756,0,1717,1,0,1,0,3,1,Gd,7,Typ,1,Gd,Detchd,1998,Unf,3,642,TA,TA,Y,0,35,272,0,0,0,NA,NA,NA,0,2,2006,WD,Abnorml,140000
                5,60,RL,84,14260,Pave,NA,IR1,Lvl,AllPub,FR2,Gtl,NoRidge,Norm,Norm,1Fam,2Story,8,5,2000,2000,Gable,CompShg,VinylSd,VinylSd,BrkFace,350,Gd,TA,PConc,Gd,TA,Av,GLQ,655,Unf,0,490,1145,GasA,Ex,Y,SBrkr,1145,1053,0,2198,1,0,2,1,4,1,Gd,9,Typ,1,TA,Attchd,2000,RFn,3,836,TA,TA,Y,192,84,0,0,0,0,NA,NA,NA,0,12,2008,WD,Normal,250000
                ```
                Only generate the python code for the model implementation.
            """
        },
        {
            "role": "assistant",
            "content": """
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_squared_error
                import pandas as pd

                # Load the training data
                train_df = pd.read_csv('../house-prices-advanced-regression-techniques/train.csv')

                # Split the data into training and validation sets
                X_train, X_val, y_train, y_val = train_test_split(train_df.drop('SalePrice', axis=1), train_df['SalePrice'], test_size=0.2, random_state=42)

                # Create a Random Forest Regressor model
                model = RandomForestRegressor(n_estimators=100, random_state=42)

                # Train the model on the training data
                model.fit(X_train, y_train)

                # Evaluate the model on the validation data
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                print(f'MSE on validation set: {mse:.2f}')

                # Make predictions on the test data
                test_df = pd.read_csv('../house-prices-advanced-regression-techniques/test.csv')
                y_pred_test = model.predict(test_df)
            """
        },
        {
            "role": "self-generated unit test fail",
            "content": """
                Traceback (most recent call last):
                File "/home/shiqi/llama3/gen_test.py", line 16, in <module>
                    model.fit(X_train, y_train)
                ValueError: could not convert string to float: 'RL'
                Reflect your code, and rewrite the complete code only.
            """
        },
        {
            "role": "assistant",
            "content": """
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_squared_error
                import pandas as pd
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline

                # Load the training data
                train_df = pd.read_csv('../house-prices-advanced-regression-techniques/train.csv')

                # Split the data into training and validation sets
                X_train, X_val, y_train, y_val = train_test_split(train_df.drop('SalePrice', axis=1), train_df['SalePrice'], test_size=0.2, random_state=42)

                # Define the preprocessing steps for categorical and numerical columns
                categorical_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'MasVnrType', 'Foundation', 'Functional', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
                numerical_cols = ['LotFrontage', 'LotArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold', 'PoolArea', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'GarageYrBlt']

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                        ('numerical', 'passthrough', numerical_cols)
                    ]
                )

                # Create a pipeline with the preprocessing step and the Random Forest Regressor model
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
                ])

                # Train the model on the training data
                model.fit(X_train, y_train)

                # Evaluate the model on the validation data
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                print(f'MSE on validation set: {mse:.2f}')

                # Make predictions on the test data
                test_df = pd.read_csv('../house-prices-advanced-regression-techniques/test.csv')
                y_pred_test = model.predict(test_df)
            """
        },
        {
            "role": "self-generated unit test fail",
            "content": """
                Traceback (most recent call last):
                File "/home/shiqi/llama3/gen_test.py", line 33, in <module>
                    model.fit(X_train, y_train)
                ValueError: Input X contains NaN.
                RandomForestRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
                Reflect your code, and rewrite the complete code only.
            """
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
