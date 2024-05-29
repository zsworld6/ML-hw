CAAFE_METHOD_1 = """
    请你用caafe来实现这一分类任务，以下是caafe的一个demo：
    from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
    from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
    from sklearn.ensemble import RandomForestClassifier

    import os
    import openai
    import torch
    from caafe import data
    from sklearn.metrics import accuracy_score
    from tabpfn.scripts import tabular_metrics
    from functools import partial
    openai.api_key = "sk-OBHPzt9qv7cnvfKl04CeEc9e5b09474b839b316390FcFa2b"
    openai.api_base = "https://api.xiaoai.plus/v1"
    metric_used = tabular_metrics.auc_metric
    cc_test_datasets_multiclass = data.load_all_data()
    ds = cc_test_datasets_multiclass[5]
    # ds
    # print(ds)
    ds, df_train, df_test, _, _ = data.get_data_split(ds, seed=0)
    print (df_train)
    target_column_name = ds[4][-1]
    dataset_description = ds[-1]
    # ds[0]
    from caafe.preprocessing import make_datasets_numeric
    df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
    train_x, train_y = data.get_X_y(df_train, target_column_name)
    test_x, test_y = data.get_X_y(df_test, target_column_name)
    clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
    ### Setup and Run CAAFE - This will be billed to your OpenAI Account!

    caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                llm_model="gpt-4",
                                iterations=2)

    caafe_clf.fit_pandas(df_train,
                        target_column_name=target_column_name,
                        dataset_description=dataset_description)

    pred = caafe_clf.predict(df_test)
    acc = accuracy_score(pred, test_y)
    print(f'Accuracy after CAAFE {acc}')
"""

CAAFE_METHOD_2 = """
    请你用caafe来实现这一分类任务，以下是caafe的用法：

    ### Usage
    To use CAAFE, first create a `CAAFEClassifier` object specifying your sklearn base classifier (clf_no_feat_eng; e.g. a random forest or [`TabPFN`](https://github.com/automl/TabPFN))
    and the language model you want to use (e.g. gpt-4):

    ```python
    clf_no_feat_eng = ...
    caafe_clf = CAAFEClassifier(
        base_classifier=clf_no_feat_eng,
        llm_model="gpt-4",
        iterations=2
    )
    ```

    Then, fit the CAAFE-enhanced classifier to your training data:
    ```python
    caafe_clf.fit_pandas(
        df_train,
        target_column_name=target_column_name,
        dataset_description=dataset_description
    )
    ```
    Finally, use the classifier to make predictions on your test data:

    ```python
    pred = caafe_clf.predict(df_test)
    ```

    View generated features:
    ```python
    print(caafe_clf.code)
    ```

    #### Why not let GPT generate your features directly (or use Code Interpreter)?
    GPT-4 is a powerful language model that can generate code.
    However, it is not designed to generate code that is useful for machine learning.
    CAAFE uses a systematic verification process to ensure that the generated features are actually useful for the machine learning task at hand by: iteratively creating new code, verifying their performance using cross validation and providing feedback to the language model.
    CAAFE makes sure that cross validation is correctly applied and formalizes the verification process.
    Also, CAAFE uses a whitelist of allowed operations to ensure that the generated code is safe(er) to execute.
    There inherent risks in generating AI generated code, however, please see [Important Usage Considerations][#important-usage-considerations].

    #### Downstream Classifiers
    Downstream classifiers should be fast and need no specific hyperparameter tuning since they are iteratively being called.
    By default we are using [`TabPFN`](https://github.com/automl/TabPFN) as the base classifier, which is a fast automated machine learning method for small tabular datasets.

    ```python
    from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets

    clf_no_feat_eng = TabPFNClassifier(
        device=('cuda' if torch.cuda.is_available() else 'cpu'),
        N_ensemble_configurations=4
    )
    clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)
    ```

    However, [`TabPFN`](https://github.com/automl/TabPFN) only works for small datasets. You can use any other sklearn classifier as the base classifier.
    For example, you can use a [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html):
    ```python
    from sklearn.ensemble import RandomForestClassifier

    clf_no_feat_eng = RandomForestClassifier(n_estimators=100, max_depth=2)
"""