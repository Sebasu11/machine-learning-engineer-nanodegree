
# Funcion para calcular el % de datos faltantes para cada columnas
def missing_values_table(df):
    import pandas as pd
    # Total valores faltantes
    mis_val = df.isnull().sum()

    # % valores faltantes
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Tabla para presentar datos
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Renombrar columnas
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing ', 1 : '% of Total Values'})

    # Ordena la tabla de mayor faltantes a menor
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print resumen
    print ("DataFrame have " + str(df.shape[1]) + " columns.\n"
     + str(mis_val_table_ren_columns.shape[0]) +
    " at least miss one value.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# Funcion para identificar mejores variables basadas en pruebas estadísticas univariadas
def Feature_Selection_k_highest_scores(df,target,stat):
    import pandas as pd
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif,chi2
    select = SelectKBest(stat)
    select.fit(df.drop([target],axis=1), df[target])
    scores = select.scores_
    feature_scores = pd.DataFrame({'columnas': df.drop([target],axis=1).columns.tolist(),'scores': scores.tolist(),})
    return feature_scores

def plt_pca(df,target):
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import matplotlib.pyplot as plt
    pca = PCA(df.shape[1]-1)
    projected = pca.fit_transform(MinMaxScaler().fit_transform(df.drop([target],axis=1)))
    pca_inversed_data = pca.inverse_transform(np.eye(df.shape[1]-1))
    plt.style.use('seaborn')

    plt.figure(figsize = (15, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), '--o')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(0, df.shape[1]-1, 1.0))
    plt.tight_layout()
    plt.xticks(rotation=90)
    upper_9= list(filter(lambda x: x > 0.9, list(np.cumsum(pca.explained_variance_ratio_))))
    print(f'90 % variance explained with : {list(np.cumsum(pca.explained_variance_ratio_)).index(upper_9[0])} components')

def box_plot(df,categories,nr_cols):
    import matplotlib.pyplot as plt
    import seaborn as sns
    nr_rows = len(categories)//nr_cols+1
    li_cat_feats = list(categories)
    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))
    for r in range(0,nr_rows):
        for c in range(0,nr_cols):
            i = r*nr_cols+c
            if i < len(li_cat_feats):
                sns.boxplot(x=li_cat_feats[i], data=df, ax = axs[r][c])
    plt.tight_layout()
    plt.show()

def count_plot(df,categories,nr_cols):
    import matplotlib.pyplot as plt
    import seaborn as sns
    nr_rows = len(categories)//nr_cols+1
    li_cat_feats = list(categories)
    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))
    for r in range(0,nr_rows):
        for c in range(0,nr_cols):
            i = r*nr_cols+c
            if i < len(li_cat_feats):
                sns.countplot(x=li_cat_feats[i], data=df, ax = axs[r][c])
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test,y_prediction):
    import itertools
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score,recall_score,precision_score,f1_score,confusion_matrix,accuracy_score
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_prediction)
    labels = ['0', '1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    fmt = '.1f'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")
    plt.grid('off')
    plt.show()


def variance_threshold_selector(df,target,threshold):
    selector = VarianceThreshold(threshold)
    X = df.drop(target,axis=1)
    selector.fit(X)
    feat_ix_keep = selector.get_support(indices=True)
    orig_feat_ix = np.arange(X.columns.size)
    feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)

    print('Se eliminarion {} columnas del dataset.'.format(len(X.columns[feat_ix_delete])))
    print('Columnas eliminadas : ')
    print('\n')
    print(df.columns[feat_ix_delete])
    df.drop(X.columns[feat_ix_delete],axis=1,inplace=True)

# Funcion para eliminar outliers a partir del rango interquartile
def drop_outliers(df, field_name):
    IQR = (df[field_name].quantile(.75) - df[field_name].quantile(.25))
    K=3#=1.5
    df.drop(df[df[field_name] > df[field_name].quantile(.75)+(IQR*K)].index, inplace=True)
    df.drop(df[df[field_name] < df[field_name].quantile(.25)-(IQR*K)].index, inplace=True)