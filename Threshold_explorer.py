import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq


from jupyter_plotly_dash import JupyterDash
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_curve, auc,accuracy_score,f1_score,recall_score,precision_score,roc_auc_score


def visualize(y_test,y_prob):
    fpr, tpr, threshold = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    d = {'AUC': ['{:.2f}'.format(0)], 'Precision': ['{:.2f}'.format(0)], 'Accuracy': ['{:.2f}'.format(0)], 'Recall': ['{:.2f}'.format(0)], 'F1': ['{:.2f}'.format(0)]}
    df = pd.DataFrame(data=d)
    app = JupyterDash('SimpleExample')
    app.layout = html.Div(children=[
        html.Div(className="banner", children=[
            # Change App Name here
            html.Div(className='container scalable', children=[
                # Change App Name here
                html.H2(html.A(
                    'Threshold Explorer',
                    href='http://localhost:8888/tree',
                    style={'text-decoration': 'none',
                           'color': 'inherit'
                    }
                )),

                #html.A(html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"),
                #                href='https://plot.ly/products/dash/')
            ]),
        ]),

    html.H1(children='DASH Plotly'),

    dcc.Graph(
        id='confusion',

        ),
        dash_table.DataTable(
         id='table',
         #style_table={'overflowX': 'scroll'},
         style_table={'width':'400px'},
         columns=[{"name": i, "id": i} for i in df.columns],
         data=df.to_dict('records'),
         css=[{'selector': '.dash-cell div.dash-cell-value','rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}]
        ),

    dcc.Graph(
        id='ROC',

        figure={
            'data': [
            {"name": f"Mean ROC curve (area ={roc_auc:.3f})",
             "type": "scatter",
             "x": fpr,
             "y": tpr
            }
        ],

            'layout' : {
              "title": "ROC Curve Classification",
              "height": 350,
              "width" : 500,
              "xaxis": {"title": "False Positive Rate"},
              "yaxis": {"title": "True Positive Rate"},
            }
        }
    ),

    html.Div([
    dcc.Slider(
        id='slider-threshold',
        min=0,
        max=1,
        value=0.5,
        step=0.01,
        #marks={'0.25': '0.25','0.5': '0.5', '0.75': '0.75'},
        #size=350,
        #handleLabel={"showCurrentValue": True,"label": "VALUE"},
    ),
    html.Div(id='updatemode-output-container', style={'margin-top': 20})])])

    @app.callback(dash.dependencies.Output('confusion', 'figure'),
                  [dash.dependencies.Input('slider-threshold', 'value')])

    def update_graph(threshold_value):
        y_pred=[1 if x >threshold_value else 0 for x in y_prob]
        cm=confusion_matrix(y_test, y_pred)
        cm_1=cm[0][0]
        cm_2=cm[0][1]
        cm_3=cm[1][0]
        cm_4=cm[1][1]

        return {'data': [{"type": "heatmap",
                          "x": ["0", "1"],#, "standing", "people", "backgroun"],
                          "y": ["1", "0"],#, "standing", "people", "backgroun"],
                          "z": [[cm_1, cm_2], [cm_3, cm_4]],
                          "showscale": True,
                          "colorscale": "RdBu"
                }
                         ],
                'layout' : {
                    "title": "Confusion Matrix",
                    "height": 300,
                    "width" : 400,
                    "xaxis": {"title": "Predicted value"},
                    "yaxis": {"title": "Real value"},
                    "annotations": [
                        {"x": 0,
                         "y": 1,
                         "font": {"color": "#FFFFFF"},
                         "text": str(cm_1),#"9",
                         "xref": "x1",
                         "yref": "y1",
                         "showarrow": False},
                        {"x": 1,
                         "y": 1,
                         "font": {"color": "#FFFFFF"},
                         "text": str(cm_2),
                         "xref": "x1",
                         "yref": "y1",
                         "showarrow": False
                        },
                        {"x": 0,
                         "y": 0,
                         "font": {"color": "#000000"},
                         "text": str(cm_3),
                         "xref": "x1",
                         "yref": "y1",
                         "showarrow": False
                         },
                        {"x": 1,
                         "y": 0,
                         "font": {"color": "#FFFFFF"},
                         "text": str(cm_4),
                         "xref": "x1",
                         "yref": "y1",
                         "showarrow": False
                         }]
                    }
                }
    @app.callback(
    dash.dependencies.Output('table', 'data'),
    [dash.dependencies.Input('slider-threshold', 'value')])

    def update_graph(threshold_value):
        y_pred=[1 if x >threshold_value else 0 for x in y_prob]
        f1=f1_score(y_test, y_pred)
        auc_score=roc_auc_score(y_test, y_pred)
        accuracy=accuracy_score(y_test, y_pred)
        recall=recall_score(y_test, y_pred)
        precision=precision_score(y_test, y_pred)
        d = {'AUC': ['{:.2f}'.format(auc_score)], 'Precision': ['{:.2f}'.format(precision)], 'Accuracy': ['{:.2f}'.format(accuracy)], 'Recall': ['{:.2f}'.format(recall)], 'F1': ['{:.2f}'.format(f1)]}
        columns=[{"name": i, "id": i} for i in df.columns]
        data=df.to_dict('records')
        return [d]

    return app
