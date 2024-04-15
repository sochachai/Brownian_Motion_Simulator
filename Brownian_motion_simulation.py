import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  # (version 4.7.0 or higher)
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)

#np.random.seed(42)

app = Dash(__name__)
'''
B(0) = 0
B(t_1) = B(0) + \sqrt{t_1}N(0,1)
B(t_2) = B(t_1) + \sqrt{t_2-t_1}N(0,1)
...
B(t_n) = B(t_{n-1}) + \sqrt{t_{n}-t_{n-1}}N(0,1)
'''

# App layout
app.layout = html.Div([

    html.H1("Brownian motion simulation with Dash", style={'text-align': 'center'}),

    dcc.Dropdown(id = "sample_point_cardinality",
                 options = [
                     {"label": "100", "value": 100},
                     {"label": "500", "value": 500},
                     {"label": "1000", "value": 1000},
                     {"label": "10000", "value": 10000}],
                 multi = False,
                 value = 100,
                 style = {'width': "40%"}
                 ),
    dcc.Dropdown(id = "step_size",
                 options = [
                     {"label": "1", "value": 1},
                     {"label": "0.5", "value": 0.5},
                     {"label": "0.1", "value": 0.1},
                     {"label": "0.01", "value": 0.01}],
                 multi = False,
                 value = 1,
                 style = {'width': "40%"}
                 ),
    html.Br(),

    dcc.Graph(id='Brownian_motion')

])

# Connect the Plotly graphs with Dash Components
@app.callback(
    Output(component_id = 'Brownian_motion', component_property = 'figure'), # Do not put brackets! Otherwise it expects a list!
    [Input(component_id = 'sample_point_cardinality', component_property = 'value'),
     Input(component_id = 'step_size', component_property = 'value')]
)
def update_graph(n, delta_t):
    # get standard normal sequence
    normal_seq = np.random.normal(0, 1, n)

    # initialize sample point array
    sample_path = np.array([0])

    # append sample points
    for i in range(n):
        new_sample_point = sample_path[-1] + np.sqrt(delta_t) * normal_seq[i]
        sample_path = np.append(sample_path, [new_sample_point])

    # Plot sample path
    t = np.arange(0,n+1)
    y = sample_path
    fig = px.line(x = t, y = y, title = 'BM_sample_path')
    return fig


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)

'''
# choose sample point cardinality and step size
n = 100000
step_size = 0.1
# get standard normal sequence
normal_seq = np.random.normal(0,1,n)
# initialize sample point array
sample_path = np.array([0])
# append sample points
for i in range(n):
    new_sample_point = sample_path[-1] + np.sqrt(step_size) * normal_seq[i]
    sample_path = np.append(sample_path,[new_sample_point])


t = np.arange(0,n+1)
y = sample_path
fig = px.line(x = t, y = y, title = 'BM_sample_path')
print(fig.show())
#plt.plot(t,y)
#plt.show()
'''
