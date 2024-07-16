from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
from main_vf import *

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/process_form', methods=['POST'])
def process_form():
    data = request.json
    
    # Convert the data to CSV format
    df = pd.DataFrame([data])
    csv_data = StringIO()
    df.to_csv(csv_data, index=False)

    # Apply your function to the dataframe (example function)
    df_result = main_function() #example_function(df)
    
    # Convert the resulting dataframe to a list of dictionaries
    result = df_result.to_dict()#orient='records')
    print("result : ", result)
    
    return jsonify(result)

parser = argparse.ArgumentParser(description = 'Train or test neural net')
parser.add_argument('--train', dest = 'train', action = 'store_true', default = False)
parser.add_argument('--test', dest = 'test', action = 'store_true', default = True)
parser.add_argument('--maxlen', type=int, default=2000, help='Max timesteps')
parser.add_argument('--gamma', type=float, default=0.90, help='Max timesteps')
parser.add_argument('--epsilon', type=float, default=0.01, help='Max timesteps')
parser.add_argument('--epsilon_min', type=float, default=0.01, help='Max timesteps')
parser.add_argument('--epsilon_decay', type=float, default= 0.995, help='Max timesteps')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Max timesteps')
parser.add_argument('--episodes', type=int, default=2, help='Max timesteps')
parser.add_argument('--episode', type=int, default=5, help='Max timesteps')
parser.add_argument('--tmax', type=int, default=3000, help='Max timesteps')
parser.add_argument('--nb_article', type=int, default=4, help='Max timesteps')

args = parser.parse_args()

def main_function():
        print("=============== Test modèle : ====================\n\n")
        df_article = pd.read_csv("data/articles_data.csv")
        new_names = {'longueur': 'Longueur', 'largeur': 'Largeur',
        "hauteur": "Hauteur", "poids": "Poids", "quantite": "Quantite"}
        df_article = df_article.rename(columns=new_names)
        df_article = df_article[['Longueur', 'Largeur', 'Hauteur', 'Poids', 'Quantite']]
        df_article = df_article.loc[df_article.index.repeat(df_article['Quantite'])].reset_index(drop=True)
        df_article['Quantite'] = 1
        print( "Nombre Articles : ", len(df_article)) 

        df_carton = pd.read_csv("data/bins.csv") # conteneurs_data
        df_carton = df_carton[['Longueur', 'Largeur', 'Hauteur', 'Poids_max','Prix', 'Quantite', 'Type']]
        # Initialiser l'env
        print("+++++++ ENVIRONNEMENT ++++++++++")
        #"""
        env = Environment( df_article , df_carton)

        state_size = len(env.items_data(0))  #env.get_state_size()
        action_size = len(df_carton)
        agent = DQNAgent(state_size, action_size, args)
        print(agent._build_model().summary())
        agent.load("save/model.weights.h5")

        #res = test(env, agent, action_size, 20)
        #print("id_carton : ", res)

        ## Eval model
        pred = evaluate(env, agent, state_size)
        #"""
        viz_result = view(pred, df_article, df_carton)
        #print(viz_result)
        #print("===========")

        ## BIN PACK
        bin = Bin(df_article, df_carton)
        return bin.pack()




def example_function(df):
    # Example function that adds a new column
    df['new_column'] = df['name'] + ' - processed'
    return df

if __name__ == '__main__':
    app.run(debug=True)
