from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
from main_vf import *
import warnings

app2 = Flask(__name__)

parser = argparse.ArgumentParser(description = 'Train or test neural net')
parser.add_argument('--train', dest = 'train', action = 'store_true', default = False)
parser.add_argument('--test', dest = 'test', action = 'store_true', default = True)
parser.add_argument('--maxlen', type=int, default=2000, help='Max timesteps')
parser.add_argument('--gamma', type=float, default=0.90, help='Max timesteps')
parser.add_argument('--epsilon', type=float, default=0.01, help='Max timesteps')
parser.add_argument('--epsilon_min', type=float, default=0.01, help='Max timesteps')
parser.add_argument('--epsilon_decay', type=float, default= 0.995, help='Max timesteps')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Max timesteps')
parser.add_argument('--episodes', type=int, default=300, help='Max timesteps')
parser.add_argument('--tmax', type=int, default=3000, help='Max timesteps')

args = parser.parse_args()

@app2.route('/')
def index():
    return app2.send_static_file('index.html')

@app2.route('/process_form', methods=['POST'])
def process_form():
    data = request.json
    
    # Convert the data to CSV format
    df = pd.DataFrame([data])
    csv_data = StringIO()
    df.to_csv(csv_data, index=False)

    # Apply your function to the dataframe (example function)
    df_result = main_function() #example_function(df)

    ## lecture en dataframe de df_carton
    #df_result = getPack(df_article, df_carton)
    
    # Convert the resulting dataframe to a list of dictionaries
    result = df_result.to_dict()#orient='records')
    
    return jsonify(result)

def main_function(model_used = False):
        print("=============== EValuation du modèle : ====================\n\n")
        df_article = pd.read_csv("data/articles_data.csv")
        new_names = {'longueur': 'Longueur', 'largeur': 'Largeur',
        "hauteur": "Hauteur", "poids": "Poids", "quantite": "Quantite"}
        df_article = df_article.rename(columns=new_names)
        #==========================================================
        df_article['key'] = df_article.index
        df_key = df_article.copy()#[['key', 'sku']].copy()
        df_key = df_key.rename(columns = {'Longueur':'Longueur_key',
                        'Largeur':'Largeur_key',
                        'Hauteur':'Hauteur_key',
                        'Poids': 'Poids_key', 
                        'Quantite':'Quantite_key'})
        #===========================================================

        df_article = df_article[['sku', 'Longueur', 'Largeur', 'Hauteur', 'Poids', 'Quantite']]
        df_article = df_article.loc[df_article.index.repeat(df_article['Quantite'])].reset_index(drop=True)
        df_article['Quantite'] = 1
        print( "Nombre Articles : ", len(df_article)) 

        df_carton = pd.read_csv("data/bins.csv")
        df_carton = df_carton[['Longueur', 'Largeur', 'Hauteur', 'Poids_max','Prix', 'Quantite', 'Type']]
        ## BIN PACK
        if model_used :
            print("+++++++ MODEL ++++++++++")
            env = Environment( df_article , df_carton)

            state_size = len(env.items_data(0))  #env.get_state_size()
            action_size = len(df_carton)
            agent = DQNAgent(state_size, action_size, args)
            agent.load("save/model.weights.h5")

            #agent.load("save/model.h5")

            pred = test(env, agent, state_size)
            #print("\n\nres : ", pred)

            res = view(pred, df_article, df_carton)
        else :
            bin = Bin(df_article, df_carton)
            res =  bin.pack()
        
        #=====================================================================
        res = res.merge(df_key, how = 'left',left_on = ['Longueur Article (cm)',
        'Largeur Article (cm)', 'Hauteur Article (cm)', 'Poids Article (kg)'],
        right_on = ['Longueur_key','Largeur_key', 'Hauteur_key', 'Poids_key'])
            
        res = res.drop_duplicates(subset = ["key","ID Carton"])
        res = res[["sku", 'ID Carton', 'Longueur Article (cm)',
       'Largeur Article (cm)', 'Hauteur Article (cm)', 'Poids Article (kg)',
       'Quantite Article', "Volume Article",
       "Volume Articles", "Poids Articles", "Longueur", 'Largeur', 'Hauteur',
       'Max Weight', 'Prix', 'Quantite', 'Type', "Volume Carton",
       'Espace inoccupé', 'Poids inoccupé', 'fragile']].copy()
        res[["Volume Articles", "Poids Articles","Volume Carton",'Espace inoccupé', "Poids Article (kg)"]] = res[["Volume Articles", "Poids Articles","Volume Carton",'Espace inoccupé', "Poids Article (kg)"]].round(2)
        #======================================================================
        
        return res




def example_function(df):
    # Example function that adds a new column
    df['new_column'] = df['name'] + ' - processed'
    return df

if __name__ == '__main__':
    app2.run(debug=True)