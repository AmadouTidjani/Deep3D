import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random

def view(pred, df_article, df_carton): 
    # Vérifier si pred est un DataFrame
    if isinstance(pred, pd.DataFrame):
        res = pred.copy()
    else:
        res = pd.DataFrame(pred, columns=["id_carton"])
        res["id_article"] = res.index
    
    df_article["id"] = df_article.index
    res = res.join(df_article.set_index('id'), on='id_article')
    res["article_volume"] = res['Longueur'] * res['Largeur'] * res['Hauteur'] * res["Quantite"]
    res["Poids_Qte"] = res["Poids"] * res["Quantite"]

    res["cumul_volume"] = res.groupby("id_carton")["article_volume"].transform("sum")
    res["cumul_poids"] = res.groupby("id_carton")["Poids_Qte"].transform("sum")
    df_carton["id"] = df_carton.index
    res = res.rename(columns = {
        'Longueur': 'Longueur Article (cm)',
        'Largeur': 'Largeur Article (cm)',
        'Hauteur': 'Hauteur Article (cm)',
        'Poids': 'Poids Article (kg)',
        'Quantite': 'Quantite Article',
        'v': 'v_last'
    })
    res = res.join(df_carton.set_index('id'), on='id_carton')
    res["box_volume"] = res['Longueur'] * res['Largeur'] * res['Hauteur']
     # Calcul des indicateurs 
    res["esp_inocc"] = np.round(100 * (res["box_volume"] - res["cumul_volume"]) / res["box_volume"], 2)
    res["poids_inocc"] = np.round(100 * (res["Poids_max"] - res["cumul_poids"]) / res["Poids_max"], 2)

    res = res.rename(columns = {
        'Longueur': 'Longueur Carton (cm)',
        'Largeur': 'Largeur Carton (cm)',
        'Hauteur': 'Hauteur Carton (cm)',
        'Poids_max': 'Poids_max Carton (kg)',
        'Quantite': 'Quantite Carton',
    })

    #list_def = ["id_article", "id_carton", "box_volume", "article_volume", "cumul_volume", "esp_inocc", "cumul_poids", "Poids_max", "poids_inocc"]
    
    #res = res.drop('v', axis =0)#[list_def]
    
    # Décommenter les lignes ci-dessous si vous souhaitez renommer les colonnes
    
    
    res = res.rename(columns={
        'id_article': 'ID Article',
        'id_carton': 'ID Carton',
        'box_volume': "Volume Carton",
        "article_volume": "Volume Article",
        "cumul_volume": "Volume Articles",
        "esp_inocc": "Espace inoccupé",
        "cumul_poids": "Poids Articles",
        "Poids_max": "Poids Max",
        "poids_inocc": "Poids inoccupé"
    })
    return res


def visualize_packing1(df):
    list_carton = list(pd.unique(df['ID Carton']))
    for id_carton in list_carton:
        df_temp = df[df['ID Carton'] == id_carton].copy()
        df_temp['Volume Article (cm^3)'] = df_temp['Longueur Article (cm)'] * df_temp['Largeur Article (cm)'] * df_temp['Hauteur Article (cm)']
        df_temp = df_temp.sort_values(by='Volume Article (cm^3)', ascending=False)
       
        visualize_packing(df_temp, id_carton)

def visualize_packing(df, id_carton):
    length = df['Longueur Carton (cm)'].iloc[0]
    width = df['Largeur Carton (cm)'].iloc[0]
    height = df['Hauteur Carton (cm)'].iloc[0]
    nb_articles = len(df)
    carton_length = max(length, width, height)
    carton_width = min(length, width, height)
    carton_height = np.median([length, width, height])
    bin_dims = sorted([length,width,height])
    carton_height,carton_width,carton_length = bin_dims

    articles = list(df[['Longueur Article (cm)', 'Largeur Article (cm)', 'Hauteur Article (cm)']].itertuples(index=False, name=None))
    list_fragiles = df[['Fragile']]
    '''
    # Attribuer des couleurs uniques aux dimensions uniques des articles
    unique_dims = list(set(articles))
    colors = go.cm.get_cmap('hsv', len(unique_dims))
    color_dict = {dims: colors(i) for i, dims in enumerate(unique_dims)}'''

# Exemple d'appel de la fonction (assurez-vous que df est correctement défini)
# visualize_packing(df, id_carton=1)
    # Fonction pour dessiner un parallélépipède avec bordure noire
    def draw_parallelepiped(x, y, z, dx, dy, dz, color):
        return go.Mesh3d(
            x=[x, x, x+dx, x+dx, x, x, x+dx, x+dx],
            y=[y, y+dy, y+dy, y, y, y+dy, y+dy, y],
            z=[z, z, z, z, z+dz, z+dz, z+dz, z+dz],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=color,
            opacity=1
        )

    def random_color():
        return f'rgb({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)})'

    # Initialize the figure
    fig = go.Figure()

    # Add the carton (box)
    fig.add_trace(draw_parallelepiped(0, 0, 0, carton_length, carton_width, carton_height, 'rgba(200, 200, 200, 0.1)'))

    # Keep track of available space in the carton
    available_space = [(0, 0, 0, carton_length, carton_width, carton_height)]
    placed_articles = 0
    traces = []  # To keep track of all article traces

    # Function to find the best position for an article
    def find_best_position(article, spaces):
        best_space_index = -1
        max_difference = float('-inf')
        best_position = None
        best_rotation = None
       
        # Toutes les rotations possibles
        rotations = [
            (article[0], article[1], article[2]),
            (article[1], article[0], article[2]),
            (article[2], article[1], article[0]),
            (article[0], article[2], article[1]),
            (article[1], article[2], article[0]),
            (article[2], article[0], article[1])
        ]
       
        for rotation in rotations:
            for i, (sx, sy, sz, sl, sw, sh) in enumerate(spaces):
                if rotation[0] <= sl and rotation[1] <= sw and rotation[2] <= sh:
                    diff_x = sl - rotation[0]
                    diff_y = sw - rotation[1]
                    diff_z = sh - rotation[2]
                    min_diff = min(diff_x, diff_y, diff_z)
                   
                    if min_diff > max_difference:
                        max_difference = min_diff
                        best_space_index = i
                        best_position = (sx, sy, sz)
                        best_rotation = rotation
       
        return best_space_index, best_position, best_rotation

    # Place the articles
    for i, article in enumerate(articles):
        best_space_index, best_position, best_rotation = find_best_position(article, available_space)
       
        if best_position is None:
            print(f"Article {i} ne peut pas être emballé.")
            break
       
        x_pos, y_pos, z_pos = best_position
       
       
        # Update available spaces
        sx, sy, sz, sl, sw, sh = available_space.pop(best_space_index)
        # Utilisez best_rotation au lieu de article pour les dimensions
        if list_fragiles.Fragile.values[i]:
            color = 'red'
        else:
            color = random_color()

        new_trace = draw_parallelepiped(x_pos, y_pos, z_pos, best_rotation[0], best_rotation[1], best_rotation[2], color=color)#random_color())
        traces.append(new_trace)
       
        # Mise à jour du reste du code comme avant...
        fig = go.Figure()
        fig.add_trace(draw_parallelepiped(0, 0, 0, carton_length, carton_width, carton_height, 'rgba(200, 200, 200, 0.1)'))
        for trace in traces:
            fig.add_trace(trace)
       
        # Lors de la mise à jour des espaces disponibles, utilisez best_rotation
        new_spaces = [
            (sx + best_rotation[0], sy, sz, sl - best_rotation[0], sw, sh),
            (sx, sy + best_rotation[1], sz, sl, sw - best_rotation[1], sh),
            (sx, sy, sz + best_rotation[2], sl, sw, sh - best_rotation[2])
        ]

       
       
        available_space.extend(new_spaces)
        placed_articles += 1

        # Set layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, carton_length], showticklabels=False),
                yaxis=dict(range=[0, carton_width], showticklabels=False),
                zaxis=dict(range=[0, carton_height], showticklabels=False),
            ),
            title=f'Visualisation de l\'emballage dans le carton {id_carton} (Nombre d\'articles: {nb_articles})',
            margin=dict(r=20, l=10, b=10, t=40),
            annotations=[
                dict(
                    x=0.05,
                    y=0.95,
                    xref='paper',
                    yref='paper',
                    text=f'Articles placé(s): {placed_articles}',
                    showarrow=False,
                )
            ]
        )

        # Set camera angle
        fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))

        # Save the updated visualization
        fig.write_html(f"app/static/images/images_emballage/viz_carton{id_carton}.html")



# If you need a static image, you can use:
# fig.write_image(f"app/static/images/images_emballage/viz_carton{id_carton}.png")
    # Sauvegarder la visualisation en PNG
   # plt.savefig(f"app/static/images/images_emballage/viz_carton{id_carton}", bbox_inches='tight')
    #plt.show()


class Bin1:

    def __init__(self, df_article, df_carton):
        self.alpha = 0.25
        self.df_article = df_article
        self.df_carton = df_carton



    def bp(a, b, df):
        # Calculate absolute differences and indicator values
        #df['diff'] = np.abs(df['v'] - float(a))
        df.loc[:, 'diff'] = np.abs(df['v'] - float(a))
        #df['indicator'] = (df['v']>a)*(df['Poids_max']>b)
        df.loc[:, 'indicator'] = (df['v'] > a) & (df['Poids_max'] > b)

        # Multiply absolute differences and indicators
        #df['diff_indicator'] = df['diff'] * df['indicator']
        df.loc[:, 'diff_indicator'] = df['diff'] * df['indicator']

        # Sort by diff_indicator and get the index of the minimum value
        df = df[df["indicator"]==1]["diff_indicator"]

        if df.empty: 
            return None
        return df.idxmin()

    def put(df_article, df_carton, alpha = 0.25): 
        
        # alpha = [0, 1]
        # alpha == 0 : tous les articles dans un seul carton
        # alpha == 0.1 : on divise les articles en 2 groupes par exemple
        # alpha == 0.2 : on divise en 5 groupes
        # ...
        # alpha == 1 : article = un carton
        # Calcul du volume total des articles et des cartons
        
        df_article["v"] = df_article["Longueur"] * df_article["Largeur"] * df_article["Hauteur"] * df_article["Quantite"]
        
        df_carton["v"] = df_carton["Longueur"] * df_carton["Largeur"] * df_carton["Hauteur"]

        used_cartons = []  # Liste pour stocker les cartons déjà utilisés
        results = []  # Liste pour stocker les résultats de l'emballage

        # Diviser les articles en groupes en fonction de alpha
        n = len(df_article)
        div = int((1-alpha) * n)
        if div ==0: div+=1
        done = True

        grouped_articles = []
        for i in range(0, n, div):
            grouped_articles.append(df_article.iloc[i:i+div].copy())

        group_results = []  # Liste pour stocker les résultats de chaque groupe
        for i, df_group in enumerate(grouped_articles):
            # Calcul du volume et du poids total du groupe d'articles
            a_group = df_group["v"].sum()
            b_group = df_group["Poids"].sum()

            # Recherche d'un carton pour emballer le groupe d'articles
            cartons_except_used = df_carton[~df_carton.index.isin(used_cartons)]
            
            packed_group = Bin.bp(a_group, b_group, cartons_except_used)

            if packed_group is not None:
                # Enregistrer le résultat du groupe
                group_results.append((list(df_group.index), packed_group))
                used_cartons.append(packed_group)
            else:
                #print("Aucun carton disponible pour emballer les articles : ", list(df_group.index))
                done *= False
        # Enregistrer les résultats pour cette valeur d'alpha
        #results.append(group_results)

        return group_results, done

    def pack(self):

        df_article = self.df_article
        df_carton = self.df_carton
        done = True
        #alpha = 0.1
        
        
        for alpha in np.arange(0,1+0.1,0.1):
            #
            res, done = Bin.put(df_article, df_carton, alpha)
            #print("===================\n\n", res,"\n\n")
            if not done:print("alpha=",np.round(alpha,2), ",done=",done," Résultat : ", res)
        
            c1 = [] #[x[0] for x in res]
            c2 = [] #[x[1] for x in res]
            c3 = []
            
            # calculer les sorties que si tous les articles sont classés ou dernier élement de la boucle
            if (alpha==1.0) or done:
                print("value: ", np.round(alpha,2))
                for x in res:
                    for i in x[0]:
                        c1.append(i)
                        c2.append(x[1])
                        c3.append(x[0])

                # Création du DataFrame
                df = pd.DataFrame({'id_article': c1, 'id_carton': c2, 'pack_together': c3})
                break
            
        # résultat d'emballage
        # Affichage des resultats
        return view(df, df_article, df_carton)

class Bin:

    def __init__(self, df_article, df_carton):
        self.alpha = 0.25
        self.df_article = df_article
        self.df_carton = df_carton

    @staticmethod
    def median_index(values):
        sorted_indices = np.argsort(values)
        median_idx = len(sorted_indices) // 2
        return sorted_indices[median_idx]

    @staticmethod
    def bp(a_dims, a_weight, df_carton):
        # a_dims is a sorted array of dimensions [Longueur, Largeur, Hauteur]
        a_dims = sorted(a_dims)
        
        def calculate_diff(carton):
            c_dims = sorted([carton['Longueur'], carton['Largeur'], carton['Hauteur']])
            
            diffs = [float(c_dims[i] - a_dims[i]) for i in range(len(a_dims))]
            return diffs

        diffs_list = df_carton.apply(lambda carton: calculate_diff(carton), axis=1)
        df_carton['diffs'] = diffs_list

        # Filter cartons that can contain the article
        df_carton['fits'] = df_carton.apply(
            lambda carton: all(np.array(sorted([carton['Longueur'], carton['Largeur'], carton['Hauteur']])) > np.array(a_dims)) 
                           and carton['Poids_max'] > a_weight, axis=1)

        # Calculate an indicator based on the diffs
        df_carton['diff_indicator'] = df_carton.apply(lambda carton: max(calculate_diff(carton)) if carton['fits'] else np.inf, axis=1)

        # Filter by indicator and get the index of the minimum diff_indicator
        filtered_df = df_carton[df_carton["fits"] == True]
        if filtered_df.empty:
            return None

        return filtered_df['diff_indicator'].idxmin()

    @staticmethod
    def bp1(a, b, df):
        # Calculate absolute differences and indicator values
        df['diff'] = np.abs(df['v'] - float(a))
        df['indicator'] = (df['v'] > a) & (df['Poids_max'] > b)

        # Multiply absolute differences and indicators
        df['diff_indicator'] = df['diff'] * df['indicator']

        # Filter by indicator and get the index of the minimum diff_indicator
        filtered_df = df[df["indicator"] == 1]
        if filtered_df.empty:
            return None
        return filtered_df['diff_indicator'].idxmin()

    @staticmethod
    def put(df_article, df_carton, alpha=0.25):
        used_cartons = []  # List to store used cartons
        group_results = []  # List to store the results for each group

        # Create random groups of articles based on alpha
        n = len(df_article)
        div = int((1 - alpha) * n)
        if div == 0: div += 1

        article_ids = df_article.index.tolist()
        grouped_articles = []

        while article_ids:
            group = np.random.choice(article_ids, size=min(div, len(article_ids)), replace=False)
            grouped_articles.append(df_article.loc[group].copy())
            article_ids = list(set(article_ids) - set(group))

        for i, df_group in enumerate(grouped_articles):
            # Calculate the total dimensions and weight for the group of articles
            lengths = df_group["Longueur"].values
            widths = df_group["Largeur"].values
            heights = df_group["Hauteur"].values
            quantities = df_group["Quantite"].values

            total_lengths = np.sum(np.sort(lengths)[:len(lengths)-1]) + np.max(lengths)
            total_widths = np.sum(np.sort(widths)[:len(widths)-1]) + np.max(widths)
            total_heights = np.sum(np.sort(heights)[:len(heights)-1]) + np.max(heights)

            a_dims = [total_lengths, total_widths, total_heights]
            a_weight = df_group["Poids"].sum()

            # Search for a carton to pack the group of articles
            cartons_except_used = df_carton[~df_carton.index.isin(used_cartons)]
            packed_group = Bin.bp(a_dims, a_weight, cartons_except_used)

            if packed_group is not None:
                # Record the result for the group
                group_results.append((list(df_group.index), packed_group))
                used_cartons.append(packed_group)
            else:
                return group_results, False

        return group_results, True
    
    #staticmethod
    def cumul_articles(df_group):
        # Initialize the total dimensions
        total_length, total_width, total_height = 0, 0, 0
        
        # Initialize previous sorted dimensions
        prev_dims = None

        for index, row in df_group.iterrows():
            current_dims = sorted([row['Longueur'], row['Largeur'], row['Hauteur']])
            
            #print(current_dims)
            
            if prev_dims is None:
                # First iteration, set previous dimensions
                prev_dims = current_dims
            else:
                # Sum the smallest dimensions
                total_length = prev_dims[0] + current_dims[0]
                total_width = max(total_width, prev_dims[1], current_dims[1])
                total_height = max(total_height, prev_dims[2], current_dims[2])
                
                # Update previous dimensions
                prev_dims = sorted([total_length, total_width, total_height])
            
        
        return sorted(prev_dims, reverse = True)

    @staticmethod
    def put2(df_article, df_carton, alpha=0.25):
        used_cartons = []  # List to store used cartons
        group_results = []  # List to store the results for each group

        # Create random groups of articles based on alpha
        n = len(df_article)
        div = int((1 - alpha) * n)
        if div == 0: div += 1

        article_ids = df_article.index.tolist()
        grouped_articles = []

        while article_ids:
            group = np.random.choice(article_ids, size=min(div, len(article_ids)), replace=False)
            grouped_articles.append(df_article.loc[group].copy())
            article_ids = list(set(article_ids) - set(group))
        grouped_articles = []
        for i in range(0, n, div):
            grouped_articles.append(df_article.iloc[i:i+div].copy())
            
        for i, df_group in enumerate(grouped_articles):
            # Calculate the total dimensions and weight for the group of articles
            a_dims = Bin.cumul_articles(df_group)#[df_group["Longueur"].max(), df_group["Largeur"].max(), df_group["Hauteur"].max()]
            a_weight = df_group["Poids"].sum()

            # Search for a carton to pack the group of articles
            cartons_except_used = df_carton[~df_carton.index.isin(used_cartons)]
            packed_group = Bin.bp(a_dims, a_weight, cartons_except_used)

            if packed_group is not None:
                # Record the result for the group
                group_results.append((list(df_group.index), packed_group))
                used_cartons.append(packed_group)
            else:
                return group_results, False

        return group_results, True
    
    @staticmethod
    def put1(df_article, df_carton, alpha=0.25):
        # Calculate volumes for articles and cartons
        df_article["v"] = df_article["Longueur"] * df_article["Largeur"] * df_article["Hauteur"] * df_article["Quantite"]
        df_carton["v"] = df_carton["Longueur"] * df_carton["Largeur"] * df_carton["Hauteur"]

        used_cartons = []  # List to store used cartons
        group_results = []  # List to store the results for each group

        # Create random groups of articles based on alpha
        n = len(df_article)
        div = int((1 - alpha) * n)
        if div == 0: div += 1

        article_ids = df_article.index.tolist()
        grouped_articles = []

        while article_ids:
            group = np.random.choice(article_ids, size=min(div, len(article_ids)), replace=False)
            grouped_articles.append(df_article.loc[group].copy())
            article_ids = list(set(article_ids) - set(group))

        for i, df_group in enumerate(grouped_articles):
            # Calculate the total volume and weight for the group of articles
            a_group = df_group["v"].sum()
            b_group = df_group["Poids"].sum()

            # Search for a carton to pack the group of articles
            cartons_except_used = df_carton[~df_carton.index.isin(used_cartons)]
            packed_group = Bin.bp(a_group, b_group, cartons_except_used)

            if packed_group is not None:
                # Record the result for the group
                group_results.append((list(df_group.index), packed_group))
                used_cartons.append(packed_group)
            else:
                return group_results, False

        return group_results, True

    def pack(self):
        df_article = self.df_article
        df_carton = self.df_carton

        for alpha in np.arange(0, 1.1, 0.1):
            res, done = Bin.put2(df_article, df_carton, alpha)
            print("alpha : ", alpha)
            if done or alpha == 1.0:
                c1 = []  # Article IDs
                c2 = []  # Carton IDs
                c3 = []  # Group of articles packed together

                for x in res:
                    for i in x[0]:
                        c1.append(i)
                        c2.append(x[1])
                        c3.append(x[0])

                # Create the resulting DataFrame
                df = pd.DataFrame({'id_article': c1, 'id_carton': c2, 'pack_together': c3})
                break

        
        return view(df, df_article, df_carton)
    