import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import os
from itertools import permutations, product

if os.getcwd().endswith('flaskblog') or os.getcwd().endswith('Deep3D'):
    from app.models_rl.viz_nsga.Main import visualize_packing_nsga
else:
    from viz_nsga.Main import visualize_packing_nsga

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
        #visualize_packing_nsga(df_temp, id_carton)

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

# Exemple d'appel de la fonction (assurez-vous que df est correctement défini)
# visualize_packing(df, id_carton=1)
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

    # Initialize the figure once
    fig = go.Figure()

    # Add the carton (box)
    fig.add_trace(draw_parallelepiped(0, 0, 0, carton_length, carton_width, carton_height, 'rgba(200, 200, 200, 0.1)'))

    # Track available space and placed articles
    available_space = [(0, 0, 0, carton_length, carton_width, carton_height)]
    placed_articles = []
    traces = []

    def find_best_position(article, spaces, placed_articles):
        best_fit = float('inf')
        best_position = None
        best_rotation = None

        # Générer toutes les rotations possibles
        rotations = [
            (article[0], article[1], article[2]),
            (article[1], article[0], article[2]),
            (article[2], article[1], article[0]),
            (article[0], article[2], article[1]),
            (article[1], article[2], article[0]),
            (article[2], article[0], article[1])
        ]

        for rotation in rotations:
            for space in spaces:
                sx, sy, sz, sl, sw, sh = space
                if rotation[0] <= sl and rotation[1] <= sw and rotation[2] <= sh:
                    # Calculer le volume laissé et utiliser la somme des écarts comme heuristique
                    leftover_volume = (sl - rotation[0]) * (sw - rotation[1]) * (sh - rotation[2])
                    if leftover_volume < best_fit:
                        # Vérifier les chevauchements
                        overlaps = False
                        for placed in placed_articles:
                            px, py, pz, pdx, pdy, pdz = placed['position']
                            if (sx < px + pdx and sx + rotation[0] > px and
                                sy < py + pdy and sy + rotation[1] > py and
                                sz < pz + pdz and sz + rotation[2] > pz):
                                overlaps = True
                                break
                        if not overlaps:
                            best_fit = leftover_volume
                            best_position = (sx, sy, sz)
                            best_rotation = rotation

        return best_position, best_rotation


    # Sort articles by volume descending
    sorted_articles = sorted(articles, key=lambda a: a[0]*a[1]*a[2], reverse=True)

    placed_articles_count = 0

    for i, article in enumerate(sorted_articles):
        position, rotation = find_best_position(article, available_space, placed_articles)
        if position and rotation:
            x_pos, y_pos, z_pos = position

            # Place the article
            placed_articles.append({
                'position': (x_pos, y_pos, z_pos, rotation[0], rotation[1], rotation[2]),
                'dimensions': rotation
            })

            placed_articles_count += 1

            # Add the article to the figure
            fig.add_trace(draw_parallelepiped(x_pos, y_pos, z_pos, rotation[0], rotation[1], rotation[2], random_color()))

            # Update available spaces
            new_spaces = [
                (x_pos + rotation[0], y_pos, z_pos, carton_length - (x_pos + rotation[0]), rotation[1], rotation[2]),
                (x_pos, y_pos + rotation[1], z_pos, rotation[0], carton_width - (y_pos + rotation[1]), rotation[2]),
                (x_pos, y_pos, z_pos + rotation[2], rotation[0], rotation[1], carton_height - (z_pos + rotation[2]))
            ]
            # Filter out any invalid spaces
            new_spaces = [s for s in new_spaces if s[3] > 0 and s[4] > 0 and s[5] > 0]
            available_space.extend(new_spaces)
        else:
            print(f"Article {i} ne peut pas être emballé.")

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, carton_length], showticklabels=False, title='X (cm)'),
            yaxis=dict(range=[0, carton_width], showticklabels=False, title='Y (cm)'),
            zaxis=dict(range=[0, carton_height], showticklabels=False, title='Z (cm)'),
            aspectmode='data'
        ),
        title=f'Visualisation de l\'emballage dans le carton {id_carton} (Nombre d\'articles placés: {placed_articles_count}/{nb_articles})',
        margin=dict(r=20, l=10, b=10, t=40),
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'Articles placé(s): {placed_articles_count}',
                showarrow=False,
            )
        ]
    )

    # Set camera angle for better view
    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))

    # Save the visualization
    fig.write_html(f"app/static/images/images_emballage/viz_carton{id_carton}.html")

class Bin1:

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
            result = np.array(diffs)
            return np.prod(result)

        diffs_list = df_carton.apply(lambda carton: calculate_diff(carton), axis=1)
        df_carton['diffs'] = diffs_list

        # Filter cartons that can contain the article
        df_carton['fits'] = df_carton.apply(
            lambda carton: all(np.array(sorted([carton['Longueur'], carton['Largeur'], carton['Hauteur']])) >= np.array(a_dims)) 
                           and carton['Poids_max'] >= a_weight, axis=1)

        
        # Calculate an indicator based on the diffs
        df_carton['diff_indicator'] = df_carton.apply(lambda carton: calculate_diff(carton) if carton['fits'] else np.inf, axis=1)

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
            res, done = Bin.put(df_article, df_carton, alpha)
            #print("alpha : ", alpha)
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
    

class Bin:

    def __init__(self, df_article, df_carton):
        self.alpha = 0.25
        self.df_article = df_article
        self.df_carton = df_carton
        self.df_article["v"] = df_article['Longueur'] * df_article['Largeur'] * df_article['Hauteur']
        self.df_carton["v"] = df_carton['Longueur'] * df_carton['Largeur'] * df_carton['Hauteur']

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
            # Calculate differences and handle negative differences
            diffs = [max(c_dims[i] - a_dims[i], 0) for i in range(len(a_dims))]
            result = np.array(diffs)
            return np.prod(result)

        df_carton['fits'] = df_carton.apply(
            lambda carton: all(np.array(sorted([carton['Longueur'], carton['Largeur'], carton['Hauteur']])) >= np.array(a_dims)) 
                           and carton['Poids_max'] >= a_weight, axis=1)

        df_carton['diff_indicator'] = df_carton.apply(lambda carton: calculate_diff(carton) if carton['fits'] else np.inf, axis=1)

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
    def cumul_articles(df_group):
        # Initialize the total dimensions
        total_length, total_width, total_height = 0, 0, 0

        for index, row in df_group.iterrows():
            current_dims = sorted([row['Longueur'], row['Largeur'], row['Hauteur']])
            # Accumulate dimensions
            total_length += current_dims[0]
            total_width = max(total_width, current_dims[1])
            total_height = max(total_height, current_dims[2])

        # Return the combined dimensions in sorted order
        return sorted([total_length, total_width, total_height], reverse=True)

    @staticmethod
    def put2(df_article, df_carton, alpha=0.25):
        used_cartons = []  # List to store used cartons
        group_results = []  # List to store the results for each group

        # Create random groups of articles based on alpha
        n = len(df_article)
        div = max(int((1 - alpha) * n), 1)

        article_ids = df_article.index.tolist()
        grouped_articles = []

        # Group articles by dividing the DataFrame into chunks
        for i in range(0, n, div):
            grouped_articles.append(df_article.iloc[i:i+div].copy())
            
        for df_group in grouped_articles:
            # Calculate the total dimensions and weight for the group of articles
            print(" shape df_group : ", df_group.columns)
            a_dims = df_group["v"].sum() #Bin.cumul_articles(df_group)
            a_weight = df_group["Poids"].sum()

            # Search for a carton to pack the group of articles
            cartons_except_used = df_carton[~df_carton.index.isin(used_cartons)]
            #print("Dimensions combinees : ", a_dims)
            #print("Dimensions cartons : ", cartons_except_used)
            #print("===================\n\n")
            #print(ddss)
            packed_group = Bin.bp1(a_dims, a_weight, cartons_except_used)

            if packed_group is not None:
                # Record the result for the group
                group_results.append((list(df_group.index), packed_group))
                used_cartons.append(packed_group)
            else:
                return group_results, False  # Stop if no suitable carton is found

        return group_results, True

    def pack(self):
        df_article = self.df_article
        df_carton = self.df_carton

        for alpha in np.arange(0, 1.1, 0.1):
            res, done = Bin.put2(df_article, df_carton, alpha)
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

class Bin2:

    def __init__(self, df_article, df_carton):
        self.alpha = 0.25
        self.df_article = df_article
        self.df_carton = df_carton
        self.df_article["v"] = df_article['Longueur'] * df_article['Largeur'] * df_article['Hauteur']
        self.df_carton["v"] = df_carton['Longueur'] * df_carton['Largeur'] * df_carton['Hauteur']

    @staticmethod
    def generate_orientations(article):
        """
        Génère toutes les orientations possibles pour un article.
        :param article: Un article avec ses dimensions [L, l, H]
        :return: Liste des permutations (orientations)
        """
        return list(permutations(article))

    @staticmethod
    def generate_combinations(articles):
        """
        Génère toutes les combinaisons possibles d'orientations pour un groupe d'articles.
        :param articles: Liste d'articles, chaque article est une liste de ses orientations possibles
        :return: Liste de combinaisons de dimensions
        """
        return list(product(*articles))

    @staticmethod
    def calculate_combined_dimensions(combination):
        """
        Calcule les dimensions totales (Longueur, Largeur, Hauteur) pour une combinaison d'articles.
        :param combination: Combinaison d'orientations des articles
        :return: Tuple (Longueur totale, Largeur totale, Hauteur totale)
        """
        total_length = sum(item[0] for item in combination)  # Somme des longueurs
        total_width = max(item[1] for item in combination)   # Largeur maximale
        total_height = max(item[2] for item in combination)  # Hauteur maximale
        return total_length, total_width, total_height

    @staticmethod
    def bp(a_dims, a_weight, df_carton):
        # a_dims is a sorted array of dimensions [Longueur, Largeur, Hauteur]
        a_dims = sorted(a_dims)

        def calculate_diff(carton):
            c_dims = sorted([carton['Longueur'], carton['Largeur'], carton['Hauteur']])
            diffs = [max(c_dims[i] - a_dims[i], 0) for i in range(len(a_dims))]
            result = np.array(diffs)
            return np.prod(result)

        df_carton['fits'] = df_carton.apply(
            lambda carton: all(np.array(sorted([carton['Longueur'], carton['Largeur'], carton['Hauteur']])) >= np.array(a_dims)) 
                           and carton['Poids_max'] >= a_weight, axis=1)

        df_carton['diff_indicator'] = df_carton.apply(lambda carton: calculate_diff(carton) if carton['fits'] else np.inf, axis=1)

        # Filter by indicator and get the index of the minimum diff_indicator
        filtered_df = df_carton[df_carton["fits"] == True]
        if filtered_df.empty:
            return None
            
        return filtered_df['diff_indicator'].idxmin()

    @staticmethod
    def put2(df_article, df_carton, alpha=0.25):
        used_cartons = []  # List to store used cartons
        group_results = []  # List to store the results for each group

        # Create random groups of articles based on alpha
        n = len(df_article)
        div = max(int((1 - alpha) * n), 1)

        article_ids = df_article.index.tolist()
        grouped_articles = []

        # Group articles by dividing the DataFrame into chunks
        for i in range(0, n, div):
            grouped_articles.append(df_article.iloc[i:i + div].copy())

        for df_group in grouped_articles:
            # Générer les orientations possibles pour chaque article du groupe
            orientations = [Bin.generate_orientations([row['Longueur'], row['Largeur'], row['Hauteur']]) for _, row in df_group.iterrows()]
            # Générer toutes les combinaisons d'orientations
            combinations = Bin.generate_combinations(orientations)

            found = False
            for combination in combinations:
                # Calculer les dimensions combinées pour chaque combinaison
                print("valeur combinaison : ", combination)
                a_dims = Bin.calculate_combined_dimensions(combination)
                a_weight = df_group["Poids"].sum()

                # Rechercher un carton pour emballer le groupe d'articles
                cartons_except_used = df_carton[~df_carton.index.isin(used_cartons)]
                packed_group = Bin.bp(a_dims, a_weight, cartons_except_used)

                if packed_group is not None:
                    # Enregistrer le résultat pour le groupe
                    group_results.append((list(df_group.index), packed_group))
                    used_cartons.append(packed_group)
                    found = True
                    break

            if not found:
                return group_results, False  # Arrêter si aucun carton approprié n'est trouvé

        return group_results, True

    def pack(self):
        df_article = self.df_article
        df_carton = self.df_carton

        for alpha in np.arange(0, 1.1, 0.1):
            res, done = Bin.put2(df_article, df_carton, alpha)
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
