
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import os
from itertools import permutations, product
import concurrent.futures

## fonction qui prend la prediction du modèle, les articles, et les cartons
## calcul les volumes, l'espace inoccupé, 
### renvoie une table (df) qui donne pour chaque article les infos du
## carton dans lequel l'article 
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

### Input : return view
### Selection d'un carton et application de la visualisation
def visualize_packing1(df):
    list_carton = list(pd.unique(df['ID Carton']))
    results = []
    for id_carton in list_carton:
        df_temp = df[df['ID Carton'] == id_carton].copy()
        df_temp['Volume Article (cm^3)'] = df_temp['Longueur Article (cm)'] * df_temp['Largeur Article (cm)'] * df_temp['Hauteur Article (cm)']
        df_temp = df_temp.sort_values(by='Volume Article (cm^3)', ascending=False)
        result = visualize_packing(df_temp, id_carton)
        results.append(result)
    return results

def pack_articles(articles, carton_length, carton_width, carton_height):
    def can_place(x, y, z, dx, dy, dz, placed):
        if x + dx > carton_length or y + dy > carton_width or z + dz > carton_height:
            return False
        for p in placed:
            px, py, pz, pdx, pdy, pdz = p['position']
            if (x < px + pdx and x + dx > px and
                y < py + pdy and y + dy > py and
                z < pz + pdz and z + dz > pz):
                return False
        return True

    def find_best_position(article, placed):
        best_position = None
        best_z = float('inf')
        best_y = float('inf')
        best_x = float('inf')
        
        rotations = [
            (article[0], article[1], article[2]),
            (article[1], article[0], article[2]),
            (article[2], article[1], article[0]),
            (article[0], article[2], article[1]),
            (article[1], article[2], article[0]),
            (article[2], article[0], article[1])
        ]

        for rotation in rotations:
            dx, dy, dz = rotation
            for z in range(int(carton_height - dz + 1)):
                for y in range(int(carton_width - dy + 1)):
                    for x in range(int(carton_length - dx + 1)):
                        if can_place(x, y, z, dx, dy, dz, placed):
                            if z < best_z or (z == best_z and y < best_y) or (z == best_z and y == best_y and x < best_x):
                                best_position = (x, y, z, dx, dy, dz)
                                best_z, best_y, best_x = z, y, x
                            break  # Move to next y if position found
                    if best_position and best_z == z and best_y == y:
                        break  # Move to next z if position found
                if best_position and best_z == z:
                    break  # Stop if best bottom position found

        return best_position

    placed = []
    for article in sorted(articles, key=lambda a: a[0]*a[1]*a[2], reverse=True):
        position = find_best_position(article, placed)
        if position:
            placed.append({'dimensions': article, 'position': position})

    volume_utilisé = sum(a['dimensions'][0] * a['dimensions'][1] * a['dimensions'][2] for a in placed) / (carton_length * carton_width * carton_height)
    return placed, volume_utilisé


def draw_parallelepiped(x, y, z, dx, dy, dz, color):
    return go.Mesh3d(
        x=[x, x, x+dx, x+dx, x, x, x+dx, x+dx],
        y=[y, y+dy, y+dy, y, y, y+dy, y+dy, y],
        z=[z, z, z, z, z+dz, z+dz, z+dz, z+dz],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=color,
        opacity=1,  # Réduire légèrement l'opacité pour améliorer les performances
        flatshading=True  # Utiliser le flat shading pour un rendu plus rapide
    )

def random_color():
    return f'rgb({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)})'

def process_orientation(args):
    articles, orientation = args
    placed_articles, volume_utilisé = pack_articles(articles, *orientation)
    return placed_articles, volume_utilisé, orientation

### faire la viz de les articles qui sont dans le meme carton
### enregistrement en format specifique
def visualize_packing(df, id_carton):
    carton_dims = df[['Longueur Carton (cm)', 'Largeur Carton (cm)', 'Hauteur Carton (cm)']].iloc[0].tolist()
    nb_articles = len(df)
    articles = list(df[['Longueur Article (cm)', 'Largeur Article (cm)', 'Hauteur Article (cm)']].itertuples(index=False, name=None))
    
    carton_orientations = list(permutations(carton_dims))
    best_orientation = None
    best_volume_utilisé = 0
    best_placed_articles = []
    
    for orientation in carton_orientations:
        placed_articles, volume_utilisé = pack_articles(articles, *orientation)
        if len(placed_articles) > len(best_placed_articles) or (len(placed_articles) == len(best_placed_articles) and volume_utilisé > best_volume_utilisé):
            best_orientation = orientation
            best_volume_utilisé = volume_utilisé
            best_placed_articles = placed_articles
    
    carton_length, carton_width, carton_height = best_orientation
    
    fig = go.Figure()
    fig.add_trace(draw_parallelepiped(0, 0, 0, carton_length, carton_width, carton_height, 'rgba(200, 200, 200, 0.1)'))

    placed_articles_count = 0
    for i, article in enumerate(articles):
        if placed_articles_count < len(best_placed_articles) and article == best_placed_articles[placed_articles_count]['dimensions']:
            x, y, z, dx, dy, dz = best_placed_articles[placed_articles_count]['position']
            fig.add_trace(draw_parallelepiped(x, y, z, dx, dy, dz, random_color()))
            placed_articles_count += 1
        else:
            print(f"Article {i+1} ne peut pas être emballé.")

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, carton_length], showticklabels=False, title='X (cm)'),
            yaxis=dict(range=[0, carton_width], showticklabels=False, title='Y (cm)'),
            zaxis=dict(range=[0, carton_height], showticklabels=False, title='Z (cm)'),
            aspectmode='data'
        ),
        title=f'Visualisation de l\'emballage dans le carton {id_carton} (Articles placés: {placed_articles_count}/{nb_articles})',
        margin=dict(r=20, l=10, b=10, t=40),
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'Volume utilisé: {best_volume_utilisé:.2%}',
                showarrow=False,
            )
        ]
    )

    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
    fig.write_html(f"app/static/images/images_emballage/viz_carton{id_carton}.html")

    return {
        'carton_id': id_carton,
        'carton_dimensions': best_orientation,
        'articles_placés': placed_articles_count,
        'total_articles': nb_articles,
        'volume_utilisé': best_volume_utilisé
    }
# Exemple d'utilisation avec un DataFrame (df)
# visualize_packing_vtk(df, id_carton=1)

### classe BIN : Algorithme de regroupement des articles 
### dans le carton. 
class Bin1:

    def __init__(self, df_article, df_carton):
        self.alpha = 0.15
        self.df_article = df_article
        self.df_carton = df_carton
        self.df_article["v"] = df_article['Longueur'] * df_article['Largeur'] * df_article['Hauteur']
        self.df_carton["v"] = df_carton['Longueur'] * df_carton['Largeur'] * df_carton['Hauteur']

    
    ### fonction - qui prend les dimensions d'un article et son poids et df des cartons
    ### on regarde les contraintes pour chacun des cartons du df
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
            a_dims = df_group["v"].sum() #
            a_dims = Bin.cumul_articles(df_group)
            a_weight = df_group["Poids"].sum()

            # Search for a carton to pack the group of articles
            cartons_except_used = df_carton[~df_carton.index.isin(used_cartons)]
            #print("Dimensions combinees : ", a_dims)
            #print("Dimensions cartons : ", cartons_except_used)
            #print("===================\n\n")
            #print(ddss)
            #packed_group = Bin.bp1(a_dims, a_weight, cartons_except_used)
            packed_group = Bin.bp(a_dims, a_weight, cartons_except_used)

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


###
from itertools import permutations, product
import numpy as np
import pandas as pd

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

from itertools import permutations, product
import numpy as np
import pandas as pd

class Bin:

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
    def calculate_combined_volume(combination):
        """
        Calcule le volume total pour une combinaison d'articles.
        :param combination: Combinaison d'orientations des articles
        :return: Volume total des articles combinés
        """
        total_volume = sum(item[0] * item[1] * item[2] for item in combination)  # Somme des volumes
        return total_volume

    @staticmethod
    def bp(a_volume, a_weight, df_carton):
        """
        Cherche un carton capable de contenir les articles en fonction du volume et du poids.
        :param a_volume: Volume total des articles
        :param a_weight: Poids total des articles
        :param df_carton: DataFrame des cartons disponibles
        :return: Index du carton sélectionné ou None si aucun carton ne convient
        """
        # Vérification des cartons par rapport au volume et au poids
        df_carton['fits'] = df_carton.apply(
            lambda carton: carton['v'] >= a_volume and carton['Poids_max'] >= a_weight, axis=1
        )

        # Filtrer les cartons qui peuvent contenir la combinaison d'articles
        filtered_df = df_carton[df_carton["fits"] == True]
        
        if filtered_df.empty:
            return None
            
        # Retourner l'index du carton ayant le volume le plus proche du volume nécessaire
        return filtered_df['v'].idxmin()

    @staticmethod
    def put2(df_article, df_carton, alpha=0.25):
        used_cartons = []  # List to store used cartons
        group_results = []  # List to store the results for each group

        # Créer des groupes d'articles en fonction de alpha
        n = len(df_article)
        div = max(int((1 - alpha) * n), 1)

        grouped_articles = []

        # Regrouper les articles en les divisant en morceaux
        for i in range(0, n, div):
            grouped_articles.append(df_article.iloc[i:i + div].copy())

        for df_group in grouped_articles:
            # Générer les orientations possibles pour chaque article du groupe
            orientations = [Bin.generate_orientations([row['Longueur'], row['Largeur'], row['Hauteur']]) for _, row in df_group.iterrows()]
            # Générer toutes les combinaisons d'orientations
            combinations = Bin.generate_combinations(orientations)

            found = False
            for combination in combinations:
                # Calculer le volume total pour chaque combinaison
                a_volume = Bin.calculate_combined_volume(combination)
                a_weight = df_group["Poids"].sum()

                # Rechercher un carton pour emballer le groupe d'articles
                cartons_except_used = df_carton[~df_carton.index.isin(used_cartons)]
                packed_group = Bin.bp(a_volume, a_weight, cartons_except_used)

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
                c1 = []  # IDs des articles
                c2 = []  # IDs des cartons
                c3 = []  # Groupes d'articles emballés ensemble

                for x in res:
                    for i in x[0]:
                        c1.append(i)
                        c2.append(x[1])
                        c3.append(x[0])

                # Créer le DataFrame résultat
                df = pd.DataFrame({'id_article': c1, 'id_carton': c2, 'pack_together': c3})
                break
        return view(df, df_article, df_carton)
