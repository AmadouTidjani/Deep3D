import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import os

def visualize_packing1(df):
    list_carton = list(pd.unique(df['ID Carton']))
    for id_carton in list_carton:
        df_temp = df[df['ID Carton'] == id_carton].copy()
        df_temp['Volume Article (cm^3)'] = df_temp['Longueur Article (cm)'] * df_temp['Largeur Article (cm)'] * df_temp['Hauteur Article (cm)']
        df_temp = df_temp.sort_values(by='Volume Article (cm^3)', ascending=False)
        
        visualize_packing(df_temp, id_carton)

def visualize_packing(df, id_carton=0):
    length = df['Longueur Carton (cm)'].iloc[0]
    width = df['Largeur Carton (cm)'].iloc[0]
    height = df['Hauteur Carton (cm)'].iloc[0]
    
    carton_length = max(length, width, height)
    carton_width = min(length, width, height)
    carton_height = np.median([length, width, height])

    articles = list(df[['Longueur Article (cm)', 'Largeur Article (cm)', 'Hauteur Article (cm)']].itertuples(index=False, name=None))

    # Attribuer des couleurs uniques aux dimensions uniques des articles
    unique_dims = list(set(articles))
    colors = plt.cm.get_cmap('hsv', len(unique_dims))
    color_dict = {dims: colors(i) for i, dims in enumerate(unique_dims)}
    
    # Fonction pour dessiner un parallélépipède avec bordure noire
    def draw_parallelepiped(ax, x, y, z, dx, dy, dz, color='b', alpha=0.5):
        vertices = [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz]
        ]

        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [4, 7, 3, 0]
        ]

        poly3d = [[vertices[vert_id] for vert_id in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha))

    images = []  # Liste pour stocker les images de chaque étape
    step_folder = "steps"
    os.makedirs(step_folder, exist_ok=True)  # Créer un dossier pour les étapes

    # Créer une nouvelle figure
    fig = plt.figure(figsize=(21, 7))  # Ajusté pour inclure trois sous-graphes
    views = [(20, 30), (90, 0), (0, 90)]  # Angles de vue pour les trois perspectives
    titles = ['Vue de face', 'Vue d\'en bas', 'Vue de profil']

    for j, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(131 + j, projection='3d')
        draw_parallelepiped(ax, 0, 0, 0, carton_length, carton_width, carton_height, color='#7B68EE', alpha=0.1)
        ax.set_xlim(0, carton_length)
        ax.set_ylim(0, carton_width)
        ax.set_zlim(0, carton_height)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(titles[j])

    plt.suptitle(f'Carton {id_carton} - Étape 0')
    image_path = os.path.join(step_folder, f"step_{id_carton}_0.png")
    plt.savefig(image_path, bbox_inches='tight')
    images.append(imageio.imread(image_path))
    plt.close(fig)

    # Garder la trace de l'espace disponible dans le carton
    available_space = [(0, 0, 0, carton_length, carton_width, carton_height)]
    placed_articles = []

    # Fonction pour trouver la meilleure position pour un article
    def find_best_position(article, spaces):
        best_space_index = -1
        max_difference = -1
        best_position = None

        for i, (sx, sy, sz, sl, sw, sh) in enumerate(spaces):
            # Vérifier si l'article peut rentrer dans cet espace
            if article[0] <= sl and article[1] <= sw and article[2] <= sh:
                # Calculer les différences d'espace
                diff_x = sl - article[0]
                diff_y = sw - article[1]
                diff_z = sh - article[2]
                min_diff = min(diff_x, diff_y, diff_z)

                if min_diff > max_difference:
                    max_difference = min_diff
                    best_space_index = i
                    best_position = (sx, sy, sz)

        return best_space_index, best_position

    # Placer les articles
    for i, article in enumerate(articles):
        best_space_index, best_position = find_best_position(article, available_space)

        if best_position is None:
            print(f"Article {i} ne rentre pas dans le carton.")
            break

        x_pos, y_pos, z_pos = best_position
        placed_articles.append((x_pos, y_pos, z_pos, article[0], article[1], article[2], color_dict[article]))

        fig = plt.figure(figsize=(21, 7))  # Ajusté pour inclure trois sous-graphes

        for j, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(131 + j, projection='3d')
            draw_parallelepiped(ax, 0, 0, 0, carton_length, carton_width, carton_height, color='#7B68EE', alpha=0.1)

            # Dessiner tous les articles placés jusqu'à présent
            for (px, py, pz, pdx, pdy, pdz, color) in placed_articles:
                draw_parallelepiped(ax, px, py, pz, pdx, pdy, pdz, color=color)

            # Dessiner une flèche indiquant l'emplacement du nouvel article
            if j == 0:  # Dessiner la flèche seulement dans la vue de face
                ax.quiver(x_pos, y_pos, z_pos, article[0], article[1], article[2], color='r', linewidth=2, arrow_length_ratio=0.1)

            ax.set_xlim(0, carton_length)
            ax.set_ylim(0, carton_width)
            ax.set_zlim(0, carton_height)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(titles[j])

        plt.suptitle(f'Carton {id_carton} - Étape {i + 1}')
        image_path = os.path.join(step_folder, f"step_{id_carton}_{i + 1}.png")
        plt.savefig(image_path, bbox_inches='tight')
        images.append(imageio.imread(image_path))
        plt.close(fig)

    # Créer et sauvegarder le GIF
    gif_path = f"packing_carton{id_carton}.gif"
    imageio.mimsave(gif_path, images, duration=1)
    print(f"GIF enregistré : {gif_path}")


def visualize_packing1_(df):
    list_carton = list(pd.unique(df['ID Carton']))
    for id_carton in list_carton:
        df_temp = df[df['ID Carton']==id_carton].copy()
        print("Articles : \n",df_temp.values)
        
        visualize_packing(df_temp, id_carton)

def visualize_packing_(df, id_carton=0):
    length = df['Longueur Carton (cm)'].iloc[0]
    width = df['Largeur Carton (cm)'].iloc[0]
    height = df['Hauteur Carton (cm)'].iloc[0]
    
    carton_length = max(length,width, height)
    carton_width = min(length,width, height)
    carton_height = np.median([length,width, height])

    articles = list(df[['Longueur Article (cm)','Largeur Article (cm)', 'Hauteur Article (cm)']].itertuples(index=False, name=None))

    # Attribuer des couleurs uniques aux dimensions uniques des articles
    unique_dims = list(set(articles))
    colors = plt.cm.get_cmap('hsv', len(unique_dims))
    color_dict = {dims: colors(i) for i, dims in enumerate(unique_dims)}
    
    # Fonction pour dessiner un parallélépipède avec bordure noire
    def draw_parallelepiped(ax, x, y, z, dx, dy, dz, color='b', alpha=0.5):
        vertices = [
            [x, y, z],
            [x+dx, y, z],
            [x+dx, y+dy, z],
            [x, y+dy, z],
            [x, y, z+dz],
            [x+dx, y, z+dz],
            [x+dx, y+dy, z+dz],
            [x, y+dy, z+dz]
        ]

        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [4, 7, 3, 0]
        ]

        poly3d = [[vertices[vert_id] for vert_id in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha))

    # Créer une nouvelle figure avec deux sous-graphiques
    fig = plt.figure(figsize=(14, 7))

    # Premier angle de vue
    ax1 = fig.add_subplot(121, projection='3d')
    # Dessiner le carton
    draw_parallelepiped(ax1, 0, 0, 0, carton_length, carton_width, carton_height, color='cyan', alpha=0.1)

    # Garder la trace de l'espace disponible dans le carton
    available_space = [(0, 0, 0, carton_length, carton_width, carton_height)]

    # Fonction pour trouver la meilleure position pour un article
    def find_best_position(article, spaces):
        best_space_index = -1
        max_difference = -1
        best_position = None

        for i, (sx, sy, sz, sl, sw, sh) in enumerate(spaces):
            # Vérifier si l'article peut rentrer dans cet espace
            if article[0] <= sl and article[1] <= sw and article[2] <= sh:
                # Calculer les différences d'espace
                diff_x = sl - article[0]
                diff_y = sw - article[1]
                diff_z = sh - article[2]
                min_diff = min(diff_x, diff_y, diff_z)

                if min_diff > max_difference:
                    max_difference = min_diff
                    best_space_index = i
                    best_position = (sx, sy, sz)

        return best_space_index, best_position

    # Placer les articles
    for i, article in enumerate(articles):
        best_space_index, best_position = find_best_position(article, available_space)

        if best_position is None:
            print(f"Article {i} ne rentre pas dans le carton.")
            break

        x_pos, y_pos, z_pos = best_position
        draw_parallelepiped(ax1, x_pos, y_pos, z_pos, article[0], article[1], article[2], color=color_dict[article])

        # Mise à jour des espaces disponibles
        sx, sy, sz, sl, sw, sh = available_space.pop(best_space_index)

        new_spaces = [
            (sx + article[0], sy, sz, sl - article[0], sw, sh),
            (sx, sy + article[1], sz, sl, sw - article[1], sh),
            (sx, sy, sz + article[2], sl, sw, sh - article[2])
        ]

        available_space.extend(new_spaces)

    # Définir les limites des axes
    ax1.set_xlim(0, carton_length)
    ax1.set_ylim(0, carton_width)
    ax1.set_zlim(0, carton_height)

    # Masquer les valeurs des axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])

    # Définir un angle de vue
    ax1.view_init(elev=20, azim=30)

    # Deuxième angle de vue
    ax2 = fig.add_subplot(122, projection='3d')
    # Dessiner le carton
    draw_parallelepiped(ax2, 0, 0, 0, carton_length, carton_width, carton_height, color='cyan', alpha=0.1)

    # Réinitialiser l'espace disponible et replacer les articles
    available_space = [(0, 0, 0, carton_length, carton_width, carton_height)]

    # Placer les articles
    for i, article in enumerate(articles):
        best_space_index, best_position = find_best_position(article, available_space)

        if best_position is None:
            print(f"Article {i} ne rentre pas dans le carton.")
            break

        x_pos, y_pos, z_pos = best_position
        draw_parallelepiped(ax2, x_pos, y_pos, z_pos, article[0], article[1], article[2], color=color_dict[article])

        # Mise à jour des espaces disponibles
        sx, sy, sz, sl, sw, sh = available_space.pop(best_space_index)

        new_spaces = [
            (sx + article[0], sy, sz, sl - article[0], sw, sh),
            (sx, sy + article[1], sz, sl, sw - article[1], sh),
            (sx, sy, sz + article[2], sl, sw, sh - article[2])
        ]

        available_space.extend(new_spaces)

    # Définir les limites des axes
    ax2.set_xlim(0, carton_length)
    ax2.set_ylim(0, carton_width)
    ax2.set_zlim(0, carton_height)

    # Masquer les valeurs des axes
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    # Définir un angle de vue différent
    ax2.view_init(elev=50, azim=120)

    # Ajouter un titre
    plt.suptitle(f'Visualisation de l\'emballage dans le carton {id_carton}')

    # Sauvegarder la visualisation en PNG
    plt.savefig(f"viz_carton{id_carton}", bbox_inches='tight')
    plt.show()

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