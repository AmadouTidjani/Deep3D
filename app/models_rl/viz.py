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
    print("res : ", res[["Poids_Qte","cumul_poids", "id_carton"]])
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
        df_temp = df[df['ID Carton']==id_carton].copy()
        print("Articles : \n",df_temp.values)
        
        visualize_packing(df_temp, id_carton)

def visualize_packing(df, id_carton=0):
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

class Bin:

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
    