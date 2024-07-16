import numpy as np
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

# Exemple d'utilisation avec des DataFrames df_article et df_carton
# df_article = pd.read_csv('/path/to/your/article/file.csv')
# df_carton = pd.read_csv('/path/to/your/carton/file.csv')
# pred = some_prediction_data
# result_df = view(pred, df_article, df_carton)
# print(result_df)
"""

def view(pred, df_article, df_carton): 

    if isinstance(pred, pd.DataFrame):
        res = pred.copy()
    else :
        res = pd.DataFrame(pred, columns = ["id_carton"])
        res["id_article"] = res.index
    
    df_article["id"] = df_article.index
    res = res.join(df_article.set_index('id'), on='id_article')
    res["article_volume"] = res['Longueur'] * res['Largeur']  * res['Hauteur']* res["Quantite"]
    res["Poids_Qte"] = res["Poids"]*res["Quantite"]
    print("columns ; ", res.columns)

    #res = res[["id_article", "id_carton", "article_volume", "Poids", "Poids_Qte"]]
    res["cumul_volume"] = res.groupby("id_carton")["article_volume"].transform("sum")
    res["cumul_poids"] = res.groupby("id_carton")["Poids_Qte"].transform("sum")
    df_carton["id"] = df_carton.index
    
    #"
    res = res.rename(columns = {
        'Longueur': 'Longueur Item (cm)',
        'Largeur': 'Largeur Item (cm)',
        'Hauteur': 'Hauteur Item (cm)',
        'Poids': 'Poids Item (kg)',
    })
    #"
    print("columns ; ", res.columns)
    print("columns ; ", df_article.columns)
    print("columns ; ", df_carton.columns)
    
    res = res.join(df_carton.set_index('id'), on='id_carton')
    res["box_volume"] = res['Longueur'] * res['Largeur']  * res['Hauteur']
    
    # calcul des indicateurs 
    res["esp_inocc"] = np.round(100*(res["box_volume"] - res["cumul_volume"])/res["box_volume"],2)
    res["poids_inocc"] = np.round(100*(res["Poids_max"] - res["cumul_poids"])/res["Poids_max"],2)

    list_def = ["id_article","id_carton","box_volume","article_volume", "cumul_volume", "esp_inocc", "cumul_poids", "Poids_max", "poids_inocc"]
    #list_def = ["id_article","id_carton", "esp_inocc", "poids_inocc"]
    #print("Descrition du resultat - Emballages : \n",
    #res[list_def])
    
    res = res[list_def]
    #"
    res = res.rename(columns = {'id_article': 'ID article',
                                'id_carton': 'ID Bin',
                                'box_volume': "Bin's volume",
                                "article_volume" : "Item's volume",
                                "cumul_volume" : "Items's volume",
                                "esp_inocc": "Espace inoccupé",
                                "cumul_poids" : "Items's weight",
                                "Poids_max": "Max Weight",
                                "poids_inocc": "Less weight"})
    #"
    #res.columns = [" ID Item "," ID Bin "," Volume Bin "," Volume Item ", " Volume all Items ", " Espace non occupé ", " Poids restant (perc) "]
    return res# [list_def] #res[["id_article","id_carton","box_volume", "article_volume", "esp_inocc"]].head())
"""
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
                print("Aucun carton disponible pour emballer les articles : ", list(df_group.index))
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
    