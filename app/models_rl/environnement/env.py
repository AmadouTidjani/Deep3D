
import numpy as np

def median_index(lst):
    # Create a list of tuples (element, index)
    indexed_lst = [(value, index) for index, value in enumerate(lst)]
    # Sort the list by the elements
    sorted_lst = sorted(indexed_lst, key=lambda x: x[0])
    # Return the index of the median element
    return sorted_lst[1][1]

class Environment:
    def __init__(self, articles_df, cartons_df):
        self.articles_df = articles_df
        self.cartons_df = cartons_df

        self.items_init = articles_df
        self.bins_init = cartons_df
        
        self.current_article_index = 0
        self.current_carton = None
        
    def reset(self):

        # update data
        self.articles_df = self.items_init.copy()
        self.cartons_df = self.bins_init.copy()

        # Reset episode by selecting a new article
        self.current_article_index = 0
        self.current_carton = None
        #article = self.articles_df.iloc[self.current_article_index]
        return self.current_article_index #self._encode_state(article)
    
    def step(self, action):
        # Get current article and available cartons
        article = self.articles_df.iloc[self.current_article_index]
        available_cartons = self.cartons_df.copy()
 
        # Validate action (ensure chosen carton can fit the article) [inf, inf]
        
        if action < 0 or action >= len(available_cartons):
            reward = 0  # Penalty for invalid action
            done = True
            info = {"message": "Invalid carton selection"}
            lost_space = np.inf
            box_volume = 0
            article_volume = 0
            #return self._encode_state(article), reward, lost_space, box_volume, article_volume, done, info
            return self.current_article_index, reward, lost_space, box_volume, article_volume, done, info
        
        chosen_carton = available_cartons.iloc[action]
        
        # Check if article fits in the carton (dimensions and weight)
        if not self._fits_in_carton(article, chosen_carton):
            reward = 0  # Penalty for selecting an unsuitable carton
            done = False
            info = {"message": "Article ne convient pas au carton"}
            #print("====== Not fit \narticle : \n", article)
            #print("carton : \n", chosen_carton, "\n\n============================\n")
            lost_space = np.inf
            box_volume = 0
            article_volume = 0
            #return self._encode_state(article), reward, lost_space, box_volume, article_volume, done, info
            return self.current_article_index, reward, lost_space, box_volume, article_volume, done, info
        
        # Calculate reward based on wasted space
        #temp_carton_volume = chosen_carton["Longueur"] * chosen_carton["Largeur"] * chosen_carton["Hauteur"]
        # Calcul reward
        box_volume = chosen_carton['Longueur'] * chosen_carton['Largeur']  * chosen_carton['Hauteur']
        article_volume = article['Longueur'] * article['Largeur']  * article['Hauteur']* article["Quantite"]
        lost_space = box_volume - article_volume
        #wasted_space = temp_carton_volume - \
        #           article["Longueur"] * article["Largeur"] * article["Hauteur"] * article["Quantite"]
        reward = 100 *(1 -  (lost_space / box_volume)) 
        #reward = 100 * (1 - wasted_space/temp_carton_volume)  # Higher reward for less wasted space
       
        
        # Check if all articles are packed
        self.current_article_index += 1
        #done = True
        done = self.current_article_index >= len(self.articles_df)
        self.current_carton = chosen_carton if not done else None
        
        ## Redimensionnement du carton choisi : *   
        values_article = [article['Longueur'], article['Largeur'], article['Hauteur']]
        val_quant = article['Quantite']
        values_article = [float(val_quant*el) for el in values_article]
    
        values_carton = [chosen_carton['Longueur'], chosen_carton['Largeur'], chosen_carton['Hauteur']]
        # Trier les valeurs
        values_article = sorted(values_article)
        values_carton = sorted(values_carton)
        values_result = [float(values_carton[i] - values_article[i]) for i in range(len(values_article))]
        idx_max = values_result.index(max(values_result))
        idx_min = values_result.index(min(values_result))
        #idx_moyen = 3 - idx_max - idx_min
        idx_moyen = median_index(values_result)
        
        ############################################################
        ### Very important : bin sizes update 
        
        self.cartons_df.iloc[action,:] = [values_result[idx_max],values_carton[idx_moyen], values_carton[idx_min],
                                          float(chosen_carton["Poids_max"]) - float(article['Quantite']*article['Poids']),\
                                          float(chosen_carton["Prix"]),\
                                          int(chosen_carton["Quantite"]),\
                                          chosen_carton["Type"]
                                        ]
        
        #return self._encode_state(article if not done else None), reward, lost_space, box_volume, article_volume, done, {}
        return self.current_article_index, reward, lost_space, box_volume, article_volume, done, {}
    
    
    def get_carton_optimal(self, article):
        # Implement an algorithm to find the optimal carton for the article
        # This could be a greedy search, dynamic programming, or other methods
        # For now, we'll just return a random suitable carton

        suitable_cartons = self.cartons_df[self._fits_in_carton(article, self.cartons_df)]
        if len(suitable_cartons) > 0:
            return suitable_cartons.sample(1).iloc[0]
        else:
            return None  # No suitable carton found
        
        
    def _encode_state(self, article):
        # Encode the state as a vector including relevant information
        if article is None:
            return np.zeros(len(self.articles_df.columns))  # All articles packed
        else:
            return article.values #np.concatenate((article.values, np.array([len(self.available_cartons())])))
        
        
    def _fits_in_carton(self, article, carton):

        article_L_l_h = float(article['Quantite'])*np.array(np.array(article[["Longueur", "Largeur", "Hauteur"]] ))
        carton_L_l_h = np.array(np.array(carton[["Longueur", "Largeur", "Hauteur"]] ))
        
        article_L_l_h = np.sort(article_L_l_h)
        carton_L_l_h = np.sort(carton_L_l_h)

        condition_dimensions = np.sort(carton_L_l_h) > np.sort(article_L_l_h)
        # index_sort_dim_article = np.argsort(article_L_l_h)
        # index_sort_dim_carton = np.argsort(carton_L_l_h)
        if False in condition_dimensions:
            return False

        return (article["Poids"] * article["Quantite"] <= carton["Poids_max"]) #and (carton["Quantite"] == 0)
        
    
    
    def available_cartons(self):
        # Return a list of available carton indices based on current state
        if self.current_carton is not None:
            return [i for i, carton in self.cartons_df.iterrows() if carton.name != self.current_carton.name]
        
        else:
            return range(len(self.cartons_df))
        
    def packed_articles(self):

        # followed articles by the algorithm
        return list(range(self.current_article_index))
    
    def return_id_articles(self):

        return self.current_article_index - 1 ## apres la fonction step qui incrémente
    """
    def return_id(self, state):
        concat_values = ''.join(map(str, state))
        idx = self.articles_df.index[self.articles_df.apply(lambda x: ''.join(map(str, x)), axis=1) == concat_values].tolist()

        if idx:
            return idx[0]
        else:
            return None 
    """

    def get_state_size(self):

        return len(self.articles_df)

    def items_data(self, id):

        article = self.articles_df.iloc[id]
        return self._encode_state(article)