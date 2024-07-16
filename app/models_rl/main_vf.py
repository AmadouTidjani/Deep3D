import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import collections
import warnings
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app/models_rl'))
#from environnement.env import Environment
from app.models_rl.environnement.env import Environment
#from dqnet.agent import DQNAgent
from app.models_rl.dqnet.agent import DQNAgent
import tensorflow as tf
import numpy as np
from time import time as t
from time import sleep
#from viz import *
from app.models_rl.viz import *

sys.path.append(os.path.join(os.path.dirname(__file__), 'gitviz'))
#from gitviz.test import pack_viz
from app.models_rl.gitviz.test import pack_viz
# Ignorer les avertissements "SettingWithCopyWarning"
warnings.filterwarnings('ignore')


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



def train1(env, agent, action_size, episodes = 20, batch_size =2, plot=True):

    state_size = state_size = len(env.items_data(0))  #env.get_state_size()
    #action_size = len(cartons_df)
    scores = []
    times = []
    for e in range(episodes):
        state = env.reset()
        val_state = env.items_data(state)
        val_state = tf.convert_to_tensor(val_state, dtype=tf.float32) 
        val_state = np.reshape(val_state, [1, state_size])
        done = False
        
        constraint = False
        action_contraint = 0
        start = t()
        score = []
        print("ietération: ", e)
        while not done:
            if constraint:
                action = action_contraint
            else:
                action = agent.act(val_state)
            
            print("action en cours: ", action)
            print("state : " , state)
            next_state, reward, lost_space, box_volume, article_volume, done, info = env.step(action)
            print("next_state : " , next_state)
            print("done : ", done)
            
            if len(info)==0:
                action_contraint = action
                constraint = True
                score.append(reward)
                print("ok pour l'article : ", state, " dans carton ", action)
            else : 
                constraint = False
            
            
            if not done :
                val_next_state = env.items_data(next_state)
                val_next_state = tf.convert_to_tensor(val_next_state, dtype=tf.float32)
                val_next_state = np.reshape(val_next_state, [1, state_size])
                agent.remember(val_state, action, reward, val_next_state, done)
            else:
                print(f"\nEpisode: {e}/{episodes}, Score: {np.mean(score)}, e: {agent.epsilon:.2}\n")
                break
            
            #agent.remember(val_state, action, reward, val_next_state, done)
            state = next_state
            val_state = val_next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
        scores.append(np.mean(score))
        times.append(t()-start)
        if e >= 20 and e % 5 :
            savepath = "save/model_" + str (e) + ".weights.h5" 
            agent.save(savepath)
    agent.save("save/model.weights.h5") 
    # Afficher les graphiques de performance
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(scores)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Espace occupé par episode')
        plt.savefig('save/train.png')
        #plt.show()
        
        plt.figure(figsize=(12, 6))
        plt.plot(times)
        plt.xlabel('Episodes')
        plt.ylabel('Time (s)')
        plt.title('Temps par episode')
        plt.savefig('save/train_times.png')
        #plt.show()

from time import time

def train(env, agent, action_size, episodes = 20, batch_size =2, plot=True):
    state_size = len(env.items_data(0))
    scores = []
    times = []
    act_times = []
    step_times = []
    remember_times = []
    replay_times = []

    for e in range(episodes):
        state = env.reset()
        val_state = env.items_data(state)
        val_state = tf.convert_to_tensor(val_state, dtype=tf.float32) 
        val_state = np.reshape(val_state, [1, state_size])
        done = False
        
        constraint = False
        action_contraint = 0
        start = time()
        score = 0
        list_states = list(range(action_size))
        while not done:
            if constraint:
                action = action_contraint
            else:
                start_act = time()
                #action = agent.act(val_state)
                action = agent.act1(val_state, list_states)
                act_times.append(time() - start_act)
            
            start_step = time()
            next_state, reward, lost_space, box_volume, article_volume, done, info = env.step(action)
            step_times.append(time() - start_step)
            
            if len(info) == 0:
                action_contraint = action
                constraint = True
                score += reward
                list_states = list(range(action_size))
            else:
                if action in list_states:list_states.remove(action)
                constraint = False
            
            if not done:
                val_next_state = env.items_data(next_state)
                val_next_state = tf.convert_to_tensor(val_next_state, dtype=tf.float32)
                val_next_state = np.reshape(val_next_state, [1, state_size])
                start_remember = time()
                agent.remember(val_state, action, reward, val_next_state, done)
                remember_times.append(time() - start_remember)
            else:
                print(f"\nEpisode: {e+1}/{episodes}, Score: {np.round(score,2)}, e: {agent.epsilon:.2}\n")
                
            
            state = next_state
            val_state = val_next_state

            if len(agent.memory) > batch_size:
                start_replay = time()
                agent.replay(batch_size)
                replay_times.append(time() - start_replay)
            
        scores.append(score)
        times.append(time() - start)
        if e >= 10 and e % 5 == 0:
            savepath = "save/model_" + str(e) + ".weights.h5" 
            agent.save(savepath)
    
    agent.save("save/model.weights.h5")
    
    # Afficher les graphiques de performance
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(scores)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Espace occupé par episode')
        plt.savefig('save/train.png')
        
        plt.figure(figsize=(12, 6))
        plt.plot(times)
        plt.xlabel('Episodes')
        plt.ylabel('Time (s)')
        plt.title('Temps par episode')
        plt.savefig('save/train_times.png')
        
        plt.figure(figsize=(12, 6))
        plt.plot(act_times)
        plt.xlabel('Steps')
        plt.ylabel('Time (s)')
        plt.title('Time for agent.act')
        plt.savefig('save/act_times.png')
        
        plt.figure(figsize=(12, 6))
        plt.plot(step_times)
        plt.xlabel('Steps')
        plt.ylabel('Time (s)')
        plt.title('Time for env.step')
        plt.savefig('save/step_times.png')
        
        plt.figure(figsize=(12, 6))
        plt.plot(remember_times)
        plt.xlabel('Steps')
        plt.ylabel('Time (s)')
        plt.title('Time for agent.remember')
        plt.savefig('save/remember_times.png')
        
        plt.figure(figsize=(12, 6))
        plt.plot(replay_times)
        plt.xlabel('Steps')
        plt.ylabel('Time (s)')
        plt.title('Time for agent.replay')
        plt.savefig('save/replay_times.png')

def evaluate(env, agent, state_size):
    # Use the trained model to make decisions in the environment
    state = env.reset()
    val_state = env.items_data(state)
    val_state = tf.convert_to_tensor(val_state, dtype=tf.float32)
    val_state = np.reshape(val_state, [1, state_size])
    done = False
    constraint = False
    action_contraint = 0
    matching = []
    tested_actions = list(range(action_size))
    
    while not done:
        if constraint:
            action = action_contraint
        else:
            action = agent.act1(val_state, tested_actions)
            #print("valeurs à tester : ", tested_actions)
            #action = agent.act(val_state)

        next_state, reward, lost_space, box_volume, article_volume, done, info = env.step(action)
        
        if len(info)==0:
            
            action_contraint = action
            constraint = True
            matching.append([state, action])
            tested_actions = list(range(action_size))
        else : 
            constraint = False
            if action in tested_actions:tested_actions.remove(action)
        
        state = next_state  # Update current state

    return pd.DataFrame(matching, columns = ['id_article', 'id_carton'])


def test(env, agent, action_size, episodes = 10, plot = True):
    state_size = len(env.items_data(0))
    #action_size = len(test_cartons_df)
    Q = np.zeros((state_size, action_size))

    # Tester l'agent avec les données de test
    scores = []
    times = []

    for e in range(episodes):
        start = t()
        state = env.reset()
        val_state = env.items_data(state)
        val_state = tf.convert_to_tensor(val_state, dtype=tf.float32)
        val_state = np.reshape(val_state, [1, state_size])
        score = []

        done = False
        constraint = False
        action_contraint = 0

        while not done:
            if constraint:
                action = action_contraint
            else:
                action = agent.act(val_state)
            
            next_state, reward, lost_space, box_volume, article_volume, done, info = env.step(action)
            
            if len(info)==0:
                Q[state, action]+=1
                action_contraint = action
                score.append(reward)
                constraint = True
            else : 
                constraint = False
            
            if not done:
                next_val_state = env.items_data(next_state)
                next_val_state = tf.convert_to_tensor(next_val_state, dtype=tf.float32)
                next_val_state = np.reshape(next_val_state, [1, state_size])
                val_state = next_val_state
            else:
                scores.append(np.mean(score))
                print(f"Episode: {e}/{episodes}, Score: {score}")
                break
        times.append(t()-start)

    # Calculer la moyenne des scores
    average_score = np.mean(scores)
    print(f"\n\nAverage Score over {episodes} episodes: {average_score}")
    
    
    # Afficher les graphiques de performance
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(scores)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Espace occupé par episode')
        plt.savefig('save/test.png')
        #plt.show()
        
        plt.figure(figsize=(12, 6))
        plt.plot(times)
        plt.xlabel('Episodes')
        plt.ylabel('Time (s)')
        plt.title('Temps par episode')
        plt.savefig('save/times.png')
        #plt.show()

    
    return Q.argmax(1)



if __name__ == '__main__':

    args = parser.parse_args()

    if args.train:
        df_article = pd.read_csv("data/articles_data.csv")
        new_names = {'longueur': 'Longueur', 'largeur': 'Largeur',
        "hauteur": "Hauteur", "poids": "Poids", "quantite": "Quantite"}
        df_article = df_article.rename(columns=new_names)
        df_article = df_article[['Longueur', 'Largeur', 'Hauteur', 'Poids', 'Quantite']]
        df_article = df_article.loc[df_article.index.repeat(df_article['Quantite'])].reset_index(drop=True)
        df_article['Quantite'] = 1
        print( "Nombre Articles : ", len(df_article)) 
        print( "Nombre Articles : \n", df_article)
        #sleep(10000) 
        df_article = df_article[:args.nb_article]
        df_carton = pd.read_csv("data/bins.csv") # conteneurs_data
        df_carton = df_carton[['Longueur', 'Largeur', 'Hauteur', 'Poids_max','Prix', 'Quantite', 'Type']]
        #df_carton = df_carton[1:]
        print( "Nombre Cartons : ", len(df_carton)) 


        # Initialiser l'env
        print("+++++++ ENVIRONNEMENT ++++++++++")
        env = Environment( df_article , df_carton)

        state_size = state_size = len(env.items_data(0))  #env.get_state_size()
        action_size = len(df_carton)
        agent = DQNAgent(state_size, action_size, args)

        train(env, agent, action_size, episodes=args.episodes)
    
    if args.test and not args.train:
        print("=============== Test modèle : ====================\n\n")
        """
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
        print(df_key)
        #===========================================================
        df_article = df_article[['Longueur', 'Largeur', 'Hauteur', 'Poids', 'Quantite']]
        
        df_article = df_article[:args.nb_article]
        df_article = df_article.loc[df_article.index.repeat(df_article['Quantite'])].reset_index(drop=True)
        df_article['Quantite'] = 1
        print( "Nombre Articles : ", len(df_article)) 

        df_carton = pd.read_csv("data/bins.csv") # conteneurs_data
        df_carton = df_carton[['Longueur', 'Largeur', 'Hauteur', 'Poids_max','Prix', 'Quantite', 'Type']]
        #for col in ['Poids_max']:
        #df_carton[col] = 4.5*df_carton[col]
        #df_carton = df_carton[1:]
        print( "Nombre Cartons : ", len(df_carton)) 


        # Initialiser l'env
        print("+++++++ ENVIRONNEMENT ++++++++++")
        
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
        
        viz_result = view(pred, df_article, df_carton)
        #print(viz_result)
        #print("===========")

        ### BIN PACK
        bin = Bin(df_article, df_carton)
        res = bin.pack()
        
        #=====================================================================
        res = res.merge(df_key, how = 'left',left_on = ['Longueur Article (cm)',
       'Largeur Article (cm)', 'Hauteur Article (cm)', 'Poids Article (kg)'],
       right_on = ['Longueur_key','Largeur_key', 'Hauteur_key', 'Poids_key'])
        
        res = res.drop_duplicates(subset = ["key","ID Bin"])
        res = res[
            ["sku", 'ID Bin', 'Longueur Article (cm)',
       'Largeur Article (cm)', 'Hauteur Article (cm)', 'Poids Article (kg)',
       'Quantite Article', "Volume Article",
       "Volume Articles", "Poids Articles", "Longueur", 'Largeur', 'Hauteur',
       'Max Weight', 'Prix', 'Quantite', 'Type', "Volume Carton",
       'Espace inoccupé', 'Poids inoccupé', 'fragile']
        ]
        #======================================================================
        print(res)

        ### 
        #df_carton['id_carton'] = df_carton.index
        #df_article['id_article'] = df_article.index
        #pack_viz(viz_result, df_article, df_carton)"""