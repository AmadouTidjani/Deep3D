import random
from collections import deque
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, args):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=args.maxlen)
        self.gamma = args.gamma   # discount rate
        self.epsilon = args.epsilon  # exploration rate
        self.epoch = args.episode
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.learning_rate
        self.model = self._build_model()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.losses = []
    
    def _build_model(self):
        """Construit le modèle de réseau de neurones pour la DQN."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke la transition (state, action, reward, next_state, done) dans la mémoire."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Retourne l'action à prendre selon l'état actuel en utilisant epsilon-greedy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def act1(self, state, ma_liste):
        """Retourne une action en tenant compte d'une liste d'actions disponibles."""
        if ma_liste:
            return random.choice(ma_liste)

        act_values = self.model.predict(state)
        sorted_actions = np.argsort(act_values[0])[::-1]  # Trier les actions par valeur prédite décroissante
        for action in sorted_actions:
            if action in ma_liste:
                return action
        return sorted_actions[0]  # Retourner la meilleure action disponible
    
    def replay(self, batch_size):
        """Entraîne le modèle en utilisant des mini-lots de la mémoire."""
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in minibatch:
            # Prédiction du modèle sur l'état actuel
            target_f = self.model.predict(state)
            
            # Si l'épisode est terminé, utiliser juste la récompense
            if done:
                target_f[0][action] = reward
            else:
                # Calculer la cible en utilisant la récompense et le futur état
                q_future = np.amax(self.model.predict(next_state)[0])
                target_f[0][action] = reward + self.gamma * q_future
            
            # Entraîner le modèle sur cet état avec la cible modifiée
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            
            # Calculer et stocker la perte moyenne sur le mini-lot
            loss = history.history['loss'][0]
            total_loss += loss
        
        # Calculer la perte moyenne sur l'ensemble du mini-lot
        self.losses.append(total_loss / batch_size)
        
        # Réduire epsilon pour diminuer l'exploration au fil du temps
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_loss(self, path):
        """Sauvegarde les pertes calculées au cours de l'entraînement."""
        np.save(path, self.losses)
        return self.losses
    
    def load(self, name):
        """Charge les poids d'un modèle sauvegardé."""
        self.model.load_weights(name)
    
    def save(self, name):
        """Sauvegarde les poids du modèle."""
        self.model.save_weights(name)


# =========================================
###########################################################
############################################################
import random
from collections import deque
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from time import time, sleep
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size,args):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=args.maxlen)
        self.gamma = args.gamma   # discount rate
        self.epsilon = args.epsilon  # exploration rate
        self.epoch = args.episode
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.learning_rate
        self.model = self._build_model()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.losses = []
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def act1(self, state, ma_liste):
        
        if ma_liste:#np.random.rand() <= self.epsilon:
            return random.choice(ma_liste)
            #possible_actions = list(set(range(self.action_size)) - set(excluded_actions))
            """
            if possible_actions:
                return random.choice(possible_actions)
            else:
                return random.randrange(self.action_size)
            """
        act_values = self.model.predict(state)
        sorted_actions = np.argsort(act_values[0])[::-1]  # Sort actions by predicted value (descending)
        for action in sorted_actions:
            if action in ma_liste:
                return action
        return sorted_actions[0]  # Return the best available action if all preferred actions are excluded
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=1)
            total_loss += self.mse(target, target_f)
        self.losses.append(total_loss.numpy()/batch_size)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def save_loss(self, path):
        np.save(path, self.losses)
        return self.losses
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)

