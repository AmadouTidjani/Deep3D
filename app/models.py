from datetime import datetime
from itsdangerous import URLSafeTimedSerializer as Serializer  
from app import db, login_manager, app
from flask_login import UserMixin


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    __tablename__ = 'User'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    is_admin = db.Column(db.Boolean, default=False)
    nom = db.Column(db.String(20), nullable=False)
    prenom = db.Column(db.String(20), nullable=False)
    adresse = db.Column(db.String(255))
    image = db.Column(db.String(20), default='default.jpg')
    mot_de_passe = db.Column(db.String(255), nullable=False)
    telephone = db.Column(db.Integer, default=None)
    email = db.Column(db.String(255), unique=True, nullable=False)
    code_enregistrement = db.Column(db.String(20), default=None)
    # Relation entre User et Article avec backref 'user'
    articles = db.relationship('Article', backref='User', lazy=True)
    commande = db.relationship('Commande', backref='user', lazy=True)
    adresses = db.relationship('Adresse', backref='user', lazy=True)

    def get_reset_token(self):
        s = Serializer(app.config['SECRET_KEY'])
        return s.dumps({'user_id': self.id})

    @staticmethod
    def verify_reset_token(token,  max_age=1800):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token,max_age=max_age)['user_id']
        except:
            return None
        return User.query.get(user_id)



    def __init__(self, is_admin, nom, prenom, adresse, image, mot_de_passe, telephone, email, code_enregistrement ):
        self.is_admin = is_admin
        self.nom = nom
        self.prenom = prenom
        self.adresse = adresse
        self.image = image 
        self.mot_de_passe = mot_de_passe
        self.telephone = telephone
        self.email = email
        self.code_enregistrement = code_enregistrement

    def __repr__(self):
        return f"User(is_admin={self.is_admin}, nom={self.nom}, prenom={self.prenom}, adresse={self.adresse}, image={self.image}, telephone={self.telephone}, email={self.email}, code_enregistrement={self.code_enregistrement})"

    def to_dict(self):
        return{
            "id": self.id,
            "is_admin": self.is_admin,
            "nom": self.nom,
            "prenom": self.prenom,
            "adresse": self.adresse,
            "image":self.image,
            "mot_de_passe": self.mot_de_passe,
            "telephone": self.telephone,
            "code_enregistrement": self.code_enregistrement
        }


class Article(db.Model):
    __tablename__ = 'Article'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('User.id'))
    sku = db.Column(db.String(30), nullable=False)
    largeur = db.Column(db.Float(precision=5), nullable=False)
    longueur = db.Column(db.Float(precision=5), nullable=False)
    hauteur = db.Column(db.Float(precision=5), nullable=False)
    poids = db.Column(db.Float(precision=5), nullable=False)
    quantite = db.Column(db.Integer(), nullable=False)
    fragile = db.Column(db.Boolean(), default=False) 

    def __init__(self, user_id,sku, largeur, longueur, hauteur, poids, quantite, fragile):
        self.user_id = user_id
        self.sku = sku
        self.largeur = largeur
        self.longueur = longueur
        self.hauteur = hauteur
        self.poids = poids
        self.quantite = quantite
        self.fragile = fragile

    def __repr__(self):
        return f"Article(id={self.id}, sku='{self.sku}', largeur={self.largeur}, longueur={self.longueur}, hauteur={self.hauteur}, poids={self.poids}, quantite={self.quantite}, fragile={self.fragile})"

    def to_dict(self):
        return {
            "id": self.id,
            "sku": self.sku,
            "largeur": float(self.largeur),
            "longueur": float(self.longueur),
            "hauteur": float(self.hauteur),
            "poids": float(self.poids),
            "quantite": self.quantite,
            "fragile": self.fragile
        }

class Adresse(db.Model):
    __tablename__ = 'Adresse'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer(),db.ForeignKey('User.id') , nullable=False)
    rue = db.Column(db.String(50), default=None)
    code_postal = db.Column(db.Integer(), default=None)
    pays = db.Column(db.String(50), nullable=False)
    ville = db.Column(db.String(50), nullable=False)

    def __init__(self, rue, code_postal, pays, ville):
        self.rue = rue
        self.code_postal = code_postal
        self.pays = pays
        self.ville = ville

    def __repr__(self):
        return f"Adresse(id={self.id}, rue={self.rue}, code_postal={self.code_postal}, pays={self.pays}, ville={self.ville})"

    def to_dict(self):
        return {
            "id": self.id,
            "rue": self.rue,
            "code_postal": self.code_postal,
            "pays": self.pays,
            "ville": self.ville
        }

association_table = db.Table('conteneur_commande',
    db.Column('conteneur_id', db.Integer, db.ForeignKey('Conteneur.id')),
    db.Column('commande_id', db.Integer, db.ForeignKey('Commande.id'))
)


class Commande(db.Model):
    __tablename__ = 'Commande'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('User.id'))
    quantite = db.Column(db.Integer(), nullable=False)
    date_commande = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    largeur = db.Column(db.Float(precision=2), nullable=False)
    longueur = db.Column(db.Float(precision=2), nullable=False)
    hauteur = db.Column(db.Float(precision=2), nullable=False)
    poids = db.Column(db.Float(precision=2), nullable=False)
    adresse = db.Column(db.String(120), nullable=False)
    conteneurs = db.relationship('Conteneur', secondary=association_table, back_populates='commandes')


    def __init__(self, user_id, quantite, largeur, longueur, hauteur, poids, paiement):
        self.user_id = user_id
        self.quantite = quantite
        self.largeur = largeur
        self.longueur = longueur
        self.hauteur = hauteur
        self.poids = poids
        self.paiement = paiement

    def __repr__(self):
        return f"Commande(id={self.id}, client_id={self.user_id}, quantite={self.quantite}, date_commande={self.date_commande}, largeur={self.largeur}, longueur={self.longueur}, hauteur={self.hauteur}, poids={self.poids}, paiement={self.paiement})"

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "quantite": self.quantite,
            "date_commande": self.date_commande,
            "largeur": self.largeur,
            "longueur": self.longueur,
            "hauteur": self.hauteur,
            "poids": self.poids,
            "paiement": self.paiement
        }
class Conteneur(db.Model):
    __tablename__ = 'Conteneur'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    type_conteneur = db.Column(db.String(50), nullable=False)
    largeur = db.Column(db.Float(precision=2), nullable=False)
    longueur = db.Column(db.Float(precision=2), nullable=False)
    hauteur = db.Column(db.Float(precision=2), nullable=False)
    Poid_maximal = db.Column(db.Float(precision=2), nullable=False)
    quantite = db.Column(db.Integer(), nullable=False)
    fragile = db.Column(db.Boolean(), default=None)
    commandes = db.relationship('Commande', secondary=association_table, back_populates='conteneurs')

    

    def __init__(self,type_conteneur, longeur, largeur, Poid_maximal, hauteur, prix, quantite):
        self.type_conteneur = type_conteneur
        self.longueur = longeur
        self.largeur = largeur
        self.poid_maximal = Poid_maximal
        self.hauteur = hauteur
        self.prix = prix
        self.quantite = quantite

    def __repr__(self):
        return f"Conteneur(IdContaineur={self.id}, TypeContaineur={self.type_conteneur}, Longeur={self.longueur}, Largeur={self.largeur}, PoidMaximal={self.poid_maximal}, hauteur={self.hauteur}, prix={self.prix}, quantite={self.quantite})"

    def to_dict(self):
        return {
            "id": self.id,
            "type_conteneur": self.type_conteneur,
            "Longeur": self.longueur,
            "Largeur": self.largeur,
            "PoidMaximal": self.poid_maximal,
            "hauteur": self.hauteur,
            "prix": self.prix,
            "quantite": self.quantite
        }



    
'''class Paiement(db.Model):
    __tablename__ = 'Paiement'
    id = db.Column(db.Integer, primary_key=True)
    montant = db.Column(db.Float)
    methode = db.Column(db.String(50))
    status = db.Column(db.String(50))
    commande = db.relationship('Commande', back_populates='paiement')

    def __init__(self, montant, methode, status):
        self.montant = montant
        self.methode = methode
        self.status = status

    def __repr__(self):
        return f"Paiement(id={self.id}, montant={self.montant}, methode={self.methode}, status={self.status})"

    def to_dict(self):
        return {
            "id": self.id,
            "montant": self.montant,
            "methode": self.methode,
            "status": self.status
        }'''
    
'''class Admin(db.Model):
    __tablename__ = 'Admin'
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.Integer(), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    mot_de_passe = db.Column(db.String(100))
    conteneur = db.relationship('Conteneur', backref='admin',lazy=True)

    def __init__(self, nom, email, motDePasse):
        self.nom = nom
        self.email = email
        self.motDePasse = motDePasse

    def __repr__(self):
        return f"Admin(id={self.id}, nom={self.nom}, login={self.email})"

    def to_dict(self):
        return {
            "idAdmin": self.idAdmin,
            "nom": self.nom,
            "login": self.email
        }'''


'''class AlgorithmeBinPacking(db.Model):
    __tablename__ = 'AlgorithmeBinPacking'
    id = db.Column(db.Integer, primary_key=True)
    list_conteneur = db.Column(db.List())
    list_article = db.Column(db.List())
    conteneur = db.relationship('Conteneur', backref='algorithme_bin_packing')
  )

    def __init__(self, conteneur, article):
        self.conteneur = conteneur
        self.article = article

    def __repr__(self):
        return f"AlgorithmeBinPacking(conteneur={self.conteneur}, article={self.article})"

    def to_dict(self):
        return {
            "conteneur": self.conteneur.to_dict(),
            "article": self.article.to_dict()
        }'''
