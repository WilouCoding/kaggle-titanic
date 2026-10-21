# Titanic : prédire la survie des passagers

Projet d'apprentissage sur la compétition Kaggle, et du ML en général.
[Titanic](https://www.kaggle.com/competitions/titanic).
Objectif : prédire quels passagers ont survécu à partir de leurs caractéristiques.

## Approche

- Exploration des données avec pandas (valeurs manquantes, taux de survie par groupe)
- Prétraitement avec un `Pipeline` et un `ColumnTransformer` scikit-learn (imputation, encodage one-hot), pour éviter toute fuite de données
- Modèles comparés : rd forest, régression logistique, XGBoost
- Évaluation par validation croisée (5 plis)
- Feature engineering : titre extrait du nom, taille de famille, indicateur « voyage seul »

## Résultats

| Version                    | Validation croisée | Score Kaggle |
| Colonnes de base + XGBoost | 0.826 ± 0.017      | 0.76794 |
| Avec feature engineering   | 0.836 ± 0.019     | 0.75598 |

## Ce que j'ai appris

- L'écart entre validation croisée et score public s'explique par la petite taille du jeu de test (418 passagers : un passager vaut environ 0,0024 de précision).
- On peut pense s'amélioré en augmentant les informations pour le modèlemais obtenir des moindres résultats.

## Lancer le projet

1. Télécharger `train.csv` et `test.csv` sur la page Kaggle de la compétition et les placer dans un dossier `data/`.
2. Installer les dépendances : pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn.
3. Ouvrir `notebooks/TitanicModel.ipynb`.